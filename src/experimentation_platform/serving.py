"""Serving API (FastAPI): experiments, assignment, ingestion, readouts.

A thin layer over the same objects the library and CLI use — the API
re-implements nothing. State is in-memory (registry + event store),
which is the right scope for a reference implementation: a production
deployment puts the registry in a config service and the event store
behind a stream processor, without changing any semantics here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .assignment import AssignmentEngine
from .errors import ExperimentationError
from .events import Event, EventStore
from .metrics import compute_all_metrics
from .models import (
    EligibilityRule,
    Experiment,
    Lifecycle,
    MetricKind,
    MetricSpec,
    Variant,
)
from .readout import build_readout
from .registry import ExperimentRegistry
from .simulator import build_demo_experiment
from .stats import required_n_proportions, srm_check


class VariantIn(BaseModel):
    name: str
    weight: float
    is_control: bool = False


class MetricIn(BaseModel):
    name: str
    kind: str = "conversion"
    is_primary: bool = False
    is_guardrail: bool = False
    higher_is_better: bool = True
    guardrail_tolerance: float = 0.0


class EligibilityIn(BaseModel):
    attribute: str
    operator: str
    value: Any = None


class ExperimentIn(BaseModel):
    id: str
    name: str
    hypothesis: str = ""
    variants: list[VariantIn]
    metrics: list[MetricIn]
    salt: str = ""
    layer: str | None = None
    allocation_fraction: float = 1.0
    eligibility: list[EligibilityIn] = Field(default_factory=list)


class TransitionIn(BaseModel):
    target: str


class AssignIn(BaseModel):
    unit_id: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class EventIn(BaseModel):
    event_id: str
    unit_id: str
    variant: str
    event_name: str
    timestamp: datetime | None = None
    value: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


def _experiment_from(req: ExperimentIn) -> Experiment:
    return Experiment(
        id=req.id,
        name=req.name,
        hypothesis=req.hypothesis,
        variants=tuple(
            Variant(v.name, v.weight, v.is_control) for v in req.variants
        ),
        metrics=tuple(
            MetricSpec(
                name=m.name,
                kind=MetricKind(m.kind),
                is_primary=m.is_primary,
                is_guardrail=m.is_guardrail,
                higher_is_better=m.higher_is_better,
                guardrail_tolerance=m.guardrail_tolerance,
            )
            for m in req.metrics
        ),
        salt=req.salt,
        layer=req.layer,
        allocation_fraction=req.allocation_fraction,
        eligibility=tuple(
            EligibilityRule(e.attribute, e.operator, e.value)
            for e in req.eligibility
        ),
    )


def _results_payload(experiment: Experiment, store: EventStore) -> dict[str, Any]:
    results = compute_all_metrics(experiment, store)
    payload: dict[str, Any] = {}
    for name, res in results.items():
        payload[name] = {
            "kind": res.metric.kind.value,
            "per_variant": {
                v: {
                    "n_exposed": s.n_exposed,
                    "n_converted": s.n_converted,
                    "rate": s.rate,
                    "mean": s.mean,
                    "variance": s.variance,
                }
                for v, s in res.per_variant.items()
            },
            "comparisons": [
                {
                    "treatment": c.treatment,
                    "control": c.control,
                    "absolute_effect": c.absolute_effect,
                    "relative_effect": c.result.relative_effect,
                    "p_value": c.p_value,
                    "significant": c.significant,
                    "ci_low": c.result.ci_low,
                    "ci_high": c.result.ci_high,
                }
                for c in res.comparisons
            ],
        }
    return payload


def create_app(
    registry: ExperimentRegistry | None = None,
    store: EventStore | None = None,
) -> FastAPI:
    registry = registry or ExperimentRegistry()
    store = store or EventStore()
    engine = AssignmentEngine()
    app = FastAPI(title="Experimentation Platform", version="0.1.0")

    def get_or_404(experiment_id: str) -> Experiment:
        try:
            return registry.get(experiment_id)
        except ExperimentationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "experiments": [e.id for e in registry.list()],
            "events": len(store),
        }

    @app.post("/experiments", status_code=201)
    def create_experiment(req: ExperimentIn) -> dict[str, Any]:
        try:
            experiment = registry.register(_experiment_from(req))
        except ExperimentationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return experiment.to_dict()

    @app.get("/experiments")
    def list_experiments() -> dict[str, Any]:
        return {"experiments": [e.to_dict() for e in registry.list()]}

    @app.get("/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        return get_or_404(experiment_id).to_dict()

    @app.post("/experiments/{experiment_id}/transition")
    def transition(experiment_id: str, req: TransitionIn) -> dict[str, Any]:
        experiment = get_or_404(experiment_id)
        try:
            experiment.transition(Lifecycle(req.target))
        except (ExperimentationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return experiment.to_dict()

    @app.post("/experiments/{experiment_id}/assign")
    def assign(experiment_id: str, req: AssignIn) -> dict[str, Any]:
        experiment = get_or_404(experiment_id)
        variant = engine.assign(experiment, req.unit_id, req.attributes)
        return {
            "experiment_id": experiment_id,
            "unit_id": req.unit_id,
            "variant": variant,
        }

    @app.post("/experiments/{experiment_id}/events")
    def ingest_event(experiment_id: str, req: EventIn) -> dict[str, Any]:
        get_or_404(experiment_id)
        result = store.append(
            Event(
                event_id=req.event_id,
                experiment_id=experiment_id,
                unit_id=req.unit_id,
                variant=req.variant,
                event_name=req.event_name,
                timestamp=req.timestamp or datetime.now(timezone.utc),
                value=req.value,
                attributes=req.attributes,
            )
        )
        return {
            "accepted": result.accepted,
            "duplicate": result.duplicate,
            "late": result.late,
        }

    @app.get("/experiments/{experiment_id}/results")
    def results(experiment_id: str) -> dict[str, Any]:
        experiment = get_or_404(experiment_id)
        return {"experiment_id": experiment_id,
                "results": _results_payload(experiment, store)}

    @app.get("/experiments/{experiment_id}/readout")
    def readout(
        experiment_id: str,
        baseline_rate: float = 0.10,
        mde: float = 0.02,
        looks: int = 1,
    ) -> dict[str, Any]:
        experiment = get_or_404(experiment_id)
        results_map = compute_all_metrics(experiment, store)
        primary = results_map[experiment.primary_metric.name]
        observed = [
            primary.per_variant[v.name].n_exposed for v in experiment.variants
        ]
        srm = (
            srm_check(observed, [v.weight for v in experiment.variants])
            if sum(observed) > 0
            else None
        )
        try:
            n_required = required_n_proportions(baseline_rate, mde)
        except ValueError:
            n_required = None
        report = build_readout(
            experiment_id=experiment.id,
            primary=primary,
            guardrails={
                m.name: results_map[m.name]
                for m in experiment.guardrail_metrics
            },
            srm=srm,
            n_required=n_required,
            looks=looks,
        )
        return report.to_dict()

    return app


def build_demo_app() -> FastAPI:
    """An app with the demo experiment registered and running."""
    registry = ExperimentRegistry()
    experiment = build_demo_experiment()
    registry.register(experiment)
    experiment.transition(Lifecycle.RUNNING)
    return create_app(registry, EventStore())


# For `uvicorn experimentation_platform.serving:app`.
app = build_demo_app()
