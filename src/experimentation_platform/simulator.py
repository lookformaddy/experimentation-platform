"""Deterministic synthetic traffic simulator.

Generates a full experiment's worth of exposures and metric events
from known ground truth, so readouts can be demonstrated — and tested
— against reality instead of hope:

- ``true_lift``: relative lift applied to the treatment arm's primary
  metric (and, for continuous primaries, to its mean).
- ``inject_srm``: simulates the classic logging bug — a fraction of
  control exposures is silently dropped, skewing the split so the SRM
  check must fire.
- ``guardrail_regression``: absolute degradation added to every
  guardrail metric on the treatment arm.

Same seed, same traffic, byte for byte.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .assignment import AssignmentEngine
from .events import Event
from .models import Experiment, MetricKind, MetricSpec

EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


def build_demo_experiment() -> Experiment:
    """The demo experiment: a generic checkout-flow test."""
    from .models import EligibilityRule, MetricSpec, Variant

    return Experiment(
        id="checkout-flow-v2",
        name="Checkout flow v2",
        hypothesis="A shorter checkout flow increases completed purchases "
        "without raising the payment error rate.",
        variants=(
            Variant("control", 0.5, is_control=True),
            Variant("treatment", 0.5),
        ),
        metrics=(
            MetricSpec("purchase", MetricKind.CONVERSION, is_primary=True),
            MetricSpec(
                "payment_error", MetricKind.CONVERSION, is_guardrail=True,
                higher_is_better=False, guardrail_tolerance=0.005,
            ),
            MetricSpec("revenue", MetricKind.CONTINUOUS),
        ),
        eligibility=(
            EligibilityRule("country", "in", ("US", "CA")),
        ),
    )


@dataclass
class SimulationResult:
    events: list[Event] = field(default_factory=list)
    n_assigned: int = 0
    n_exposed: int = 0
    n_dropped_srm: int = 0


def simulate_traffic(
    experiment: Experiment,
    n_units: int = 6000,
    seed: int = 42,
    base_rate: float = 0.10,
    true_lift: float = 0.0,
    guardrail_base_rate: float = 0.02,
    guardrail_regression: float = 0.0,
    revenue_mean: float = 40.0,
    revenue_std: float = 12.0,
    srm_drop_fraction: float = 0.30,
    inject_srm: bool = False,
    engine: AssignmentEngine | None = None,
) -> SimulationResult:
    """Simulate ``n_units`` eligible units through ``experiment``."""
    rng = random.Random(seed)
    engine = engine or AssignmentEngine()
    result = SimulationResult()
    seq = 0

    def next_id(prefix: str) -> str:
        nonlocal seq
        seq += 1
        return f"{prefix}-{seq:07d}"

    for i in range(n_units):
        unit_id = f"unit-{i + 1:06d}"
        variant = engine.assign(
            experiment, unit_id, attributes={"country": "US"}
        )
        if variant is None:
            continue
        result.n_assigned += 1
        is_control = variant == experiment.control.name
        if inject_srm and is_control and rng.random() < srm_drop_fraction:
            result.n_dropped_srm += 1
            continue  # the dropped-exposure bug: unit ran, never logged
        ts = EPOCH + timedelta(seconds=i)
        result.events.append(
            Event(next_id("exp"), experiment.id, unit_id, variant,
                  "exposure", ts)
        )
        result.n_exposed += 1
        for metric in experiment.metrics:
            _emit_metric(
                rng, experiment, metric, unit_id, variant, is_control, ts,
                base_rate, true_lift, guardrail_base_rate,
                guardrail_regression, revenue_mean, revenue_std, next_id,
                result,
            )
    return result


def _emit_metric(
    rng: random.Random,
    experiment: Experiment,
    metric: MetricSpec,
    unit_id: str,
    variant: str,
    is_control: bool,
    ts: datetime,
    base_rate: float,
    true_lift: float,
    guardrail_base_rate: float,
    guardrail_regression: float,
    revenue_mean: float,
    revenue_std: float,
    next_id,
    result: SimulationResult,
) -> None:
    if metric.kind is MetricKind.CONVERSION:
        if metric.is_primary:
            rate = base_rate if is_control else base_rate * (1.0 + true_lift)
        elif metric.is_guardrail:
            rate = (
                guardrail_base_rate
                if is_control
                else guardrail_base_rate
                + (guardrail_regression if metric.higher_is_better is False
                   else -guardrail_regression)
            )
        else:
            rate = base_rate if is_control else base_rate * (1.0 + true_lift)
        rate = min(max(rate, 0.0), 1.0)
        if rng.random() < rate:
            result.events.append(
                Event(next_id("evt"), experiment.id, unit_id, variant,
                      metric.name, ts + timedelta(seconds=30))
            )
    else:
        if metric.is_primary:
            mean = revenue_mean if is_control else revenue_mean * (1.0 + true_lift)
        elif metric.is_guardrail:
            mean = revenue_mean if is_control else revenue_mean + guardrail_regression
        else:
            mean = revenue_mean if is_control else revenue_mean * (1.0 + true_lift)
        value = rng.gauss(mean, revenue_std)
        result.events.append(
            Event(next_id("evt"), experiment.id, unit_id, variant,
                  metric.name, ts + timedelta(seconds=30), value=value)
        )
