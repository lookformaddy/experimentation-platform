"""Experiment definitions and the lifecycle state machine.

An :class:`Experiment` is a frozen, validated contract: variants with
traffic weights, one primary metric, any number of secondary and
guardrail metrics, eligibility rules over unit attributes, an optional
mutual-exclusion layer, and a lifecycle

    draft -> running -> paused -> running -> concluded
    draft -> concluded            (cancelled before launch)

Validation happens at construction, so an invalid experiment can never
reach assignment or analysis: weights must sum to 1, exactly one
variant must be the control, metric names must be unique, and exactly
one metric must be primary.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from .errors import ExperimentValidationError, InvalidTransitionError


class Lifecycle(enum.Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    CONCLUDED = "concluded"


# The complete transition table. Anything not listed is rejected.
TRANSITIONS: dict[Lifecycle, frozenset[Lifecycle]] = {
    Lifecycle.DRAFT: frozenset({Lifecycle.RUNNING, Lifecycle.CONCLUDED}),
    Lifecycle.RUNNING: frozenset({Lifecycle.PAUSED, Lifecycle.CONCLUDED}),
    Lifecycle.PAUSED: frozenset({Lifecycle.RUNNING, Lifecycle.CONCLUDED}),
    Lifecycle.CONCLUDED: frozenset(),
}


class MetricKind(enum.Enum):
    CONVERSION = "conversion"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class Variant:
    name: str
    weight: float
    is_control: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ExperimentValidationError("variant name must be non-empty")
        if not (0.0 < self.weight <= 1.0):
            raise ExperimentValidationError(
                f"variant {self.name!r} weight must be in (0, 1], got {self.weight}"
            )


@dataclass(frozen=True)
class MetricSpec:
    """One measured outcome.

    ``guardrail_tolerance`` is the largest absolute degradation (in the
    metric's own units) a guardrail may show before it counts as a
    breach when it is also statistically significant.
    """

    name: str
    kind: MetricKind
    is_primary: bool = False
    is_guardrail: bool = False
    higher_is_better: bool = True
    guardrail_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ExperimentValidationError("metric name must be non-empty")
        if self.is_primary and self.is_guardrail:
            raise ExperimentValidationError(
                f"metric {self.name!r} cannot be both primary and guardrail"
            )
        if self.guardrail_tolerance < 0:
            raise ExperimentValidationError("guardrail_tolerance must be >= 0")


@dataclass(frozen=True)
class EligibilityRule:
    """``attributes[attribute] <operator> value`` must hold to enter.

    Operators: eq, ne, in, not_in, gt, gte, lt, lte, exists.
    A missing attribute fails every operator except ``ne``/``not_in``
    semantics are *not* special-cased: missing means not eligible,
    except for ``exists`` with value False.
    """

    attribute: str
    operator: str
    value: object = None

    _OPERATORS = frozenset(
        {"eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "exists"}
    )

    def __post_init__(self) -> None:
        if self.operator not in self._OPERATORS:
            raise ExperimentValidationError(
                f"unknown eligibility operator: {self.operator!r}"
            )

    def evaluate(self, attributes: dict) -> bool:
        present = self.attribute in attributes
        if self.operator == "exists":
            return present == bool(self.value)
        if not present:
            return False
        actual = attributes[self.attribute]
        try:
            if self.operator == "eq":
                return bool(actual == self.value)
            if self.operator == "ne":
                return bool(actual != self.value)
            if self.operator == "in":
                return actual in self.value  # type: ignore[operator]
            if self.operator == "not_in":
                return actual not in self.value  # type: ignore[operator]
            if self.operator == "gt":
                return bool(actual > self.value)  # type: ignore[operator]
            if self.operator == "gte":
                return bool(actual >= self.value)  # type: ignore[operator]
            if self.operator == "lt":
                return bool(actual < self.value)  # type: ignore[operator]
            if self.operator == "lte":
                return bool(actual <= self.value)  # type: ignore[operator]
        except TypeError:
            return False
        return False


@dataclass
class Experiment:
    id: str
    name: str
    hypothesis: str
    variants: tuple[Variant, ...]
    metrics: tuple[MetricSpec, ...]
    salt: str = ""
    layer: str | None = None
    allocation_fraction: float = 1.0
    eligibility: tuple[EligibilityRule, ...] = ()
    status: Lifecycle = Lifecycle.DRAFT
    # Set by the registry when the experiment joins a layer: the
    # [layer_start, layer_start + allocation_fraction) slice of the
    # layer's unit space this experiment owns.
    layer_start: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if not self.salt:
            self.salt = self.id
        self.validate()

    def validate(self) -> None:
        if not self.id:
            raise ExperimentValidationError("experiment id must be non-empty")
        if len(self.variants) < 2:
            raise ExperimentValidationError("an experiment needs at least 2 variants")
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ExperimentValidationError(f"duplicate variant names: {names}")
        total = sum(v.weight for v in self.variants)
        if abs(total - 1.0) > 1e-9:
            raise ExperimentValidationError(
                f"variant weights must sum to 1.0, got {total}"
            )
        controls = [v for v in self.variants if v.is_control]
        if len(controls) != 1:
            raise ExperimentValidationError(
                f"exactly one control variant is required, got {len(controls)}"
            )
        metric_names = [m.name for m in self.metrics]
        if len(set(metric_names)) != len(metric_names):
            raise ExperimentValidationError(f"duplicate metric names: {metric_names}")
        primaries = [m for m in self.metrics if m.is_primary]
        if len(primaries) != 1:
            raise ExperimentValidationError(
                f"exactly one primary metric is required, got {len(primaries)}"
            )
        if not (0.0 < self.allocation_fraction <= 1.0):
            raise ExperimentValidationError(
                "allocation_fraction must be in (0, 1]"
            )

    # -- lookups ---------------------------------------------------------
    @property
    def control(self) -> Variant:
        return next(v for v in self.variants if v.is_control)

    @property
    def primary_metric(self) -> MetricSpec:
        return next(m for m in self.metrics if m.is_primary)

    @property
    def guardrail_metrics(self) -> tuple[MetricSpec, ...]:
        return tuple(m for m in self.metrics if m.is_guardrail)

    def metric(self, name: str) -> MetricSpec:
        for m in self.metrics:
            if m.name == name:
                return m
        raise ExperimentValidationError(
            f"experiment {self.id!r} has no metric {name!r}"
        )

    def is_eligible(self, attributes: dict | None) -> bool:
        attrs = attributes or {}
        return all(rule.evaluate(attrs) for rule in self.eligibility)

    # -- lifecycle ---------------------------------------------------------
    def transition(self, target: Lifecycle) -> None:
        if target not in TRANSITIONS[self.status]:
            raise InvalidTransitionError(
                f"cannot move experiment {self.id!r} from "
                f"{self.status.value} to {target.value}"
            )
        self.status = target

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "hypothesis": self.hypothesis,
            "status": self.status.value,
            "salt": self.salt,
            "layer": self.layer,
            "allocation_fraction": self.allocation_fraction,
            "variants": [
                {"name": v.name, "weight": v.weight, "is_control": v.is_control}
                for v in self.variants
            ],
            "metrics": [
                {
                    "name": m.name,
                    "kind": m.kind.value,
                    "is_primary": m.is_primary,
                    "is_guardrail": m.is_guardrail,
                    "higher_is_better": m.higher_is_better,
                    "guardrail_tolerance": m.guardrail_tolerance,
                }
                for m in self.metrics
            ],
            "eligibility": [
                {"attribute": r.attribute, "operator": r.operator, "value": r.value}
                for r in self.eligibility
            ],
        }
