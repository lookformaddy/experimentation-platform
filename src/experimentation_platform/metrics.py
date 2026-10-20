"""Metric computation: events in, per-variant comparisons out.

Attribution rule (documented, and the only one): a unit's metric
outcome belongs to the variant of its *first exposure*. Units with no
exposure never count, however many metric events they emit — an
unexposed converter is an instrumentation bug, not a result.

- Conversion metrics: a unit converts if it has at least one event
  named after the metric. The comparison is a two-proportion z-test.
- Continuous metrics: a unit contributes its *first* metric event's
  value (repeat purchases etc. should be pre-aggregated upstream into
  one per-unit event). The comparison is Welch's t-test.
"""

from __future__ import annotations

from dataclasses import dataclass

from .events import EXPOSURE, EventStore
from .models import Experiment, MetricKind, MetricSpec
from .stats import (
    ProportionResult,
    WelchResult,
    two_proportion_ztest,
    welch_ttest_from_stats,
)


@dataclass(frozen=True)
class VariantStats:
    variant: str
    n_exposed: int
    # conversion
    n_converted: int = 0
    rate: float = 0.0
    # continuous
    mean: float = 0.0
    variance: float = 0.0


@dataclass(frozen=True)
class Comparison:
    treatment: str
    control: str
    result: ProportionResult | WelchResult

    @property
    def absolute_effect(self) -> float:
        return self.result.absolute_effect

    @property
    def p_value(self) -> float:
        return self.result.p_value

    @property
    def significant(self) -> bool:
        return self.result.significant


@dataclass(frozen=True)
class MetricResult:
    metric: MetricSpec
    per_variant: dict[str, VariantStats]
    comparisons: tuple[Comparison, ...]  # one per non-control variant

    @property
    def total_exposed(self) -> int:
        return sum(v.n_exposed for v in self.per_variant.values())


def _mean_var(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def compute_metric(
    experiment: Experiment,
    store: EventStore,
    metric: MetricSpec,
    alpha: float = 0.05,
    include_late: bool = False,
) -> MetricResult:
    events = store.events_for(experiment.id, include_late=include_late)
    exposed = store.exposed_units(experiment.id)
    # First-exposure variant per unit, intersected with the late filter:
    # rebuild from the (possibly filtered) event list so late exposures
    # drop out consistently with late metric events.
    first_exposure: dict[str, str] = {}
    for e in events:
        if e.event_name == EXPOSURE and e.unit_id not in first_exposure:
            first_exposure[e.unit_id] = e.variant
    by_variant_units: dict[str, set[str]] = {v.name: set() for v in experiment.variants}
    for unit, variant in first_exposure.items():
        if variant in by_variant_units:
            by_variant_units[variant].add(unit)

    per_variant: dict[str, VariantStats] = {}
    if metric.kind is MetricKind.CONVERSION:
        converted: dict[str, set[str]] = {name: set() for name in by_variant_units}
        for e in events:
            if e.event_name == metric.name and e.unit_id in first_exposure:
                variant = first_exposure[e.unit_id]
                if variant in converted:
                    converted[variant].add(e.unit_id)
        for name, units in by_variant_units.items():
            n = len(units)
            conv = len(converted[name] & units)
            per_variant[name] = VariantStats(
                variant=name, n_exposed=n, n_converted=conv,
                rate=(conv / n) if n else 0.0,
            )
    else:
        first_value: dict[str, float] = {}
        for e in events:
            if (
                e.event_name == metric.name
                and e.value is not None
                and e.unit_id in first_exposure
                and e.unit_id not in first_value
            ):
                first_value[e.unit_id] = e.value
        for name, units in by_variant_units.items():
            values = [first_value[u] for u in units if u in first_value]
            mean, var = _mean_var(values)
            per_variant[name] = VariantStats(
                variant=name, n_exposed=len(units), mean=mean, variance=var,
            )
        # Exposed counts for continuous metrics still describe exposure;
        # units without a metric event contribute no observation but
        # remain in n_exposed for SRM / power accounting.

    control = experiment.control.name
    comparisons: list[Comparison] = []
    c_stats = per_variant[control]
    for variant in experiment.variants:
        if variant.is_control:
            continue
        t_stats = per_variant[variant.name]
        if metric.kind is MetricKind.CONVERSION:
            if c_stats.n_exposed == 0 or t_stats.n_exposed == 0:
                continue
            result: ProportionResult | WelchResult = two_proportion_ztest(
                c_stats.n_converted, c_stats.n_exposed,
                t_stats.n_converted, t_stats.n_exposed, alpha=alpha,
            )
        else:
            # Observations = units with a value; recover counts from the
            # stored aggregates by re-deriving n with values.
            c_values = _values_for(events, first_exposure, metric.name, control)
            t_values = _values_for(events, first_exposure, metric.name, variant.name)
            if len(c_values) < 2 or len(t_values) < 2:
                continue
            cm, cv = _mean_var(c_values)
            tm, tv = _mean_var(t_values)
            result = welch_ttest_from_stats(
                cm, cv, len(c_values), tm, tv, len(t_values), alpha=alpha
            )
        comparisons.append(
            Comparison(treatment=variant.name, control=control, result=result)
        )
    return MetricResult(
        metric=metric, per_variant=per_variant, comparisons=tuple(comparisons)
    )


def _values_for(
    events: list, first_exposure: dict[str, str], metric_name: str, variant: str
) -> list[float]:
    seen: set[str] = set()
    values: list[float] = []
    for e in events:
        if (
            e.event_name == metric_name
            and e.value is not None
            and first_exposure.get(e.unit_id) == variant
            and e.unit_id not in seen
        ):
            seen.add(e.unit_id)
            values.append(e.value)
    return values


def compute_all_metrics(
    experiment: Experiment,
    store: EventStore,
    alpha: float = 0.05,
    include_late: bool = False,
) -> dict[str, MetricResult]:
    return {
        m.name: compute_metric(experiment, store, m, alpha, include_late)
        for m in experiment.metrics
    }
