"""Experimentation Platform.

A clean-room reference implementation of an enterprise A/B
experimentation platform: validated experiment definitions with a
lifecycle state machine, deterministic hash-based assignment with
mutual-exclusion layers, an append-only exposure/event store,
pure-stdlib statistics (z-test, Welch's t-test, SRM chi-square),
an O'Brien-Fleming-style peeking guard, decision readouts, a
deterministic synthetic traffic simulator, and a FastAPI service.
All data is synthetic.
"""

from .assignment import AssignmentEngine, unit_bucket, unit_interval
from .events import Event, EventStore
from .metrics import MetricResult, compute_all_metrics, compute_metric
from .models import (
    EligibilityRule,
    Experiment,
    Lifecycle,
    MetricKind,
    MetricSpec,
    Variant,
)
from .readout import Decision, Readout, build_readout
from .registry import ExperimentRegistry
from .stats import (
    required_n_continuous,
    required_n_proportions,
    srm_check,
    two_proportion_ztest,
    welch_ttest,
)

__version__ = "0.1.0"

__all__ = [
    "AssignmentEngine",
    "Decision",
    "EligibilityRule",
    "Event",
    "EventStore",
    "Experiment",
    "ExperimentRegistry",
    "Lifecycle",
    "MetricKind",
    "MetricResult",
    "MetricSpec",
    "Readout",
    "Variant",
    "build_readout",
    "compute_all_metrics",
    "compute_metric",
    "required_n_continuous",
    "required_n_proportions",
    "srm_check",
    "two_proportion_ztest",
    "unit_bucket",
    "unit_interval",
    "welch_ttest",
]
