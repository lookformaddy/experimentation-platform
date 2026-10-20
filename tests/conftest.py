from datetime import datetime, timedelta, timezone

import pytest

from experimentation_platform.events import Event, EventStore
from experimentation_platform.models import (
    EligibilityRule,
    Experiment,
    MetricKind,
    MetricSpec,
    Variant,
)
from experimentation_platform.registry import ExperimentRegistry

UTC = timezone.utc
BASE = datetime(2024, 1, 1, 9, 0, tzinfo=UTC)


def make_experiment(**overrides) -> Experiment:
    kwargs = dict(
        id="exp-1",
        name="Test experiment",
        hypothesis="Treatment improves signup.",
        variants=(
            Variant("control", 0.5, is_control=True),
            Variant("treatment", 0.5),
        ),
        metrics=(
            MetricSpec("signup", MetricKind.CONVERSION, is_primary=True),
            MetricSpec(
                "error_rate", MetricKind.CONVERSION, is_guardrail=True,
                higher_is_better=False, guardrail_tolerance=0.005,
            ),
            MetricSpec("revenue", MetricKind.CONTINUOUS),
        ),
    )
    kwargs.update(overrides)
    return Experiment(**kwargs)


@pytest.fixture()
def experiment() -> Experiment:
    return make_experiment()


@pytest.fixture()
def registry(experiment) -> ExperimentRegistry:
    reg = ExperimentRegistry()
    reg.register(experiment)
    return reg


@pytest.fixture()
def store() -> EventStore:
    return EventStore()


def event(
    seq: int,
    unit: str,
    variant: str,
    name: str = "exposure",
    value: float | None = None,
    when: datetime | None = None,
    experiment_id: str = "exp-1",
) -> Event:
    return Event(
        event_id=f"e-{seq}",
        experiment_id=experiment_id,
        unit_id=unit,
        variant=variant,
        event_name=name,
        timestamp=when or (BASE + timedelta(seconds=seq)),
        value=value,
    )
