import pytest

from experimentation_platform.errors import (
    ExperimentValidationError,
    InvalidTransitionError,
)
from experimentation_platform.models import (
    EligibilityRule,
    Lifecycle,
    MetricKind,
    MetricSpec,
    Variant,
)
from experimentation_platform.registry import ExperimentRegistry

from conftest import make_experiment


def test_weights_must_sum_to_one():
    with pytest.raises(ExperimentValidationError):
        make_experiment(variants=(
            Variant("control", 0.4, is_control=True),
            Variant("treatment", 0.4),
        ))


def test_exactly_one_control_required():
    with pytest.raises(ExperimentValidationError):
        make_experiment(variants=(
            Variant("a", 0.5), Variant("b", 0.5),
        ))
    with pytest.raises(ExperimentValidationError):
        make_experiment(variants=(
            Variant("a", 0.5, is_control=True),
            Variant("b", 0.5, is_control=True),
        ))


def test_exactly_one_primary_required():
    with pytest.raises(ExperimentValidationError):
        make_experiment(metrics=(
            MetricSpec("m", MetricKind.CONVERSION),
        ))


def test_duplicate_names_rejected():
    with pytest.raises(ExperimentValidationError):
        make_experiment(variants=(
            Variant("x", 0.5, is_control=True), Variant("x", 0.5),
        ))
    with pytest.raises(ExperimentValidationError):
        make_experiment(metrics=(
            MetricSpec("m", MetricKind.CONVERSION, is_primary=True),
            MetricSpec("m", MetricKind.CONTINUOUS),
        ))


def test_salt_defaults_to_id():
    assert make_experiment(id="abc").salt == "abc"


def test_lifecycle_happy_path(experiment):
    assert experiment.status is Lifecycle.DRAFT
    experiment.transition(Lifecycle.RUNNING)
    experiment.transition(Lifecycle.PAUSED)
    experiment.transition(Lifecycle.RUNNING)
    experiment.transition(Lifecycle.CONCLUDED)
    assert experiment.status is Lifecycle.CONCLUDED


def test_lifecycle_rejects_skips_and_reversals(experiment):
    with pytest.raises(InvalidTransitionError):
        experiment.transition(Lifecycle.PAUSED)  # draft -> paused
    experiment.transition(Lifecycle.RUNNING)
    with pytest.raises(InvalidTransitionError):
        experiment.transition(Lifecycle.DRAFT)
    experiment.transition(Lifecycle.CONCLUDED)
    with pytest.raises(InvalidTransitionError):
        experiment.transition(Lifecycle.RUNNING)  # concluded is terminal


def test_draft_can_be_cancelled(experiment):
    experiment.transition(Lifecycle.CONCLUDED)
    assert experiment.status is Lifecycle.CONCLUDED


def test_eligibility_operators():
    attrs = {"country": "US", "age": 30, "plan": "pro"}
    assert EligibilityRule("country", "eq", "US").evaluate(attrs)
    assert not EligibilityRule("country", "eq", "CA").evaluate(attrs)
    assert EligibilityRule("age", "gte", 18).evaluate(attrs)
    assert not EligibilityRule("age", "lt", 18).evaluate(attrs)
    assert EligibilityRule("plan", "in", ("pro", "team")).evaluate(attrs)
    assert EligibilityRule("plan", "not_in", ("free",)).evaluate(attrs)
    assert EligibilityRule("country", "exists", True).evaluate(attrs)
    assert not EligibilityRule("missing", "exists", True).evaluate(attrs)
    assert not EligibilityRule("missing", "eq", "x").evaluate(attrs)


def test_registry_layer_overflow_rejected():
    reg = ExperimentRegistry()
    reg.register(make_experiment(id="a", layer="L", allocation_fraction=0.6))
    with pytest.raises(Exception):
        reg.register(make_experiment(id="b", layer="L", allocation_fraction=0.6))


def test_registry_assigns_disjoint_layer_slices():
    reg = ExperimentRegistry()
    a = reg.register(make_experiment(id="a", layer="L", allocation_fraction=0.3))
    b = reg.register(make_experiment(id="b", layer="L", allocation_fraction=0.3))
    assert a.layer_start == 0.0
    assert b.layer_start == pytest.approx(0.3)
    assert reg.layer_usage("L") == pytest.approx(0.6)
