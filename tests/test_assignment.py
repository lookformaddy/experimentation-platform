from experimentation_platform.assignment import AssignmentEngine, unit_bucket, unit_interval
from experimentation_platform.models import EligibilityRule, Variant
from experimentation_platform.registry import ExperimentRegistry

from conftest import make_experiment

ENGINE = AssignmentEngine()


def test_assignment_is_stable_across_calls(experiment):
    for i in range(200):
        uid = f"u-{i}"
        assert ENGINE.assign(experiment, uid) == ENGINE.assign(experiment, uid)


def test_assignment_covers_both_variants(experiment):
    assigned = {ENGINE.assign(experiment, f"u-{i}") for i in range(500)}
    assert assigned == {"control", "treatment"}


def test_unit_interval_in_range_and_deterministic():
    for i in range(100):
        v = unit_interval(f"u-{i}", "salt")
        assert 0.0 <= v < 1.0
        assert v == unit_interval(f"u-{i}", "salt")


def test_different_salt_changes_assignments(experiment):
    other = make_experiment(id="exp-1", salt="another-salt")
    differ = sum(
        ENGINE.assign(experiment, f"u-{i}") != ENGINE.assign(other, f"u-{i}")
        for i in range(500)
    )
    assert differ > 100  # roughly half should flip; 100 is a safe floor


def test_bucketing_uniformity_50_50(experiment):
    n = 20_000
    counts = {"control": 0, "treatment": 0}
    for i in range(n):
        counts[ENGINE.assign(experiment, f"u-{i}")] += 1
    for count in counts.values():
        assert abs(count / n - 0.5) < 0.02


def test_assignment_respects_unequal_weights():
    exp = make_experiment(
        variants=(
            Variant("control", 0.7, is_control=True),
            Variant("treatment", 0.3),
        )
    )
    n = 20_000
    treatment = sum(ENGINE.assign(exp, f"u-{i}") == "treatment" for i in range(n))
    assert abs(treatment / n - 0.3) < 0.02


def test_bucket_range():
    for i in range(200):
        assert 0 <= unit_bucket(f"u-{i}", "s", buckets=100) < 100


def test_eligibility_rule_filters_units():
    exp = make_experiment(
        eligibility=(EligibilityRule("country", "in", ("US", "CA")),)
    )
    assert ENGINE.assign(exp, "u-1", {"country": "US"}) is not None
    assert ENGINE.assign(exp, "u-1", {"country": "FR"}) is None
    assert ENGINE.assign(exp, "u-1", {}) is None
    assert ENGINE.assign(exp, "u-1", None) is None


def test_layer_exclusion_no_unit_in_two_experiments():
    reg = ExperimentRegistry()
    exp_a = make_experiment(id="a", layer="pricing", allocation_fraction=0.5)
    exp_b = make_experiment(id="b", layer="pricing", allocation_fraction=0.5)
    reg.register(exp_a)
    reg.register(exp_b)
    overlap = 0
    either = 0
    for i in range(2000):
        uid = f"u-{i}"
        a = ENGINE.assign(exp_a, uid)
        b = ENGINE.assign(exp_b, uid)
        if a is not None and b is not None:
            overlap += 1
        if a is not None or b is not None:
            either += 1
    assert overlap == 0
    assert either > 1500  # the two halves together cover ~all units


def test_layer_partial_allocation_leaves_units_unassigned():
    reg = ExperimentRegistry()
    exp = make_experiment(id="a", layer="small", allocation_fraction=0.2)
    reg.register(exp)
    assigned = sum(ENGINE.assign(exp, f"u-{i}") is not None for i in range(5000))
    assert abs(assigned / 5000 - 0.2) < 0.03
