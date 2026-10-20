from experimentation_platform.events import EventStore
from experimentation_platform.metrics import compute_all_metrics, compute_metric
from experimentation_platform.readout import Decision, assess_peeking, build_readout
from experimentation_platform.simulator import build_demo_experiment, simulate_traffic
from experimentation_platform.stats import srm_check


def _readout_for(experiment, store, n_required=3839, looks=1):
    results = compute_all_metrics(experiment, store)
    primary = results[experiment.primary_metric.name]
    observed = [primary.per_variant[v.name].n_exposed for v in experiment.variants]
    srm = srm_check(observed, [v.weight for v in experiment.variants])
    return build_readout(
        experiment_id=experiment.id,
        primary=primary,
        guardrails={m.name: results[m.name] for m in experiment.guardrail_metrics},
        srm=srm,
        n_required=n_required,
        looks=looks,
    )


def _run(kwargs, n_units=6000, seed=42):
    experiment = build_demo_experiment()
    sim = simulate_traffic(experiment, n_units=n_units, seed=seed, **kwargs)
    store = EventStore()
    store.extend(sim.events)
    return experiment, _readout_for(experiment, store)


def test_ship_on_real_lift():
    _, readout = _run(dict(true_lift=0.5))
    assert readout.decision is Decision.SHIP


def test_srm_failure_is_inconclusive():
    _, readout = _run(dict(true_lift=0.5, inject_srm=True))
    assert readout.srm is not None and not readout.srm.passed
    assert readout.decision is Decision.INCONCLUSIVE


def test_guardrail_breach_is_no_ship():
    _, readout = _run(dict(true_lift=0.5, guardrail_regression=0.03))
    assert readout.guardrail_breaches
    assert readout.decision is Decision.NO_SHIP


def test_no_lift_is_inconclusive():
    _, readout = _run(dict(true_lift=0.0))
    assert readout.decision is Decision.INCONCLUSIVE


def test_negative_lift_is_no_ship():
    _, readout = _run(dict(true_lift=-0.5))
    assert readout.decision is Decision.NO_SHIP


def test_peeking_guard_flags_early_naive_significance():
    # z=2.1 is naively significant at alpha=.05, but at 10% information
    # the OBF boundary is ~6.2: a textbook peeking flag.
    a = assess_peeking(z_value=2.1, p_value=0.036, n_current=100,
                       n_required=1000, looks=5)
    assert a.peeking_risk
    assert a.warning is not None
    assert a.obf_critical_z > 6.0


def test_peeking_guard_quiet_at_planned_end():
    a = assess_peeking(z_value=2.1, p_value=0.036, n_current=1000,
                       n_required=1000, looks=1)
    assert not a.peeking_risk


def test_peeking_guard_quiet_when_not_significant():
    a = assess_peeking(z_value=0.5, p_value=0.62, n_current=100,
                       n_required=1000, looks=5)
    assert not a.peeking_risk
