from experimentation_platform.events import EventStore
from experimentation_platform.metrics import compute_metric
from experimentation_platform.simulator import build_demo_experiment, simulate_traffic


def test_simulator_is_deterministic():
    exp = build_demo_experiment()
    a = simulate_traffic(exp, n_units=500, seed=7, true_lift=0.3)
    b = simulate_traffic(exp, n_units=500, seed=7, true_lift=0.3)
    assert a.events == b.events


def test_simulator_different_seed_differs():
    exp = build_demo_experiment()
    a = simulate_traffic(exp, n_units=500, seed=7)
    b = simulate_traffic(exp, n_units=500, seed=8)
    assert a.events != b.events


def test_simulator_lift_shows_up_in_metrics():
    exp = build_demo_experiment()
    sim = simulate_traffic(exp, n_units=6000, seed=42, true_lift=0.5)
    store = EventStore()
    store.extend(sim.events)
    result = compute_metric(exp, store, exp.primary_metric)
    assert result.comparisons[0].absolute_effect > 0.02
    assert result.comparisons[0].significant


def test_srm_injection_drops_control_exposures():
    exp = build_demo_experiment()
    clean = simulate_traffic(exp, n_units=2000, seed=42)
    bugged = simulate_traffic(exp, n_units=2000, seed=42, inject_srm=True)
    assert bugged.n_dropped_srm > 100
    assert bugged.n_exposed < clean.n_exposed


def test_simulator_respects_eligibility():
    exp = build_demo_experiment()  # requires country in (US, CA)
    sim = simulate_traffic(exp, n_units=200, seed=1)
    assert sim.n_assigned == 200  # simulator supplies country=US
