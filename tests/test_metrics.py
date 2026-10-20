from experimentation_platform.metrics import compute_metric

from conftest import event


def _hand_store(store):
    """10 control / 10 treatment exposed; 2 vs 5 signups; revenue known."""
    seq = 0

    def nxt(*args, **kwargs):
        nonlocal seq
        seq += 1
        return event(seq, *args, **kwargs)

    for i in range(10):
        store.append(nxt(f"c-{i}", "control"))
        store.append(nxt(f"t-{i}", "treatment"))
    for i in range(2):
        store.append(nxt(f"c-{i}", "control", name="signup"))
    for i in range(5):
        store.append(nxt(f"t-{i}", "treatment", name="signup"))
    # revenue: first value per unit counts; the repeat for t-0 must not
    for i, v in enumerate([10.0, 20.0, 30.0]):
        store.append(nxt(f"c-{i}", "control", name="revenue", value=v))
    for i, v in enumerate([20.0, 30.0, 40.0]):
        store.append(nxt(f"t-{i}", "treatment", name="revenue", value=v))
    store.append(nxt("t-0", "treatment", name="revenue", value=999.0))
    return store


def test_conversion_metric_counts(experiment, store):
    result = compute_metric(experiment, _hand_store(store),
                            experiment.metric("signup"))
    assert result.per_variant["control"].n_exposed == 10
    assert result.per_variant["control"].n_converted == 2
    assert result.per_variant["control"].rate == 0.2
    assert result.per_variant["treatment"].n_converted == 5
    assert result.per_variant["treatment"].rate == 0.5
    comp = result.comparisons[0]
    assert comp.absolute_effect == 0.3


def test_continuous_metric_means(experiment, store):
    result = compute_metric(experiment, _hand_store(store),
                            experiment.metric("revenue"))
    assert result.per_variant["control"].mean == 20.0
    assert result.per_variant["treatment"].mean == 30.0  # 999.0 ignored
    assert result.comparisons[0].absolute_effect == 10.0


def test_unexposed_units_never_count(experiment, store):
    store.append(event(1, "ghost", "control", name="signup"))
    result = compute_metric(experiment, store, experiment.metric("signup"))
    assert result.total_exposed == 0
    assert result.comparisons == ()


def test_empty_store_gives_no_comparisons(experiment, store):
    result = compute_metric(experiment, store, experiment.metric("signup"))
    assert result.total_exposed == 0
