"""End-to-end demo: three scenarios, three different readouts.

1. A real lift, cleanly logged            -> SHIP
2. A dropped-exposure logging bug (SRM)   -> INCONCLUSIVE (invalid)
3. A lift that breaks a guardrail         -> NO-SHIP

Each scenario simulates deterministic synthetic traffic, ingests it
into the event store, computes metrics, runs the SRM check, and
builds the decision readout the same way the API does.
"""

from __future__ import annotations

from .events import EventStore
from .metrics import compute_all_metrics, compute_metric
from .readout import build_readout
from .simulator import build_demo_experiment, simulate_traffic
from .stats import required_n_proportions, srm_check

SCENARIOS = [
    (
        "Real lift, clean logging",
        dict(true_lift=0.50),
        "ship",
    ),
    (
        "Dropped control exposures (SRM bug)",
        dict(true_lift=0.50, inject_srm=True),
        "inconclusive",
    ),
    (
        "Lift that breaks the payment-error guardrail",
        dict(true_lift=0.50, guardrail_regression=0.03),
        "no_ship",
    ),
]


def run_scenario(label: str, kwargs: dict, n_units: int = 6000, seed: int = 42):
    experiment = build_demo_experiment()
    sim = simulate_traffic(experiment, n_units=n_units, seed=seed, **kwargs)
    store = EventStore()
    store.extend(sim.events)
    results = compute_all_metrics(experiment, store)
    primary = results[experiment.primary_metric.name]
    observed = [
        primary.per_variant[v.name].n_exposed for v in experiment.variants
    ]
    srm = srm_check(observed, [v.weight for v in experiment.variants])
    n_required = required_n_proportions(0.10, 0.02)
    readout = build_readout(
        experiment_id=experiment.id,
        primary=primary,
        guardrails={
            name: results[name] for name in
            [m.name for m in experiment.guardrail_metrics]
        },
        srm=srm,
        n_required=n_required,
        looks=1,
    )
    print(f"Scenario: {label}")
    print(f"  exposed per variant: {observed} "
          f"(dropped by SRM bug: {sim.n_dropped_srm})")
    for name, res in results.items():
        for comp in res.comparisons:
            print(
                f"  {name:<14s} control vs {comp.treatment}: "
                f"effect {comp.absolute_effect:+.4f}  p={comp.p_value:.4f}"
            )
    print(f"  SRM: chi2={srm.chi2:.2f} p={srm.p_value:.3g} "
          f"passed={srm.passed}")
    print(f"  READOUT: {readout.decision.value.upper()}")
    for reason in readout.reasons:
        print(f"    - {reason}")
    print()
    return readout


def run_demo() -> None:
    print("Experimentation platform demo — synthetic traffic, known truth")
    print("=" * 72)
    print()
    for label, kwargs, expected in SCENARIOS:
        readout = run_scenario(label, kwargs)
        assert readout.decision.value == expected, (
            f"{label}: expected {expected}, got {readout.decision.value}"
        )
    print("Demo complete: lift ships, SRM invalidates, guardrail vetoes.")


if __name__ == "__main__":
    run_demo()
