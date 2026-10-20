"""Deterministic assignment: hash bucketing and exclusion layers.

Bucketing hashes ``salt:unit_id`` with SHA-256 and maps the first
eight bytes onto [0, 1). The same unit therefore always lands in the
same variant — across processes, restarts, and languages that can hash
the same string — with no assignment state to store or lose. Changing
an experiment's salt re-randomizes it, which is why the salt defaults
to the experiment id and is part of the definition.

Mutual-exclusion layers solve a different problem: two experiments
that must never see the same unit (say, two pricing tests) share a
*layer*. Every unit hashes once into the layer's [0, 1) space; each
experiment in the layer owns a disjoint slice of that space sized by
its ``allocation_fraction``. A unit can fall into at most one slice,
so overlap is impossible by construction — the registry rejects any
registration that would push a layer past 100%.
"""

from __future__ import annotations

import hashlib

from .models import Experiment


def unit_interval(unit_id: str, salt: str) -> float:
    """Hash ``salt:unit_id`` deterministically onto [0, 1)."""
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def unit_bucket(unit_id: str, salt: str, buckets: int = 10_000) -> int:
    """The unit's bucket in ``range(buckets)`` under ``salt``."""
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


class AssignmentEngine:
    """Stateless assigner; all state lives in the experiment definition."""

    def assign(
        self,
        experiment: Experiment,
        unit_id: str,
        attributes: dict | None = None,
    ) -> str | None:
        """Return the variant name for ``unit_id``, or None.

        None means the unit does not enter the experiment: it failed
        an eligibility rule, or it fell outside this experiment's
        slice of its mutual-exclusion layer.
        """
        if not experiment.is_eligible(attributes):
            return None
        if experiment.layer is not None:
            layer_pos = unit_interval(unit_id, f"layer:{experiment.layer}")
            start = experiment.layer_start
            if not (start <= layer_pos < start + experiment.allocation_fraction):
                return None
        pos = unit_interval(unit_id, f"exp:{experiment.id}:{experiment.salt}")
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.weight
            if pos < cumulative:
                return variant.name
        return experiment.variants[-1].name  # float-rounding tail

    def assign_many(
        self, experiment: Experiment, unit_ids: list[str]
    ) -> dict[str, str | None]:
        return {uid: self.assign(experiment, uid) for uid in unit_ids}
