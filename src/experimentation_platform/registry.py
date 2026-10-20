"""Experiment registry: storage plus layer bookkeeping.

The registry owns the one cross-experiment invariant — layer slices
never overlap. Slices are handed out first-fit in registration order:
each experiment joining a layer starts where the previous one ended.
Deregistration is deliberately not offered; concluded experiments keep
their slice so historical assignments stay interpretable.
"""

from __future__ import annotations

from .errors import ExperimentNotFoundError, LayerAllocationError
from .models import Experiment


class ExperimentRegistry:
    def __init__(self) -> None:
        self._experiments: dict[str, Experiment] = {}
        self._layer_used: dict[str, float] = {}

    def register(self, experiment: Experiment) -> Experiment:
        if experiment.id in self._experiments:
            raise LayerAllocationError(
                f"experiment already registered: {experiment.id!r}"
            )
        if experiment.layer is not None:
            used = self._layer_used.get(experiment.layer, 0.0)
            if used + experiment.allocation_fraction > 1.0 + 1e-9:
                raise LayerAllocationError(
                    f"layer {experiment.layer!r} is full: {used:.2f} allocated, "
                    f"cannot add {experiment.allocation_fraction:.2f}"
                )
            experiment.layer_start = used
            self._layer_used[experiment.layer] = used + experiment.allocation_fraction
        self._experiments[experiment.id] = experiment
        return experiment

    def get(self, experiment_id: str) -> Experiment:
        try:
            return self._experiments[experiment_id]
        except KeyError:
            raise ExperimentNotFoundError(experiment_id) from None

    def list(self) -> list[Experiment]:
        return list(self._experiments.values())

    def layer_usage(self, layer: str) -> float:
        return self._layer_used.get(layer, 0.0)
