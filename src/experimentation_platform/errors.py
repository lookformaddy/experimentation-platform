"""Typed errors for the experimentation platform.

Every failure mode callers are expected to handle is a subclass of
``ExperimentationError`` so API layers can map them uniformly
(validation problems -> 4xx, missing experiments -> 404).
"""

from __future__ import annotations


class ExperimentationError(Exception):
    """Base class for all experimentation-platform errors."""


class ExperimentValidationError(ExperimentationError):
    """An experiment definition failed validation."""


class ExperimentNotFoundError(ExperimentationError):
    def __init__(self, experiment_id: str):
        super().__init__(f"experiment not found: {experiment_id!r}")
        self.experiment_id = experiment_id


class InvalidTransitionError(ExperimentationError):
    """A lifecycle transition that the state machine does not allow."""


class LayerAllocationError(ExperimentationError):
    """Experiments in a mutual-exclusion layer would overlap."""
