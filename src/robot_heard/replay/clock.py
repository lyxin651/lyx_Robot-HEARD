"""Deterministic logical clocks for R1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from robot_heard.replay.events import ContractValidationError


class LogicalClock(Protocol):
    """Minimal sample-position clock consumed by deterministic replay."""

    @property
    def current_sample(self) -> int:
        ...

    def advance_to(self, sample_position: int) -> None:
        ...


@dataclass
class FakeClock:
    """A monotonic integer-sample clock with no wall-clock dependency."""

    current_sample: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.current_sample, bool) or not isinstance(self.current_sample, int):
            raise ContractValidationError("current_sample must be an integer")
        if self.current_sample < 0:
            raise ContractValidationError("current_sample must be non-negative")

    def advance_to(self, sample_position: int) -> None:
        if isinstance(sample_position, bool) or not isinstance(sample_position, int):
            raise ContractValidationError("sample_position must be an integer")
        if sample_position < self.current_sample:
            raise ContractValidationError(
                f"logical clock cannot move backwards: {sample_position} < {self.current_sample}"
            )
        self.current_sample = sample_position

    def reset(self) -> None:
        """Return the clock to sample zero for an isolated synthetic run."""

        self.current_sample = 0
