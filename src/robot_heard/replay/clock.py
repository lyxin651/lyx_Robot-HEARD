"""Logical and wall-clock abstractions for deterministic and paced replay."""

from __future__ import annotations

import time
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


class WallClock(Protocol):
    """Monotonic wall-clock surface used by compute-aware replay."""

    def now(self) -> float:
        """Return monotonic seconds."""

    def sleep_until(self, target_monotonic: float) -> None:
        """Wait until the target monotonic timestamp."""


class MonotonicWallClock:
    """Production wall clock backed only by ``perf_counter`` and sleep."""

    def now(self) -> float:
        return time.perf_counter()

    def sleep_until(self, target_monotonic: float) -> None:
        while True:
            remaining = target_monotonic - self.now()
            if remaining <= 0:
                return
            time.sleep(remaining)


@dataclass
class FakeWallClock:
    """Deterministic wall clock with no real sleeping."""

    current_time: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.current_time, bool) or not isinstance(self.current_time, (int, float)):
            raise ContractValidationError("current_time must be a number")
        if self.current_time < 0:
            raise ContractValidationError("current_time must be non-negative")

    def now(self) -> float:
        return float(self.current_time)

    def advance(self, seconds: float) -> None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ContractValidationError("seconds must be a number")
        if seconds < 0:
            raise ContractValidationError("wall clock cannot advance backwards")
        self.current_time += seconds

    def sleep_until(self, target_monotonic: float) -> None:
        if isinstance(target_monotonic, bool) or not isinstance(target_monotonic, (int, float)):
            raise ContractValidationError("target_monotonic must be a number")
        if target_monotonic < self.current_time:
            raise ContractValidationError(
                "fake wall clock cannot sleep backwards: "
                f"{target_monotonic} < {self.current_time}"
            )
        self.current_time = float(target_monotonic)

    def reset(self) -> None:
        self.current_time = 0.0


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
