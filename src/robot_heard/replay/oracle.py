"""Runtime-owned oracle release gates for bounded replay metadata.

The gate may retain evaluator-owned timeline rows, but the descriptor exposed
to a consumer contains only the released segment identity and integer sample
interval.  References, speakers, and future rows never cross this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Tuple, Union

from robot_heard.replay.events import ContractValidationError


@dataclass(frozen=True)
class OracleSegmentDescriptor:
    """Consumer-visible descriptor for one causally released interval."""

    segment_id: str
    start_sample: int
    end_sample: int

    def __post_init__(self) -> None:
        if not isinstance(self.segment_id, str) or not self.segment_id.strip():
            raise ContractValidationError("segment_id must be a non-empty string")
        if (
            isinstance(self.start_sample, bool)
            or not isinstance(self.start_sample, int)
            or self.start_sample < 0
        ):
            raise ContractValidationError("start_sample must be a non-negative integer")
        if (
            isinstance(self.end_sample, bool)
            or not isinstance(self.end_sample, int)
            or self.end_sample <= self.start_sample
        ):
            raise ContractValidationError("end_sample must be greater than start_sample")


@dataclass(frozen=True)
class OracleRelease:
    """Evidence returned when a descriptor becomes visible to a consumer."""

    descriptor: OracleSegmentDescriptor
    segment_end_available_wall_offset_sec: float
    oracle_release_wall_offset_sec: float
    oracle_release_quantization_sec: float
    condition: str

    def __post_init__(self) -> None:
        values = (
            self.segment_end_available_wall_offset_sec,
            self.oracle_release_wall_offset_sec,
            self.oracle_release_quantization_sec,
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ContractValidationError("oracle release timing values must be numeric")
        if self.segment_end_available_wall_offset_sec < 0:
            raise ContractValidationError("segment end availability must be non-negative")
        if self.oracle_release_wall_offset_sec + 1e-12 < self.segment_end_available_wall_offset_sec:
            raise ContractValidationError("oracle release cannot precede segment end availability")
        if self.oracle_release_quantization_sec < -1e-12:
            raise ContractValidationError("oracle release quantization cannot be negative")
        if not isinstance(self.condition, str) or not self.condition.strip():
            raise ContractValidationError("oracle condition must be non-empty")


class OracleReleaseGate(Protocol):
    """Minimal generic runtime/consumer release boundary."""

    condition: str

    def on_source_available(
        self,
        available_through_sample: int,
        scheduled_availability_wall_offset_sec: float,
    ) -> Tuple[OracleRelease, ...]:
        ...

    @property
    def remaining_count(self) -> int:
        ...


@dataclass(frozen=True)
class ScopeStartRelease:
    """Causal release of a scope start boundary only.

    This object intentionally has no end sample.  It is the only Stage-2
    boundary metadata visible to a rolling consumer before the end boundary
    becomes available.
    """

    segment_id: str
    start_sample: int
    scope_start_available_wall_offset_sec: float
    oracle_release_wall_offset_sec: float
    oracle_release_quantization_sec: float
    condition: str

    def __post_init__(self) -> None:
        OracleSegmentDescriptor(self.segment_id, self.start_sample, self.start_sample + 1)
        _validate_release_timing(
            self.scope_start_available_wall_offset_sec,
            self.oracle_release_wall_offset_sec,
            self.oracle_release_quantization_sec,
            self.condition,
        )


@dataclass(frozen=True)
class ScopeEndRelease:
    """Causal release of a scope end boundary only."""

    segment_id: str
    end_sample: int
    scope_end_available_wall_offset_sec: float
    oracle_release_wall_offset_sec: float
    oracle_release_quantization_sec: float
    condition: str

    def __post_init__(self) -> None:
        if not isinstance(self.segment_id, str) or not self.segment_id.strip():
            raise ContractValidationError("segment_id must be a non-empty string")
        if isinstance(self.end_sample, bool) or not isinstance(self.end_sample, int) or self.end_sample <= 0:
            raise ContractValidationError("end_sample must be a positive integer")
        _validate_release_timing(
            self.scope_end_available_wall_offset_sec,
            self.oracle_release_wall_offset_sec,
            self.oracle_release_quantization_sec,
            self.condition,
        )


def _validate_release_timing(
    boundary_available: float,
    release_offset: float,
    quantization: float,
    condition: str,
) -> None:
    values = (boundary_available, release_offset, quantization)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ContractValidationError("scope release timing values must be numeric")
    if boundary_available < 0 or release_offset < 0:
        raise ContractValidationError("scope release timing cannot be negative")
    if release_offset + 1e-12 < boundary_available:
        raise ContractValidationError("scope release cannot precede boundary availability")
    if quantization < -1e-12:
        raise ContractValidationError("scope release quantization cannot be negative")
    if not isinstance(condition, str) or not condition.strip():
        raise ContractValidationError("oracle condition must be non-empty")


ScopeBoundaryRelease = Union[ScopeStartRelease, ScopeEndRelease]


class ScopeBoundaryReleaseGate(Protocol):
    """Generic runtime-owned gate for causally released scope boundaries."""

    condition: str

    def on_source_available(
        self,
        available_through_sample: int,
        scheduled_availability_wall_offset_sec: float,
    ) -> Tuple[ScopeBoundaryRelease, ...]:
        ...

    @property
    def remaining_count(self) -> int:
        ...


class Stage2OracleReleaseGate:
    """Release Stage-2 intervals only after their end sample is available.

    The complete timeline is evaluator/runtime-owned.  ``on_source_available``
    returns descriptor-only release evidence to the runtime, which may forward
    individual releases to a consumer.  Rows are sorted by end sample, with
    segment ID as deterministic tie-break.
    """

    def __init__(
        self,
        segments: Iterable[object],
        *,
        sample_rate: int,
        condition: str = "r4_precomputed_segment_end_gate_raw_ch0",
        packet_duration_sec: Optional[float] = None,
    ) -> None:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ContractValidationError("sample_rate must be a positive integer")
        if not isinstance(condition, str) or not condition.strip():
            raise ContractValidationError("condition must be a non-empty string")
        if packet_duration_sec is not None and packet_duration_sec <= 0:
            raise ContractValidationError("packet_duration_sec must be positive")
        normalized = []
        seen = set()
        for row in segments:
            segment_id = getattr(row, "segment_id", None)
            start_sample = getattr(row, "start_sample", None)
            end_sample = getattr(row, "end_sample", None)
            descriptor = OracleSegmentDescriptor(segment_id, start_sample, end_sample)
            if descriptor.segment_id in seen:
                raise ContractValidationError(f"duplicate oracle segment_id={descriptor.segment_id!r}")
            seen.add(descriptor.segment_id)
            normalized.append(descriptor)
        normalized.sort(key=lambda item: (item.end_sample, item.segment_id))
        self._segments: Tuple[OracleSegmentDescriptor, ...] = tuple(normalized)
        self._sample_rate = sample_rate
        self.condition = condition
        self._packet_duration_sec = packet_duration_sec
        self._next_index = 0

    @property
    def release_count(self) -> int:
        return self._next_index

    @property
    def remaining_count(self) -> int:
        return len(self._segments) - self._next_index

    def on_source_available(
        self,
        available_through_sample: int,
        scheduled_availability_wall_offset_sec: float,
    ) -> Tuple[OracleRelease, ...]:
        if (
            isinstance(available_through_sample, bool)
            or not isinstance(available_through_sample, int)
            or available_through_sample < 0
        ):
            raise ContractValidationError("available_through_sample must be non-negative integer")
        if scheduled_availability_wall_offset_sec < 0:
            raise ContractValidationError("scheduled availability offset must be non-negative")
        releases = []
        while self._next_index < len(self._segments):
            descriptor = self._segments[self._next_index]
            if descriptor.end_sample > available_through_sample:
                break
            end_offset = descriptor.end_sample / self._sample_rate
            quantization = scheduled_availability_wall_offset_sec - end_offset
            if quantization < -1e-12:
                raise ContractValidationError(
                    f"oracle release precedes segment end: {descriptor.segment_id!r}"
                )
            if (
                self._packet_duration_sec is not None
                and quantization >= self._packet_duration_sec + 1e-12
            ):
                raise ContractValidationError(
                    f"oracle release quantization exceeds packet duration: {descriptor.segment_id!r}"
                )
            release = OracleRelease(
                descriptor=descriptor,
                segment_end_available_wall_offset_sec=end_offset,
                oracle_release_wall_offset_sec=scheduled_availability_wall_offset_sec,
                oracle_release_quantization_sec=max(0.0, quantization),
                condition=self.condition,
            )
            releases.append(release)
            self._next_index += 1
        return tuple(releases)


class Stage2ScopeBoundaryReleaseGate:
    """Release Stage-2 starts at S and ends at E, independently.

    The complete timeline is retained only by this runtime-owned object.
    Release ordering is deterministic by boundary sample, then START before
    END at the same sample, then segment ID.  A consumer receives only one
    boundary record at a time through the runtime hook.
    """

    def __init__(
        self,
        segments: Iterable[object],
        *,
        sample_rate: int,
        condition: str = "r5a_oracle_segment_boundaries_raw_ch0",
        packet_duration_sec: Optional[float] = None,
    ) -> None:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ContractValidationError("sample_rate must be a positive integer")
        if not isinstance(condition, str) or not condition.strip():
            raise ContractValidationError("condition must be a non-empty string")
        if packet_duration_sec is not None and packet_duration_sec <= 0:
            raise ContractValidationError("packet_duration_sec must be positive")
        rows = []
        seen = set()
        for row in segments:
            segment_id = getattr(row, "segment_id", None)
            start_sample = getattr(row, "start_sample", None)
            end_sample = getattr(row, "end_sample", None)
            descriptor = OracleSegmentDescriptor(segment_id, start_sample, end_sample)
            if descriptor.segment_id in seen:
                raise ContractValidationError(f"duplicate oracle segment_id={descriptor.segment_id!r}")
            seen.add(descriptor.segment_id)
            rows.append(descriptor)
        events = []
        for descriptor in rows:
            events.append((descriptor.start_sample, 0, descriptor.segment_id, descriptor))
            events.append((descriptor.end_sample, 1, descriptor.segment_id, descriptor))
        events.sort(key=lambda item: (item[0], item[1], item[2]))
        self._events = tuple(events)
        self._sample_rate = sample_rate
        self._packet_duration_sec = packet_duration_sec
        self.condition = condition
        self._next_index = 0

    @property
    def release_count(self) -> int:
        return self._next_index

    @property
    def remaining_count(self) -> int:
        return len(self._events) - self._next_index

    def on_source_available(
        self,
        available_through_sample: int,
        scheduled_availability_wall_offset_sec: float,
    ) -> Tuple[ScopeBoundaryRelease, ...]:
        if (
            isinstance(available_through_sample, bool)
            or not isinstance(available_through_sample, int)
            or available_through_sample < 0
        ):
            raise ContractValidationError("available_through_sample must be non-negative integer")
        if (
            isinstance(scheduled_availability_wall_offset_sec, bool)
            or not isinstance(scheduled_availability_wall_offset_sec, (int, float))
            or scheduled_availability_wall_offset_sec < 0
        ):
            raise ContractValidationError("scheduled availability offset must be non-negative")
        releases = []
        while self._next_index < len(self._events):
            boundary_sample, boundary_kind, _, descriptor = self._events[self._next_index]
            if boundary_sample > available_through_sample:
                break
            boundary_offset = boundary_sample / self._sample_rate
            quantization = scheduled_availability_wall_offset_sec - boundary_offset
            if quantization < -1e-12:
                raise ContractValidationError(
                    f"scope release precedes boundary: {descriptor.segment_id!r}"
                )
            if (
                self._packet_duration_sec is not None
                and quantization >= self._packet_duration_sec + 1e-12
            ):
                raise ContractValidationError(
                    f"scope release quantization exceeds packet duration: {descriptor.segment_id!r}"
                )
            if boundary_kind == 0:
                releases.append(
                    ScopeStartRelease(
                        segment_id=descriptor.segment_id,
                        start_sample=descriptor.start_sample,
                        scope_start_available_wall_offset_sec=boundary_offset,
                        oracle_release_wall_offset_sec=scheduled_availability_wall_offset_sec,
                        oracle_release_quantization_sec=max(0.0, quantization),
                        condition=self.condition,
                    )
                )
            else:
                releases.append(
                    ScopeEndRelease(
                        segment_id=descriptor.segment_id,
                        end_sample=descriptor.end_sample,
                        scope_end_available_wall_offset_sec=boundary_offset,
                        oracle_release_wall_offset_sec=scheduled_availability_wall_offset_sec,
                        oracle_release_quantization_sec=max(0.0, quantization),
                        condition=self.condition,
                    )
                )
            self._next_index += 1
        return tuple(releases)
