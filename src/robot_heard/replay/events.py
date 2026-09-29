"""R1 replay contract types.

The types in this module use integer sample positions as their audio-time
authority.  They intentionally contain no model or audio-library dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


class ContractValidationError(ValueError):
    """Raised when a replay contract invariant is violated."""


class ReplayMode(str, Enum):
    """R0 replay modes implemented or reserved by the runtime."""

    COMPUTE_UNAWARE = "COMPUTE_UNAWARE"
    COMPUTE_AWARE = "COMPUTE_AWARE"


class EventType(str, Enum):
    """Persisted runtime event categories."""

    SOURCE_AVAILABLE = "SOURCE_AVAILABLE"
    PACKET_DELIVERED = "PACKET_DELIVERED"
    CONSUMER_CALL_START = "CONSUMER_CALL_START"
    CONSUMER_CALL_RETURN = "CONSUMER_CALL_RETURN"
    HYPOTHESIS = "HYPOTHESIS"
    ENDPOINT = "ENDPOINT"
    ERROR = "ERROR"
    ORACLE_METADATA_RELEASE = "ORACLE_METADATA_RELEASE"
    SOURCE_END = "SOURCE_END"
    FINISH_START = "FINISH_START"
    FINISH_RETURN = "FINISH_RETURN"
    RESET = "RESET"


class HypothesisKind(str, Enum):
    """The two ASR hypothesis lifecycle states in the R1 contract."""

    PARTIAL = "PARTIAL"
    FINAL = "FINAL"


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractValidationError(f"{field_name} must be a positive integer")
    return value


def _validate_optional_sample(value: Any, field_name: str) -> None:
    if value is not None:
        _require_non_negative_int(value, field_name)


def _validate_optional_wall_time(value: Any, field_name: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
    ):
        raise ContractValidationError(f"{field_name} must be a non-negative number or null")


def _normalize_channel_ids(channel_ids: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(channel_ids, (str, bytes)):
        raise ContractValidationError("channel_ids must be a non-empty sequence")
    normalized = tuple(channel_ids)
    if not normalized:
        raise ContractValidationError("channel_ids must be non-empty")
    for index, channel_id in enumerate(normalized):
        _require_non_empty_string(channel_id, f"channel_ids[{index}]")
    if len(set(normalized)) != len(normalized):
        raise ContractValidationError("channel_ids must be unique")
    return normalized


def _normalize_samples(
    samples: Sequence[Sequence[Any]],
) -> Tuple[Tuple[Any, ...], ...]:
    if isinstance(samples, (str, bytes)):
        raise ContractValidationError("samples must be a channel-major sequence")
    try:
        normalized = tuple(tuple(channel) for channel in samples)
    except TypeError as exc:
        raise ContractValidationError("samples must be a channel-major sequence") from exc
    if not normalized:
        raise ContractValidationError("samples must contain at least one channel")
    frame_count = len(normalized[0])
    for index, channel in enumerate(normalized):
        if len(channel) != frame_count:
            raise ContractValidationError(
                f"samples channel {index} has {len(channel)} frames; expected {frame_count}"
            )
    return normalized


@dataclass(frozen=True)
class StreamMetadata:
    """Whitelist of metadata visible to a consumer at stream startup."""

    stream_id: str
    sample_rate: int
    channel_ids: Tuple[str, ...]
    sample_format: Optional[str] = None
    channel_layout: Optional[str] = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.stream_id, "stream_id")
        _require_positive_int(self.sample_rate, "sample_rate")
        object.__setattr__(self, "channel_ids", _normalize_channel_ids(self.channel_ids))
        if self.sample_format is not None:
            _require_non_empty_string(self.sample_format, "sample_format")
        if self.channel_layout is not None:
            _require_non_empty_string(self.channel_layout, "channel_layout")

    @property
    def num_channels(self) -> int:
        return len(self.channel_ids)

    def consumer_visible_dict(self) -> Mapping[str, Any]:
        """Return only the R0 startup metadata whitelist."""

        payload = {
            "stream_id": self.stream_id,
            "sample_rate": self.sample_rate,
            "num_channels": self.num_channels,
            "channel_ids": list(self.channel_ids),
        }
        if self.sample_format is not None:
            payload["sample_format"] = self.sample_format
        if self.channel_layout is not None:
            payload["channel_layout"] = self.channel_layout
        return MappingProxyType(payload)


@dataclass(frozen=True)
class AudioChunk:
    """One exact half-open source packet.

    ``samples`` is normalized to an immutable channel-major tuple.  This is a
    container normalization only; the runtime never pads, truncates, resamples,
    or downmixes sample values.
    """

    stream_id: str
    sequence_id: int
    start_sample: int
    end_sample: int
    sample_rate: int
    channel_ids: Tuple[str, ...]
    samples: Tuple[Tuple[Any, ...], ...]
    is_final_source_chunk: bool
    scheduled_availability_logical_sample: int
    scheduled_availability_wall_time: Optional[float] = None
    delivery_wall_time: Optional[float] = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.stream_id, "stream_id")
        _require_non_negative_int(self.sequence_id, "sequence_id")
        _require_non_negative_int(self.start_sample, "start_sample")
        _require_non_negative_int(self.end_sample, "end_sample")
        if self.end_sample <= self.start_sample:
            raise ContractValidationError("end_sample must be greater than start_sample")
        _require_positive_int(self.sample_rate, "sample_rate")
        if not isinstance(self.is_final_source_chunk, bool):
            raise ContractValidationError("is_final_source_chunk must be boolean")
        _require_non_negative_int(
            self.scheduled_availability_logical_sample,
            "scheduled_availability_logical_sample",
        )
        if self.scheduled_availability_logical_sample != self.end_sample:
            raise ContractValidationError(
                "scheduled_availability_logical_sample must equal end_sample"
            )
        object.__setattr__(self, "channel_ids", _normalize_channel_ids(self.channel_ids))
        object.__setattr__(self, "samples", _normalize_samples(self.samples))
        if len(self.channel_ids) != len(self.samples):
            raise ContractValidationError(
                "channel count must equal samples channel count"
            )
        expected_frames = self.end_sample - self.start_sample
        actual_frames = len(self.samples[0])
        if actual_frames != expected_frames:
            raise ContractValidationError(
                f"frame count {actual_frames} does not equal end-start {expected_frames}"
            )
        _validate_optional_wall_time(
            self.scheduled_availability_wall_time,
            "scheduled_availability_wall_time",
        )
        _validate_optional_wall_time(self.delivery_wall_time, "delivery_wall_time")

    @property
    def frame_count(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def num_channels(self) -> int:
        return len(self.channel_ids)


@dataclass(frozen=True)
class ConsumerCapabilities:
    """Explicit adapter capability declaration.

    All capabilities default to false.  The runtime never fabricates an event
    or metric for a false capability.
    """

    supports_partial: bool = False
    supports_revision: bool = False
    supports_stable_prefix: bool = False
    supports_endpoint: bool = False
    supports_token_timestamps: bool = False
    supports_word_timestamps: bool = False
    reports_consumed_position: bool = False
    native_streaming: bool = False
    stateful: bool = False

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, bool):
                raise ContractValidationError(f"{name} must be boolean")

    def to_dict(self) -> Mapping[str, bool]:
        return MappingProxyType(
            {
                "supports_partial": self.supports_partial,
                "supports_revision": self.supports_revision,
                "supports_stable_prefix": self.supports_stable_prefix,
                "supports_endpoint": self.supports_endpoint,
                "supports_token_timestamps": self.supports_token_timestamps,
                "supports_word_timestamps": self.supports_word_timestamps,
                "reports_consumed_position": self.reports_consumed_position,
                "native_streaming": self.native_streaming,
                "stateful": self.stateful,
            }
        )


@dataclass(frozen=True)
class HypothesisEvent:
    """A complete PARTIAL or FINAL hypothesis snapshot."""

    stream_id: str
    scope_id: str
    event_id: str
    revision_index: int
    text: str
    kind: HypothesisKind
    replaces_event_id: Optional[str] = None
    stable_prefix: Optional[str] = None
    consumed_through_sample: Optional[int] = None
    observed_available_through_sample: Optional[int] = None
    observed_delivered_through_sample: Optional[int] = None
    observed_wall_time: Optional[float] = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.stream_id, "stream_id")
        _require_non_empty_string(self.scope_id, "scope_id")
        _require_non_empty_string(self.event_id, "event_id")
        _require_non_negative_int(self.revision_index, "revision_index")
        if not isinstance(self.text, str):
            raise ContractValidationError("text must be a string")
        try:
            kind = HypothesisKind(self.kind)
        except ValueError as exc:
            raise ContractValidationError("kind must be PARTIAL or FINAL") from exc
        object.__setattr__(self, "kind", kind)
        if self.replaces_event_id is not None:
            _require_non_empty_string(self.replaces_event_id, "replaces_event_id")
        if self.stable_prefix is not None and not isinstance(self.stable_prefix, str):
            raise ContractValidationError("stable_prefix must be a string or null")
        _validate_optional_sample(self.consumed_through_sample, "consumed_through_sample")
        _validate_optional_sample(
            self.observed_available_through_sample,
            "observed_available_through_sample",
        )
        _validate_optional_sample(
            self.observed_delivered_through_sample,
            "observed_delivered_through_sample",
        )
        _validate_optional_wall_time(self.observed_wall_time, "observed_wall_time")


@dataclass(frozen=True)
class EndpointEvent:
    """A consumer endpoint belief, distinct from FINAL."""

    stream_id: str
    scope_id: str
    event_id: str
    consumed_through_sample: Optional[int] = None
    observed_available_through_sample: Optional[int] = None
    observed_delivered_through_sample: Optional[int] = None
    observed_wall_time: Optional[float] = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.stream_id, "stream_id")
        _require_non_empty_string(self.scope_id, "scope_id")
        _require_non_empty_string(self.event_id, "event_id")
        _validate_optional_sample(self.consumed_through_sample, "consumed_through_sample")
        _validate_optional_sample(
            self.observed_available_through_sample,
            "observed_available_through_sample",
        )
        _validate_optional_sample(
            self.observed_delivered_through_sample,
            "observed_delivered_through_sample",
        )
        _validate_optional_wall_time(self.observed_wall_time, "observed_wall_time")


@dataclass(frozen=True)
class ErrorEvent:
    """A fatal or explicitly recoverable consumer error."""

    stream_id: str
    scope_id: str
    event_id: str
    message: str
    fatal: bool
    recoverable: bool
    consumed_through_sample: Optional[int] = None
    observed_available_through_sample: Optional[int] = None
    observed_delivered_through_sample: Optional[int] = None
    observed_wall_time: Optional[float] = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.stream_id, "stream_id")
        _require_non_empty_string(self.scope_id, "scope_id")
        _require_non_empty_string(self.event_id, "event_id")
        _require_non_empty_string(self.message, "message")
        if not isinstance(self.fatal, bool) or not isinstance(self.recoverable, bool):
            raise ContractValidationError("fatal and recoverable must be boolean")
        if self.fatal == self.recoverable:
            raise ContractValidationError("exactly one of fatal and recoverable must be true")
        _validate_optional_sample(self.consumed_through_sample, "consumed_through_sample")
        _validate_optional_sample(
            self.observed_available_through_sample,
            "observed_available_through_sample",
        )
        _validate_optional_sample(
            self.observed_delivered_through_sample,
            "observed_delivered_through_sample",
        )
        _validate_optional_wall_time(self.observed_wall_time, "observed_wall_time")


OutputEvent = Union[HypothesisEvent, EndpointEvent, ErrorEvent]


@dataclass(frozen=True)
class TraceEvent:
    """One persisted JSONL runtime event record."""

    schema_version: str
    event_index: int
    stream_id: str
    event_type: EventType
    logical_sample_position: Optional[int]
    payload: Mapping[str, Any]
    sequence_id: Optional[int] = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.schema_version, "schema_version")
        _require_non_negative_int(self.event_index, "event_index")
        _require_non_empty_string(self.stream_id, "stream_id")
        try:
            event_type = EventType(self.event_type)
        except ValueError as exc:
            raise ContractValidationError("event_type is not a known EventType") from exc
        object.__setattr__(self, "event_type", event_type)
        _validate_optional_sample(self.logical_sample_position, "logical_sample_position")
        if self.sequence_id is not None:
            _require_non_negative_int(self.sequence_id, "sequence_id")
        if not isinstance(self.payload, Mapping):
            raise ContractValidationError("payload must be a mapping")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_record(self) -> Mapping[str, Any]:
        """Return a JSON-compatible record without mutating the event."""

        record = {
            "schema_version": self.schema_version,
            "event_index": self.event_index,
            "stream_id": self.stream_id,
            "event_type": self.event_type.value,
            "logical_sample_position": self.logical_sample_position,
            "payload": dict(self.payload),
        }
        if self.sequence_id is not None:
            record["sequence_id"] = self.sequence_id
        return MappingProxyType(record)
