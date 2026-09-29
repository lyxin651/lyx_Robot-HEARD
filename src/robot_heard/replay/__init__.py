"""Deterministic synthetic replay contracts and runtime."""

from robot_heard.replay.clock import (
    FakeClock,
    FakeWallClock,
    LogicalClock,
    MonotonicWallClock,
    WallClock,
)
from robot_heard.replay.events import (
    AudioChunk,
    ConsumerCapabilities,
    ContractValidationError,
    EndpointEvent,
    ErrorEvent,
    EventType,
    HypothesisEvent,
    HypothesisKind,
    ReplayMode,
    StreamMetadata,
    TraceEvent,
)
from robot_heard.replay.runtime import (
    DeterministicReplayRuntime,
    ReplayRunResult,
)
from robot_heard.replay.paced_runtime import PacedReplayRunResult, PacedReplayRuntime
from robot_heard.replay.metrics import MetricsError, compute_paced_metrics, write_metrics_json
from robot_heard.replay.source import ReplaySource, SyntheticSource
from robot_heard.replay.trace import TraceDurabilityPolicy, TraceWriter
from robot_heard.replay.wav_source import MultiChannelPcmWavSource, WavSourceError

__all__ = [
    "AudioChunk",
    "ConsumerCapabilities",
    "ContractValidationError",
    "DeterministicReplayRuntime",
    "EndpointEvent",
    "ErrorEvent",
    "EventType",
    "FakeClock",
    "FakeWallClock",
    "HypothesisEvent",
    "HypothesisKind",
    "LogicalClock",
    "MonotonicWallClock",
    "WallClock",
    "PacedReplayRuntime",
    "PacedReplayRunResult",
    "ReplayMode",
    "ReplayRunResult",
    "ReplaySource",
    "StreamMetadata",
    "SyntheticSource",
    "TraceEvent",
    "TraceDurabilityPolicy",
    "TraceWriter",
    "MultiChannelPcmWavSource",
    "WavSourceError",
    "MetricsError",
    "compute_paced_metrics",
    "write_metrics_json",
]
