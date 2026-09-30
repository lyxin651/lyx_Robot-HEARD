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
from robot_heard.replay.metrics import (
    MetricsError,
    compute_paced_metrics,
    compute_r4_metrics,
    compute_r5a_metrics,
    write_metrics_json,
)
from robot_heard.replay.source import ReplaySource, SyntheticSource
from robot_heard.replay.trace import TraceDurabilityPolicy, TraceWriter
from robot_heard.replay.wav_source import MultiChannelPcmWavSource, WavSourceError
from robot_heard.replay.oracle import (
    OracleRelease,
    OracleReleaseGate,
    OracleSegmentDescriptor,
    ScopeBoundaryRelease,
    ScopeBoundaryReleaseGate,
    ScopeEndRelease,
    ScopeStartRelease,
    Stage2OracleReleaseGate,
    Stage2ScopeBoundaryReleaseGate,
)

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
    "OracleRelease",
    "OracleReleaseGate",
    "OracleSegmentDescriptor",
    "ScopeBoundaryRelease",
    "ScopeBoundaryReleaseGate",
    "ScopeEndRelease",
    "ScopeStartRelease",
    "Stage2OracleReleaseGate",
    "Stage2ScopeBoundaryReleaseGate",
    "MetricsError",
    "compute_paced_metrics",
    "compute_r4_metrics",
    "compute_r5a_metrics",
    "write_metrics_json",
]
