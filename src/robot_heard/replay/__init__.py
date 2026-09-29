"""Deterministic synthetic replay contracts and runtime."""

from robot_heard.replay.clock import FakeClock, LogicalClock
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
from robot_heard.replay.source import ReplaySource, SyntheticSource
from robot_heard.replay.trace import TraceWriter
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
    "HypothesisEvent",
    "HypothesisKind",
    "LogicalClock",
    "ReplayMode",
    "ReplayRunResult",
    "ReplaySource",
    "StreamMetadata",
    "SyntheticSource",
    "TraceEvent",
    "TraceWriter",
    "MultiChannelPcmWavSource",
    "WavSourceError",
]
