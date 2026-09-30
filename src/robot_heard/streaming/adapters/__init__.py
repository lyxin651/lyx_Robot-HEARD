"""Streaming adapters built on top of the model-agnostic consumer contract."""

from robot_heard.streaming.adapters.offline_whisper import (
    OfflineWhisperReferenceConsumer,
)
from robot_heard.streaming.adapters.rolling_whisper import (
    RollingWhisperConsumer,
    RollingWhisperError,
)

__all__ = [
    "OfflineWhisperReferenceConsumer",
    "RollingWhisperConsumer",
    "RollingWhisperError",
]
