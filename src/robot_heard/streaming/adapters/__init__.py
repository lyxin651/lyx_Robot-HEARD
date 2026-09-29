"""Streaming adapters built on top of the model-agnostic consumer contract."""

from robot_heard.streaming.adapters.offline_whisper import (
    OfflineWhisperReferenceConsumer,
)

__all__ = ["OfflineWhisperReferenceConsumer"]

