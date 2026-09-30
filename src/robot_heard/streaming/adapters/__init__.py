"""Streaming adapters built on top of the model-agnostic consumer contract."""

from robot_heard.streaming.adapters.offline_whisper import (
    OfflineWhisperReferenceConsumer,
)
from robot_heard.streaming.adapters.rolling_whisper import (
    RollingWhisperConsumer,
    RollingWhisperError,
)
from robot_heard.streaming.adapters.wenet_u2pp import (
    CtypesWeNetDecoderFactory,
    NativeDecodeResult,
    WeNetU2PPError,
    WeNetU2PPStreamingConsumer,
    parse_native_result,
    pcm32_to_pcm16le,
)

__all__ = [
    "OfflineWhisperReferenceConsumer",
    "RollingWhisperConsumer",
    "RollingWhisperError",
    "CtypesWeNetDecoderFactory",
    "NativeDecodeResult",
    "WeNetU2PPError",
    "WeNetU2PPStreamingConsumer",
    "parse_native_result",
    "pcm32_to_pcm16le",
]
