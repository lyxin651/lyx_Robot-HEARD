"""Independent streaming consumer interface.

This contract is intentionally parallel to ``robot_heard.asr.ASRBackend`` and
does not inherit from or modify the offline file-based ASR interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from robot_heard.replay.events import (
    AudioChunk,
    ConsumerCapabilities,
    OutputEvent,
    StreamMetadata,
)


class StreamingConsumer(ABC):
    """Consumer lifecycle used by deterministic replay and future adapters."""

    @abstractmethod
    def start(self, metadata: StreamMetadata) -> None:
        """Initialize one stream using only consumer-visible metadata."""

    @abstractmethod
    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        """Consume one delivered chunk and return zero or more output events."""

    @abstractmethod
    def finish(self) -> Sequence[OutputEvent]:
        """Flush after source end has become visible."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all state before the next isolated stream."""

    @abstractmethod
    def capabilities(self) -> ConsumerCapabilities:
        """Return the explicit, immutable capability declaration."""
