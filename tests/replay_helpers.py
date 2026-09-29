"""Synthetic consumers shared by R1 contract tests."""

from __future__ import annotations

from typing import List, Optional, Sequence

from robot_heard.replay.events import (
    AudioChunk,
    ConsumerCapabilities,
    EndpointEvent,
    ErrorEvent,
    HypothesisEvent,
    HypothesisKind,
    OutputEvent,
    StreamMetadata,
)
from robot_heard.streaming.base import StreamingConsumer


class FakeLatencyConsumer(StreamingConsumer):
    """Test-only synchronous consumer whose work advances a FakeWallClock."""

    def __init__(
        self,
        wall_clock,
        *,
        compute_sec: float,
        finish_compute_sec: float = 0.0,
        emit_hypothesis: bool = True,
        reports_consumed_position: bool = False,
    ) -> None:
        self.wall_clock = wall_clock
        self.compute_sec = compute_sec
        self.finish_compute_sec = finish_compute_sec
        self.emit_hypothesis = emit_hypothesis
        self.reports_consumed_position = reports_consumed_position
        self.stream_id: Optional[str] = None
        self.chunks: List[AudioChunk] = []
        self._last_event_id: Optional[str] = None
        self._revision = -1

    def start(self, metadata: StreamMetadata) -> None:
        self.stream_id = metadata.stream_id

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        self.wall_clock.advance(self.compute_sec)
        self.chunks.append(chunk)
        if not self.emit_hypothesis:
            return ()
        self._revision += 1
        event_id = f"latency-partial-{chunk.sequence_id}"
        event = HypothesisEvent(
            stream_id=chunk.stream_id,
            scope_id="latency-scope",
            event_id=event_id,
            revision_index=self._revision,
            text=f"chunk-{chunk.sequence_id}",
            kind=HypothesisKind.PARTIAL,
            replaces_event_id=self._last_event_id,
            consumed_through_sample=(
                chunk.end_sample if self.reports_consumed_position else None
            ),
        )
        self._last_event_id = event_id
        return (event,)

    def finish(self) -> Sequence[OutputEvent]:
        self.wall_clock.advance(self.finish_compute_sec)
        if not self.emit_hypothesis:
            return ()
        self._revision += 1
        event = HypothesisEvent(
            stream_id=self.stream_id,
            scope_id="latency-scope",
            event_id="latency-final",
            revision_index=self._revision,
            text="final",
            kind=HypothesisKind.FINAL,
            replaces_event_id=self._last_event_id,
            consumed_through_sample=(
                self.chunks[-1].end_sample
                if self.reports_consumed_position and self.chunks
                else None
            ),
        )
        self._last_event_id = event.event_id
        return (event,)

    def reset(self) -> None:
        self.stream_id = None
        self.chunks.clear()
        self._last_event_id = None
        self._revision = -1

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities(
            supports_partial=self.emit_hypothesis,
            supports_revision=self.emit_hypothesis,
            reports_consumed_position=self.reports_consumed_position,
        )


class RecordingConsumer(StreamingConsumer):
    def __init__(self) -> None:
        self.metadata = None
        self.chunks: List[AudioChunk] = []
        self.reset_count = 0

    def start(self, metadata: StreamMetadata) -> None:
        self.metadata = metadata.consumer_visible_dict()

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        self.chunks.append(chunk)
        return ()

    def finish(self) -> Sequence[OutputEvent]:
        return ()

    def reset(self) -> None:
        self.reset_count += 1
        self.metadata = None
        self.chunks.clear()

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities()

class SnapshotConsumer(StreamingConsumer):
    def __init__(self, *, reports_consumed_position: bool = True) -> None:
        self.reports_consumed_position = reports_consumed_position
        self.metadata = None
        self.chunks: List[AudioChunk] = []
        self._last_event_id = None
        self._revision = -1
        self._last_sample = None

    def start(self, metadata: StreamMetadata) -> None:
        self.metadata = metadata.consumer_visible_dict()

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        self.chunks.append(chunk)
        self._revision += 1
        event_id = f"partial-{chunk.sequence_id}"
        event = HypothesisEvent(
            stream_id=chunk.stream_id,
            scope_id="scope-0",
            event_id=event_id,
            revision_index=self._revision,
            text=f"chunk-{chunk.sequence_id}",
            kind=HypothesisKind.PARTIAL,
            replaces_event_id=self._last_event_id,
            consumed_through_sample=(chunk.end_sample if self.reports_consumed_position else None),
        )
        self._last_event_id = event_id
        self._last_sample = chunk.end_sample
        return (event,)

    def finish(self) -> Sequence[OutputEvent]:
        self._revision += 1
        event = HypothesisEvent(
            stream_id=self.metadata["stream_id"],
            scope_id="scope-0",
            event_id="final-0",
            revision_index=self._revision,
            text="final",
            kind=HypothesisKind.FINAL,
            replaces_event_id=self._last_event_id,
            consumed_through_sample=(self._last_sample if self.reports_consumed_position else None),
        )
        return (event,)

    def reset(self) -> None:
        self.metadata = None
        self.chunks.clear()
        self._last_event_id = None
        self._revision = -1
        self._last_sample = None

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities(
            supports_partial=True,
            supports_revision=True,
            reports_consumed_position=self.reports_consumed_position,
            stateful=True,
        )


class EndpointThenFinalConsumer(StreamingConsumer):
    def __init__(self) -> None:
        self.stream_id = None

    def start(self, metadata: StreamMetadata) -> None:
        self.stream_id = metadata.stream_id

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        return (
            EndpointEvent(
                stream_id=self.stream_id,
                scope_id="scope-endpoint",
                event_id="endpoint-0",
            ),
            HypothesisEvent(
                stream_id=self.stream_id,
                scope_id="scope-endpoint",
                event_id="final-0",
                revision_index=0,
                text="done",
                kind=HypothesisKind.FINAL,
            ),
        )

    def finish(self) -> Sequence[OutputEvent]:
        return ()

    def reset(self) -> None:
        self.stream_id = None

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities(supports_endpoint=True)


class FinalThenEndpointConsumer(StreamingConsumer):
    def __init__(self) -> None:
        self.stream_id = None

    def start(self, metadata: StreamMetadata) -> None:
        self.stream_id = metadata.stream_id

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        return (
            HypothesisEvent(
                stream_id=self.stream_id,
                scope_id="scope-final-endpoint",
                event_id="final-0",
                revision_index=0,
                text="done",
                kind=HypothesisKind.FINAL,
            ),
            EndpointEvent(
                stream_id=self.stream_id,
                scope_id="scope-final-endpoint",
                event_id="endpoint-0",
            ),
        )

    def finish(self) -> Sequence[OutputEvent]:
        return ()

    def reset(self) -> None:
        self.stream_id = None

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities(supports_endpoint=True)


class FinalThenPartialConsumer(StreamingConsumer):
    def __init__(self) -> None:
        self.stream_id = None

    def start(self, metadata: StreamMetadata) -> None:
        self.stream_id = metadata.stream_id

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        return (
            HypothesisEvent(
                stream_id=self.stream_id,
                scope_id="scope-final",
                event_id="final-0",
                revision_index=0,
                text="done",
                kind=HypothesisKind.FINAL,
            ),
            HypothesisEvent(
                stream_id=self.stream_id,
                scope_id="scope-final",
                event_id="partial-after-final",
                revision_index=1,
                text="changed",
                kind=HypothesisKind.PARTIAL,
                replaces_event_id="final-0",
            ),
        )

    def finish(self) -> Sequence[OutputEvent]:
        return ()

    def reset(self) -> None:
        self.stream_id = None

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities(supports_partial=True, supports_revision=True)


class FinalThenSecondFinalConsumer(StreamingConsumer):
    def __init__(self) -> None:
        self.stream_id = None

    def start(self, metadata: StreamMetadata) -> None:
        self.stream_id = metadata.stream_id

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        return (
            HypothesisEvent(
                stream_id=self.stream_id,
                scope_id="scope-final",
                event_id="final-0",
                revision_index=0,
                text="done",
                kind=HypothesisKind.FINAL,
            ),
            HypothesisEvent(
                stream_id=self.stream_id,
                scope_id="scope-final",
                event_id="final-1",
                revision_index=1,
                text="changed",
                kind=HypothesisKind.FINAL,
                replaces_event_id="final-0",
            ),
        )

    def finish(self) -> Sequence[OutputEvent]:
        return ()

    def reset(self) -> None:
        self.stream_id = None

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities(supports_revision=True)


class RecoverableThenPartialConsumer(StreamingConsumer):
    def __init__(self) -> None:
        self.stream_id = None

    def start(self, metadata: StreamMetadata) -> None:
        self.stream_id = metadata.stream_id

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        return (
            ErrorEvent(
                stream_id=self.stream_id,
                scope_id="scope-recoverable",
                event_id="recoverable-0",
                message="synthetic recoverable error",
                fatal=False,
                recoverable=True,
            ),
            HypothesisEvent(
                stream_id=self.stream_id,
                scope_id="scope-recoverable",
                event_id="partial-after-recoverable",
                revision_index=0,
                text="recovered",
                kind=HypothesisKind.PARTIAL,
            ),
        )

    def finish(self) -> Sequence[OutputEvent]:
        return ()

    def reset(self) -> None:
        self.stream_id = None

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities(supports_partial=True)


class FatalThenFinalConsumer(StreamingConsumer):
    def __init__(self) -> None:
        self.stream_id = None

    def start(self, metadata: StreamMetadata) -> None:
        self.stream_id = metadata.stream_id

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        return (
            ErrorEvent(
                stream_id=self.stream_id,
                scope_id="scope-error",
                event_id="fatal-0",
                message="synthetic fatal",
                fatal=True,
                recoverable=False,
            ),
            HypothesisEvent(
                stream_id=self.stream_id,
                scope_id="scope-error",
                event_id="final-after-fatal",
                revision_index=0,
                text="invalid",
                kind=HypothesisKind.FINAL,
            ),
        )

    def finish(self) -> Sequence[OutputEvent]:
        return ()

    def reset(self) -> None:
        self.stream_id = None

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities()
