"""Compute-aware, single-threaded wall-clock paced replay for R3.

The deterministic runtime remains the implementation of COMPUTE_UNAWARE.  This
module owns the separate COMPUTE_AWARE scheduler: source availability is driven
by a monotonic wall clock, while a synchronous consumer may block and allow the
source queue to accumulate losslessly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, List, Optional, Tuple

from robot_heard.replay.clock import MonotonicWallClock, WallClock
from robot_heard.replay.events import (
    AudioChunk,
    ConsumerCapabilities,
    ContractValidationError,
    EndpointEvent,
    ErrorEvent,
    EventType,
    HypothesisEvent,
    OutputEvent,
    TraceEvent,
)
from robot_heard.replay.runtime import LifecycleValidator
from robot_heard.replay.source import ReplaySource
from robot_heard.replay.trace import TraceWriter
from robot_heard.streaming.base import StreamingConsumer


@dataclass(frozen=True)
class PacedReplayRunResult:
    """Summary of one completed compute-aware run."""

    trace_events: Tuple[TraceEvent, ...]
    source_end_sample: int
    run_wall_origin: float
    measured_wall_interval_sec: float
    source_end_wall_offset_sec: float
    finish_return_wall_offset_sec: float


class PacedReplayRuntime:
    """Replay a source against wall availability with a FIFO lossless queue.

    The scheduler is intentionally single-threaded.  A blocking synchronous
    ``consume`` call advances real wall time (or a test fake), and the return
    path catches up every source packet whose scheduled availability has
    arrived before recording the call return and its outputs.
    """

    def __init__(
        self,
        source: ReplaySource,
        consumer: StreamingConsumer,
        trace_writer: TraceWriter,
        *,
        wall_clock: Optional[WallClock] = None,
    ) -> None:
        self.source = source
        self.consumer = consumer
        self.trace_writer = trace_writer
        self.wall_clock = wall_clock or MonotonicWallClock()
        self.capabilities = consumer.capabilities()
        if not isinstance(self.capabilities, ConsumerCapabilities):
            raise ContractValidationError(
                "consumer.capabilities() must return ConsumerCapabilities"
            )
        self._validator = LifecycleValidator(source.metadata.stream_id, self.capabilities)
        self._trace_events: List[TraceEvent] = []
        self._available_through_sample = 0
        self._delivered_through_sample = 0
        self._consumed_through_sample: Optional[int] = None
        self._source_end_visible = False
        self._started = False
        self._finished = False
        self._run_wall_origin: Optional[float] = None
        self._last_wall_offset = 0.0

    @property
    def trace_events(self) -> Tuple[TraceEvent, ...]:
        return tuple(self._trace_events)

    @property
    def available_through_sample(self) -> int:
        return self._available_through_sample

    @property
    def delivered_through_sample(self) -> int:
        return self._delivered_through_sample

    @property
    def consumed_through_sample(self) -> Optional[int]:
        return self._consumed_through_sample

    @property
    def run_wall_origin(self) -> Optional[float]:
        return self._run_wall_origin

    def _wall_offset(self, wall_time: float) -> float:
        if self._run_wall_origin is None:
            raise RuntimeError("run wall origin has not been established")
        offset = wall_time - self._run_wall_origin
        if offset < -1e-12:
            raise ContractValidationError("wall time precedes run wall origin")
        return max(0.0, offset)

    def _scheduled_wall_time(self, end_sample: int) -> float:
        if self._run_wall_origin is None:
            raise RuntimeError("run wall origin has not been established")
        return self._run_wall_origin + (end_sample / self.source.metadata.sample_rate)

    def _clock_origin_kind(self) -> str:
        class_name = type(self.wall_clock).__name__
        if class_name == "FakeWallClock":
            return "fake_wall_clock"
        if class_name == "MonotonicWallClock":
            return "monotonic_wall_clock"
        return f"custom_wall_clock:{class_name}"

    def _state_payload(self, queue: Deque[AudioChunk]) -> dict:
        consumed = self._consumed_through_sample
        return {
            "available_through_sample": self._available_through_sample,
            "delivered_through_sample": self._delivered_through_sample,
            "consumed_through_sample": consumed,
            "queue_backlog_samples": (
                self._available_through_sample - self._delivered_through_sample
            ),
            "processing_lag_samples": (
                None if consumed is None else self._available_through_sample - consumed
            ),
            "queue_length_packets": len(queue),
        }

    def _append_trace(
        self,
        event_type: EventType,
        *,
        logical_sample_position: Optional[int],
        payload: dict,
        wall_time: float,
        sequence_id: Optional[int] = None,
    ) -> None:
        wall_offset = self._wall_offset(wall_time)
        if wall_offset + 1e-12 < self._last_wall_offset:
            raise ContractValidationError(
                "compute-aware trace wall_offset_sec must be monotonic: "
                f"{wall_offset} < {self._last_wall_offset}"
            )
        self._last_wall_offset = max(self._last_wall_offset, wall_offset)
        event = TraceEvent(
            schema_version=self.trace_writer.schema_version,
            event_index=len(self._trace_events),
            stream_id=self.source.metadata.stream_id,
            event_type=event_type,
            logical_sample_position=logical_sample_position,
            payload=payload,
            sequence_id=sequence_id,
            wall_offset_sec=wall_offset,
        )
        self.trace_writer.append(event)
        self._trace_events.append(event)

    def _validate_source_chunk(self, chunk: AudioChunk) -> None:
        if chunk.stream_id != self.source.metadata.stream_id:
            raise ContractValidationError("source chunk stream_id does not match metadata")
        if chunk.sample_rate != self.source.metadata.sample_rate:
            raise ContractValidationError("source chunk sample_rate does not match metadata")
        if chunk.channel_ids != self.source.metadata.channel_ids:
            raise ContractValidationError("source chunk channel_ids do not match metadata")
        if chunk.end_sample > self.source.total_samples:
            raise ContractValidationError("source chunk exceeds source total_samples")

    def _enqueue_chunk(self, chunk: AudioChunk, queue: Deque[AudioChunk]) -> None:
        self._validate_source_chunk(chunk)
        if chunk.start_sample != self._available_through_sample:
            raise ContractValidationError(
                "source packet overlap or gap before availability: "
                f"start={chunk.start_sample}, available={self._available_through_sample}"
            )
        if self._source_end_visible:
            raise ContractValidationError("source yielded a packet after its final chunk")
        self._available_through_sample = chunk.end_sample
        scheduled_time = self._scheduled_wall_time(chunk.end_sample)
        scheduled_offset = self._wall_offset(scheduled_time)
        queue.append(chunk)
        self._append_trace(
            EventType.SOURCE_AVAILABLE,
            logical_sample_position=chunk.end_sample,
            payload={
                "start_sample": chunk.start_sample,
                "end_sample": chunk.end_sample,
                "scheduled_availability_wall_time": scheduled_time,
                "scheduled_availability_offset_sec": scheduled_offset,
                **self._state_payload(queue),
            },
            wall_time=scheduled_time,
            sequence_id=chunk.sequence_id,
        )
        if chunk.is_final_source_chunk:
            if chunk.end_sample != self.source.total_samples:
                raise ContractValidationError(
                    "final source chunk does not end at source.total_samples"
                )
            self._source_end_visible = True
            self._append_trace(
                EventType.SOURCE_END,
                logical_sample_position=chunk.end_sample,
                payload={
                    "source_end_sample": chunk.end_sample,
                    "scheduled_availability_wall_time": scheduled_time,
                    "scheduled_availability_offset_sec": scheduled_offset,
                    **self._state_payload(queue),
                },
                wall_time=scheduled_time,
                sequence_id=chunk.sequence_id,
            )

    def _catch_up(
        self,
        next_chunk: Optional[AudioChunk],
        source_iterator,
        queue: Deque[AudioChunk],
        now: float,
    ) -> Optional[AudioChunk]:
        """Enqueue every packet available by ``now`` in source order."""

        while next_chunk is not None:
            scheduled_time = self._scheduled_wall_time(next_chunk.end_sample)
            if scheduled_time > now + 1e-12:
                break
            self._enqueue_chunk(next_chunk, queue)
            try:
                next_chunk = next(source_iterator)
            except StopIteration:
                next_chunk = None
        return next_chunk

    def _observe_output(
        self,
        event: OutputEvent,
        *,
        queue: Deque[AudioChunk],
        observation_time: float,
    ) -> None:
        self._validator.validate(
            event,
            available=self._available_through_sample,
            delivered=self._delivered_through_sample,
        )
        if event.consumed_through_sample is not None:
            self._consumed_through_sample = event.consumed_through_sample
        payload = {
            "event_id": event.event_id,
            "scope_id": event.scope_id,
            "consumed_through_sample": event.consumed_through_sample,
            "observed_available_through_sample": self._available_through_sample,
            "observed_delivered_through_sample": self._delivered_through_sample,
            "observed_wall_time": observation_time,
            "observed_wall_offset_sec": self._wall_offset(observation_time),
            **self._state_payload(queue),
        }
        if isinstance(event, HypothesisEvent):
            payload.update(
                {
                    "kind": event.kind.value,
                    "revision_index": event.revision_index,
                    "text": event.text,
                    "replaces_event_id": event.replaces_event_id,
                    "stable_prefix": event.stable_prefix,
                }
            )
            event_type = EventType.HYPOTHESIS
        elif isinstance(event, EndpointEvent):
            event_type = EventType.ENDPOINT
        elif isinstance(event, ErrorEvent):
            payload.update(
                {
                    "message": event.message,
                    "fatal": event.fatal,
                    "recoverable": event.recoverable,
                }
            )
            event_type = EventType.ERROR
        else:  # pragma: no cover - LifecycleValidator rejects this first.
            raise ContractValidationError("unsupported consumer output event")
        self._append_trace(
            event_type,
            logical_sample_position=self._available_through_sample,
            payload=payload,
            wall_time=observation_time,
        )

    def _consume_one(
        self, chunk: AudioChunk, queue: Deque[AudioChunk], delivery_time: float
    ) -> Tuple[float, Tuple[OutputEvent, ...]]:
        if chunk.start_sample != self._delivered_through_sample:
            raise ContractValidationError(
                "source packet overlap or gap before delivery: "
                f"start={chunk.start_sample}, delivered={self._delivered_through_sample}"
            )
        scheduled_time = self._scheduled_wall_time(chunk.end_sample)
        delivered_chunk = replace(
            chunk,
            scheduled_availability_wall_time=scheduled_time,
            delivery_wall_time=delivery_time,
        )
        self._delivered_through_sample = chunk.end_sample
        delivery_offset = self._wall_offset(delivery_time)
        scheduled_offset = self._wall_offset(scheduled_time)
        self._append_trace(
            EventType.PACKET_DELIVERED,
            logical_sample_position=chunk.end_sample,
            payload={
                "start_sample": chunk.start_sample,
                "end_sample": chunk.end_sample,
                "scheduled_availability_offset_sec": scheduled_offset,
                "delivery_wall_time": delivery_time,
                "delivery_offset_sec": delivery_offset,
                "delivery_lag_sec": delivery_offset - scheduled_offset,
                **self._state_payload(queue),
            },
            wall_time=delivery_time,
            sequence_id=chunk.sequence_id,
        )
        self._append_trace(
            EventType.CONSUMER_CALL_START,
            logical_sample_position=chunk.end_sample,
            payload={
                "sequence_id": chunk.sequence_id,
                "call_start_wall_time": delivery_time,
                **self._state_payload(queue),
            },
            wall_time=delivery_time,
            sequence_id=chunk.sequence_id,
        )
        returned = self.consumer.consume(delivered_chunk)
        if returned is None:
            raise ContractValidationError("consume() must return an iterable of OutputEvent")
        try:
            output_events = tuple(returned)
        except TypeError as exc:
            raise ContractValidationError(
                "consume() must return an iterable of OutputEvent"
            ) from exc

        return_time = self.wall_clock.now()
        # This deliberately precedes CALL_RETURN and output observation.  A
        # blocking consumer cannot pause source availability.
        # Catch-up is performed by run(), because it owns the source iterator.
        return return_time, output_events

    def _record_return_and_outputs(
        self,
        chunk: AudioChunk,
        queue: Deque[AudioChunk],
        output_events: Tuple[OutputEvent, ...],
        return_time: float,
    ) -> None:
        self._append_trace(
            EventType.CONSUMER_CALL_RETURN,
            logical_sample_position=chunk.end_sample,
            payload={
                "sequence_id": chunk.sequence_id,
                "output_count": len(output_events),
                "call_return_wall_time": return_time,
                **self._state_payload(queue),
            },
            wall_time=return_time,
            sequence_id=chunk.sequence_id,
        )
        for event in output_events:
            self._observe_output(event, queue=queue, observation_time=return_time)

    def run(self) -> PacedReplayRunResult:
        """Run once, persist wall-stamped evidence, and drain the FIFO queue."""

        if self._started:
            raise RuntimeError("paced replay runtime can run only once")
        self._started = True
        try:
            # start() is preparation and is excluded from the measured stream.
            self.consumer.start(self.source.metadata)
            self._run_wall_origin = self.wall_clock.now()
            queue: Deque[AudioChunk] = deque()
            source_iterator = iter(self.source)
            try:
                next_chunk: Optional[AudioChunk] = next(source_iterator)
            except StopIteration as exc:
                raise ContractValidationError("source yielded no packets") from exc

            while next_chunk is not None or queue:
                now = self.wall_clock.now()
                next_chunk = self._catch_up(next_chunk, source_iterator, queue, now)
                if not queue:
                    if next_chunk is None:
                        break
                    self.wall_clock.sleep_until(
                        self._scheduled_wall_time(next_chunk.end_sample)
                    )
                    next_chunk = self._catch_up(
                        next_chunk,
                        source_iterator,
                        queue,
                        self.wall_clock.now(),
                    )
                if not queue:
                    continue

                # Wall time can advance even between an availability check and
                # the actual delivery (for example, OS scheduling).  Refresh
                # once more so packets that became available in that gap are
                # traced before this delivery.
                next_chunk = self._catch_up(
                    next_chunk,
                    source_iterator,
                    queue,
                    self.wall_clock.now(),
                )
                delivery_time = self.wall_clock.now()
                next_chunk = self._catch_up(
                    next_chunk,
                    source_iterator,
                    queue,
                    delivery_time,
                )
                chunk = queue.popleft()
                return_time, output_events = self._consume_one(
                    chunk, queue, delivery_time
                )
                next_chunk = self._catch_up(
                    next_chunk,
                    source_iterator,
                    queue,
                    return_time,
                )
                self._record_return_and_outputs(
                    chunk,
                    queue,
                    output_events,
                    return_time,
                )

            if not self._source_end_visible:
                raise ContractValidationError("source ended without a final source chunk")
            finish_start = self.wall_clock.now()
            self._append_trace(
                EventType.FINISH_START,
                logical_sample_position=self._available_through_sample,
                payload={
                    "finish_start_wall_time": finish_start,
                    **self._state_payload(queue),
                },
                wall_time=finish_start,
            )
            returned = self.consumer.finish()
            if returned is None:
                raise ContractValidationError("finish() must return an iterable of OutputEvent")
            try:
                output_events = tuple(returned)
            except TypeError as exc:
                raise ContractValidationError(
                    "finish() must return an iterable of OutputEvent"
                ) from exc
            finish_return = self.wall_clock.now()
            self._append_trace(
                EventType.FINISH_RETURN,
                logical_sample_position=self._available_through_sample,
                payload={
                    "output_count": len(output_events),
                    "finish_return_wall_time": finish_return,
                    **self._state_payload(queue),
                },
                wall_time=finish_return,
            )
            for event in output_events:
                self._observe_output(event, queue=queue, observation_time=finish_return)
            source_end_event = next(
                event for event in self._trace_events if event.event_type is EventType.SOURCE_END
            )
            source_end_offset = source_end_event.wall_offset_sec
            if source_end_offset is None:
                raise ContractValidationError("SOURCE_END must have a wall offset")
            finish_return_offset = self._wall_offset(finish_return)
            self.trace_writer.finalize_run_provenance(
                {
                    "clock_origin": {
                        "kind": self._clock_origin_kind(),
                        "run_wall_origin": self._run_wall_origin,
                        "wall_offset_reference": "monotonic_seconds - run_wall_origin",
                    },
                    "metric_window": {
                        "start_sample": 0,
                        "end_sample": self.source.total_samples,
                        "wall_start_offset_sec": 0.0,
                        "source_end_available_wall_offset_sec": source_end_offset,
                        "measured_streaming_interval_end_wall_offset_sec": source_end_offset,
                        "tail_interval_start_wall_offset_sec": source_end_offset,
                        "finish_return_wall_offset_sec": finish_return_offset,
                        "measured_run_interval_end_wall_offset_sec": finish_return_offset,
                        "boundary_convention": (
                            "streaming window ends at SOURCE_END; tail window ends at FINISH_RETURN"
                        ),
                    },
                }
            )
            self._finished = True
            return PacedReplayRunResult(
                trace_events=self.trace_events,
                source_end_sample=self.source.total_samples,
                run_wall_origin=self._run_wall_origin,
                measured_wall_interval_sec=finish_return_offset,
                source_end_wall_offset_sec=source_end_offset,
                finish_return_wall_offset_sec=finish_return_offset,
            )
        finally:
            self.trace_writer.close()
