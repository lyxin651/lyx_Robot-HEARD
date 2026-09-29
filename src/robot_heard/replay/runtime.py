"""Deterministic, compute-unaware replay runtime for R1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    OutputEvent,
    TraceEvent,
)
from robot_heard.replay.source import SyntheticSource
from robot_heard.replay.trace import TraceWriter
from robot_heard.streaming.base import StreamingConsumer


@dataclass
class _ScopeState:
    current_hypothesis: Optional[HypothesisEvent] = None
    stable_prefix: Optional[str] = None
    final: bool = False
    fatal_error: bool = False
    endpoint_seen: bool = False


class LifecycleValidator:
    """Validate output lifecycle and capability-dependent invariants.

    ``ErrorEvent(recoverable=True)`` is the adapter's explicit declaration
    that the scope may continue.  No implicit recovery is inferred from the
    error message or from later output.
    """

    def __init__(self, stream_id: str, capabilities: ConsumerCapabilities) -> None:
        self.stream_id = stream_id
        self.capabilities = capabilities
        self._scopes: Dict[str, _ScopeState] = {}
        self._event_ids = set()

    def _scope(self, scope_id: str) -> _ScopeState:
        return self._scopes.setdefault(scope_id, _ScopeState())

    def _validate_consumed(
        self,
        consumed: Optional[int],
        *,
        available: int,
        delivered: int,
    ) -> None:
        if not self.capabilities.reports_consumed_position:
            if consumed is not None:
                raise ContractValidationError(
                    "consumed_through_sample must be null when "
                    "reports_consumed_position is false"
                )
            return
        if consumed is None:
            raise ContractValidationError(
                "consumed_through_sample is required when "
                "reports_consumed_position is true"
            )
        if consumed > delivered or delivered > available:
            raise ContractValidationError(
                "consumed <= delivered <= available invariant violated"
            )

    def _validate_common(
        self,
        event: OutputEvent,
        *,
        available: int,
        delivered: int,
    ) -> _ScopeState:
        if event.stream_id != self.stream_id:
            raise ContractValidationError(
                f"event stream_id={event.stream_id!r} does not match "
                f"runtime stream_id={self.stream_id!r}"
            )
        if event.event_id in self._event_ids:
            raise ContractValidationError(f"duplicate event_id: {event.event_id}")
        self._validate_consumed(
            event.consumed_through_sample,
            available=available,
            delivered=delivered,
        )
        state = self._scope(event.scope_id)
        if state.fatal_error:
            raise ContractValidationError(
                f"scope {event.scope_id!r} has fatal error; later events are forbidden"
            )
        return state

    def validate(self, event: OutputEvent, *, available: int, delivered: int) -> None:
        """Validate one output before it is persisted as observed evidence."""

        if not isinstance(event, (HypothesisEvent, EndpointEvent, ErrorEvent)):
            raise ContractValidationError("consumer returned an unsupported output event")
        state = self._validate_common(event, available=available, delivered=delivered)

        if isinstance(event, HypothesisEvent):
            if state.final:
                raise ContractValidationError(
                    f"scope {event.scope_id!r} is final; later hypotheses are forbidden"
                )
            if event.kind is HypothesisKind.PARTIAL and not self.capabilities.supports_partial:
                raise ContractValidationError("consumer emitted PARTIAL without capability")
            if state.current_hypothesis is None:
                if event.revision_index != 0 or event.replaces_event_id is not None:
                    raise ContractValidationError(
                        "first hypothesis must have revision_index=0 and no replacement"
                    )
            else:
                if not self.capabilities.supports_revision:
                    raise ContractValidationError(
                        "consumer replaced a hypothesis without revision capability"
                    )
                expected_index = state.current_hypothesis.revision_index + 1
                if event.revision_index != expected_index:
                    raise ContractValidationError(
                        f"revision_index must be {expected_index}; got {event.revision_index}"
                    )
                if event.replaces_event_id != state.current_hypothesis.event_id:
                    raise ContractValidationError(
                        "replaces_event_id must identify the immediately previous "
                        "active hypothesis"
                    )

            previous_stable = state.stable_prefix
            if event.stable_prefix is not None and not self.capabilities.supports_stable_prefix:
                raise ContractValidationError(
                    "stable_prefix provided without stable-prefix capability"
                )
            if previous_stable is not None and not event.text.startswith(previous_stable):
                raise ContractValidationError(
                    "text withdraws a previously declared stable prefix"
                )
            if event.stable_prefix is not None:
                if not event.text.startswith(event.stable_prefix):
                    raise ContractValidationError("stable_prefix must be an exact text prefix")
                if previous_stable is not None and (
                    len(event.stable_prefix) < len(previous_stable)
                    or not event.stable_prefix.startswith(previous_stable)
                ):
                    raise ContractValidationError(
                        "stable_prefix may only remain equal or grow"
                    )
                state.stable_prefix = event.stable_prefix

            state.current_hypothesis = event
            if event.kind is HypothesisKind.FINAL:
                state.final = True
            self._event_ids.add(event.event_id)
            return

        if isinstance(event, EndpointEvent):
            if not self.capabilities.supports_endpoint:
                raise ContractValidationError("consumer emitted ENDPOINT without capability")
            state.endpoint_seen = True
            self._event_ids.add(event.event_id)
            return

        if state.final:
            raise ContractValidationError(
                f"scope {event.scope_id!r} is final; later errors are forbidden"
            )
        self._event_ids.add(event.event_id)
        if event.fatal:
            state.fatal_error = True


@dataclass(frozen=True)
class ReplayRunResult:
    """Deterministic run summary returned after trace persistence closes."""

    trace_events: Tuple[TraceEvent, ...]
    source_end_sample: int


class DeterministicReplayRuntime:
    """Replay synthetic packets without wall-clock pacing or model execution."""

    def __init__(
        self,
        source: SyntheticSource,
        consumer: StreamingConsumer,
        trace_writer: TraceWriter,
        *,
        clock: Optional[LogicalClock] = None,
    ) -> None:
        self.source = source
        self.consumer = consumer
        self.trace_writer = trace_writer
        self.clock = clock or FakeClock()
        self.capabilities = consumer.capabilities()
        if not isinstance(self.capabilities, ConsumerCapabilities):
            raise ContractValidationError("consumer.capabilities() must return ConsumerCapabilities")
        self._validator = LifecycleValidator(source.metadata.stream_id, self.capabilities)
        self._trace_events: List[TraceEvent] = []
        self._available_through_sample = 0
        self._delivered_through_sample = 0
        self._consumed_through_sample: Optional[int] = None
        self._source_end_visible = False
        self._started = False
        self._finished = False

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

    def _append_trace(
        self,
        event_type: EventType,
        *,
        logical_sample_position: Optional[int],
        payload: dict,
        sequence_id: Optional[int] = None,
    ) -> None:
        event = TraceEvent(
            schema_version=self.trace_writer.schema_version,
            event_index=len(self._trace_events),
            stream_id=self.source.metadata.stream_id,
            event_type=event_type,
            logical_sample_position=logical_sample_position,
            payload=payload,
            sequence_id=sequence_id,
        )
        self.trace_writer.append(event)
        self._trace_events.append(event)

    def _state_payload(self) -> dict:
        return {
            "available_through_sample": self._available_through_sample,
            "delivered_through_sample": self._delivered_through_sample,
            "consumed_through_sample": self._consumed_through_sample,
        }

    def _observe_output(self, event: OutputEvent) -> None:
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
            "observed_wall_time": None,
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
        else:
            payload.update(
                {
                    "message": event.message,
                    "fatal": event.fatal,
                    "recoverable": event.recoverable,
                }
            )
            event_type = EventType.ERROR
        self._append_trace(
            event_type,
            logical_sample_position=self._available_through_sample,
            payload=payload,
        )

    def _consume_one(self, chunk: AudioChunk) -> None:
        if chunk.start_sample != self._delivered_through_sample:
            raise ContractValidationError(
                "source packet overlap or gap before delivery: "
                f"start={chunk.start_sample}, delivered={self._delivered_through_sample}"
            )
        self._delivered_through_sample = chunk.end_sample
        self._append_trace(
            EventType.PACKET_DELIVERED,
            logical_sample_position=chunk.end_sample,
            payload={
                "start_sample": chunk.start_sample,
                "end_sample": chunk.end_sample,
                **self._state_payload(),
            },
            sequence_id=chunk.sequence_id,
        )
        self._append_trace(
            EventType.CONSUMER_CALL_START,
            logical_sample_position=chunk.end_sample,
            payload={"sequence_id": chunk.sequence_id, **self._state_payload()},
            sequence_id=chunk.sequence_id,
        )
        returned = self.consumer.consume(chunk)
        if returned is None:
            raise ContractValidationError("consume() must return an iterable of OutputEvent")
        try:
            output_events = tuple(returned)
        except TypeError as exc:
            raise ContractValidationError(
                "consume() must return an iterable of OutputEvent"
            ) from exc
        self._append_trace(
            EventType.CONSUMER_CALL_RETURN,
            logical_sample_position=chunk.end_sample,
            payload={"sequence_id": chunk.sequence_id, "output_count": len(output_events), **self._state_payload()},
            sequence_id=chunk.sequence_id,
        )
        for event in output_events:
            self._observe_output(event)

    def run(self) -> ReplayRunResult:
        """Run once from sample zero and close the raw trace in all cases."""

        if self._started:
            raise RuntimeError("deterministic replay runtime can run only once")
        self._started = True
        try:
            self.consumer.start(self.source.metadata)
            queue: List[AudioChunk] = []
            for chunk in self.source:
                self.clock.advance_to(chunk.end_sample)
                self._available_through_sample = chunk.end_sample
                self._append_trace(
                    EventType.SOURCE_AVAILABLE,
                    logical_sample_position=chunk.end_sample,
                    payload={
                        "available_through_sample": self._available_through_sample,
                        "start_sample": chunk.start_sample,
                        "end_sample": chunk.end_sample,
                        "sequence_id": chunk.sequence_id,
                    },
                    sequence_id=chunk.sequence_id,
                )
                queue.append(chunk)
                if chunk.is_final_source_chunk:
                    self._source_end_visible = True
                    self._append_trace(
                        EventType.SOURCE_END,
                        logical_sample_position=chunk.end_sample,
                        payload={"source_end_sample": chunk.end_sample},
                        sequence_id=chunk.sequence_id,
                    )
                while queue:
                    self._consume_one(queue.pop(0))

            if not self._source_end_visible:
                raise ContractValidationError("source ended without a final source chunk")

            self._append_trace(
                EventType.FINISH_START,
                logical_sample_position=self._available_through_sample,
                payload=self._state_payload(),
            )
            returned = self.consumer.finish()
            if returned is None:
                raise ContractValidationError("finish() must return an iterable of OutputEvent")
            output_events = tuple(returned)
            self._append_trace(
                EventType.FINISH_RETURN,
                logical_sample_position=self._available_through_sample,
                payload={"output_count": len(output_events), **self._state_payload()},
            )
            for event in output_events:
                self._observe_output(event)
            self._finished = True
            return ReplayRunResult(
                trace_events=self.trace_events,
                source_end_sample=self.source.total_samples,
            )
        finally:
            self.trace_writer.close()

    def reset(self) -> None:
        """Reset a consumer and logical state before a new isolated run."""

        if self._started:
            raise RuntimeError("reset must occur before a replay run starts")
        self.consumer.reset()
        reset = getattr(self.clock, "reset", None)
        if reset is not None:
            reset()
        self._validator = LifecycleValidator(self.source.metadata.stream_id, self.capabilities)
        self._available_through_sample = 0
        self._delivered_through_sample = 0
        self._consumed_through_sample = None
        self._append_trace(
            EventType.RESET,
            logical_sample_position=0,
            payload={"state": "reset"},
        )
