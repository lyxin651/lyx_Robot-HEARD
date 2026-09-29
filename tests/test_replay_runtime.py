from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from robot_heard.replay.events import (
    AudioChunk,
    ConsumerCapabilities,
    ContractValidationError,
    HypothesisEvent,
    HypothesisKind,
    OutputEvent,
    StreamMetadata,
)
from robot_heard.replay.runtime import DeterministicReplayRuntime, LifecycleValidator
from robot_heard.replay.source import SyntheticSource
from robot_heard.replay.trace import TraceWriter
from robot_heard.streaming.base import StreamingConsumer

from replay_helpers import (
    EndpointThenFinalConsumer,
    FatalThenFinalConsumer,
    RecordingConsumer,
    SnapshotConsumer,
)


def _writer(tmp_path: Path, source: SyntheticSource, consumer, run_id: str = "run-0"):
    trace_path = (tmp_path / f"{run_id}.trace.jsonl").resolve()
    run_path = (tmp_path / f"{run_id}.trace.run.json").resolve()
    capabilities = consumer.capabilities()
    provenance = {
        "schema_version": "r1.event.v1",
        "run_id": run_id,
        "replay_mode": "COMPUTE_UNAWARE",
        "source_identity": source.identity(),
        "source_sample_rate": source.metadata.sample_rate,
        "channel_ids": list(source.metadata.channel_ids),
        "packet_samples": source.packet_samples,
        "queue_policy": "lossless_unbounded",
        "consumer_identity": type(consumer).__name__,
        "consumer_capabilities": dict(capabilities.to_dict()),
        "code_commit": "test",
        "oracle_condition": "synthetic_none",
        "clock_policy": "integer_sample_fake_clock",
        "trace_path": str(trace_path),
    }
    return TraceWriter(trace_path, run_path, provenance), trace_path, run_path


def _source(sample_count=5, packet_samples=2):
    return SyntheticSource(
        (tuple(range(sample_count)), tuple(range(100, 100 + sample_count))),
        sample_rate=1000,
        packet_samples=packet_samples,
        stream_id="synthetic-runtime",
        channel_ids=("ch0", "ch1"),
    )


def _records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _SampleEchoConsumer(StreamingConsumer):
    def __init__(self) -> None:
        self.stream_id = None
        self.received = []

    def start(self, metadata: StreamMetadata) -> None:
        self.stream_id = metadata.stream_id

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        self.received.append(tuple(chunk.samples[0]))
        return (
            HypothesisEvent(
                stream_id=self.stream_id,
                scope_id="sample-scope",
                event_id=f"sample-{chunk.sequence_id}",
                revision_index=chunk.sequence_id,
                text=str(chunk.samples[0]),
                kind=HypothesisKind.PARTIAL,
                replaces_event_id=(
                    f"sample-{chunk.sequence_id - 1}" if chunk.sequence_id else None
                ),
            ),
        )

    def finish(self) -> Sequence[OutputEvent]:
        return ()

    def reset(self) -> None:
        self.stream_id = None
        self.received.clear()

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities(supports_partial=True, supports_revision=True)


def test_runtime_is_compute_unaware_and_has_fixed_event_order(tmp_path):
    source = _source(sample_count=3, packet_samples=2)
    consumer = SnapshotConsumer()
    writer, trace_path, _ = _writer(tmp_path, source, consumer)

    result = DeterministicReplayRuntime(source, consumer, writer).run()
    records = _records(trace_path)
    event_types = [record["event_type"] for record in records]

    assert event_types[:5] == [
        "SOURCE_AVAILABLE",
        "PACKET_DELIVERED",
        "CONSUMER_CALL_START",
        "CONSUMER_CALL_RETURN",
        "HYPOTHESIS",
    ]
    assert event_types.count("PACKET_DELIVERED") == 2
    assert event_types.count("CONSUMER_CALL_START") == 2
    assert event_types.count("CONSUMER_CALL_RETURN") == 2
    assert event_types[-3:] == ["FINISH_START", "FINISH_RETURN", "HYPOTHESIS"]
    assert result.source_end_sample == 3
    assert [chunk.start_sample for chunk in consumer.chunks] == [0, 2]
    assert [chunk.end_sample for chunk in consumer.chunks] == [2, 3]


def test_runtime_records_revisions_from_zero_and_final_after_partials(tmp_path):
    source = _source(sample_count=3, packet_samples=1)
    consumer = SnapshotConsumer()
    writer, trace_path, _ = _writer(tmp_path, source, consumer)

    DeterministicReplayRuntime(source, consumer, writer).run()
    hypotheses = [
        record["payload"]
        for record in _records(trace_path)
        if record["event_type"] == "HYPOTHESIS"
    ]

    assert [item["revision_index"] for item in hypotheses] == [0, 1, 2, 3]
    assert [item["kind"] for item in hypotheses] == ["PARTIAL", "PARTIAL", "PARTIAL", "FINAL"]
    assert hypotheses[0]["replaces_event_id"] is None
    assert hypotheses[1]["replaces_event_id"] == hypotheses[0]["event_id"]
    assert hypotheses[-1]["replaces_event_id"] == hypotheses[-2]["event_id"]


def test_consumed_position_true_is_recorded_and_bounded(tmp_path):
    source = _source(sample_count=2, packet_samples=1)
    consumer = SnapshotConsumer(reports_consumed_position=True)
    writer, trace_path, _ = _writer(tmp_path, source, consumer)

    runtime = DeterministicReplayRuntime(source, consumer, writer)
    runtime.run()
    hypotheses = [
        record["payload"]
        for record in _records(trace_path)
        if record["event_type"] == "HYPOTHESIS"
    ]

    assert [item["consumed_through_sample"] for item in hypotheses] == [1, 2, 2]
    assert runtime.consumed_through_sample == 2


def test_consumed_position_false_is_null_and_never_substituted(tmp_path):
    source = _source(sample_count=2, packet_samples=1)
    consumer = SnapshotConsumer(reports_consumed_position=False)
    writer, trace_path, _ = _writer(tmp_path, source, consumer)

    runtime = DeterministicReplayRuntime(source, consumer, writer)
    runtime.run()
    hypotheses = [
        record["payload"]
        for record in _records(trace_path)
        if record["event_type"] == "HYPOTHESIS"
    ]

    assert all(item["consumed_through_sample"] is None for item in hypotheses)
    assert runtime.consumed_through_sample is None


def test_endpoint_is_not_final_and_later_final_is_allowed(tmp_path):
    source = _source(sample_count=1, packet_samples=1)
    consumer = EndpointThenFinalConsumer()
    writer, trace_path, _ = _writer(tmp_path, source, consumer)

    DeterministicReplayRuntime(source, consumer, writer).run()
    output_types = [
        record["event_type"]
        for record in _records(trace_path)
        if record["event_type"] in {"ENDPOINT", "HYPOTHESIS"}
    ]

    assert output_types == ["ENDPOINT", "HYPOTHESIS"]


def test_fatal_error_closes_scope_and_rejects_later_final(tmp_path):
    source = _source(sample_count=1, packet_samples=1)
    consumer = FatalThenFinalConsumer()
    writer, _, _ = _writer(tmp_path, source, consumer)

    with pytest.raises(ContractValidationError, match="fatal error"):
        DeterministicReplayRuntime(source, consumer, writer).run()


def test_stable_prefix_can_only_remain_or_grow():
    capabilities = ConsumerCapabilities(
        supports_partial=True,
        supports_revision=True,
        supports_stable_prefix=True,
    )
    validator = LifecycleValidator("s", capabilities)
    first = HypothesisEvent(
        stream_id="s",
        scope_id="scope",
        event_id="e0",
        revision_index=0,
        text="abc",
        kind=HypothesisKind.PARTIAL,
        stable_prefix="a",
    )
    validator.validate(first, available=3, delivered=3)
    second = HypothesisEvent(
        stream_id="s",
        scope_id="scope",
        event_id="e1",
        revision_index=1,
        text="abd",
        kind=HypothesisKind.PARTIAL,
        replaces_event_id="e0",
        stable_prefix="a",
    )
    validator.validate(second, available=3, delivered=3)

    shortened = HypothesisEvent(
        stream_id="s",
        scope_id="scope",
        event_id="e2",
        revision_index=2,
        text="abd",
        kind=HypothesisKind.PARTIAL,
        replaces_event_id="e1",
        stable_prefix="",
    )
    with pytest.raises(ContractValidationError, match="remain equal or grow"):
        validator.validate(shortened, available=3, delivered=3)


def test_future_perturbation_cannot_change_prefix_trace_or_input_visibility(tmp_path):
    source_a = SyntheticSource(
        ((10, 20, 30, 40),),
        sample_rate=1000,
        packet_samples=1,
        stream_id="perturb",
    )
    source_b = SyntheticSource(
        ((10, 20, 300, 400),),
        sample_rate=1000,
        packet_samples=1,
        stream_id="perturb",
    )
    consumer_a = _SampleEchoConsumer()
    consumer_b = _SampleEchoConsumer()
    writer_a, trace_a, _ = _writer(tmp_path / "a", source_a, consumer_a, run_id="a")
    writer_b, trace_b, _ = _writer(tmp_path / "b", source_b, consumer_b, run_id="b")
    DeterministicReplayRuntime(source_a, consumer_a, writer_a).run()
    DeterministicReplayRuntime(source_b, consumer_b, writer_b).run()

    records_a = _records(trace_a)
    records_b = _records(trace_b)
    prefix_a = [record for record in records_a if record["logical_sample_position"] <= 2]
    prefix_b = [record for record in records_b if record["logical_sample_position"] <= 2]
    assert prefix_a == prefix_b
    assert consumer_a.received[:2] == consumer_b.received[:2] == [(10,), (20,)]
    assert consumer_a.received[2:] != consumer_b.received[2:]


def test_consumer_metadata_is_whitelist_not_full_provenance(tmp_path):
    source = _source(sample_count=1, packet_samples=1)
    consumer = RecordingConsumer()
    writer, _, run_path = _writer(tmp_path, source, consumer)

    DeterministicReplayRuntime(source, consumer, writer).run()
    provenance = json.loads(run_path.read_text(encoding="utf-8"))

    assert set(consumer.metadata) == {
        "stream_id",
        "sample_rate",
        "num_channels",
        "channel_ids",
    }
    assert "future_segment_list" not in consumer.metadata
    assert "reference" not in consumer.metadata
    assert "speaker_id" not in consumer.metadata
    assert "future_segment_list" not in provenance
