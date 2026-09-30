from __future__ import annotations

import json

import pytest

from robot_heard.replay.events import ContractValidationError, EventType, TraceEvent
from robot_heard.replay.runtime import DeterministicReplayRuntime
from robot_heard.replay.source import SyntheticSource
from robot_heard.replay import TraceDurabilityPolicy
from robot_heard.replay.trace import TraceWriter

from replay_helpers import RecordingConsumer, SnapshotConsumer
from test_replay_runtime import _writer, _source


def test_trace_jsonl_and_run_sidecar_are_separate_durable_artifacts(tmp_path):
    source = _source(sample_count=2, packet_samples=1)
    consumer = RecordingConsumer()
    writer, trace_path, run_path = _writer(tmp_path, source, consumer)

    DeterministicReplayRuntime(source, consumer, writer).run()

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    provenance = json.loads(run_path.read_text(encoding="utf-8"))
    assert records
    assert all(record["schema_version"] == "r1.event.v1" for record in records)
    assert [record["event_index"] for record in records] == list(range(len(records)))
    assert all("event_type" in record and "payload" in record for record in records)
    assert all(record["wall_offset_sec"] is None for record in records)
    assert provenance["trace_path"] == str(trace_path.resolve())
    assert provenance["consumer_capabilities"]["reports_consumed_position"] is False


def test_trace_writer_refuses_existing_trace_or_sidecar(tmp_path):
    source = _source(sample_count=1, packet_samples=1)
    consumer = RecordingConsumer()
    writer, trace_path, run_path = _writer(tmp_path, source, consumer)
    writer.close()

    with pytest.raises(FileExistsError, match="trace already exists"):
        _writer(tmp_path, source, consumer)

    other_trace = tmp_path / "other.jsonl"
    other_run = tmp_path / "other.run.json"
    other_run.write_text("{}\n", encoding="utf-8")
    provenance = {
        "schema_version": "r1.event.v1",
        "run_id": "other",
        "replay_mode": "COMPUTE_UNAWARE",
        "source_identity": source.identity(),
        "source_sample_rate": source.metadata.sample_rate,
        "channel_ids": list(source.metadata.channel_ids),
        "packet_samples": source.packet_samples,
        "queue_policy": "lossless_unbounded",
        "consumer_identity": "RecordingConsumer",
        "consumer_capabilities": dict(consumer.capabilities().to_dict()),
        "code_commit": "test",
        "oracle_condition": "synthetic_none",
        "clock_policy": "integer_sample_fake_clock",
        "trace_path": str(other_trace.resolve()),
    }
    with pytest.raises(FileExistsError, match="run provenance already exists"):
        TraceWriter(other_trace, other_run, provenance)


def test_trace_writer_rejects_wrong_event_index_and_schema(tmp_path):
    source = _source(sample_count=1, packet_samples=1)
    consumer = RecordingConsumer()
    writer, _, _ = _writer(tmp_path, source, consumer, run_id="index")
    event = TraceEvent(
        schema_version="r1.event.v1",
        event_index=1,
        stream_id=source.metadata.stream_id,
        event_type=EventType.RESET,
        logical_sample_position=0,
        payload={"state": "reset"},
    )
    with pytest.raises(ValueError, match="event_index"):
        writer.append(event)
    writer.close()


def test_trace_writer_requires_complete_r0_provenance(tmp_path):
    trace_path = (tmp_path / "missing.trace.jsonl").resolve()
    run_path = (tmp_path / "missing.trace.run.json").resolve()

    with pytest.raises(ContractValidationError, match="missing required fields"):
        TraceWriter(
            trace_path,
            run_path,
            {
                "schema_version": "r1.event.v1",
                "run_id": "missing",
                "trace_path": str(trace_path),
            },
        )


def test_deterministic_logical_trace_repeats_for_same_input(tmp_path):
    source_a = _source(sample_count=4, packet_samples=2)
    consumer_a = SnapshotConsumer()
    writer_a, trace_a, _ = _writer(tmp_path / "a", source_a, consumer_a, run_id="a")
    DeterministicReplayRuntime(source_a, consumer_a, writer_a).run()

    source_b = _source(sample_count=4, packet_samples=2)
    consumer_b = SnapshotConsumer()
    writer_b, trace_b, _ = _writer(tmp_path / "b", source_b, consumer_b, run_id="b")
    DeterministicReplayRuntime(source_b, consumer_b, writer_b).run()

    records_a = [json.loads(line) for line in trace_a.read_text(encoding="utf-8").splitlines()]
    records_b = [json.loads(line) for line in trace_b.read_text(encoding="utf-8").splitlines()]
    assert records_a == records_b


def test_trace_event_serialization_is_deterministic():
    event = TraceEvent(
        schema_version="r1.event.v1",
        event_index=0,
        stream_id="s",
        event_type=EventType.SOURCE_AVAILABLE,
        logical_sample_position=2,
        payload={"end_sample": 2, "start_sample": 0},
        sequence_id=0,
    )

    assert event.to_record()["payload"] == {"end_sample": 2, "start_sample": 0}
    assert event.to_record()["wall_offset_sec"] is None


def _single_event_writer(tmp_path, policy):
    source = _source(sample_count=1, packet_samples=1)
    trace_path = (tmp_path / "trace.jsonl").resolve()
    run_path = (tmp_path / "trace.run.json").resolve()
    provenance = {
        "schema_version": "r1.event.v1",
        "run_id": "policy",
        "replay_mode": "COMPUTE_UNAWARE",
        "source_identity": source.identity(),
        "source_sha256": source.content_sha256,
        "source_sample_rate": source.metadata.sample_rate,
        "channel_ids": list(source.metadata.channel_ids),
        "packet_samples": source.packet_samples,
        "queue_policy": "lossless_unbounded",
        "timeline_authority": "integer_sample_index",
        "consumer_identity": "test",
        "consumer_config": {},
        "consumer_capabilities": {},
        "code_commit": "test",
        "oracle_condition": "none",
        "warmup_load_policy": {},
        "clock_policy": "logical",
        "clock_origin": {"kind": "logical"},
        "metric_window": {},
        "trace_path": str(trace_path),
        "device_provenance": {"status": "not_applicable"},
    }
    writer = TraceWriter(trace_path, run_path, provenance, durability_policy=policy)
    event = TraceEvent(
        schema_version="r1.event.v1",
        event_index=0,
        stream_id=source.metadata.stream_id,
        event_type=EventType.RESET,
        logical_sample_position=0,
        payload={"state": "reset"},
    )
    return writer, event, trace_path, run_path


def test_trace_durability_policy_is_recorded_and_buffered_close_is_durable(tmp_path, monkeypatch):
    import robot_heard.replay.trace as trace_module

    fsync_calls = []
    real_fsync = trace_module.os.fsync

    def count_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(trace_module.os, "fsync", count_fsync)
    writer, event, trace_path, run_path = _single_event_writer(
        tmp_path, TraceDurabilityPolicy.BUFFERED_CLOSE_FSYNC
    )
    writer.append(event)
    assert len(fsync_calls) == 1  # sidecar initialization only
    writer.close()
    assert len(fsync_calls) == 2  # final trace flush/fsync
    assert json.loads(run_path.read_text(encoding="utf-8"))["trace_durability_policy"] == (
        "buffered_close_fsync"
    )
    assert len(trace_path.read_text(encoding="utf-8").splitlines()) == 1


def test_per_event_fsync_policy_remains_available(tmp_path, monkeypatch):
    import robot_heard.replay.trace as trace_module

    fsync_calls = []
    real_fsync = trace_module.os.fsync
    monkeypatch.setattr(
        trace_module.os,
        "fsync",
        lambda fd: (fsync_calls.append(fd), real_fsync(fd))[1],
    )
    writer, event, _, run_path = _single_event_writer(
        tmp_path, TraceDurabilityPolicy.PER_EVENT_FSYNC
    )
    writer.append(event)
    writer.close()
    assert len(fsync_calls) == 3  # sidecar, append, close
    assert json.loads(run_path.read_text(encoding="utf-8"))["trace_durability_policy"] == (
        "per_event_fsync"
    )


def test_trace_event_semantics_are_identical_across_durability_policies(tmp_path):
    writer_a, event_a, trace_a, _ = _single_event_writer(
        tmp_path / "per", TraceDurabilityPolicy.PER_EVENT_FSYNC
    )
    writer_b, event_b, trace_b, _ = _single_event_writer(
        tmp_path / "buffered", TraceDurabilityPolicy.BUFFERED_CLOSE_FSYNC
    )
    writer_a.append(event_a)
    writer_b.append(event_b)
    writer_a.close()
    writer_b.close()
    assert trace_a.read_text(encoding="utf-8") == trace_b.read_text(encoding="utf-8")
