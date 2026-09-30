from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_heard.replay.clock import FakeWallClock
from robot_heard.replay.metrics import compute_paced_metrics
from robot_heard.replay.runtime import DeterministicReplayRuntime
from robot_heard.replay.paced_runtime import PacedReplayRuntime
from robot_heard.replay.source import SyntheticSource
from robot_heard.replay.trace import TraceWriter

from replay_helpers import FakeLatencyConsumer, SnapshotConsumer


def _source(sample_count: int, packet_samples: int, *, sample_rate: int = 1000):
    return SyntheticSource(
        (tuple(range(sample_count)),),
        sample_rate=sample_rate,
        packet_samples=packet_samples,
        stream_id="paced-source",
        channel_ids=("ch0",),
    )


def _writer(tmp_path: Path, source, consumer, run_id: str = "paced"):
    trace_path = (tmp_path / f"{run_id}.trace.jsonl").resolve()
    run_path = (tmp_path / f"{run_id}.trace.run.json").resolve()
    provenance = {
        "schema_version": "r3.event.v1",
        "run_id": run_id,
        "replay_mode": "COMPUTE_AWARE",
        "source_identity": source.identity(),
        "source_sha256": source.content_sha256,
        "source_sample_rate": source.metadata.sample_rate,
        "channel_ids": list(source.metadata.channel_ids),
        "packet_samples": source.packet_samples,
        "queue_policy": "lossless_unbounded_fifo",
        "timeline_authority": "integer_sample_index",
        "consumer_identity": type(consumer).__name__,
        "consumer_config": {},
        "consumer_capabilities": dict(consumer.capabilities().to_dict()),
        "code_commit": "test",
        "oracle_condition": "synthetic_none",
        "warmup_load_policy": {"model_load": "not_applicable", "warmup": "not_applicable"},
        "clock_policy": "fake_wall_clock",
        "clock_origin": {"kind": "fake_wall", "offset_sec": 0.0},
        "metric_window": {"start_sample": 0, "end_sample": source.total_samples},
        "trace_path": str(trace_path),
        "device_provenance": {"status": "not_applicable"},
    }
    return TraceWriter(trace_path, run_path, provenance), trace_path


def _records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_fake_wall_clock_is_deterministic_and_cannot_go_backwards():
    clock = FakeWallClock()
    clock.sleep_until(0.02)
    clock.advance(0.3)
    assert clock.now() == pytest.approx(0.32)
    with pytest.raises(ValueError, match="backwards"):
        clock.sleep_until(0.31)


def test_canonical_20ms_300ms_order_backlog_and_delivery_lag(tmp_path):
    source = _source(16 * 320, 320, sample_rate=16000)
    clock = FakeWallClock()
    consumer = FakeLatencyConsumer(
        clock,
        compute_sec=0.3,
        emit_hypothesis=True,
        reports_consumed_position=False,
    )
    writer, trace_path = _writer(tmp_path, source, consumer)

    result = PacedReplayRuntime(source, consumer, writer, wall_clock=clock).run()
    records = _records(trace_path)
    source_available = [r for r in records if r["event_type"] == "SOURCE_AVAILABLE"]
    assert [r["sequence_id"] for r in source_available] == list(range(16))
    assert source_available[0]["wall_offset_sec"] == pytest.approx(0.02)
    assert source_available[15]["wall_offset_sec"] == pytest.approx(0.32)

    return_index = next(
        index
        for index, record in enumerate(records)
        if record["event_type"] == "CONSUMER_CALL_RETURN"
        and record["sequence_id"] == 0
    )
    output_index = next(
        index
        for index, record in enumerate(records)
        if record["event_type"] == "HYPOTHESIS"
        and record["payload"]["event_id"] == "latency-partial-0"
    )
    packet_one_index = next(
        index
        for index, record in enumerate(records)
        if record["event_type"] == "PACKET_DELIVERED"
        and record["sequence_id"] == 1
    )
    assert all(
        next(
            index
            for index, record in enumerate(records)
            if record["event_type"] == "SOURCE_AVAILABLE"
            and record["sequence_id"] == sequence_id
        )
        < return_index
        for sequence_id in range(1, 16)
    )
    assert return_index < output_index < packet_one_index

    assert records[return_index]["payload"]["available_through_sample"] == 5120
    assert records[return_index]["payload"]["delivered_through_sample"] == 320
    assert records[return_index]["payload"]["queue_backlog_samples"] == 4800
    packet_one = records[packet_one_index]
    assert packet_one["wall_offset_sec"] == pytest.approx(0.32)
    assert packet_one["payload"]["scheduled_availability_offset_sec"] == pytest.approx(0.04)
    assert packet_one["payload"]["delivery_lag_sec"] == pytest.approx(0.28)
    assert packet_one["payload"]["queue_backlog_samples"] == 4480
    assert result.source_end_sample == 5120


def test_sustained_overload_is_lossless_fifo_and_drains_after_source_end(tmp_path):
    source = _source(10 * 20, 20)
    clock = FakeWallClock()
    consumer = FakeLatencyConsumer(clock, compute_sec=0.03, emit_hypothesis=False)
    writer, trace_path = _writer(tmp_path, source, consumer)

    result = PacedReplayRuntime(source, consumer, writer, wall_clock=clock).run()
    records = _records(trace_path)
    delivered = [r["sequence_id"] for r in records if r["event_type"] == "PACKET_DELIVERED"]
    assert delivered == list(range(10))
    assert records[-1]["event_type"] == "FINISH_RETURN"
    assert records[-1]["payload"]["queue_backlog_samples"] == 0
    source_end = next(r for r in records if r["event_type"] == "SOURCE_END")
    assert source_end["wall_offset_sec"] == pytest.approx(0.2)
    assert result.measured_wall_interval_sec > source_end["wall_offset_sec"]

    metrics = compute_paced_metrics(
        result.trace_events,
        sample_rate=1000,
        source_total_samples=200,
    )
    assert metrics["RTF_consume"] == pytest.approx(1.5)
    assert metrics["queue_backlog_samples"]["max"] > 0


def test_fast_consumer_does_not_sustain_backlog(tmp_path):
    source = _source(10 * 20, 20)
    clock = FakeWallClock()
    consumer = FakeLatencyConsumer(clock, compute_sec=0.005, emit_hypothesis=False)
    writer, trace_path = _writer(tmp_path, source, consumer)

    PacedReplayRuntime(source, consumer, writer, wall_clock=clock).run()
    records = _records(trace_path)
    deliveries = [r for r in records if r["event_type"] == "PACKET_DELIVERED"]
    assert all(r["payload"]["queue_backlog_samples"] == 0 for r in deliveries)
    assert all(r["payload"]["delivery_lag_sec"] == pytest.approx(0.0) for r in deliveries)


def test_consumer_at_packet_interval_keeps_fifo_without_sustained_backlog(tmp_path):
    source = _source(5 * 20, 20)
    clock = FakeWallClock()
    consumer = FakeLatencyConsumer(clock, compute_sec=0.02, emit_hypothesis=False)
    writer, trace_path = _writer(tmp_path, source, consumer)

    PacedReplayRuntime(source, consumer, writer, wall_clock=clock).run()
    deliveries = [
        record for record in _records(trace_path) if record["event_type"] == "PACKET_DELIVERED"
    ]
    assert [record["sequence_id"] for record in deliveries] == list(range(5))
    assert all(record["payload"]["queue_backlog_samples"] == 0 for record in deliveries)


def test_wall_offsets_are_non_null_monotonic_and_chunk_fields_are_absolute(tmp_path):
    source = _source(3, 1)
    clock = FakeWallClock(current_time=10.0)
    consumer = FakeLatencyConsumer(clock, compute_sec=0.0, emit_hypothesis=False)
    writer, trace_path = _writer(tmp_path, source, consumer)

    PacedReplayRuntime(source, consumer, writer, wall_clock=clock).run()
    records = _records(trace_path)
    offsets = [record["wall_offset_sec"] for record in records]
    assert all(value is not None for value in offsets)
    assert offsets == sorted(offsets)
    assert offsets[0] == pytest.approx(0.001)
    assert all(chunk.scheduled_availability_wall_time is not None for chunk in consumer.chunks)
    assert all(chunk.delivery_wall_time is not None for chunk in consumer.chunks)


def test_runtime_finalizes_origin_and_metric_window_provenance(tmp_path):
    source = _source(3, 1)
    clock = FakeWallClock()
    consumer = FakeLatencyConsumer(clock, compute_sec=0.005, emit_hypothesis=False)
    writer, _ = _writer(tmp_path, source, consumer)

    result = PacedReplayRuntime(source, consumer, writer, wall_clock=clock).run()
    provenance = json.loads(
        (tmp_path / "paced.trace.run.json").read_text(encoding="utf-8")
    )
    assert provenance["clock_origin"]["kind"] == "fake_wall_clock"
    assert provenance["clock_origin"]["run_wall_origin"] == pytest.approx(
        result.run_wall_origin
    )
    window = provenance["metric_window"]
    assert window["wall_start_offset_sec"] == 0.0
    assert window["source_end_available_wall_offset_sec"] == pytest.approx(
        result.source_end_wall_offset_sec
    )
    assert window["finish_return_wall_offset_sec"] == pytest.approx(
        result.finish_return_wall_offset_sec
    )
    assert provenance["trace_durability_policy"] == "per_event_fsync"


def test_compute_unaware_and_compute_aware_preserve_logical_delivery_and_outputs(tmp_path):
    source_a = _source(6, 2)
    source_b = _source(6, 2)
    deterministic_consumer = SnapshotConsumer()
    paced_clock = FakeWallClock()
    paced_consumer = FakeLatencyConsumer(
        paced_clock,
        compute_sec=0.0,
        emit_hypothesis=True,
        reports_consumed_position=True,
    )

    from test_replay_runtime import _writer as deterministic_writer

    writer_a, trace_a, _ = deterministic_writer(tmp_path / "det", source_a, deterministic_consumer)
    DeterministicReplayRuntime(source_a, deterministic_consumer, writer_a).run()
    writer_b, trace_b = _writer(tmp_path / "paced", source_b, paced_consumer)
    PacedReplayRuntime(source_b, paced_consumer, writer_b, wall_clock=paced_clock).run()

    det = _records(trace_a)
    paced = _records(trace_b)
    det_delivered = [
        (r["sequence_id"], r["payload"]["start_sample"], r["payload"]["end_sample"])
        for r in det
        if r["event_type"] == "PACKET_DELIVERED"
    ]
    paced_delivered = [
        (r["sequence_id"], r["payload"]["start_sample"], r["payload"]["end_sample"])
        for r in paced
        if r["event_type"] == "PACKET_DELIVERED"
    ]
    assert paced_delivered == det_delivered
    det_hypotheses = [
        (r["payload"]["text"], r["payload"]["revision_index"], r["payload"]["kind"])
        for r in det
        if r["event_type"] == "HYPOTHESIS"
    ]
    paced_hypotheses = [
        (r["payload"]["text"], r["payload"]["revision_index"], r["payload"]["kind"])
        for r in paced
        if r["event_type"] == "HYPOTHESIS"
    ]
    assert paced_hypotheses == det_hypotheses
