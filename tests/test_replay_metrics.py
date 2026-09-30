from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_heard.replay.clock import FakeWallClock
from robot_heard.replay.metrics import compute_paced_metrics, write_metrics_json
from robot_heard.replay.paced_runtime import PacedReplayRuntime

from replay_helpers import FakeLatencyConsumer
from test_replay_paced_runtime import _records, _source, _writer


def _run(tmp_path: Path, *, compute: float, finish_compute: float = 0.0, consumed: bool = False):
    source = _source(10 * 20, 20)
    clock = FakeWallClock()
    consumer = FakeLatencyConsumer(
        clock,
        compute_sec=compute,
        finish_compute_sec=finish_compute,
        emit_hypothesis=False,
        reports_consumed_position=consumed,
    )
    writer, trace_path = _writer(tmp_path, source, consumer)
    result = PacedReplayRuntime(source, consumer, writer, wall_clock=clock).run()
    return source, result, trace_path


def test_rtf_compute_duty_completion_and_backlog_metrics(tmp_path):
    source, result, trace_path = _run(tmp_path, compute=0.03, finish_compute=0.04)
    before = trace_path.read_bytes()
    metrics = compute_paced_metrics(
        result.trace_events,
        sample_rate=source.metadata.sample_rate,
        source_total_samples=source.total_samples,
        run_id="metrics-run",
    )
    assert metrics["run_id"] == "metrics-run"
    assert metrics["D_audio"] == pytest.approx(0.2)
    assert metrics["C_consume"] == pytest.approx(0.3)
    assert metrics["C_finish"] == pytest.approx(0.04)
    assert metrics["RTF_consume"] == pytest.approx(1.5)
    assert metrics["RTF_total_compute"] == pytest.approx(1.7)
    assert metrics["compute_duty"] > 0
    assert metrics["queue_backlog_samples"]["max"] > 0
    assert metrics["queue_backlog_samples"]["p50"] is not None
    assert metrics["queue_backlog_samples"]["p95"] is not None
    assert metrics["queue_backlog_samples"]["sample_count"] == 10
    assert metrics["queue_backlog_samples"]["sample_unit"] == "samples"
    assert metrics["queue_backlog_samples"]["sampling_policy"] == (
        "SOURCE_AVAILABLE transitions before delivery"
    )
    assert metrics["queue_backlog_samples"]["integration"] == {
        "integration_start_wall_offset_sec": 0.0,
        "integration_end_wall_offset_sec": metrics["measured_wall_interval_sec"],
        "boundary_convention": (
            "piecewise-constant; post-event state applies until next event"
        ),
    }
    assert metrics["queue_backlog_samples"]["final"] == 0
    assert metrics["queue_backlog_samples"]["sampling"].startswith("SOURCE_AVAILABLE")
    assert metrics["delivery_lag"]["max"] >= 0
    assert metrics["completion_overrun_sec"] > 0
    assert metrics["deadline"] is None
    assert metrics["compute_duty_window"]["includes_finish_tail"] is True
    assert metrics["completion_evidence"]["completion_event_type"] == "FINISH_RETURN"
    assert metrics["completion_evidence"]["source_end_event_index"] is not None
    assert metrics["completion_evidence"]["completion_event_index"] is not None
    assert metrics["streaming_wall_interval_sec"] < metrics["measured_wall_interval_sec"]
    assert metrics["tail_interval_sec"] > 0
    assert trace_path.read_bytes() == before


def test_deadline_misses_are_optional_and_exact(tmp_path):
    source, result, _ = _run(tmp_path, compute=0.03)
    metrics = compute_paced_metrics(
        result.trace_events,
        sample_rate=source.metadata.sample_rate,
        source_total_samples=source.total_samples,
        deadline_budget_sec=0.0,
    )
    deadline = metrics["deadline"]
    assert deadline["deadline_budget_sec"] == 0.0
    assert deadline["miss_count"] > 0
    assert deadline["miss_ratio"] == pytest.approx(deadline["miss_count"] / 10)


def test_processing_lag_is_unavailable_without_consumed_position(tmp_path):
    source, result, _ = _run(tmp_path, compute=0.005, consumed=False)
    metrics = compute_paced_metrics(
        result.trace_events,
        sample_rate=source.metadata.sample_rate,
        source_total_samples=source.total_samples,
    )
    assert metrics["processing_lag_samples"]["available"] is False


def test_processing_lag_is_derived_when_consumer_reports_position(tmp_path):
    source = _source(10 * 20, 20)
    clock = FakeWallClock()
    consumer = FakeLatencyConsumer(
        clock,
        compute_sec=0.005,
        emit_hypothesis=True,
        reports_consumed_position=True,
    )
    writer, _ = _writer(tmp_path, source, consumer)
    result = PacedReplayRuntime(source, consumer, writer, wall_clock=clock).run()
    metrics = compute_paced_metrics(
        result.trace_events,
        sample_rate=source.metadata.sample_rate,
        source_total_samples=source.total_samples,
    )
    assert metrics["processing_lag_samples"]["available"] is True
    assert metrics["processing_lag_samples"]["final_samples"] == 0


def test_metrics_artifact_is_separate_and_atomic_writer_round_trips(tmp_path):
    source, result, trace_path = _run(tmp_path, compute=0.005)
    metrics = compute_paced_metrics(
        result.trace_events,
        sample_rate=source.metadata.sample_rate,
        source_total_samples=source.total_samples,
    )
    output = tmp_path / "r3.metrics.json"
    write_metrics_json(output, metrics)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "r3.metrics.v1"
    assert loaded["percentile_method"] == "nearest_rank"
    assert trace_path.exists()
