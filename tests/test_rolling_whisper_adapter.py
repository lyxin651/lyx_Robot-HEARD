from __future__ import annotations

import inspect
import json
import wave
from pathlib import Path
from typing import Sequence

import pytest

from robot_heard.asr.base import ASRBackend, ASRResult
from robot_heard.replay.clock import FakeWallClock
from robot_heard.replay.events import (
    ConsumerCapabilities,
    HypothesisEvent,
    HypothesisKind,
)
from robot_heard.replay.oracle import Stage2ScopeBoundaryReleaseGate
from robot_heard.replay.paced_runtime import PacedReplayRuntime
from robot_heard.replay.source import SyntheticSource
from robot_heard.replay.trace import TraceWriter
from robot_heard.streaming.adapters.rolling_whisper import RollingWhisperConsumer


class _Row:
    def __init__(self, segment_id: str, start_sample: int, end_sample: int):
        self.segment_id = segment_id
        self.start_sample = start_sample
        self.end_sample = end_sample


class _FakeBackend(ASRBackend):
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path):
        path = Path(audio_path)
        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == 16000
            assert handle.getsampwidth() == 4
            frames = handle.getnframes()
        self.calls.append((path, frames))
        return ASRResult(text=f"frames-{frames}", language="zh", decode_sec=0.0)


def _source(sample_count: int = 40000, packet_samples: int = 320):
    return SyntheticSource(
        (tuple(range(sample_count)), tuple(100000 + value for value in range(sample_count))),
        sample_rate=16000,
        packet_samples=packet_samples,
        stream_id="r5a-test",
        channel_ids=("ch0", "ch1"),
        sample_format="pcm_s32le",
        channel_layout="separated_mono",
    )


def _writer(tmp_path: Path, source, consumer):
    trace_path = (tmp_path / "trace.jsonl").resolve()
    run_path = (tmp_path / "trace.run.json").resolve()
    provenance = {
        "schema_version": "r5a.event.v1",
        "run_id": "r5a-test",
        "replay_mode": "COMPUTE_AWARE",
        "source_identity": source.identity(),
        "source_sha256": source.content_sha256,
        "source_sample_rate": source.metadata.sample_rate,
        "channel_ids": list(source.metadata.channel_ids),
        "packet_samples": source.packet_samples,
        "queue_policy": "lossless_unbounded_fifo",
        "timeline_authority": "integer_sample_index",
        "consumer_identity": type(consumer).__name__,
        "consumer_config": {"update_interval_samples": consumer.update_interval_samples},
        "consumer_capabilities": dict(consumer.capabilities().to_dict()),
        "code_commit": "test",
        "oracle_condition": "r5a_test",
        "warmup_load_policy": {"model_load": "not_applicable", "warmup": "not_applicable"},
        "clock_policy": "fake_wall_clock",
        "clock_origin": {"kind": "fake_wall", "run_wall_origin": None},
        "metric_window": {"start_sample": 0, "end_sample": source.total_samples},
        "trace_path": str(trace_path),
        "device_provenance": {"status": "not_applicable"},
        "trace_durability_policy": "buffered_close_fsync",
    }
    return TraceWriter(trace_path, run_path, provenance), trace_path


def _run(tmp_path: Path, rows, *, sample_count=40000, update_interval=16000):
    source = _source(sample_count=sample_count)
    backend = _FakeBackend()
    consumer = RollingWhisperConsumer(
        backend,
        materialized_dir=tmp_path / "materialized",
        spool_path=tmp_path / "audio.pcm",
        snapshot_evidence_path=tmp_path / "snapshots.jsonl",
        update_interval_samples=update_interval,
        wall_clock=FakeWallClock(),
    )
    writer, trace_path = _writer(tmp_path, source, consumer)
    gate = Stage2ScopeBoundaryReleaseGate(rows, sample_rate=16000, packet_duration_sec=0.02)
    result = PacedReplayRuntime(
        source, consumer, writer, wall_clock=consumer.wall_clock, oracle_gate=gate
    ).run()
    return source, backend, consumer, result, trace_path


def test_constructor_has_no_gate_and_capabilities_are_not_native():
    assert "oracle_gate" not in inspect.signature(RollingWhisperConsumer).parameters
    consumer = object.__new__(RollingWhisperConsumer)
    assert not hasattr(consumer, "_oracle_gate")


def test_short_scope_direct_final_and_exact_input(tmp_path):
    source, backend, consumer, result, trace_path = _run(
        tmp_path, [_Row("short", 0, 8000)], sample_count=8000
    )
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    hypotheses = [r["payload"] for r in records if r["event_type"] == "HYPOTHESIS"]
    assert len(hypotheses) == 1
    assert hypotheses[0]["kind"] == "FINAL"
    assert hypotheses[0]["revision_index"] == 0
    assert hypotheses[0]["replaces_event_id"] is None
    assert hypotheses[0]["consumed_through_sample"] == 8000
    assert backend.calls[-1][1] == 8000
    assert consumer.capabilities().supports_stable_prefix is False
    assert consumer.capabilities().native_streaming is False
    assert result.source_end_sample == source.total_samples


def test_rolling_prefix_revisions_are_complete_and_final_is_exact(tmp_path):
    _, backend, consumer, _, trace_path = _run(
        tmp_path, [_Row("long", 0, 40000)], sample_count=40000
    )
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    hypotheses = [r["payload"] for r in records if r["event_type"] == "HYPOTHESIS"]
    assert [item["kind"] for item in hypotheses] == ["PARTIAL", "PARTIAL", "FINAL"]
    assert [item["revision_index"] for item in hypotheses] == [0, 1, 2]
    assert hypotheses[0]["replaces_event_id"] is None
    assert hypotheses[1]["replaces_event_id"] == hypotheses[0]["event_id"]
    assert hypotheses[2]["replaces_event_id"] == hypotheses[1]["event_id"]
    assert [item["consumed_through_sample"] for item in hypotheses] == [16000, 32000, 40000]
    assert [frames for _, frames in backend.calls] == [16000, 32000, 40000]
    assert all(item["text"] == f"frames-{frames}" for item, (_, frames) in zip(hypotheses, backend.calls))
    assert consumer.active_scope_ids == ()


def test_overlapping_scopes_are_each_processed_without_future_metadata(tmp_path):
    _, backend, consumer, _, trace_path = _run(
        tmp_path,
        [_Row("a", 0, 24000), _Row("b", 8000, 32000)],
        sample_count=32000,
    )
    assert consumer.startup_metadata == {
        "stream_id": "r5a-test",
        "sample_rate": 16000,
        "num_channels": 2,
        "channel_ids": ["ch0", "ch1"],
        "sample_format": "pcm_s32le",
        "channel_layout": "separated_mono",
    }
    assert not hasattr(consumer, "_oracle_gate")
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    releases = [r["payload"] for r in records if r["event_type"] == "ORACLE_METADATA_RELEASE"]
    assert all(not ({"reference", "speaker_id", "future_segments"} & set(item)) for item in releases)
    finals = [r["payload"] for r in records if r["event_type"] == "HYPOTHESIS" and r["payload"]["kind"] == "FINAL"]
    assert {item["scope_id"] for item in finals} == {"a", "b"}
    assert len(backend.calls) > 2
