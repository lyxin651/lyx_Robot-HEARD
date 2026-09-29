import json
import inspect
import wave
from pathlib import Path

import pytest

from robot_heard.asr.base import ASRBackend, ASRResult
from robot_heard.replay.clock import FakeWallClock
from robot_heard.replay.events import ContractValidationError
from robot_heard.replay.oracle import Stage2OracleReleaseGate
from robot_heard.replay.paced_runtime import PacedReplayRuntime
from robot_heard.replay.source import SyntheticSource
from robot_heard.replay.trace import TraceDurabilityPolicy, TraceWriter
from robot_heard.streaming.adapters.offline_whisper import (
    OfflineWhisperReferenceConsumer,
    OfflineWhisperReferenceError,
)


class FakeASRBackend(ASRBackend):
    def __init__(self, *, text="decoded"):
        self.text = text
        self.paths = []

    def transcribe(self, audio_path):
        self.paths.append(Path(audio_path))
        with wave.open(str(audio_path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 4
            assert handle.getframerate() == 16000
            frame_count = handle.getnframes()
        return ASRResult(text=self.text, language="zh", decode_sec=0.001 * frame_count)


def _source():
    return SyntheticSource(
        (
            tuple(range(960)),
            tuple(10000 + value for value in range(960)),
        ),
        sample_rate=16000,
        packet_samples=320,
        stream_id="r4-test",
        channel_ids=("ch0", "ch1"),
        sample_format="pcm_s32le",
        channel_layout="separate_mono_files",
    )


def _writer(tmp_path, source, consumer):
    trace_path = (tmp_path / "trace.jsonl").resolve()
    run_path = (tmp_path / "trace.run.json").resolve()
    provenance = {
        "schema_version": "r4.event.v1",
        "run_id": "r4-test",
        "replay_mode": "COMPUTE_AWARE",
        "source_identity": source.identity(),
        "source_sha256": source.content_sha256,
        "source_sample_rate": source.metadata.sample_rate,
        "channel_ids": list(source.metadata.channel_ids),
        "packet_samples": source.packet_samples,
        "queue_policy": "lossless_unbounded_fifo",
        "timeline_authority": "integer_sample_index",
        "consumer_identity": type(consumer).__name__,
        "consumer_config": {"frontend": "raw_ch0"},
        "consumer_capabilities": dict(consumer.capabilities().to_dict()),
        "code_commit": "test",
        "oracle_condition": "r4_precomputed_segment_end_gate_raw_ch0",
        "warmup_load_policy": {"model_load": "pre_stream", "warmup": "synthetic"},
        "clock_policy": "fake_wall_clock",
        "clock_origin": {"kind": "pending", "run_wall_origin": "pending"},
        "metric_window": {"start_sample": 0, "end_sample": source.total_samples},
        "trace_path": str(trace_path),
        "device_provenance": {"status": "not_applicable"},
    }
    return TraceWriter(
        trace_path,
        run_path,
        provenance,
        durability_policy=TraceDurabilityPolicy.BUFFERED_CLOSE_FSYNC,
    ), trace_path, run_path


def test_raw_ch0_adapter_materializes_exact_arrived_ranges_and_final_only(tmp_path):
    source = _source()
    gate = Stage2OracleReleaseGate(
        [
            type("Row", (), {"segment_id": "seg-500", "start_sample": 0, "end_sample": 500})(),
            type("Row", (), {"segment_id": "seg-960", "start_sample": 640, "end_sample": 960})(),
        ],
        sample_rate=16000,
        packet_duration_sec=0.02,
    )
    backend = FakeASRBackend(text="")
    clock = FakeWallClock()
    consumer = OfflineWhisperReferenceConsumer(
        backend,
        materialized_dir=tmp_path / "materialized",
        spool_path=tmp_path / "audio.pcm",
        segment_evidence_path=tmp_path / "segments.jsonl",
        wall_clock=clock,
    )
    assert "oracle_gate" not in inspect.signature(OfflineWhisperReferenceConsumer).parameters
    assert not hasattr(consumer, "_oracle_gate")
    assert not hasattr(consumer, "remaining_count")
    writer, trace_path, _ = _writer(tmp_path, source, consumer)
    result = PacedReplayRuntime(
        source, consumer, writer, wall_clock=clock, oracle_gate=gate
    ).run()
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    releases = [record for record in records if record["event_type"] == "ORACLE_METADATA_RELEASE"]
    finals = [record for record in records if record["event_type"] == "HYPOTHESIS"]
    assert len(releases) == 2
    assert len(finals) == 2
    assert all(item["payload"]["kind"] == "FINAL" for item in finals)
    assert [item["payload"]["revision_index"] for item in finals] == [0, 0]
    assert [item["payload"]["consumed_through_sample"] for item in finals] == [500, 960]
    assert releases[0]["payload"]["oracle_release_wall_offset_sec"] == pytest.approx(0.04)
    assert releases[0]["payload"]["segment_end_available_wall_offset_sec"] == pytest.approx(500 / 16000)
    assert result.source_end_sample == 960
    assert consumer.startup_metadata == {
        "stream_id": "r4-test",
        "sample_rate": 16000,
        "num_channels": 2,
        "channel_ids": ["ch0", "ch1"],
        "sample_format": "pcm_s32le",
        "channel_layout": "separate_mono_files",
    }
    assert not hasattr(consumer, "all_segments")
    assert not hasattr(consumer, "reference")
    evidence = [json.loads(line) for line in (tmp_path / "segments.jsonl").read_text().splitlines()]
    assert [item["frame_count"] for item in evidence] == [500, 320]
    assert all(item["text_empty"] for item in evidence)
    assert all("reference" not in item for item in evidence)
    assert all("speaker_id" not in item for item in evidence)
    assert len(backend.paths) == 2
    with wave.open(str(backend.paths[0]), "rb") as handle:
        assert handle.readframes(500)[:16] == (0).to_bytes(4, "little", signed=True) + (1).to_bytes(4, "little", signed=True) + (2).to_bytes(4, "little", signed=True) + (3).to_bytes(4, "little", signed=True)


def test_r4_adapter_does_not_decode_until_released_audio_has_arrived(tmp_path):
    source = _source()
    gate = Stage2OracleReleaseGate(
        [type("Row", (), {"segment_id": "late", "start_sample": 0, "end_sample": 960})()],
        sample_rate=16000,
    )
    backend = FakeASRBackend()
    consumer = OfflineWhisperReferenceConsumer(
        backend,
        materialized_dir=tmp_path / "materialized",
        spool_path=tmp_path / "audio.pcm",
        segment_evidence_path=tmp_path / "segments.jsonl",
    )
    consumer.start(source.metadata)
    consumer.on_run_wall_origin(0.0)
    consumer.on_oracle_release(gate.on_source_available(960, 0.06)[0])
    chunks = iter(source)
    assert consumer.consume(next(chunks)) == ()
    assert backend.paths == []
    assert consumer.consume(next(chunks)) == ()
    assert backend.paths == []
    assert len(consumer.consume(next(chunks))) == 1
    assert len(backend.paths) == 1


def test_consumer_pending_state_contains_only_released_descriptors(tmp_path):
    source = _source()
    gate_a = Stage2OracleReleaseGate(
        [type("Row", (), {"segment_id": "same", "start_sample": 0, "end_sample": 320})()],
        sample_rate=16000,
    )
    gate_b = Stage2OracleReleaseGate(
        [type("Row", (), {"segment_id": "same", "start_sample": 0, "end_sample": 320})()],
        sample_rate=16000,
    )
    consumer_a = OfflineWhisperReferenceConsumer(
        FakeASRBackend(),
        materialized_dir=tmp_path / "a-materialized",
        spool_path=tmp_path / "a.pcm",
        segment_evidence_path=tmp_path / "a.jsonl",
    )
    consumer_b = OfflineWhisperReferenceConsumer(
        FakeASRBackend(),
        materialized_dir=tmp_path / "b-materialized",
        spool_path=tmp_path / "b.pcm",
        segment_evidence_path=tmp_path / "b.jsonl",
    )
    release_a = gate_a.on_source_available(320, 0.02)[0]
    release_b = gate_b.on_source_available(320, 0.02)[0]
    consumer_a.on_oracle_release(release_a)
    consumer_b.on_oracle_release(release_b)
    assert consumer_a._pending == consumer_b._pending
    assert not hasattr(consumer_a, "_oracle_gate")
    assert not any("future" in key or "reference" in key for key in vars(consumer_a))
    with pytest.raises(OfflineWhisperReferenceError, match="duplicate oracle release"):
        consumer_a.on_oracle_release(release_a)


def test_runtime_validates_unreleased_oracle_rows_before_consumer_finish(tmp_path):
    source = _source()
    gate = Stage2OracleReleaseGate(
        [type("Row", (), {"segment_id": "too-late", "start_sample": 0, "end_sample": 1000})()],
        sample_rate=16000,
        packet_duration_sec=0.02,
    )
    consumer = OfflineWhisperReferenceConsumer(
        FakeASRBackend(),
        materialized_dir=tmp_path / "materialized",
        spool_path=tmp_path / "audio.pcm",
        segment_evidence_path=tmp_path / "segments.jsonl",
        wall_clock=FakeWallClock(),
    )
    writer, _, _ = _writer(tmp_path, source, consumer)
    try:
        with pytest.raises(ContractValidationError, match="unreleased oracle segments"):
            PacedReplayRuntime(
                source, consumer, writer, wall_clock=FakeWallClock(), oracle_gate=gate
            ).run()
    finally:
        consumer.reset()
