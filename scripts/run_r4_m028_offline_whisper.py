#!/usr/bin/env python3
"""Run the R4 oracle-gated offline Whisper reference on one MISP recording.

This is intentionally a slow, wall-paced reference experiment.  It does not
modify MISP artifacts and never exposes Stage-2 references to the consumer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_heard.asr.factory import create_asr_backend
from robot_heard.asr.scoring import score_result_record
from robot_heard.misp.kaldi_adapter import read_kaldi_text, resolve_misp_channel_path
from robot_heard.misp.timeline import read_stage2_segment_samples
from robot_heard.replay.metrics import compute_paced_metrics, compute_r4_metrics, write_metrics_json
from robot_heard.replay.oracle import Stage2OracleReleaseGate
from robot_heard.replay.paced_runtime import PacedReplayRuntime
from robot_heard.replay.trace import TraceDurabilityPolicy, TraceWriter
from robot_heard.replay.wav_source import MultiChannelPcmWavSource
from robot_heard.streaming.adapters.offline_whisper import OfflineWhisperReferenceConsumer


CONDITION = "r4_precomputed_segment_end_gate_raw_ch0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or len(value) != 40:
        raise RuntimeError("unable to resolve exact git HEAD; code_commit is required")
    return value


def _gpu_provenance() -> Mapping[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,compute_mode,memory.used,memory.free",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi GPU provenance query failed") from exc
    return {"nvidia_smi": completed.stdout.strip(), "visible_device": "CUDA_VISIBLE_DEVICES=0"}


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Whisper config must be a mapping: {path}")
    expected = {
        "backend": "openai_whisper",
        "model": "large-v3",
        "language": "zh",
        "task": "transcribe",
        "device": "cuda",
        "fp16": True,
        "temperature": 0.0,
    }
    decode = value.get("decode")
    if not isinstance(decode, Mapping):
        raise ValueError("Whisper config decode mapping is required")
    actual = dict(value)
    actual["temperature"] = decode.get("temperature")
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise ValueError(
                f"R4 requires frozen Whisper config {key}={expected_value!r}; "
                f"got {actual.get(key)!r}"
            )
    return value


def _write_silence(path: Path, frame_count: int = 1600) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(4)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00\x00\x00" * frame_count)


def _sync_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for R4")
    torch.cuda.synchronize()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_root = args.output_root.expanduser().resolve(strict=False)
    if output_root.exists():
        raise FileExistsError(f"R4 output already exists; refusing overwrite: {output_root}")
    materialized_dir = output_root / "materialized"
    trace_path = output_root / "trace.jsonl"
    run_path = output_root / "trace.run.json"
    spool_path = output_root / "raw_ch0.pcm"
    segment_evidence_path = output_root / "r4_segments.jsonl"

    code_commit = args.code_commit or _git_head()
    config = _load_config(args.config.expanduser().resolve(strict=True))
    gpu = _gpu_provenance()
    if config["device"] == "cuda":
        _sync_cuda()

    channel_ids = tuple(f"ch{index}" for index in range(args.channel_count))
    channel_paths = tuple(
        resolve_misp_channel_path(args.channel_stem, index)
        for index in range(args.channel_count)
    )
    source = MultiChannelPcmWavSource(
        channel_paths,
        channel_ids=channel_ids,
        stream_id=args.recording_id,
        packet_samples=args.packet_samples,
    )
    if (
        source.metadata.sample_rate != 16000
        or source.sample_width != 4
        or source.compression_type != "NONE"
        or source.metadata.num_channels != 8
    ):
        raise ValueError("R4 canonical source must be 8-channel, 16 kHz, PCM32, uncompressed")
    segments = read_stage2_segment_samples(
        args.segments,
        recording_id=args.recording_id,
        sample_rate=source.metadata.sample_rate,
        source_total_samples=source.total_samples,
    )
    if len(segments) != 144:
        raise ValueError(f"R4 M028 reference expects 144 Stage-2 segments, got {len(segments)}")
    gate = Stage2OracleReleaseGate(
        segments,
        sample_rate=source.metadata.sample_rate,
        condition=CONDITION,
        packet_duration_sec=args.packet_samples / source.metadata.sample_rate,
    )

    load_start = time.perf_counter()
    backend = create_asr_backend(config)
    model_load_sec = time.perf_counter() - load_start
    with tempfile.TemporaryDirectory(prefix="r4-whisper-warmup-") as temp_dir:
        warmup_path = Path(temp_dir) / "synthetic_silence.wav"
        _write_silence(warmup_path)
        if config["device"] == "cuda":
            _sync_cuda()
        warmup_start = time.perf_counter()
        backend.transcribe(warmup_path)
        if config["device"] == "cuda":
            _sync_cuda()
        warmup_sec = time.perf_counter() - warmup_start

    output_root.mkdir(parents=True)
    consumer = OfflineWhisperReferenceConsumer(
        backend,
        gate,
        materialized_dir=materialized_dir,
        spool_path=spool_path,
        segment_evidence_path=segment_evidence_path,
        selected_channel_id="ch0",
        frontend_condition="raw_ch0",
        gpu_sync_policy=(
            "cuda_synchronize_before_and_after_backend_transcribe"
            if config["device"] == "cuda"
            else "none"
        ),
        wall_clock=None,
    )
    # The adapter needs the same production clock used by the scheduler for
    # invocation evidence.  Importing it here keeps the replay core model-free.
    from robot_heard.replay.clock import MonotonicWallClock

    wall_clock = MonotonicWallClock()
    consumer.wall_clock = wall_clock
    provenance = {
        "schema_version": "r4.event.v1",
        "run_id": args.run_id,
        "replay_mode": "COMPUTE_AWARE",
        "source_identity": source.identity(),
        "source_sha256": source.content_sha256,
        "source_sample_rate": source.metadata.sample_rate,
        "channel_ids": list(source.metadata.channel_ids),
        "packet_samples": source.packet_samples,
        "queue_policy": "lossless_unbounded_fifo",
        "timeline_authority": "integer_sample_index",
        "consumer_identity": type(consumer).__name__,
        "consumer_config": {
            "condition": CONDITION,
            "frontend_condition": "raw_ch0",
            "selected_channel_id": "ch0",
            "materialization": "delivered_pcm_spool_exact_sample_range",
        },
        "consumer_capabilities": dict(consumer.capabilities().to_dict()),
        "code_commit": code_commit,
        "oracle_condition": CONDITION,
        "warmup_load_policy": {
            "model_load_sec": model_load_sec,
            "warmup_sec": warmup_sec,
            "warmup_audio": "synthetic_silence",
            "warmup_output_ignored": True,
        },
        "clock_policy": "monotonic_perf_counter_wall_clock",
        "clock_origin": {"kind": "pending_until_consumer_start", "run_wall_origin": "pending"},
        "metric_window": {"start_sample": 0, "end_sample": source.total_samples},
        "trace_path": str(trace_path.resolve()),
        "device_provenance": dict(gpu),
        "gpu_sync_policy": consumer.gpu_sync_policy,
        "trace_durability_policy": TraceDurabilityPolicy.BUFFERED_CLOSE_FSYNC.value,
    }
    writer = TraceWriter(
        trace_path,
        run_path,
        provenance,
        schema_version="r4.event.v1",
        durability_policy=TraceDurabilityPolicy.BUFFERED_CLOSE_FSYNC,
    )
    result = PacedReplayRuntime(
        source,
        consumer,
        writer,
        wall_clock=wall_clock,
        oracle_gate=gate,
    ).run()
    run_provenance = json.loads(run_path.read_text(encoding="utf-8"))
    global_metrics = compute_paced_metrics(
        result.trace_events,
        sample_rate=source.metadata.sample_rate,
        source_total_samples=source.total_samples,
        run_id=args.run_id,
        run_provenance=run_provenance,
        measured_wall_interval_sec=result.finish_return_wall_offset_sec,
    )
    write_metrics_json(output_root / "r3.metrics.json", global_metrics)
    r4_metrics = compute_r4_metrics(consumer.segment_evidence, global_metrics=global_metrics)
    write_metrics_json(output_root / "r4.metrics.json", r4_metrics)

    references = read_kaldi_text(args.text, required=True)
    evaluator_path = output_root / "r4_evaluator_results.jsonl"
    scored_path = output_root / "r4_scored.jsonl"
    scored_rows = []
    with evaluator_path.open("x", encoding="utf-8") as evaluator_handle, scored_path.open(
        "x", encoding="utf-8"
    ) as scored_handle:
        for row in consumer.segment_evidence:
            segment_id = row["segment_id"]
            if segment_id not in references:
                raise ValueError(f"missing post-decode reference for {segment_id!r}")
            result_row = {
                "status": "success",
                "segment_id": segment_id,
                "text_raw": row["text_raw"],
                "reference": references[segment_id],
                "audio_sec": row["frame_count"] / source.metadata.sample_rate,
                "decode_sec": row["backend_decode_sec"],
            }
            evaluator_handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            scored = score_result_record(result_row)
            scored_handle.write(json.dumps(scored, ensure_ascii=False) + "\n")
            scored_rows.append(scored)
    reference_chars = sum(row["cer_n"] for row in scored_rows)
    errors = sum(row["cer_s"] + row["cer_d"] + row["cer_i"] for row in scored_rows)
    cer = errors / reference_chars if reference_chars else None
    report = {
        "schema_version": "r4.m028.report.v1",
        "code_commit": code_commit,
        "recording_id": args.recording_id,
        "condition": CONDITION,
        "source_packet_count": sum(
            event.event_type.value == "SOURCE_AVAILABLE" for event in result.trace_events
        ),
        "stage2_segment_count": len(segments),
        "oracle_release_count": r4_metrics["segment_count"],
        "final_count": r4_metrics["final_count"],
        "unique_scope_ids": len({row["segment_id"] for row in consumer.segment_evidence}),
        "materialized_segment_count": len(list(materialized_dir.glob("*.wav"))),
        "empty_count": r4_metrics["empty_count"],
        "non_empty_count": r4_metrics["non_empty_count"],
        "cer": cer,
        "evaluated_segments": len(scored_rows),
        "model_load_sec": model_load_sec,
        "warmup_sec": warmup_sec,
        "whisper_config": config,
        "gpu_sync_policy": consumer.gpu_sync_policy,
        "gpu_provenance": gpu,
        "source_duration_sec": source.total_samples / source.metadata.sample_rate,
        "r3_metrics_path": str((output_root / "r3.metrics.json").resolve()),
        "r4_metrics_path": str((output_root / "r4.metrics.json").resolve()),
        "trace_path": str(trace_path.resolve()),
        "run_provenance_path": str(run_path.resolve()),
        "r4_segments_path": str(segment_evidence_path.resolve()),
        "r4_evaluator_results_path": str(evaluator_path.resolve()),
        "r4_scored_path": str(scored_path.resolve()),
    }
    _atomic_json(output_root / "r4_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-stem", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/whisper_openai_v0.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--packet-samples", type=int, default=320)
    parser.add_argument("--channel-count", type=int, default=8)
    parser.add_argument("--run-id", default="r4-m028-rawch0-offline-whisper")
    parser.add_argument("--code-commit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
