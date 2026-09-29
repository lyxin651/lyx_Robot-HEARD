#!/usr/bin/env python3
"""Run the R5A oracle-segmented rolling-prefix Whisper reference.

The runtime owns the complete Stage-2 timeline and forwards only causal scope
boundary releases.  This script is an experiment entry point; it never writes
authoritative MISP directories and labels the result as pseudo-online.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_heard.asr.factory import create_asr_backend
from robot_heard.asr.scoring import score_result_record
from robot_heard.misp.kaldi_adapter import read_kaldi_text, resolve_misp_channel_path
from robot_heard.misp.timeline import read_stage2_segment_samples
from robot_heard.replay.clock import MonotonicWallClock
from robot_heard.replay.metrics import (
    compute_paced_metrics,
    compute_r5a_metrics,
    write_metrics_json,
)
from robot_heard.replay.oracle import Stage2ScopeBoundaryReleaseGate
from robot_heard.replay.paced_runtime import PacedReplayRuntime
from robot_heard.replay.trace import TraceDurabilityPolicy, TraceWriter
from robot_heard.replay.wav_source import MultiChannelPcmWavSource
from robot_heard.streaming.adapters.rolling_whisper import RollingWhisperConsumer

# Reuse the R4 server-validated config/GPU provenance implementation.  It does
# not import or alter replay runtime state.
from run_r4_m028_offline_whisper import (  # type: ignore[import-not-found]
    _atomic_json,
    _gpu_provenance,
    _git_head,
    _load_config,
    _read_jsonl,
    _sha256_file,
    _sync_cuda,
    _write_silence,
)


CONDITION = "r5a_oracle_segment_boundaries_raw_ch0"


@dataclass
class _PrefixSource:
    """Read-only packet-prefix view used only for bounded GPU engineering smoke."""

    base: Any
    limit_samples: int

    @property
    def metadata(self):
        return self.base.metadata

    @property
    def total_samples(self) -> int:
        return self.limit_samples

    @property
    def packet_samples(self) -> int:
        return self.base.packet_samples

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(
            f"{self.base.content_sha256}:prefix:{self.limit_samples}".encode("ascii")
        ).hexdigest()

    def identity(self) -> str:
        return f"{self.base.identity()}:prefix:{self.limit_samples}"

    def __iter__(self):
        for chunk in self.base:
            if chunk.start_sample >= self.limit_samples:
                break
            if chunk.end_sample > self.limit_samples:
                raise ValueError("prefix limit must align to a source packet")
            yield replace(chunk, is_final_source_chunk=chunk.end_sample == self.limit_samples)


def _final_rows(snapshots: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in snapshots if row.get("kind") == "FINAL"]


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_root = args.output_root.expanduser().resolve(strict=False)
    if output_root.exists():
        raise FileExistsError(f"R5A output already exists; refusing overwrite: {output_root}")
    if args.max_scopes is not None and args.max_scopes <= 0:
        raise ValueError("max-scopes must be positive")

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
    full_source = MultiChannelPcmWavSource(
        channel_paths,
        channel_ids=channel_ids,
        stream_id=args.recording_id,
        packet_samples=args.packet_samples,
    )
    if (
        full_source.metadata.sample_rate != 16000
        or full_source.sample_width != 4
        or full_source.compression_type != "NONE"
        or full_source.metadata.num_channels != 8
    ):
        raise ValueError("R5A canonical source must be 8-channel, 16 kHz, PCM32, uncompressed")
    segments = read_stage2_segment_samples(
        args.segments,
        recording_id=args.recording_id,
        sample_rate=full_source.metadata.sample_rate,
        source_total_samples=full_source.total_samples,
    )
    if args.max_scopes is not None:
        segments = segments[: args.max_scopes]
        required_end = max(row.end_sample for row in segments)
        limit = math.ceil(required_end / args.packet_samples) * args.packet_samples
        source = _PrefixSource(full_source, min(limit, full_source.total_samples))
        segments = [row for row in segments if row.end_sample <= source.total_samples]
    else:
        source = full_source
    if not segments:
        raise ValueError("R5A run has no Stage-2 scopes")
    if args.max_scopes is None and len(segments) != 144:
        raise ValueError(f"R5A M028 expects 144 Stage-2 scopes, got {len(segments)}")

    gate = Stage2ScopeBoundaryReleaseGate(
        segments,
        sample_rate=source.metadata.sample_rate,
        condition=CONDITION,
        packet_duration_sec=args.packet_samples / source.metadata.sample_rate,
    )

    load_start = time.perf_counter()
    backend = create_asr_backend(config)
    model_load_sec = time.perf_counter() - load_start
    with tempfile.TemporaryDirectory(prefix="r5a-whisper-warmup-") as temp_dir:
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
    materialized_dir = output_root / "materialized"
    trace_path = output_root / "trace.jsonl"
    run_path = output_root / "trace.run.json"
    spool_path = output_root / "raw_ch0.pcm"
    snapshot_path = output_root / "r5a_snapshots.jsonl"
    wall_clock = MonotonicWallClock()
    consumer = RollingWhisperConsumer(
        backend,
        materialized_dir=materialized_dir,
        spool_path=spool_path,
        snapshot_evidence_path=snapshot_path,
        update_interval_samples=args.update_interval_samples,
        selected_channel_id="ch0",
        frontend_condition="raw_ch0",
        gpu_sync_policy=(
            "cuda_synchronize_before_and_after_backend_transcribe"
            if config["device"] == "cuda"
            else "none"
        ),
        wall_clock=wall_clock,
    )
    provenance = {
        "schema_version": "r5a.event.v1",
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
            "update_interval_samples": args.update_interval_samples,
            "materialization": "delivered_pcm_spool_exact_sample_prefix",
            "label": "oracle-segmented rolling-prefix Whisper; not native streaming",
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
        schema_version="r5a.event.v1",
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
    r5a_metrics = compute_r5a_metrics(
        consumer.snapshots,
        sample_rate=source.metadata.sample_rate,
        global_metrics=global_metrics,
    )
    write_metrics_json(output_root / "r5a.metrics.json", r5a_metrics)

    final_rows = _final_rows(consumer.snapshots)
    references = read_kaldi_text(args.text, required=True)
    scored_rows = []
    evaluator_path = output_root / "r5a_evaluator_results.jsonl"
    scored_path = output_root / "r5a_scored.jsonl"
    with evaluator_path.open("x", encoding="utf-8") as evaluator, scored_path.open(
        "x", encoding="utf-8"
    ) as scored_handle:
        for row in final_rows:
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
            evaluator.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            scored = score_result_record(result_row)
            scored_handle.write(json.dumps(scored, ensure_ascii=False) + "\n")
            scored_rows.append(scored)
    reference_chars = sum(row["cer_n"] for row in scored_rows)
    errors = sum(row["cer_s"] + row["cer_d"] + row["cer_i"] for row in scored_rows)
    cer = errors / reference_chars if reference_chars else None

    compatibility = None
    if args.r4_segments is not None:
        r4_rows = _read_jsonl(args.r4_segments.expanduser().resolve(strict=True))
        r4_by_id = {row["segment_id"]: row for row in r4_rows}
        r5_by_id = {row["segment_id"]: row for row in final_rows}
        compatibility = {
            "r4_path": str(args.r4_segments.expanduser().resolve()),
            "segment_id_set_equal": set(r4_by_id) == set(r5_by_id),
            "exact_text_equal_count": sum(
                segment_id in r4_by_id
                and r4_by_id[segment_id].get("text_raw") == row.get("text_raw")
                for segment_id, row in r5_by_id.items()
            ),
            "r5_final_count": len(r5_by_id),
            "r4_empty_count": sum(row.get("text_raw", "") == "" for row in r4_rows),
            "r5_empty_count": sum(row.get("text_raw", "") == "" for row in final_rows),
            "hard_text_compatibility_pass": (
                set(r4_by_id) == set(r5_by_id)
                and len(r4_by_id) == len(r5_by_id)
                and all(r4_by_id[key].get("text_raw") == r5_by_id[key].get("text_raw") for key in r5_by_id)
            ),
        }
        _atomic_json(output_root / "r4_final_compatibility.json", compatibility)

    trace_records = _read_jsonl(trace_path)
    start_release_count = sum(
        row["event_type"] == "ORACLE_METADATA_RELEASE" and row["payload"].get("boundary") == "START"
        for row in trace_records
    )
    end_release_count = sum(
        row["event_type"] == "ORACLE_METADATA_RELEASE" and row["payload"].get("boundary") == "END"
        for row in trace_records
    )
    report = {
        "schema_version": "r5a.m028.report.v1",
        "code_commit": code_commit,
        "recording_id": args.recording_id,
        "condition": CONDITION,
        "label": "oracle-segmented rolling-prefix Whisper; segment-level pseudo-online; not native streaming",
        "source_packet_count": sum(row["event_type"] == "SOURCE_AVAILABLE" for row in trace_records),
        "stage2_scope_count": len(segments),
        "scope_start_release_count": start_release_count,
        "scope_end_release_count": end_release_count,
        "partial_count": r5a_metrics["partial_count"],
        "final_count": r5a_metrics["final_count"],
        "unique_scope_ids": len({row["segment_id"] for row in final_rows}),
        "materialized_snapshot_count": len(list(materialized_dir.glob("*.wav"))),
        "empty_count": sum(row["text_raw"] == "" for row in final_rows),
        "non_empty_count": sum(row["text_raw"] != "" for row in final_rows),
        "cer": cer,
        "evaluated_segments": len(scored_rows),
        "model_load_sec": model_load_sec,
        "warmup_sec": warmup_sec,
        "whisper_config": config,
        "gpu_sync_policy": consumer.gpu_sync_policy,
        "gpu_provenance": gpu,
        "update_interval_samples": args.update_interval_samples,
        "source_duration_sec": source.total_samples / source.metadata.sample_rate,
        "r3_metrics_path": str((output_root / "r3.metrics.json").resolve()),
        "r5a_metrics_path": str((output_root / "r5a.metrics.json").resolve()),
        "trace_path": str(trace_path.resolve()),
        "run_provenance_path": str(run_path.resolve()),
        "snapshot_evidence_path": str((output_root / "r5a_snapshots.jsonl").resolve()),
        "r5a_evaluator_results_path": str(evaluator_path.resolve()),
        "r5a_scored_path": str(scored_path.resolve()),
        "r4_compatibility": compatibility,
        "reference_leakage": False,
        "future_segment_leakage": False,
        "native_streaming": False,
    }
    _atomic_json(output_root / "r5a_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-stem", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/whisper_openai_v0.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--r4-segments", type=Path)
    parser.add_argument("--packet-samples", type=int, default=320)
    parser.add_argument("--channel-count", type=int, default=8)
    parser.add_argument("--update-interval-samples", type=int, default=16000)
    parser.add_argument("--max-scopes", type=int)
    parser.add_argument("--run-id", default="r5a-m028-rawch0-rolling-whisper")
    parser.add_argument("--code-commit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = _run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
