#!/usr/bin/env python3
"""Run the R5B oracle-segmented WeNet U2++ native streaming condition.

The runtime remains the sole owner of the Stage-2 boundary gate.  This script
only assembles the canonical M028 source/evaluator and records adapter/model
provenance; references are opened after replay completion.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_heard.asr.scoring import score_result_record
from robot_heard.misp.kaldi_adapter import read_kaldi_text, resolve_misp_channel_path
from robot_heard.misp.timeline import read_stage2_segment_samples
from robot_heard.replay.clock import MonotonicWallClock
from robot_heard.replay.metrics import (
    compute_paced_metrics,
    compute_r5b_metrics,
    write_metrics_json,
)
from robot_heard.replay.oracle import Stage2ScopeBoundaryReleaseGate
from robot_heard.replay.paced_runtime import PacedReplayRuntime
from robot_heard.replay.trace import TraceDurabilityPolicy, TraceWriter
from robot_heard.replay.wav_source import MultiChannelPcmWavSource
from robot_heard.streaming.adapters.wenet_u2pp import (
    CtypesWeNetDecoderFactory,
    DEFAULT_WENET_CHUNK_SIZE,
    pcm32_to_pcm16le,
    WeNetU2PPStreamingConsumer,
)


CONDITION = "r5b_wenet_u2pp_oracle_segment_boundaries_raw_ch0"
DEFAULT_LIBRARY = Path(
    "/home/gc6_challenge/code/native_backend_qualification/wenet/"
    "runtime/libtorch/build2/api/libwenet_api.so"
)
DEFAULT_MODEL_DIR = Path(
    "/home/gc6_challenge/models/native_streaming/wenet/"
    "20210601_u2++_conformer_libtorch"
)
MODEL_ARCHIVE_SHA256 = "ace400d74a0eb073b0ee284e0c21be18796a7ab095a8b013d0e703e0180b1968"
LIBTORCH_SHA256 = "e1f6bc48403022ff4680c7299cc8b160df146892c414b8a6b6f7d5aff65bcbce"
WENET_REVISION = "d17059667d6afe0680d19b3a4948ab825ef25105"
R4_WHISPER_REFERENCE_CER = 0.3974947807933194


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
    ).strip()


def _rss_bytes() -> int:
    # Linux ru_maxrss is KiB; this is a process-level peak, not a model-only
    # allocation claim.  It is recorded with its source and unit.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass
class _PrefixSource:
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
        raise FileExistsError(f"R5B output already exists; refusing overwrite: {output_root}")
    if args.max_scopes is not None and args.max_scopes <= 0:
        raise ValueError("max-scopes must be positive")

    code_commit = args.code_commit or _git_head()
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
        raise ValueError("R5B canonical source must be 8-channel, 16 kHz, PCM32, uncompressed")
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
        raise ValueError("R5B run has no Stage-2 scopes")
    if args.max_scopes is None and len(segments) != 144:
        raise ValueError(f"full M028 expects 144 Stage-2 scopes, got {len(segments)}")

    library_path = args.library_path.expanduser().resolve(strict=True)
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    factory = CtypesWeNetDecoderFactory(
        library_path,
        model_dir,
        wenet_chunk_size=args.wenet_chunk_size,
        language="chs",
    )
    warmup_pcm16le = pcm32_to_pcm16le([0] * args.warmup_samples) if args.warmup_samples else None
    wall_clock = MonotonicWallClock()
    consumer = WeNetU2PPStreamingConsumer(
        decoder_factory=factory,
        model_chunk_samples=args.model_chunk_samples,
        selected_channel_id="ch0",
        frontend_condition="raw_ch0",
        warmup_pcm16le=warmup_pcm16le,
        snapshot_evidence_path=output_root / "r5b_snapshots.jsonl",
        wall_clock=wall_clock,
    )
    gate = Stage2ScopeBoundaryReleaseGate(
        segments,
        sample_rate=source.metadata.sample_rate,
        condition=CONDITION,
        packet_duration_sec=args.packet_samples / source.metadata.sample_rate,
    )

    output_root.mkdir(parents=True)
    trace_path = output_root / "trace.jsonl"
    run_path = output_root / "trace.run.json"
    rss_before = _rss_bytes()
    provenance = {
        "schema_version": "r5b.wenet.event.v1",
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
            "model_chunk_samples": args.model_chunk_samples,
            "wenet_chunk_size_post_subsampling_frames": args.wenet_chunk_size,
            "pcm_conversion_policy": "signed_pcm32le_arithmetic_right_shift_16_to_pcm16le_with_saturation",
            "scope_state": "one native decoder state per active oracle scope; reset after FINAL; pooled reuse",
            "partial_event_policy": "suppress initial empty and repeated visible text; emit changed text; FINAL once",
            "label": "oracle-segmented WeNet U2++ native streaming ASR",
        },
        "consumer_capabilities": dict(consumer.capabilities().to_dict()),
        "code_commit": code_commit,
        "oracle_condition": CONDITION,
        "warmup_load_policy": {
            "warmup_samples": args.warmup_samples,
            "warmup_sec": "recorded after native model load; output ignored; decoder reset before replay",
            "model_load_excluded_from_measured_stream": True,
        },
        "clock_policy": "monotonic_perf_counter_wall_clock",
        "clock_origin": {"kind": "pending_until_consumer_start", "run_wall_origin": "pending"},
        "metric_window": {"start_sample": 0, "end_sample": source.total_samples},
        "trace_path": str(trace_path.resolve()),
        "native_runtime": {
            "api": "official WeNet LibTorch C API wenet_init/wenet_decode/wenet_reset",
            "wenet_revision": WENET_REVISION,
            "library_path": str(library_path),
            "library_sha256": _sha256_file(library_path),
            "model_dir": str(model_dir),
            "model_archive_sha256": MODEL_ARCHIVE_SHA256,
            "libtorch_cpu_runtime_sha256": LIBTORCH_SHA256,
        },
        "cpu_memory_provenance": {
            "measurement": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "unit": "bytes",
            "before_replay_start": rss_before,
        },
        "device_provenance": {
            "device": "cpu",
            "status": "qualified_official_libtorch_cpu_runtime",
            "runtime_sha256": LIBTORCH_SHA256,
        },
        "trace_durability_policy": TraceDurabilityPolicy.BUFFERED_CLOSE_FSYNC.value,
    }
    writer = TraceWriter(
        trace_path,
        run_path,
        provenance,
        schema_version="r5b.wenet.event.v1",
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
    r5b_metrics = compute_r5b_metrics(
        consumer.snapshots,
        sample_rate=source.metadata.sample_rate,
        native_audit=consumer.native_audit,
        global_metrics=global_metrics,
    )
    write_metrics_json(output_root / "r5b.metrics.json", r5b_metrics)

    final_rows = _final_rows(consumer.snapshots)
    references = read_kaldi_text(args.text, required=True)
    scored_rows = []
    evaluator_path = output_root / "r5b_evaluator_results.jsonl"
    scored_path = output_root / "r5b_scored.jsonl"
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
                "audio_sec": (row["end_sample"] - row["start_sample"]) / source.metadata.sample_rate,
                "decode_sec": (
                    None
                    if row["native_call_start_wall_offset_sec"] is None
                    or row["native_call_return_wall_offset_sec"] is None
                    else row["native_call_return_wall_offset_sec"]
                    - row["native_call_start_wall_offset_sec"]
                ),
            }
            evaluator.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            scored = score_result_record(result_row)
            scored_handle.write(json.dumps(scored, ensure_ascii=False) + "\n")
            scored_rows.append(scored)
    reference_chars = sum(row["cer_n"] for row in scored_rows)
    errors = sum(row["cer_s"] + row["cer_d"] + row["cer_i"] for row in scored_rows)
    cer = errors / reference_chars if reference_chars else None

    trace_records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    start_release_count = sum(
        row["event_type"] == "ORACLE_METADATA_RELEASE" and row["payload"].get("boundary") == "START"
        for row in trace_records
    )
    end_release_count = sum(
        row["event_type"] == "ORACLE_METADATA_RELEASE" and row["payload"].get("boundary") == "END"
        for row in trace_records
    )
    report = {
        "schema_version": "r5b.wenet.m028.report.v1",
        "code_commit": code_commit,
        "recording_id": args.recording_id,
        "condition": CONDITION,
        "label": "oracle-segmented WeNet U2++ native streaming ASR",
        "source_packet_count": sum(row["event_type"] == "SOURCE_AVAILABLE" for row in trace_records),
        "stage2_scope_count": len(segments),
        "scope_start_release_count": start_release_count,
        "scope_end_release_count": end_release_count,
        "partial_count": r5b_metrics["partial_count"],
        "revision_count": r5b_metrics["revision_count"],
        "final_count": r5b_metrics["final_count"],
        "unique_scope_ids": len({row["segment_id"] for row in final_rows}),
        "empty_count": sum(row["text_raw"] == "" for row in final_rows),
        "non_empty_count": sum(row["text_raw"] != "" for row in final_rows),
        "cer": cer,
        "quality_review": {
            "status": (
                "MODEL_QUALITY_REVIEW_REQUIRED"
                if cer is not None and cer > R4_WHISPER_REFERENCE_CER
                else "DIAGNOSTIC_ONLY"
            ),
            "r4_whisper_reference_cer": R4_WHISPER_REFERENCE_CER,
            "cer_delta_vs_r4_whisper": (
                None if cer is None else cer - R4_WHISPER_REFERENCE_CER
            ),
            "interpretation": (
                "WeNet quality is evaluated diagnostically on MISP; do not require "
                "bit-identical agreement with Whisper. If unacceptable, continue "
                "qualification with AISHELL2 U2++ or WenetSpeech U2++ using the "
                "same adapter, without changing replay core semantics."
            ),
        },
        "evaluated_segments": len(scored_rows),
        "model_load_sec": consumer.model_load_sec,
        "warmup_sec": consumer.warmup_sec,
        "cpu_memory_peak_bytes": _rss_bytes(),
        "native_audit": dict(consumer.native_audit),
        "native_runtime": provenance["native_runtime"],
        "r3_metrics_path": str((output_root / "r3.metrics.json").resolve()),
        "r5b_metrics_path": str((output_root / "r5b.metrics.json").resolve()),
        "trace_path": str(trace_path.resolve()),
        "run_provenance_path": str(run_path.resolve()),
        "snapshot_evidence_path": str((output_root / "r5b_snapshots.jsonl").resolve()),
        "evaluator_results_path": str(evaluator_path.resolve()),
        "scored_path": str(scored_path.resolve()),
        "reference_leakage": False,
        "future_segment_leakage": False,
        "native_streaming": True,
        "replay_core_semantics_changed": False,
    }
    _atomic_json(output_root / "r5b_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-stem", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--library-path", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--packet-samples", type=int, default=320)
    parser.add_argument("--channel-count", type=int, default=8)
    parser.add_argument("--model-chunk-samples", type=int, default=320)
    parser.add_argument("--wenet-chunk-size", type=int, default=DEFAULT_WENET_CHUNK_SIZE)
    parser.add_argument("--warmup-samples", type=int, default=16000)
    parser.add_argument("--max-scopes", type=int)
    parser.add_argument("--run-id", default="r5b-wenet-u2pp-m028-rawch0")
    parser.add_argument("--code-commit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = _run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
