#!/usr/bin/env python3
"""Validate one prepared MISP continuous source and Stage-2 timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from robot_heard.misp.kaldi_adapter import resolve_misp_channel_path
from robot_heard.misp.timeline import read_stage2_segment_samples
from robot_heard.replay.wav_source import MultiChannelPcmWavSource


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _validate_packets(source: MultiChannelPcmWavSource) -> Dict[str, Any]:
    packet_count = 0
    emitted_samples = 0
    first_range: Optional[List[int]] = None
    last_range: Optional[List[int]] = None
    tail_samples = 0
    final_chunk_count = 0
    for chunk in source:
        if chunk.start_sample != emitted_samples:
            raise ValueError(
                f"packet gap/overlap at packet {chunk.sequence_id}: "
                f"start={chunk.start_sample}, expected={emitted_samples}"
            )
        if chunk.num_channels != len(source.channel_ids):
            raise ValueError("packet channel count changed during traversal")
        if any(len(channel) != chunk.frame_count for channel in chunk.samples):
            raise ValueError(f"packet {chunk.sequence_id} has misaligned channels")
        current_range = [chunk.start_sample, chunk.end_sample]
        if first_range is None:
            first_range = current_range
        last_range = current_range
        if chunk.is_final_source_chunk:
            final_chunk_count += 1
            tail_samples = chunk.frame_count
        emitted_samples = chunk.end_sample
        packet_count += 1

    if emitted_samples != source.total_samples:
        raise ValueError(
            f"emitted samples {emitted_samples} != source length {source.total_samples}"
        )
    if final_chunk_count != 1:
        raise ValueError(f"expected one final source chunk, got {final_chunk_count}")
    return {
        "packet_count": packet_count,
        "first_range": first_range,
        "last_range": last_range,
        "emitted_samples": emitted_samples,
        "tail_samples": tail_samples,
        "final_chunk_count": final_chunk_count,
    }


def validate_source(args: argparse.Namespace) -> Dict[str, Any]:
    channel_ids = tuple(f"ch{index}" for index in range(args.expected_channel_count))
    channel_paths = tuple(
        resolve_misp_channel_path(args.channel_stem, index)
        for index in range(args.expected_channel_count)
    )
    source = MultiChannelPcmWavSource(
        channel_paths,
        channel_ids=channel_ids,
        stream_id=args.recording_id,
        packet_samples=args.packet_samples,
    )
    if source.metadata.num_channels != args.expected_channel_count:
        raise ValueError("unexpected channel count")
    if source.metadata.sample_rate != args.expected_sample_rate:
        raise ValueError(
            f"expected {args.expected_sample_rate} Hz, got {source.metadata.sample_rate}"
        )
    if source.sample_width != args.expected_sample_width:
        raise ValueError(
            f"expected sample width {args.expected_sample_width}, got {source.sample_width}"
        )
    if source.compression_type != "NONE":
        raise ValueError(f"expected uncompressed PCM, got {source.compression_type!r}")
    if source.total_samples != args.expected_frame_count:
        raise ValueError(
            f"expected {args.expected_frame_count} frames, got {source.total_samples}"
        )

    segments = read_stage2_segment_samples(
        args.segments,
        recording_id=args.recording_id,
        sample_rate=source.metadata.sample_rate,
        source_total_samples=source.total_samples,
    )
    if len(segments) != args.expected_segments:
        raise ValueError(
            f"expected {args.expected_segments} segments, got {len(segments)}"
        )
    min_start = min(segment.start_sample for segment in segments)
    max_end = max(segment.end_sample for segment in segments)
    if min_start != args.expected_min_start_sample:
        raise ValueError(
            f"expected minimum start {args.expected_min_start_sample}, got {min_start}"
        )
    if max_end != args.expected_max_end_sample:
        raise ValueError(
            f"expected maximum end {args.expected_max_end_sample}, got {max_end}"
        )

    packet_report = _validate_packets(source)
    report = {
        "schema_version": "r2.misp-replay-validation.v1",
        "code_commit": args.code_commit or _git_commit(),
        "recording_id": args.recording_id,
        "stream_id": source.metadata.stream_id,
        "source_identity": source.identity(),
        "channel_paths": [
            {
                "channel_id": channel_id,
                "path": str(path),
                "file_sha256": file_hash,
            }
            for channel_id, path, file_hash in zip(
                source.channel_ids, source.channel_paths, source.channel_hashes
            )
        ],
        "composite_source_sha256": source.content_sha256,
        "sample_rate": source.metadata.sample_rate,
        "sample_width": source.sample_width,
        "compression_type": source.compression_type,
        "frame_count_per_channel": source.total_samples,
        "duration_sec": source.total_samples / source.metadata.sample_rate,
        "packet_samples": source.packet_samples,
        "packet_validation": packet_report,
        "segments_path": str(Path(args.segments).expanduser().resolve(strict=True)),
        "segments_sha256": _sha256_file(Path(args.segments).expanduser().resolve(strict=True)),
        "segment_count": len(segments),
        "min_start_sample": min_start,
        "max_end_sample": max_end,
        "all_bounds_valid": True,
        "all_id_suffix_checks_pass": True,
        "all_recording_ids_match": all(
            segment.recording_id == args.recording_id for segment in segments
        ),
        "r3_wall_pacing": False,
    }
    output = Path(args.output).expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-stem", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--packet-samples", type=int, default=320)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit")
    parser.add_argument("--expected-channel-count", type=int, default=8)
    parser.add_argument("--expected-sample-rate", type=int, default=16000)
    parser.add_argument("--expected-sample-width", type=int, default=4)
    parser.add_argument("--expected-frame-count", type=int, default=12_772_784)
    parser.add_argument("--expected-segments", type=int, default=144)
    parser.add_argument("--expected-min-start-sample", type=int, default=110_080)
    parser.add_argument("--expected-max-end-sample", type=int, default=12_769_280)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = validate_source(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
