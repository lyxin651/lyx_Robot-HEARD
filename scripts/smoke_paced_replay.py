#!/usr/bin/env python3
"""Short synthetic smoke for the R3 monotonic paced runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from robot_heard.replay.clock import MonotonicWallClock
from robot_heard.replay.events import AudioChunk, ConsumerCapabilities, OutputEvent, StreamMetadata
from robot_heard.replay.metrics import compute_paced_metrics
from robot_heard.replay.paced_runtime import PacedReplayRuntime
from robot_heard.replay.source import SyntheticSource
from robot_heard.replay.trace import TraceDurabilityPolicy, TraceWriter
from robot_heard.streaming.base import StreamingConsumer


class _ZeroWorkConsumer(StreamingConsumer):
    def __init__(self) -> None:
        self.stream_id = None

    def start(self, metadata: StreamMetadata) -> None:
        self.stream_id = metadata.stream_id

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        return ()

    def finish(self) -> Sequence[OutputEvent]:
        return ()

    def reset(self) -> None:
        self.stream_id = None

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _resolve_code_commit(override: Optional[str]) -> str:
    if override is not None:
        if not override.strip():
            raise RuntimeError("--code-commit cannot be empty")
        return override.strip()
    repository = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("unable to resolve exact git code commit") from exc
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("git rev-parse HEAD returned an empty commit")
    return commit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/gc6_challenge/work/realtime_replay/r3_paced_smoke.json"),
    )
    parser.add_argument(
        "--code-commit",
        default=None,
        help="optional exact commit override; otherwise git rev-parse HEAD is required",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve(strict=False)
    trace_path = output.with_suffix(".trace.jsonl")
    run_path = output.with_suffix(".trace.run.json")
    if output.exists() or trace_path.exists() or run_path.exists():
        raise FileExistsError("smoke artifacts already exist; refusing overwrite")

    sample_rate = 1000
    packet_samples = 20
    source = SyntheticSource(
        (tuple(range(200)),),
        sample_rate=sample_rate,
        packet_samples=packet_samples,
        stream_id="r3-paced-smoke",
        channel_ids=("ch0",),
    )
    consumer = _ZeroWorkConsumer()
    code_commit = _resolve_code_commit(args.code_commit)
    durability_policy = TraceDurabilityPolicy.BUFFERED_CLOSE_FSYNC
    provenance = {
        "schema_version": "r3.event.v1",
        "run_id": "r3-paced-smoke",
        "replay_mode": "COMPUTE_AWARE",
        "source_identity": source.identity(),
        "source_sha256": source.content_sha256,
        "source_sample_rate": sample_rate,
        "channel_ids": ["ch0"],
        "packet_samples": packet_samples,
        "queue_policy": "lossless_unbounded_fifo",
        "timeline_authority": "integer_sample_index",
        "consumer_identity": type(consumer).__name__,
        "consumer_config": {"compute": "zero_work"},
        "consumer_capabilities": dict(consumer.capabilities().to_dict()),
        "code_commit": code_commit,
        "oracle_condition": "synthetic_none",
        "warmup_load_policy": {"model_load": "not_applicable", "warmup": "not_applicable"},
        "clock_policy": "monotonic_wall_clock",
        "clock_origin": {"kind": "pending_runtime_origin"},
        "metric_window": {
            "start_sample": 0,
            "end_sample": source.total_samples,
            "status": "pending_runtime_window",
        },
        "trace_path": str(trace_path.resolve()),
        "device_provenance": {"device": "none", "status": "not_applicable"},
        "trace_durability_policy": durability_policy.value,
    }
    writer = TraceWriter(
        trace_path,
        run_path,
        provenance,
        durability_policy=durability_policy,
    )
    result = PacedReplayRuntime(
        source,
        consumer,
        writer,
        wall_clock=MonotonicWallClock(),
    ).run()
    final_provenance = json.loads(run_path.read_text(encoding="utf-8"))
    metrics = compute_paced_metrics(
        result.trace_events,
        sample_rate=sample_rate,
        source_total_samples=source.total_samples,
        run_id="r3-paced-smoke",
        measured_wall_interval_sec=result.measured_wall_interval_sec,
        run_provenance=final_provenance,
    )
    source_end = next(
        event for event in result.trace_events if event.event_type.value == "SOURCE_END"
    )
    first_delivery = next(
        event
        for event in result.trace_events
        if event.event_type.value == "PACKET_DELIVERED"
    )
    expected_duration = source.total_samples / sample_rate
    wall_offsets = [event.wall_offset_sec for event in result.trace_events]
    if any(offset is None for offset in wall_offsets):
        raise RuntimeError("compute-aware smoke produced a null wall offset")
    if wall_offsets != sorted(wall_offsets):
        raise RuntimeError("compute-aware smoke produced non-monotonic wall offsets")
    runtime_overrun = result.measured_wall_interval_sec - expected_duration
    report = {
        "schema_version": "r3.paced.smoke.v1",
        "run_id": "r3-paced-smoke",
        "source_total_samples": source.total_samples,
        "sample_rate": sample_rate,
        "packet_samples": packet_samples,
        "source_duration_sec": expected_duration,
        "expected_source_duration_sec": expected_duration,
        "measured_wall_interval_sec": result.measured_wall_interval_sec,
        "runtime_overrun_sec": runtime_overrun,
        "runtime_overrun_ratio": result.measured_wall_interval_sec / expected_duration,
        "first_delivery_wall_offset_sec": first_delivery.wall_offset_sec,
        "source_end_wall_offset_sec": source_end.wall_offset_sec,
        "wall_duration_check": result.measured_wall_interval_sec >= expected_duration,
        "first_packet_waited_check": (first_delivery.wall_offset_sec or 0.0) > 0.0,
        "source_end_schedule_check": (
            abs((source_end.wall_offset_sec or 0.0) - expected_duration) <= 1e-9
        ),
        "all_event_wall_offsets_monotonic": True,
        "trace_durability_policy": durability_policy.value,
        "code_commit": code_commit,
        "trace_path": str(trace_path),
        "run_provenance_path": str(run_path),
        "metrics": metrics,
    }
    _atomic_json(output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
