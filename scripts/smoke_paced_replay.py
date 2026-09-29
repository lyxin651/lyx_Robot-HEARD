#!/usr/bin/env python3
"""Short synthetic smoke for the R3 monotonic paced runtime."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

from robot_heard.replay.clock import MonotonicWallClock
from robot_heard.replay.events import AudioChunk, ConsumerCapabilities, OutputEvent, StreamMetadata
from robot_heard.replay.metrics import compute_paced_metrics
from robot_heard.replay.paced_runtime import PacedReplayRuntime
from robot_heard.replay.source import SyntheticSource
from robot_heard.replay.trace import TraceWriter
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/gc6_challenge/work/realtime_replay/r3_paced_smoke.json"),
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
        "code_commit": "caller_supplied_or_smoke",
        "oracle_condition": "synthetic_none",
        "warmup_load_policy": {"model_load": "not_applicable", "warmup": "not_applicable"},
        "clock_policy": "monotonic_wall_clock",
        "clock_origin": {"kind": "runtime_established_after_consumer_start"},
        "metric_window": {"start_sample": 0, "end_sample": source.total_samples},
        "trace_path": str(trace_path.resolve()),
        "device_provenance": {"device": "none", "status": "not_applicable"},
    }
    writer = TraceWriter(trace_path, run_path, provenance)
    result = PacedReplayRuntime(
        source,
        consumer,
        writer,
        wall_clock=MonotonicWallClock(),
    ).run()
    metrics = compute_paced_metrics(
        result.trace_events,
        sample_rate=sample_rate,
        source_total_samples=source.total_samples,
        run_id="r3-paced-smoke",
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
    report = {
        "schema_version": "r3.paced.smoke.v1",
        "run_id": "r3-paced-smoke",
        "source_total_samples": source.total_samples,
        "sample_rate": sample_rate,
        "packet_samples": packet_samples,
        "expected_source_duration_sec": expected_duration,
        "measured_wall_interval_sec": result.measured_wall_interval_sec,
        "first_delivery_wall_offset_sec": first_delivery.wall_offset_sec,
        "source_end_wall_offset_sec": source_end.wall_offset_sec,
        "wall_duration_check": result.measured_wall_interval_sec >= expected_duration,
        "first_packet_waited_check": (first_delivery.wall_offset_sec or 0.0) > 0.0,
        "trace_path": str(trace_path),
        "run_provenance_path": str(run_path),
        "metrics": metrics,
    }
    _atomic_json(output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
