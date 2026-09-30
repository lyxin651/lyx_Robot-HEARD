"""Derived metrics for COMPUTE_AWARE replay traces.

Metrics are read-only derivations from persisted raw trace evidence.  This
module never rewrites or appends to ``trace.jsonl``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from robot_heard.replay.events import ContractValidationError, EventType, TraceEvent


class MetricsError(ContractValidationError):
    """Raised when a trace cannot support a requested metric."""


def _payload(event: TraceEvent) -> Mapping[str, Any]:
    return event.payload


def _nearest_rank(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    if not 0 < percentile <= 1:
        raise MetricsError("percentile must be in (0, 1]")
    rank = max(0, math.ceil(percentile * len(values)) - 1)
    return sorted(values)[rank]


def _summary(values: Sequence[float]) -> Mapping[str, Optional[float]]:
    if not values:
        return {
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
        }
    return {
        "max": max(values),
        "mean": sum(values) / len(values),
        "p50": _nearest_rank(values, 0.50),
        "p95": _nearest_rank(values, 0.95),
    }


def _pair_durations(
    events: Sequence[TraceEvent],
    start_type: EventType,
    return_type: EventType,
) -> list[float]:
    starts: list[float] = []
    durations: list[float] = []
    for event in events:
        if event.wall_offset_sec is None:
            raise MetricsError("COMPUTE_AWARE metrics require wall_offset_sec on every event")
        if event.event_type is start_type:
            if starts:
                raise MetricsError(f"overlapping {start_type.value} intervals")
            starts.append(event.wall_offset_sec)
        elif event.event_type is return_type:
            if not starts:
                raise MetricsError(f"{return_type.value} has no matching start")
            duration = event.wall_offset_sec - starts.pop()
            if duration < -1e-12:
                raise MetricsError(f"negative {start_type.value} duration")
            durations.append(max(0.0, duration))
    if starts:
        raise MetricsError(f"unterminated {start_type.value} interval")
    return durations


def _validate_wall_order(events: Sequence[TraceEvent]) -> None:
    previous = -float("inf")
    for event in events:
        value = event.wall_offset_sec
        if value is None or value < 0 or value + 1e-12 < previous:
            raise MetricsError("trace wall offsets must be non-null, non-negative, and monotonic")
        previous = value


def _state_value(event: TraceEvent, name: str) -> Optional[int]:
    value = _payload(event).get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MetricsError(f"trace payload {name} must be a non-negative integer or null")
    return value


def _time_weighted_mean_backlog(
    events: Sequence[TraceEvent],
    measured_interval: float,
) -> float:
    if measured_interval <= 0:
        raise MetricsError("measured wall interval must be positive")
    previous_time = 0.0
    backlog = 0
    area = 0.0
    for event in events:
        current_time = event.wall_offset_sec
        if current_time is None:
            raise MetricsError("wall offset is required for backlog integration")
        if current_time > measured_interval + 1e-12:
            raise MetricsError("event occurs outside measured wall interval")
        area += backlog * max(0.0, current_time - previous_time)
        if "queue_backlog_samples" in _payload(event):
            value = _state_value(event, "queue_backlog_samples")
            if value is None:
                raise MetricsError("queue_backlog_samples cannot be null")
            backlog = value
        previous_time = max(previous_time, current_time)
    area += backlog * max(0.0, measured_interval - previous_time)
    return area / measured_interval


def compute_paced_metrics(
    trace_events: Sequence[TraceEvent],
    *,
    sample_rate: int,
    source_total_samples: int,
    run_id: Optional[str] = None,
    deadline_budget_sec: Optional[float] = None,
    measured_wall_interval_sec: Optional[float] = None,
    run_provenance: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Compute R3 metrics from raw trace events without mutating them."""

    events = tuple(trace_events)
    provenance = {} if run_provenance is None else dict(run_provenance)
    if not events:
        raise MetricsError("cannot compute metrics from an empty trace")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise MetricsError("sample_rate must be a positive integer")
    if (
        isinstance(source_total_samples, bool)
        or not isinstance(source_total_samples, int)
        or source_total_samples <= 0
    ):
        raise MetricsError("source_total_samples must be a positive integer")
    if deadline_budget_sec is not None and (
        isinstance(deadline_budget_sec, bool)
        or not isinstance(deadline_budget_sec, (int, float))
        or not math.isfinite(deadline_budget_sec)
        or deadline_budget_sec < 0
    ):
        raise MetricsError("deadline_budget_sec must be null or a non-negative finite number")
    _validate_wall_order(events)

    source_duration = source_total_samples / sample_rate
    last_wall = events[-1].wall_offset_sec
    assert last_wall is not None  # validated above
    measured_interval = last_wall if measured_wall_interval_sec is None else measured_wall_interval_sec
    if (
        isinstance(measured_interval, bool)
        or not isinstance(measured_interval, (int, float))
        or not math.isfinite(measured_interval)
        or measured_interval <= 0
    ):
        raise MetricsError("measured_wall_interval_sec must be a positive finite number")

    consume_durations = _pair_durations(
        events, EventType.CONSUMER_CALL_START, EventType.CONSUMER_CALL_RETURN
    )
    finish_durations = _pair_durations(events, EventType.FINISH_START, EventType.FINISH_RETURN)
    consume_time = sum(consume_durations)
    finish_time = sum(finish_durations)

    backlog_samples = [
        _state_value(event, "queue_backlog_samples")
        for event in events
        if event.event_type is EventType.SOURCE_AVAILABLE
    ]
    if any(value is None for value in backlog_samples):
        raise MetricsError("SOURCE_AVAILABLE events must expose queue backlog")
    backlog_values = [int(value) for value in backlog_samples]
    backlog_summary = _summary([float(value) for value in backlog_values])
    final_backlog = _state_value(events[-1], "queue_backlog_samples")
    if final_backlog is None:
        # FINISH_RETURN is expected to carry state; this also fails closed for
        # arbitrary incomplete traces.
        raise MetricsError("final trace event must expose queue backlog")
    time_weighted_mean_backlog = _time_weighted_mean_backlog(events, measured_interval)

    delivery_lags: list[float] = []
    delivery_records: list[Mapping[str, Any]] = []
    for event in events:
        if event.event_type is not EventType.PACKET_DELIVERED:
            continue
        scheduled = _payload(event).get("scheduled_availability_offset_sec")
        if not isinstance(scheduled, (int, float)) or isinstance(scheduled, bool):
            raise MetricsError("PACKET_DELIVERED requires scheduled availability offset")
        assert event.wall_offset_sec is not None
        lag = event.wall_offset_sec - float(scheduled)
        if lag < -1e-12:
            raise MetricsError("delivery lag cannot be negative")
        lag = max(0.0, lag)
        delivery_lags.append(lag)
        delivery_records.append(
            {
                "sequence_id": event.sequence_id,
                "delivery_lag_sec": lag,
            }
        )
    delivery_summary = _summary(delivery_lags)
    if deadline_budget_sec is None:
        deadline = None
    else:
        misses = sum(lag > deadline_budget_sec for lag in delivery_lags)
        deadline = {
            "deadline_budget_sec": deadline_budget_sec,
            "miss_count": misses,
            "miss_ratio": (misses / len(delivery_lags)) if delivery_lags else None,
        }

    available_values = [
        _state_value(event, "available_through_sample")
        for event in events
        if "available_through_sample" in _payload(event)
    ]
    consumed_values = [
        _state_value(event, "consumed_through_sample")
        for event in events
        if "consumed_through_sample" in _payload(event)
    ]
    processing_lag = None
    if any(value is not None for value in consumed_values):
        if not available_values or any(value is None for value in available_values):
            raise MetricsError("consumed position requires available position evidence")
        processing_values = []
        for event in events:
            available = _state_value(event, "available_through_sample")
            consumed = _state_value(event, "consumed_through_sample")
            if available is not None and consumed is not None:
                if consumed > available:
                    raise MetricsError("consumed position cannot exceed available position")
                processing_values.append(available - consumed)
        processing_lag = {
            "available": True,
            "max_samples": max(processing_values) if processing_values else None,
            "final_samples": processing_values[-1] if processing_values else None,
        }
    else:
        processing_lag = {"available": False, "max_samples": None, "final_samples": None}

    source_end_events = [event for event in events if event.event_type is EventType.SOURCE_END]
    if len(source_end_events) != 1:
        raise MetricsError("trace must contain exactly one SOURCE_END event")
    source_end = source_end_events[0].wall_offset_sec
    assert source_end is not None
    final_events = [
        event
        for event in events
        if event.event_type is EventType.HYPOTHESIS
        and _payload(event).get("kind") == "FINAL"
    ]
    completion_event = final_events[-1] if final_events else next(
        (event for event in reversed(events) if event.event_type is EventType.FINISH_RETURN),
        None,
    )
    if completion_event is None or completion_event.wall_offset_sec is None:
        raise MetricsError("trace has no completion observation")

    source_end_index = source_end_events[0].event_index
    completion_event_id = None
    if completion_event.event_type is EventType.HYPOTHESIS:
        completion_event_id = _payload(completion_event).get("event_id")
    completion_evidence = {
        "source_end_event_index": source_end_index,
        "source_end_wall_offset_sec": source_end,
        "completion_event_type": completion_event.event_type.value,
        "completion_event_index": completion_event.event_index,
        "completion_event_id": completion_event_id,
        "completion_wall_offset_sec": completion_event.wall_offset_sec,
    }
    streaming_interval = source_end
    tail_interval = measured_interval - source_end
    if tail_interval < -1e-12:
        raise MetricsError("finish return precedes source end")
    metric_evidence = {
        "code_commit": provenance.get("code_commit"),
        "trace_durability_policy": provenance.get("trace_durability_policy"),
        "clock_origin": provenance.get("clock_origin"),
        "metric_window": provenance.get("metric_window"),
        "source_end_event_index": source_end_index,
        "completion_event_index": completion_event.event_index,
    }

    queue_backlog_metric = {
        "max": backlog_summary["max"],
        "time_weighted_mean": time_weighted_mean_backlog,
        "p50": backlog_summary["p50"],
        "p95": backlog_summary["p95"],
        "final": final_backlog,
        "sampling_policy": "SOURCE_AVAILABLE transitions before delivery",
        "sample_count": len(backlog_values),
        "sample_unit": "samples",
        "sampling": "SOURCE_AVAILABLE transitions before delivery",
        "unit": "samples",
        "seconds_per_sample": 1 / sample_rate,
        "integration": {
            "integration_start_wall_offset_sec": 0.0,
            "integration_end_wall_offset_sec": measured_interval,
            "boundary_convention": (
                "piecewise-constant; post-event state applies until next event"
            ),
        },
    }
    queue_backlog_sec = {
        "max": None if queue_backlog_metric["max"] is None else queue_backlog_metric["max"] / sample_rate,
        "time_weighted_mean": queue_backlog_metric["time_weighted_mean"] / sample_rate,
        "p50": None if queue_backlog_metric["p50"] is None else queue_backlog_metric["p50"] / sample_rate,
        "p95": None if queue_backlog_metric["p95"] is None else queue_backlog_metric["p95"] / sample_rate,
        "final": final_backlog / sample_rate,
        "unit": "seconds",
    }

    return {
        "schema_version": "r3.metrics.v1",
        "run_id": run_id if run_id is not None else provenance.get("run_id"),
        "metric_evidence": metric_evidence,
        "source_duration_sec": source_duration,
        "measured_wall_interval_sec": measured_interval,
        "streaming_wall_interval_sec": streaming_interval,
        "tail_interval_sec": tail_interval,
        "metric_definitions": {
            "D_audio": "source_total_samples / sample_rate",
            "C_consume": "sum(CONSUMER_CALL_RETURN - CONSUMER_CALL_START)",
            "C_finish": "FINISH_RETURN - FINISH_START",
            "queue_backlog": "available_through_sample - delivered_through_sample",
            "delivery_lag": "delivery_wall_offset - scheduled_availability_offset",
            "percentile_method": "nearest_rank",
        },
        "percentile_method": "nearest_rank",
        "D_audio": source_duration,
        "C_consume": consume_time,
        "C_finish": finish_time,
        "RTF_consume": consume_time / source_duration,
        "RTF_total_compute": (consume_time + finish_time) / source_duration,
        "compute_duty": (consume_time + finish_time) / measured_interval,
        "compute_duty_window": {
            "start_wall_offset_sec": 0.0,
            "end_wall_offset_sec": measured_interval,
            "includes_finish_tail": True,
            "definition": "(C_consume + C_finish) / measured_run_interval",
        },
        "max_queue_backlog": queue_backlog_metric["max"],
        "mean_queue_backlog": queue_backlog_metric["time_weighted_mean"],
        "P50_queue_backlog": queue_backlog_metric["p50"],
        "P95_queue_backlog": queue_backlog_metric["p95"],
        "final_queue_backlog": queue_backlog_metric["final"],
        "queue_backlog_samples": queue_backlog_metric,
        "queue_backlog_sec": queue_backlog_sec,
        "delivery_lag": {
            "per_packet": delivery_records,
            **delivery_summary,
        },
        "deadline": deadline,
        "processing_lag_samples": processing_lag,
        "completion_overrun_sec": completion_event.wall_offset_sec - source_end,
        "completion_evidence": completion_evidence,
    }


def write_metrics_json(path: Union[str, Path], metrics: Mapping[str, Any]) -> None:
    """Atomically write derived metrics as a separate JSON artifact."""

    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(metrics), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def compute_r4_metrics(
    segment_evidence: Sequence[Mapping[str, Any]],
    *,
    global_metrics: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Derive R4 segment metrics without changing trace or segment evidence."""

    rows = [dict(row) for row in segment_evidence]
    if any(not row.get("segment_id") for row in rows):
        raise MetricsError("each R4 segment evidence row requires segment_id")
    if len({row["segment_id"] for row in rows}) != len(rows):
        raise MetricsError("R4 segment evidence contains duplicate segment_id")

    def durations(field: str) -> list[float]:
        values = []
        for row in rows:
            value = row.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise MetricsError(f"R4 evidence field {field} must be numeric")
            if not math.isfinite(float(value)):
                raise MetricsError(f"R4 evidence field {field} must be finite")
            values.append(float(value))
        return values

    segment_delays = durations("segment_end_to_final_wall_delay_sec") if rows else []
    oracle_delays = []
    asr_compute = []
    for row in rows:
        end_sample = row.get("end_sample")
        sample_rate = row.get("sample_rate")
        release = row.get("oracle_release_wall_offset_sec")
        invocation_start = row.get("asr_invocation_start_wall_offset_sec")
        invocation_return = row.get("asr_invocation_return_wall_offset_sec")
        if not isinstance(end_sample, int) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise MetricsError("R4 evidence requires integer end_sample/sample_rate")
        if not isinstance(release, (int, float)) or release < 0:
            raise MetricsError("R4 evidence requires oracle release wall offset")
        oracle_delay = float(release) - (end_sample / sample_rate)
        if oracle_delay < -1e-12:
            raise MetricsError("oracle release delay cannot be negative")
        oracle_delays.append(max(0.0, oracle_delay))
        if not isinstance(invocation_start, (int, float)) or not isinstance(invocation_return, (int, float)):
            raise MetricsError("R4 evidence requires ASR invocation boundaries")
        duration = float(invocation_return) - float(invocation_start)
        if duration < -1e-12:
            raise MetricsError("ASR wall compute cannot be negative")
        asr_compute.append(max(0.0, duration))

    result = {
        "schema_version": "r4.metrics.v1",
        "segment_count": len(rows),
        "final_count": sum(row.get("final_event_index") is not None for row in rows),
        "empty_count": sum(bool(row.get("text_empty")) for row in rows),
        "non_empty_count": sum(not bool(row.get("text_empty")) for row in rows),
        "segment_end_to_final_wall_delay": _summary(segment_delays),
        "oracle_release_delay": _summary(oracle_delays),
        "asr_wall_compute": _summary(asr_compute),
        "metric_definitions": {
            "segment_end_to_final_wall_delay_sec": "final_observed_wall_offset_sec - end_sample / sample_rate",
            "oracle_release_delay_sec": "oracle_release_wall_offset_sec - end_sample / sample_rate",
            "asr_wall_compute_sec": "asr_invocation_return_wall_offset_sec - asr_invocation_start_wall_offset_sec",
            "percentile_method": "nearest_rank",
        },
        "percentile_method": "nearest_rank",
    }
    if global_metrics is not None:
        result["global_r3_metrics"] = {
            key: global_metrics.get(key)
            for key in (
                "RTF_consume",
                "RTF_total_compute",
                "compute_duty",
                "max_queue_backlog",
                "P95_queue_backlog",
                "delivery_lag",
                "completion_overrun_sec",
            )
        }
    return result


def _levenshtein_characters(left: str, right: str) -> int:
    """Return character-level Levenshtein distance with bounded row memory."""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def compute_r5a_metrics(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    sample_rate: int,
    global_metrics: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Derive R5A rolling-prefix metrics from snapshot evidence only.

    ``scope_TTFT`` is intentionally measured from the Stage-2 oracle start
    boundary ``S / Fs``.  It is not a fully-online utterance-onset metric.
    References are not accepted by this function and therefore cannot enter
    the decode-side evidence path.
    """

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise MetricsError("sample_rate must be a positive integer")
    rows = [dict(row) for row in snapshots]
    by_scope: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        scope_id = row.get("scope_id", row.get("segment_id"))
        if not isinstance(scope_id, str) or not scope_id:
            raise MetricsError("each R5A snapshot requires scope_id")
        kind = row.get("kind")
        if kind not in ("PARTIAL", "FINAL"):
            raise MetricsError("R5A snapshot kind must be PARTIAL or FINAL")
        revision = row.get("revision_index")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise MetricsError("R5A revision_index must be a non-negative integer")
        observed = row.get("observed_wall_offset_sec")
        start_sample = row.get("start_sample")
        if (
            isinstance(start_sample, bool)
            or not isinstance(start_sample, int)
            or start_sample < 0
            or not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not math.isfinite(float(observed))
        ):
            raise MetricsError("R5A snapshots require valid start_sample and observation time")
        by_scope.setdefault(scope_id, []).append(row)

    for scope_id, scope_rows in by_scope.items():
        scope_rows.sort(key=lambda row: row["revision_index"])
        expected = list(range(len(scope_rows)))
        if [row["revision_index"] for row in scope_rows] != expected:
            raise MetricsError(f"scope {scope_id!r} has non-contiguous revisions")
        for index, row in enumerate(scope_rows):
            if index == 0:
                if row.get("replaces_event_id") is not None:
                    raise MetricsError("first R5A snapshot must not replace an event")
            elif not row.get("replaces_event_id"):
                raise MetricsError("replacement snapshot must identify previous event")

    first_observations = []
    ttlts = []
    segment_end_delays = []
    oracle_release_delays = []
    asr_compute = []
    partial_counts: dict[str, int] = {}
    revision_counts: dict[str, int] = {}
    flicker_distances: list[float] = []
    final_rows = []
    for scope_id, scope_rows in by_scope.items():
        first = scope_rows[0]
        start_sample = first["start_sample"]
        start_time = start_sample / sample_rate
        first_observations.append(float(first["observed_wall_offset_sec"]) - start_time)
        partial_counts[scope_id] = sum(row["kind"] == "PARTIAL" for row in scope_rows)
        revision_counts[scope_id] = max(0, len(scope_rows) - 1)
        for previous, current in zip(scope_rows, scope_rows[1:]):
            left = previous.get("text_raw")
            right = current.get("text_raw")
            if not isinstance(left, str) or not isinstance(right, str):
                raise MetricsError("R5A snapshots require text_raw strings for flicker")
            flicker_distances.append(float(_levenshtein_characters(left, right)))
        finals = [row for row in scope_rows if row["kind"] == "FINAL"]
        if len(finals) != 1:
            raise MetricsError(f"scope {scope_id!r} must have exactly one FINAL")
        final = finals[0]
        final_rows.append(final)
        final_observed = float(final["observed_wall_offset_sec"])
        ttlts.append(final_observed - start_time)
        end_sample = final.get("end_sample")
        if isinstance(end_sample, int) and end_sample > start_sample:
            segment_end_delays.append(final_observed - (end_sample / sample_rate))
        start_release = first.get("scope_start_release_wall_offset_sec")
        if isinstance(start_release, (int, float)):
            oracle_release_delays.append(float(start_release) - start_time)
        invocation_start = final.get("asr_invocation_start_wall_offset_sec")
        invocation_return = final.get("asr_invocation_return_wall_offset_sec")
        if isinstance(invocation_start, (int, float)) and isinstance(invocation_return, (int, float)):
            value = float(invocation_return) - float(invocation_start)
            if value < -1e-12:
                raise MetricsError("R5A ASR wall compute cannot be negative")
            asr_compute.append(max(0.0, value))

    result: dict[str, Any] = {
        "schema_version": "r5a.metrics.v1",
        "segment_count": len(by_scope),
        "final_count": len(final_rows),
        "partial_count_per_scope": partial_counts,
        "revision_count_per_scope": revision_counts,
        "partial_count": sum(partial_counts.values()),
        "revision_count": sum(revision_counts.values()),
        "scope_TTFT": _summary(first_observations),
        "scope_TTFT_authority": "Stage-2 oracle start boundary S/Fs",
        "scope_TTFT_online_interpretation": "NOT fully-online onset; oracle-segmented boundary reference",
        "scope_TTFT_stable": None,
        "TTLT": _summary(ttlts),
        "segment_end_to_final": _summary(segment_end_delays),
        "oracle_release_delay": _summary(oracle_release_delays),
        "asr_wall_compute": _summary(asr_compute),
        "flicker": {
            "algorithm": "character-level Levenshtein distance",
            "unit": "characters",
            "between": "consecutive complete hypothesis snapshots within each scope",
            "count": len(flicker_distances),
            **_summary(flicker_distances),
        },
        "metric_definitions": {
            "scope_TTFT": "first observed PARTIAL or FINAL wall offset - Stage-2 start_sample / sample_rate",
            "TTLT": "FINAL observed wall offset - Stage-2 start_sample / sample_rate",
            "segment_end_to_final": "FINAL observed wall offset - Stage-2 end_sample / sample_rate",
            "TTFT_stable": "UNAVAILABLE because supports_stable_prefix=false",
            "percentile_method": "nearest_rank",
        },
        "percentile_method": "nearest_rank",
    }
    if global_metrics is not None:
        result["global_r3_metrics"] = {
            key: global_metrics.get(key)
            for key in (
                "RTF_consume",
                "RTF_total_compute",
                "compute_duty",
                "max_queue_backlog",
                "P95_queue_backlog",
                "delivery_lag",
                "completion_overrun_sec",
            )
        }
    return result


def compute_r5b_metrics(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    sample_rate: int,
    native_audit: Optional[Mapping[str, Any]] = None,
    global_metrics: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Derive WeNet R5B metrics from adapter evidence and the frozen R3 trace.

    The lifecycle/flicker summaries intentionally share the validated R5A
    derivation, while the schema and provenance identify this as native WeNet
    evidence.  References are not accepted here; scoring remains a separate
    post-run evaluator step.
    """

    result = dict(
        compute_r5a_metrics(
            snapshots,
            sample_rate=sample_rate,
            global_metrics=global_metrics,
        )
    )
    result["schema_version"] = "r5b.metrics.v1"
    result["native_streaming"] = True
    result["stateful"] = True
    result["supports_stable_prefix"] = False
    result["reports_consumed_position"] = False
    result["reports_consumed_position_reason"] = (
        "The official C API synchronously accepts PCM16 bytes, but its feature "
        "pipeline may retain raw samples until a complete feature frame; no "
        "strict incorporated-sample position is exposed, so consumed is null."
    )
    result["endpoint_capability"] = False
    result["endpoint_capability_reason"] = (
        "The qualified API exposes partial_result/final_result only for this "
        "adapter condition; endpoint detection is not enabled or inferred."
    )
    result["emitted_partial_policy"] = {
        "feed_policy": "feed every complete fixed model chunk incrementally",
        "visible_event_policy": (
            "suppress initial empty partials and repeated visible text; emit "
            "first non-empty or changed visible text"
        ),
        "final_policy": "emit FINAL exactly once at released scope end",
        "raw_decoder_evidence": "native call/partial/final counters retained separately",
    }
    result["native_audit"] = None if native_audit is None else dict(native_audit)
    return result
