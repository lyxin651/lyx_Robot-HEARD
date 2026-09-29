"""Exact MISP Stage-2 segment-to-sample timeline mapping for R2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Union


PathLike = Union[str, Path]
_SEGMENT_SUFFIX = re.compile(r"_(\d+)-(\d+)$")


class Stage2TimelineError(ValueError):
    """Raised when authoritative Stage-2 timeline data is inconsistent."""


@dataclass(frozen=True)
class Stage2SegmentSamples:
    """One Stage-2 segment represented on the integer sample timeline."""

    segment_id: str
    recording_id: str
    start_sample: int
    end_sample: int
    start_text: str
    end_text: str

    @property
    def frame_count(self) -> int:
        return self.end_sample - self.start_sample


def _require_regular_file(path: PathLike) -> Path:
    source = Path(path).expanduser().resolve(strict=False)
    if not source.is_file():
        raise FileNotFoundError(f"segments file does not exist: {source}")
    return source


def _parse_decimal(value: str, *, field: str, source: Path, line_number: int) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise Stage2TimelineError(
            f"{source}: line {line_number}: invalid {field}: {value!r}"
        ) from exc
    if not parsed.is_finite():
        raise Stage2TimelineError(
            f"{source}: line {line_number}: {field} must be finite"
        )
    return parsed


def _seconds_to_samples(
    seconds: Decimal,
    *,
    field: str,
    sample_rate: int,
    source: Path,
    line_number: int,
) -> int:
    sign, digits, exponent = seconds.as_tuple()
    numerator = int("".join(str(digit) for digit in digits) or "0")
    if sign:
        numerator = -numerator
    if exponent >= 0:
        numerator *= 10**exponent
        denominator = 1
    else:
        denominator = 10 ** (-exponent)
    numerator *= sample_rate
    integral, remainder = divmod(numerator, denominator)
    if remainder:
        raise Stage2TimelineError(
            f"{source}: line {line_number}: {field}={seconds} does not map to an "
            f"integral sample at {sample_rate} Hz"
        )
    return integral


def _validate_id_suffix(
    segment_id: str,
    start: Decimal,
    end: Decimal,
    *,
    source: Path,
    line_number: int,
) -> None:
    match = _SEGMENT_SUFFIX.search(segment_id)
    if match is None:
        raise Stage2TimelineError(
            f"{source}: line {line_number}: segment_id has no centisecond suffix: "
            f"{segment_id!r}"
        )
    suffix_start = Decimal(match.group(1)) / Decimal(100)
    suffix_end = Decimal(match.group(2)) / Decimal(100)
    if start != suffix_start or end != suffix_end:
        raise Stage2TimelineError(
            f"{source}: line {line_number}: segment_id suffix mismatch for "
            f"{segment_id!r}; fields={start}/{end}, suffix={suffix_start}/{suffix_end}"
        )


def read_stage2_segment_samples(
    path: PathLike,
    *,
    recording_id: str,
    sample_rate: int,
    source_total_samples: int,
) -> List[Stage2SegmentSamples]:
    """Read Stage-2 rows and map exact decimal seconds to sample intervals.

    The decimal fields are authoritative.  Segment-ID centisecond suffixes
    are checked only as a consistency constraint.  No float conversion or
    rounding is used.
    """

    source = _require_regular_file(path)
    if not recording_id:
        raise Stage2TimelineError("recording_id must be non-empty")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise Stage2TimelineError("sample_rate must be a positive integer")
    if (
        isinstance(source_total_samples, bool)
        or not isinstance(source_total_samples, int)
        or source_total_samples <= 0
    ):
        raise Stage2TimelineError("source_total_samples must be a positive integer")

    segments: List[Stage2SegmentSamples] = []
    seen = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise Stage2TimelineError(
                    f"{source}: line {line_number}: expected 4 fields "
                    "'segment_id recording_id start_sec end_sec'"
                )
            segment_id, row_recording_id, start_text, end_text = parts
            if segment_id in seen:
                raise Stage2TimelineError(
                    f"{source}: line {line_number}: duplicate segment_id {segment_id!r}"
                )
            if row_recording_id != recording_id:
                raise Stage2TimelineError(
                    f"{source}: line {line_number}: recording_id {row_recording_id!r} "
                    f"does not match expected {recording_id!r}"
                )
            start = _parse_decimal(
                start_text, field="start_sec", source=source, line_number=line_number
            )
            end = _parse_decimal(
                end_text, field="end_sec", source=source, line_number=line_number
            )
            if start < 0:
                raise Stage2TimelineError(
                    f"{source}: line {line_number}: start_sec must be >= 0"
                )
            if end <= start:
                raise Stage2TimelineError(
                    f"{source}: line {line_number}: end_sec must be > start_sec"
                )
            _validate_id_suffix(
                segment_id,
                start,
                end,
                source=source,
                line_number=line_number,
            )
            start_sample = _seconds_to_samples(
                start,
                field="start_sec",
                sample_rate=sample_rate,
                source=source,
                line_number=line_number,
            )
            end_sample = _seconds_to_samples(
                end,
                field="end_sec",
                sample_rate=sample_rate,
                source=source,
                line_number=line_number,
            )
            if not 0 <= start_sample < end_sample <= source_total_samples:
                raise Stage2TimelineError(
                    f"{source}: line {line_number}: sample bounds "
                    f"[{start_sample},{end_sample}) exceed source [0,{source_total_samples})"
                )
            segments.append(
                Stage2SegmentSamples(
                    segment_id=segment_id,
                    recording_id=row_recording_id,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_text=start_text,
                    end_text=end_text,
                )
            )
            seen.add(segment_id)

    if not segments:
        raise Stage2TimelineError(f"{source}: no segments found")
    return segments
