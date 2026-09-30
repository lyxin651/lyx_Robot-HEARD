from __future__ import annotations

from pathlib import Path

import pytest

from robot_heard.misp.timeline import Stage2TimelineError, read_stage2_segment_samples


def _segments(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "segments"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_stage2_decimal_seconds_map_to_exact_integer_samples(tmp_path):
    path = _segments(
        tmp_path,
        "S197_M028_S197199201241_F8N_014152-015076 "
        "M028_S197199201241_F8N_Far 141.52 150.76",
    )

    segments = read_stage2_segment_samples(
        path,
        recording_id="M028_S197199201241_F8N_Far",
        sample_rate=16000,
        source_total_samples=12_772_784,
    )

    assert len(segments) == 1
    assert segments[0].start_sample == 2_264_320
    assert segments[0].end_sample == 2_412_160
    assert segments[0].frame_count == 147_840


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [
                "seg_000100-000200 rec 1.00 2.00",
                "seg_000100-000200 rec 1.00 2.00",
            ],
            "duplicate",
        ),
        (["malformed rec 1.00"], "4 fields"),
        (["seg_000000-000100 rec -1.00 1.00"], "start_sec must be >= 0"),
        (["seg_000100-000100 rec 1.00 1.00"], "end_sec must be > start_sec"),
        (["seg_000100-000200 other 1.00 2.00"], "does not match expected"),
        (["seg_000100-000200 rec 1.00 2.01"], "suffix mismatch"),
        (["seg_000100-000200 rec 1.00 2.00"], "exceed source"),
    ],
)
def test_stage2_timeline_rejects_invalid_rows(tmp_path, rows, message):
    path = _segments(tmp_path, *rows)
    sample_rate = 16000
    source_total_samples = 16000 * 3
    if message == "exceed source":
        source_total_samples = 16000

    with pytest.raises(Stage2TimelineError, match=message):
        read_stage2_segment_samples(
            path,
            recording_id="rec",
            sample_rate=sample_rate,
            source_total_samples=source_total_samples,
        )


def test_stage2_timeline_rejects_non_integral_sample_conversion(tmp_path):
    path = _segments(tmp_path, "seg_000001-000002 rec 0.01 0.02")

    with pytest.raises(Stage2TimelineError, match="integral sample"):
        read_stage2_segment_samples(
            path,
            recording_id="rec",
            sample_rate=16001,
            source_total_samples=1000,
        )


def test_stage2_timeline_uses_half_open_bounds_and_rejects_missing_suffix(tmp_path):
    path = _segments(tmp_path, "seg_without_suffix rec 0.00 0.01")

    with pytest.raises(Stage2TimelineError, match="centisecond suffix"):
        read_stage2_segment_samples(
            path,
            recording_id="rec",
            sample_rate=16000,
            source_total_samples=1000,
        )
