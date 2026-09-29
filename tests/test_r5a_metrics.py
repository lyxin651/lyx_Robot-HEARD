from __future__ import annotations

from pathlib import Path

from robot_heard.replay.metrics import compute_r5a_metrics

from test_rolling_whisper_adapter import _Row, _run


def test_r5a_metrics_are_derived_from_snapshots_and_mark_ttft_authority(tmp_path: Path):
    _, _, consumer, _, _ = _run(tmp_path, [_Row("scope", 0, 40000)], sample_count=40000)
    metrics = compute_r5a_metrics(consumer.snapshots, sample_rate=16000)
    assert metrics["segment_count"] == 1
    assert metrics["partial_count_per_scope"] == {"scope": 2}
    assert metrics["revision_count_per_scope"] == {"scope": 2}
    assert metrics["scope_TTFT_authority"] == "Stage-2 oracle start boundary S/Fs"
    assert metrics["scope_TTFT_online_interpretation"].startswith("NOT fully-online")
    assert metrics["scope_TTFT_stable"] is None
    assert metrics["flicker"]["algorithm"] == "character-level Levenshtein distance"
    assert metrics["flicker"]["unit"] == "characters"
    assert metrics["flicker"]["count"] == 2


def test_r5a_metrics_require_one_final_per_scope():
    try:
        compute_r5a_metrics(
            [
                {
                    "scope_id": "scope",
                    "start_sample": 0,
                    "revision_index": 0,
                    "kind": "PARTIAL",
                    "replaces_event_id": None,
                    "observed_wall_offset_sec": 1.0,
                    "text_raw": "a",
                }
            ],
            sample_rate=16000,
        )
    except Exception as exc:
        assert "exactly one FINAL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing FINAL must fail closed")
