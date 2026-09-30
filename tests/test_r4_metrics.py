import pytest

from robot_heard.replay.metrics import compute_r4_metrics


def test_r4_metrics_summarize_segment_evidence_without_reference_fields():
    rows = [
        {
            "segment_id": "a",
            "end_sample": 160,
            "sample_rate": 16000,
            "oracle_release_wall_offset_sec": 0.011,
            "asr_invocation_start_wall_offset_sec": 0.012,
            "asr_invocation_return_wall_offset_sec": 0.022,
            "segment_end_to_final_wall_delay_sec": 0.012,
            "final_event_index": 7,
            "text_empty": False,
        },
        {
            "segment_id": "b",
            "end_sample": 320,
            "sample_rate": 16000,
            "oracle_release_wall_offset_sec": 0.021,
            "asr_invocation_start_wall_offset_sec": 0.023,
            "asr_invocation_return_wall_offset_sec": 0.043,
            "segment_end_to_final_wall_delay_sec": 0.023,
            "final_event_index": 12,
            "text_empty": True,
        },
    ]
    result = compute_r4_metrics(
        rows,
        global_metrics={"RTF_consume": 1.2, "P95_queue_backlog": 320},
    )
    assert result["segment_count"] == 2
    assert result["final_count"] == 2
    assert result["empty_count"] == 1
    assert result["non_empty_count"] == 1
    assert result["segment_end_to_final_wall_delay"]["p50"] == pytest.approx(0.012)
    assert result["oracle_release_delay"]["max"] == pytest.approx(0.001)
    assert result["asr_wall_compute"]["p95"] == pytest.approx(0.02)
    assert result["global_r3_metrics"]["RTF_consume"] == 1.2
    assert "reference" not in result

