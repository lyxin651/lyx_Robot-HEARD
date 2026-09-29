import json
from pathlib import Path

import pytest

from robot_heard.asr.scoring import (
    NORMALIZATION_POLICY_V0,
    ScoringError,
    character_error_counts,
    compute_rtf,
    normalize_text,
    score_result_record,
    score_results_jsonl,
)


def _record(segment_id, *, text, reference, audio_sec, decode_sec):
    return {
        "result_schema_version": 1,
        "status": "success",
        "segment_id": segment_id,
        "audio_path": f"/{segment_id}.wav",
        "frontend": "near_field_f8n",
        "reference": reference,
        "text_raw": text,
        "text_norm": None,
        "reference_norm": None,
        "audio_sec": audio_sec,
        "decode_sec": decode_sec,
        "rtf": None,
        "backend": "openai_whisper",
        "model": "large-v3",
        "language": "zh",
    }


def test_normalize_text_v0_policy():
    assert normalize_text("ＡＢＣ， 你 好！\n") == "abc你好"


def test_character_error_counts_substitution():
    counts = character_error_counts("abc", "adc")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 0)
    assert counts.reference_chars == 3
    assert counts.cer == pytest.approx(1 / 3)


def test_character_error_counts_deletion():
    counts = character_error_counts("abc", "ac")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 1, 0)
    assert counts.reference_chars == 3


def test_character_error_counts_insertion():
    counts = character_error_counts("ac", "abc")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 0, 1)
    assert counts.reference_chars == 2


def test_character_error_counts_matches_misp_equal_cost_tie_breaking():
    counts = character_error_counts("ab", "ba")
    assert counts.errors == 2
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 1, 1)
    assert counts.reference_chars == 2
    assert counts.cer == pytest.approx(1.0)


def test_character_error_counts_empty_reference_has_no_rate():
    counts = character_error_counts("", "a")
    assert counts.insertions == 1
    assert counts.reference_chars == 0
    assert counts.cer is None


def test_compute_rtf_validates_inputs():
    assert compute_rtf(decode_sec=1.5, audio_sec=3.0) == pytest.approx(0.5)
    with pytest.raises(ScoringError, match="audio_sec"):
        compute_rtf(decode_sec=1.0, audio_sec=0.0)
    with pytest.raises(ScoringError, match="decode_sec"):
        compute_rtf(decode_sec=float("nan"), audio_sec=1.0)


def test_score_result_record_preserves_raw_and_scores_reference():
    original = _record(
        "seg-a",
        text="ＡＢＣ，你 好！",
        reference="abc你好",
        audio_sec=2.0,
        decode_sec=1.0,
    )
    scored = score_result_record(original)

    assert original["text_norm"] is None
    assert scored["text_raw"] == "ＡＢＣ，你 好！"
    assert scored["text_norm"] == "abc你好"
    assert scored["reference_norm"] == "abc你好"
    assert scored["normalization_policy"] == NORMALIZATION_POLICY_V0
    assert scored["result_schema_version"] == 2
    assert scored["rtf"] == pytest.approx(0.5)
    assert (
        scored["cer_s"],
        scored["cer_d"],
        scored["cer_i"],
        scored["cer_n"],
    ) == (0, 0, 0, 5)
    assert scored["cer"] == 0.0


def test_score_results_jsonl_computes_global_metrics(tmp_path):
    records = [
        _record(
            "seg-a",
            text="ＡＢＣ，你 好！",
            reference="abc你好",
            audio_sec=2.0,
            decode_sec=1.0,
        ),
        _record(
            "seg-b",
            text="adc",
            reference="abc",
            audio_sec=4.0,
            decode_sec=2.0,
        ),
        _record(
            "seg-c",
            text="无参考",
            reference=None,
            audio_sec=4.0,
            decode_sec=1.0,
        ),
    ]
    source = tmp_path / "raw.jsonl"
    output = tmp_path / "scored.jsonl"
    source.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records),
        encoding="utf-8",
    )

    summary = score_results_jsonl(source, output)
    rows = [
        json.loads(x)
        for x in output.read_text(encoding="utf-8").splitlines()
        if x
    ]

    assert len(rows) == 3
    assert rows[2]["reference_norm"] is None
    assert rows[2]["cer"] is None
    assert rows[2]["rtf"] == pytest.approx(0.25)
    assert summary.records == 3
    assert summary.reference_scored == 2
    assert summary.audio_sec_sum == pytest.approx(10.0)
    assert summary.decode_sec_sum == pytest.approx(4.0)
    assert summary.global_rtf == pytest.approx(0.4)
    assert (summary.cer_s, summary.cer_d, summary.cer_i, summary.cer_n) == (
        1,
        0,
        0,
        8,
    )
    assert summary.global_cer == pytest.approx(1 / 8)


def test_score_results_rejects_bad_input_without_partial_output(tmp_path):
    source = tmp_path / "raw.jsonl"
    output = tmp_path / "scored.jsonl"
    source.write_text(
        json.dumps(
            _record(
                "seg-a",
                text="ok",
                reference="ok",
                audio_sec=1.0,
                decode_sec=0.5,
            )
        )
        + "\n{not-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ScoringError, match="invalid JSON"):
        score_results_jsonl(source, output)

    assert not output.exists()
    assert not output.with_name(output.name + ".tmp").exists()
