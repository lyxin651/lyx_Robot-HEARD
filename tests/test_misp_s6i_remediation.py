import json

import pytest

from robot_heard.asr.scoring import score_result_record
from robot_heard.misp.s6i_remediation import (
    MISP2025_S6I_NUMERIC_SURFACE_POLICY,
    S6IRemediationError,
    build_s6i_remediated_records,
    normalize_misp2025_s6i_numeric_surface,
    remediate_s6i_record,
    write_s6i_remediated_jsonl,
)
from robot_heard.misp.task2_export import export_misp2025_task2_submission


def _primary(segment_id, text):
    return {
        "result_schema_version": 1,
        "status": "success",
        "segment_id": segment_id,
        "audio_path": f"/{segment_id}.flac",
        "frontend": "misp_baseline_gss",
        "reference": "仅用于测试评分",
        "text_raw": text,
        "audio_sec": 1.0,
        "decode_sec": 0.5,
        "backend": "openai_whisper",
        "model": "large-v3",
        "language": "zh",
    }


def _retry(segment_id, text):
    return {
        "segment_id": segment_id,
        "audio_path": f"/{segment_id}.flac",
        "frontend": "misp_baseline_gss",
        "retry_text_raw": text,
        "retry_decode": {
            "language": "zh",
            "task": "transcribe",
            "temperature": 0.0,
            "fp16": True,
            "no_speech_threshold": None,
        },
        "provenance": {
            "code_commit": "retry-head",
            "reference_used_for": "scoring_only",
        },
    }


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("17", "十七"),
        ("30岁", "三十岁"),
        ("40岁的", "四十岁的"),
        ("7和10以及99", "七和十以及九十九"),
        ("0 9 10 11 20 30 40 99", "零 九 十 十一 二十 三十 四十 九十九"),
    ],
)
def test_numeric_surface_normalizes_safe_ascii_integers(source, expected):
    result = normalize_misp2025_s6i_numeric_surface(source)
    assert result.text == expected
    assert result.policy == MISP2025_S6I_NUMERIC_SURFACE_POLICY
    assert result.changed is True


@pytest.mark.parametrize("source", ["001", "100", "-1", "1.5", "17a", "2024-01-01", "123-456"])
def test_numeric_surface_fails_closed_for_unsafe_numeric_forms(source):
    result = normalize_misp2025_s6i_numeric_surface(source)
    assert result.text == source
    assert result.replacements == ()


def test_numeric_surface_preserves_all_non_digit_characters():
    source = "前缀，17岁和40岁！后缀"
    result = normalize_misp2025_s6i_numeric_surface(source)
    assert result.text == "前缀，十七岁和四十岁！后缀"
    assert "，" in result.text and "！" in result.text


def test_numeric_surface_is_deterministic():
    source = "30岁和40岁的裁员问题"
    first = normalize_misp2025_s6i_numeric_surface(source)
    second = normalize_misp2025_s6i_numeric_surface(source)
    assert first == second


def test_nonempty_primary_never_triggers_retry():
    with pytest.raises(S6IRemediationError, match="not eligible"):
        remediate_s6i_record(
            _primary("seg-a", "你好"),
            retry=_retry("seg-a", "世界"),
        )


def test_empty_primary_binds_retry_and_preserves_provenance():
    primary = _primary("seg-a", "")
    retry = _retry("seg-a", "17")
    result = remediate_s6i_record(primary, retry=retry, code_commit="adapter-head")

    assert result["segment_id"] == "seg-a"
    assert result["text_raw"] == ""
    assert result["text_recognition"] == "17"
    assert result["text_submission"] == "十七"
    assert result["submission_valid"] is True
    assert result["s6i_remediation"]["recognition_source"] == "gss_no_speech_retry"
    assert result["s6i_remediation"]["retry_provenance"] == retry
    assert result["s6i_remediation"]["code_commit"] == "adapter-head"
    assert result["s6i_remediation"]["reference_used_for_decision"] is False


def test_empty_primary_without_retry_fails_closed():
    result = remediate_s6i_record(_primary("seg-a", ""))
    assert result["text_raw"] == ""
    assert result["text_recognition"] == ""
    assert result["text_submission"] is None
    assert result["submission_valid"] is False


def test_empty_retry_fails_closed():
    result = remediate_s6i_record(_primary("seg-a", ""), retry=_retry("seg-a", ""))
    assert result["text_recognition"] == ""
    assert result["text_submission"] is None
    assert result["submission_valid"] is False


def test_retry_unsupported_text_cannot_bypass_surface_validation():
    result = remediate_s6i_record(_primary("seg-a", ""), retry=_retry("seg-a", "abc"))
    assert result["text_recognition"] == "abc"
    assert result["text_submission"] is None
    assert result["submission_valid"] is False
    assert "non-Han" in result["submission_invalid_reason"]


def test_builder_rejects_retry_ids_outside_primary_and_duplicate_ids():
    with pytest.raises(S6IRemediationError, match="not in primary"):
        build_s6i_remediated_records([_primary("seg-a", "你好")], [_retry("seg-b", "世界")])

    with pytest.raises(S6IRemediationError, match="duplicate retry"):
        build_s6i_remediated_records(
            [_primary("seg-a", "")],
            [_retry("seg-a", "你好"), _retry("seg-a", "世界")],
        )


def test_scoring_explicit_field_preserves_primary_raw_text():
    record = _primary("seg-a", "17")
    record["text_recognition"] = "17"
    record["text_submission"] = "十七"
    scored = score_result_record(record, text_field="text_submission")
    assert scored["text_raw"] == "17"
    assert scored["text_norm"] == "十七"
    assert scored["scoring_text_field"] == "text_submission"


def test_export_explicit_submission_field(tmp_path):
    source = tmp_path / "remediated.jsonl"
    record = _primary("seg-a", "17")
    record["text_recognition"] = "17"
    record["text_submission"] = "十七"
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    expected = tmp_path / "expected.txt"
    expected.write_text("seg-a\n", encoding="utf-8")

    summary = export_misp2025_task2_submission(
        source,
        tmp_path / "submission",
        expected_segments_path=expected,
        text_field="text_submission",
    )
    assert summary.input_text_field == "text_submission"
    assert (tmp_path / "submission" / "summission.txt").read_text(encoding="utf-8") == "seg-a 十七\n"
    metadata = json.loads(
        (tmp_path / "submission" / "summission.export.json").read_text(encoding="utf-8")
    )
    assert metadata["input_text_field"] == "text_submission"


def test_write_remediated_jsonl_records_provenance_and_preserves_raw(tmp_path):
    primary = tmp_path / "primary.jsonl"
    retry = tmp_path / "retry.jsonl"
    output = tmp_path / "remediated.jsonl"
    metadata = tmp_path / "remediated.metadata.json"
    primary.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in [_primary("seg-a", ""), _primary("seg-b", "你好")]
        ),
        encoding="utf-8",
    )
    retry.write_text(
        json.dumps(_retry("seg-a", "17"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    payload = write_s6i_remediated_jsonl(
        primary,
        output,
        retry_path=retry,
        metadata_path=metadata,
        code_commit="adapter-head",
    )
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [row["segment_id"] for row in rows] == ["seg-a", "seg-b"]
    assert rows[0]["text_raw"] == ""
    assert rows[0]["text_recognition"] == "17"
    assert rows[0]["text_submission"] == "十七"
    assert payload["records"] == 2
    assert payload["retry_records"] == 1
    assert payload["code_commit"] == "adapter-head"
    assert json.loads(metadata.read_text(encoding="utf-8"))["output_sha256"] == payload["output_sha256"]
