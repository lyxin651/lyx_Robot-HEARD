import json
import zipfile
from pathlib import Path

import pytest

from robot_heard.misp.task2_export import (
    MISP2025_TASK2_ARCHIVE_NAME,
    MISP2025_TASK2_TRANSCRIPT_NAME,
    MISPTask2ExportError,
    export_misp2025_task2_submission,
    normalize_misp2025_task2_text,
    read_expected_segment_ids,
)


def _result(segment_id, text_raw, *, status="success"):
    return {
        "result_schema_version": 1,
        "status": status,
        "segment_id": segment_id,
        "audio_path": f"/{segment_id}.wav",
        "frontend": "near_field_f8n",
        "reference": None,
        "text_raw": text_raw,
        "text_norm": None,
        "reference_norm": None,
        "audio_sec": 1.0,
        "decode_sec": 0.5,
        "rtf": None,
        "backend": "openai_whisper",
        "model": "large-v3",
        "language": "zh",
    }


def _write_jsonl(path: Path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_task2_text_removes_space_and_punctuation_only():
    assert normalize_misp2025_task2_text("你 好，世界！\n") == "你好世界"


def test_task2_text_rejects_semantic_non_han_content():
    with pytest.raises(MISPTask2ExportError, match="non-Han"):
        normalize_misp2025_task2_text("你好2025", segment_id="seg-a")


def test_task2_export_creates_exact_transcript_and_zip(tmp_path):
    source = tmp_path / "results.jsonl"
    _write_jsonl(
        source,
        [
            _result("seg-a", "你 好，世界！"),
            _result("seg-b", "测试。"),
        ],
    )
    output_dir = tmp_path / "submission"

    summary = export_misp2025_task2_submission(
        source,
        output_dir,
        code_commit="abc123",
    )

    transcript = output_dir / MISP2025_TASK2_TRANSCRIPT_NAME
    archive = output_dir / MISP2025_TASK2_ARCHIVE_NAME
    metadata = output_dir / "summission.export.json"

    assert summary.records == 2
    assert transcript.read_text(encoding="utf-8") == (
        "seg-a 你好世界\n"
        "seg-b 测试\n"
    )
    assert archive.is_file()
    assert metadata.is_file()

    with zipfile.ZipFile(archive, "r") as handle:
        assert handle.namelist() == [MISP2025_TASK2_TRANSCRIPT_NAME]
        assert handle.read(MISP2025_TASK2_TRANSCRIPT_NAME) == transcript.read_bytes()

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["records"] == 2
    assert payload["code_commit"] == "abc123"
    assert payload["expected_segments_validated"] is False


def test_task2_export_validates_authoritative_expected_ids(tmp_path):
    source = tmp_path / "results.jsonl"
    expected = tmp_path / "expected.txt"
    _write_jsonl(
        source,
        [_result("seg-a", "你好"), _result("seg-b", "测试")],
    )
    expected.write_text("seg-b reference\nseg-a\n", encoding="utf-8")

    summary = export_misp2025_task2_submission(
        source,
        tmp_path / "out",
        expected_segments_path=expected,
    )
    assert summary.expected_segments_validated is True
    assert summary.expected_segments_path == str(expected.resolve())


def test_task2_export_rejects_missing_or_extra_ids_before_writing(tmp_path):
    source = tmp_path / "results.jsonl"
    expected = tmp_path / "expected.txt"
    _write_jsonl(source, [_result("seg-a", "你好")])
    expected.write_text("seg-a\nseg-b\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    with pytest.raises(MISPTask2ExportError, match="missing"):
        export_misp2025_task2_submission(
            source,
            output_dir,
            expected_segments_path=expected,
        )

    assert not (output_dir / MISP2025_TASK2_TRANSCRIPT_NAME).exists()
    assert not (output_dir / MISP2025_TASK2_ARCHIVE_NAME).exists()
    assert not (output_dir / "summission.export.json").exists()


def test_expected_ids_accept_manifest_jsonl_and_reject_duplicates(tmp_path):
    expected = tmp_path / "manifest.jsonl"
    expected.write_text(
        '{"segment_id":"seg-a","audio_path":"/a.wav"}\n'
        '{"segment_id":"seg-b","audio_path":"/b.wav"}\n',
        encoding="utf-8",
    )
    assert read_expected_segment_ids(expected) == ["seg-a", "seg-b"]

    expected.write_text("seg-a\nseg-a transcript\n", encoding="utf-8")
    with pytest.raises(MISPTask2ExportError, match="duplicate expected"):
        read_expected_segment_ids(expected)


def test_task2_export_rejects_duplicate_result_ids(tmp_path):
    source = tmp_path / "results.jsonl"
    _write_jsonl(
        source,
        [_result("seg-a", "你好"), _result("seg-a", "测试")],
    )
    with pytest.raises(MISPTask2ExportError, match="duplicate result"):
        export_misp2025_task2_submission(source, tmp_path / "out")


def test_task2_export_rejects_non_success_and_empty_text(tmp_path):
    source = tmp_path / "results.jsonl"
    _write_jsonl(source, [_result("seg-a", "你好", status="failure")])
    with pytest.raises(MISPTask2ExportError, match="status='success'"):
        export_misp2025_task2_submission(source, tmp_path / "out-a")

    _write_jsonl(source, [_result("seg-a", "， ！")])
    with pytest.raises(MISPTask2ExportError, match="empty"):
        export_misp2025_task2_submission(source, tmp_path / "out-b")


def test_task2_export_refuses_overwrite(tmp_path):
    source = tmp_path / "results.jsonl"
    _write_jsonl(source, [_result("seg-a", "你好")])
    output_dir = tmp_path / "out"

    export_misp2025_task2_submission(source, output_dir)
    with pytest.raises(FileExistsError, match="already exist"):
        export_misp2025_task2_submission(source, output_dir)
