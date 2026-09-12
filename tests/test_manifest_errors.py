from pathlib import Path

import pytest

from robot_heard.io.manifest import ManifestValidationError, read_manifest, validate_manifest


def test_missing_required_field_rejected(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.touch()
    with pytest.raises(ManifestValidationError, match="segment_id"):
        validate_manifest([{"audio_path": str(audio)}])


def test_missing_audio_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    with pytest.raises(ManifestValidationError, match="audio file does not exist"):
        validate_manifest([{"segment_id": "seg-1", "audio_path": str(missing)}])


def test_invalid_duration_rejected(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.touch()
    with pytest.raises(ManifestValidationError, match="duration must be a positive number"):
        validate_manifest([{"segment_id": "seg-1", "audio_path": str(audio), "duration": 0}])


def test_malformed_json_reports_line_number(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="line=1"):
        read_manifest(manifest, require_audio_exists=False)
