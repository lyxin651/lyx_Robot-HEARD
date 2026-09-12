import json
from pathlib import Path

import pytest

from robot_heard.io.manifest import ManifestValidationError, read_manifest, validate_manifest


def test_valid_manifest(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"segment_id": "seg-1", "audio_path": str(audio)}) + "\n",
        encoding="utf-8",
    )
    items = read_manifest(manifest)
    assert items[0]["segment_id"] == "seg-1"


def test_duplicate_segment_id_rejected(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.touch()
    items = [
        {"segment_id": "dup", "audio_path": str(audio)},
        {"segment_id": "dup", "audio_path": str(audio)},
    ]
    with pytest.raises(ManifestValidationError, match="duplicate segment_id"):
        validate_manifest(items)
