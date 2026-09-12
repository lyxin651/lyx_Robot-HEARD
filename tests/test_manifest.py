import json
from pathlib import Path

import pytest

from robot_heard.io.manifest import (
    ManifestValidationError,
    read_manifest,
    resolve_audio_path,
    validate_manifest,
)


def test_valid_manifest_resolves_relative_audio_path(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"segment_id": "seg-1", "audio_path": "sample.wav"}) + "\n",
        encoding="utf-8",
    )

    items = read_manifest(manifest)

    assert items[0]["segment_id"] == "seg-1"
    assert resolve_audio_path(items[0], base_dir=manifest.parent) == audio.resolve()


def test_duplicate_segment_id_rejected(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.touch()
    items = [
        {"segment_id": "dup", "audio_path": str(audio)},
        {"segment_id": "dup", "audio_path": str(audio)},
    ]
    with pytest.raises(ManifestValidationError, match="duplicate segment_id"):
        validate_manifest(items)
