from pathlib import Path

from robot_heard.io.manifest import read_manifest, write_jsonl


def test_write_jsonl_round_trip(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.touch()
    output = tmp_path / "written.jsonl"
    items = [{"segment_id": "seg-1", "audio_path": str(audio), "frontend": "gss"}]
    write_jsonl(items, output)
    assert read_manifest(output) == items
