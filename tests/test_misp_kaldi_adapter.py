import json
import shutil
import struct
import wave
from pathlib import Path

import pytest

from robot_heard.audio import probe_audio
from robot_heard.misp.kaldi_adapter import (
    MISPKaldiAdapterError,
    prepare_misp_kaldi_recording,
    read_kaldi_scp,
    read_kaldi_segments,
    read_kaldi_text,
    resolve_misp_channel_path,
    validate_segment_text_ids,
)

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE,
    reason="ffmpeg/ffprobe required for audio materialization tests",
)


def _write_pcm16_wav(
    path: Path,
    *,
    seconds: float = 2.0,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(seconds * sample_rate)
    frame = struct.pack("<h", 0) * channels
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frame * frame_count)


def _build_kaldi_fixture(tmp_path: Path):
    kaldi_dir = tmp_path / "kaldi"
    kaldi_dir.mkdir()
    stem = tmp_path / "audio" / "rec_Far"
    for channel_id in range(8):
        _write_pcm16_wav(Path(f"{stem}_{channel_id}.wav"))

    (kaldi_dir / "segments").write_text(
        "seg1 rec 0.10 0.30\n"
        "seg2 rec 0.40 0.70\n",
        encoding="utf-8",
    )
    (kaldi_dir / "text").write_text(
        "seg1 你好\n"
        "seg2 世界\n",
        encoding="utf-8",
    )
    (kaldi_dir / "wav.scp").write_text(f"rec {stem}\n", encoding="utf-8")
    (kaldi_dir / "channels.scp").write_text(f"rec {stem}\n", encoding="utf-8")
    return kaldi_dir, stem


def test_read_contract_and_id_equality(tmp_path):
    kaldi_dir, _ = _build_kaldi_fixture(tmp_path)
    segments = read_kaldi_segments(kaldi_dir / "segments")
    references = read_kaldi_text(kaldi_dir / "text")

    validate_segment_text_ids(segments, references)

    assert segments[0].segment_id == "seg1"
    assert segments[0].recording_id == "rec"
    assert segments[0].duration_sec == 0.2
    assert references["seg2"] == "世界"


def test_id_mismatch_is_rejected(tmp_path):
    kaldi_dir, _ = _build_kaldi_fixture(tmp_path)
    (kaldi_dir / "text").write_text("seg1 你好\n", encoding="utf-8")

    with pytest.raises(MISPKaldiAdapterError, match="ID sets differ"):
        validate_segment_text_ids(
            read_kaldi_segments(kaldi_dir / "segments"),
            read_kaldi_text(kaldi_dir / "text"),
        )


def test_channel_range_is_rejected(tmp_path):
    with pytest.raises(MISPKaldiAdapterError, match="0..7"):
        resolve_misp_channel_path(tmp_path / "recording", 8)


def test_kaldi_pipe_command_is_rejected(tmp_path):
    scp = tmp_path / "wav.scp"
    scp.write_text("rec sox a.wav -t wav - |\n", encoding="utf-8")

    with pytest.raises(MISPKaldiAdapterError, match="pipe"):
        read_kaldi_scp(scp)


@requires_ffmpeg
def test_prepare_materializes_explicit_channel_and_manifest(tmp_path):
    kaldi_dir, stem = _build_kaldi_fixture(tmp_path)
    output_dir = tmp_path / "out"

    summary = prepare_misp_kaldi_recording(
        kaldi_dir,
        "rec",
        output_dir,
        channel_id=3,
        limit=1,
        code_commit="abc",
    )

    assert summary.emitted_segments == 1
    assert summary.recording_segments == 2
    assert summary.frontend == "raw_ch3"

    row = json.loads(
        (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["channel_id"] == 3
    assert row["frontend"] == "raw_ch3"
    assert row["reference"] == "你好"
    assert row["duration"] == 0.2
    assert row["source_audio_path"] == str(Path(f"{stem}_3.wav").resolve())

    metadata = json.loads(
        (output_dir / "adapter.json").read_text(encoding="utf-8")
    )
    assert metadata["authoritative_id_set_equal"] is True
    assert metadata["selection_limit"] == 1
    assert metadata["emitted_segments"] == 1
    assert metadata["code_commit"] == "abc"

    audio = probe_audio(output_dir / "audio" / "seg1.wav")
    assert audio.sample_rate == 16000
    assert audio.channels == 1
    assert audio.duration_sec == pytest.approx(0.2, abs=0.02)


@requires_ffmpeg
def test_scp_stem_mismatch_fails_without_output(tmp_path):
    kaldi_dir, _ = _build_kaldi_fixture(tmp_path)
    (kaldi_dir / "channels.scp").write_text(
        "rec /tmp/other\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    with pytest.raises(MISPKaldiAdapterError, match="stems differ"):
        prepare_misp_kaldi_recording(kaldi_dir, "rec", output_dir, channel_id=0)

    assert not output_dir.exists()


@requires_ffmpeg
def test_missing_channel_fails_without_output(tmp_path):
    kaldi_dir, stem = _build_kaldi_fixture(tmp_path)
    Path(f"{stem}_7.wav").unlink()
    output_dir = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="missing channel 7"):
        prepare_misp_kaldi_recording(kaldi_dir, "rec", output_dir, channel_id=0)

    assert not output_dir.exists()


@requires_ffmpeg
def test_segment_beyond_source_fails_without_output(tmp_path):
    kaldi_dir, _ = _build_kaldi_fixture(tmp_path)
    (kaldi_dir / "segments").write_text(
        "seg1 rec 1.9 2.2\n",
        encoding="utf-8",
    )
    (kaldi_dir / "text").write_text("seg1 你好\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    with pytest.raises(MISPKaldiAdapterError, match="exceeds source duration"):
        prepare_misp_kaldi_recording(kaldi_dir, "rec", output_dir, channel_id=0)

    assert not output_dir.exists()


@requires_ffmpeg
def test_overwrite_replaces_only_after_successful_build(tmp_path):
    kaldi_dir, _ = _build_kaldi_fixture(tmp_path)
    output_dir = tmp_path / "out"

    prepare_misp_kaldi_recording(
        kaldi_dir,
        "rec",
        output_dir,
        channel_id=0,
        limit=1,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_misp_kaldi_recording(
            kaldi_dir,
            "rec",
            output_dir,
            channel_id=0,
            limit=1,
        )

    summary = prepare_misp_kaldi_recording(
        kaldi_dir,
        "rec",
        output_dir,
        channel_id=0,
        limit=2,
        overwrite=True,
    )
    assert summary.emitted_segments == 2
    assert len(
        (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ) == 2
