import shutil
import struct
import wave
from pathlib import Path

import pytest

from robot_heard.audio import AudioInputError, AudioToolError, prepare_audio, probe_audio


FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe required")


def _write_pcm16_wav(path: Path, *, sample_rate: int, channels: int, duration_sec: float = 0.1) -> None:
    frame_count = max(1, int(sample_rate * duration_sec))
    frame = struct.pack("<h", 0) * channels
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frame * frame_count)


def test_probe_audio_reports_self_describing_metadata(tmp_path):
    path = tmp_path / "mono16k.wav"
    _write_pcm16_wav(path, sample_rate=16000, channels=1, duration_sec=0.2)

    metadata = probe_audio(path)

    assert metadata.path == path.resolve()
    assert metadata.sample_rate == 16000
    assert metadata.channels == 1
    assert metadata.duration_sec == pytest.approx(0.2, abs=0.01)
    assert metadata.codec_name == "pcm_s16le"


def test_probe_audio_rejects_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_text("not audio", encoding="utf-8")

    with pytest.raises(AudioToolError, match="ffprobe failed"):
        probe_audio(path)


def test_prepare_audio_passes_through_16k_mono(tmp_path):
    path = tmp_path / "mono16k.wav"
    _write_pcm16_wav(path, sample_rate=16000, channels=1)

    prepared = prepare_audio(path)

    assert prepared.converted is False
    assert prepared.path == path.resolve()
    assert prepared.source == prepared.prepared


def test_prepare_audio_rejects_multichannel_without_downmix(tmp_path):
    path = tmp_path / "stereo16k.wav"
    _write_pcm16_wav(path, sample_rate=16000, channels=2)

    with pytest.raises(AudioInputError, match="multichannel input rejected"):
        prepare_audio(path)


def test_prepare_audio_resamples_mono_to_16k(tmp_path):
    source = tmp_path / "mono8k.wav"
    destination = tmp_path / "prepared.wav"
    _write_pcm16_wav(source, sample_rate=8000, channels=1, duration_sec=0.2)

    prepared = prepare_audio(source, output_path=destination)

    assert prepared.converted is True
    assert prepared.source.sample_rate == 8000
    assert prepared.prepared.sample_rate == 16000
    assert prepared.prepared.channels == 1
    assert prepared.path == destination.resolve()
    assert destination.is_file()
    assert prepared.prepared.duration_sec == pytest.approx(0.2, abs=0.02)


def test_prepare_audio_requires_explicit_output_for_resampling(tmp_path):
    source = tmp_path / "mono8k.wav"
    _write_pcm16_wav(source, sample_rate=8000, channels=1)

    with pytest.raises(AudioInputError, match="requires output_path"):
        prepare_audio(source)


def test_prepare_audio_refuses_to_overwrite_by_default(tmp_path):
    source = tmp_path / "mono8k.wav"
    destination = tmp_path / "prepared.wav"
    _write_pcm16_wav(source, sample_rate=8000, channels=1)
    _write_pcm16_wav(destination, sample_rate=16000, channels=1)

    with pytest.raises(FileExistsError, match="prepared audio already exists"):
        prepare_audio(source, output_path=destination)
