import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_heard.audio import AudioInputError, AudioMetadata
from robot_heard.asr.openai_whisper import OpenAIWhisperBackend


class _FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True

    def requires_grad_(self, value: bool):
        self.requires_grad = value
        return self


class _FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.parameters_list = [_FakeParameter()]
        self.transcribe_calls = []

    def eval(self) -> None:
        self.eval_called = True

    def parameters(self):
        return iter(self.parameters_list)

    def transcribe(self, path, **kwargs):
        self.transcribe_calls.append((path, kwargs))
        return {"text": "test transcript", "language": "zh"}


def _build_backend(monkeypatch):
    load_calls = []
    fake_model = _FakeModel()

    def fake_load_model(model_name, **kwargs):
        load_calls.append((model_name, kwargs))
        return fake_model

    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=fake_load_model))
    backend = OpenAIWhisperBackend(
        model_name="large-v3",
        language="zh",
        task="transcribe",
        device="cuda",
        fp16=True,
        temperature=0.0,
    )
    return backend, fake_model, load_calls


def test_backend_loads_once_reuses_model_and_accepts_prepared_audio(monkeypatch, tmp_path: Path) -> None:
    backend, fake_model, load_calls = _build_backend(monkeypatch)

    audio = tmp_path / "sample.wav"
    audio.touch()
    monkeypatch.setattr(
        "robot_heard.asr.openai_whisper.probe_audio",
        lambda path: AudioMetadata(
            path=Path(path),
            sample_rate=16000,
            channels=1,
            duration_sec=1.0,
            codec_name="pcm_s16le",
        ),
    )

    first = backend.transcribe(audio)
    second = backend.transcribe(audio)

    assert len(load_calls) == 1
    assert len(fake_model.transcribe_calls) == 2
    assert fake_model.eval_called
    assert fake_model.parameters_list[0].requires_grad is False
    assert first.text == "test transcript"
    assert second.language == "zh"
    assert first.decode_sec >= 0.0


def test_backend_rejects_multichannel_audio_before_whisper(monkeypatch, tmp_path: Path) -> None:
    backend, fake_model, _ = _build_backend(monkeypatch)
    audio = tmp_path / "stereo.wav"
    audio.touch()
    monkeypatch.setattr(
        "robot_heard.asr.openai_whisper.probe_audio",
        lambda path: AudioMetadata(
            path=Path(path),
            sample_rate=16000,
            channels=2,
            duration_sec=1.0,
            codec_name="pcm_s16le",
        ),
    )

    with pytest.raises(AudioInputError, match="requires an explicit mono target waveform"):
        backend.transcribe(audio)
    assert fake_model.transcribe_calls == []


def test_backend_rejects_non_16k_audio_before_whisper(monkeypatch, tmp_path: Path) -> None:
    backend, fake_model, _ = _build_backend(monkeypatch)
    audio = tmp_path / "mono48k.wav"
    audio.touch()
    monkeypatch.setattr(
        "robot_heard.asr.openai_whisper.probe_audio",
        lambda path: AudioMetadata(
            path=Path(path),
            sample_rate=48000,
            channels=1,
            duration_sec=1.0,
            codec_name="pcm_s16le",
        ),
    )

    with pytest.raises(AudioInputError, match="requires 16000 Hz prepared audio"):
        backend.transcribe(audio)
    assert fake_model.transcribe_calls == []
