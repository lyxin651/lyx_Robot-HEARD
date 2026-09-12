import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_backend_loads_once_and_reuses_model(monkeypatch, tmp_path: Path) -> None:
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

    audio = tmp_path / "sample.wav"
    audio.touch()
    first = backend.transcribe(audio)
    second = backend.transcribe(audio)

    assert len(load_calls) == 1
    assert len(fake_model.transcribe_calls) == 2
    assert fake_model.eval_called
    assert fake_model.parameters_list[0].requires_grad is False
    assert first.text == "test transcript"
    assert second.language == "zh"
    assert first.decode_sec >= 0.0
