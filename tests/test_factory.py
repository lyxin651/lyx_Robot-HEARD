from pathlib import Path

import pytest

from robot_heard.asr import factory
from robot_heard.asr.factory import BackendConfigError, create_asr_backend


class _DummyBackend:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _valid_config():
    return {
        "backend": "openai_whisper",
        "model": "large-v3",
        "language": "zh",
        "task": "transcribe",
        "device": "cuda",
        "fp16": True,
        "decode": {"temperature": 0.0},
        "runtime": {"download_root": None},
    }


def test_factory_passes_explicit_config(monkeypatch) -> None:
    monkeypatch.setattr(factory, "OpenAIWhisperBackend", _DummyBackend)
    backend = create_asr_backend(_valid_config())
    assert backend.kwargs["model_name"] == "large-v3"
    assert backend.kwargs["language"] == "zh"
    assert backend.kwargs["fp16"] is True
    assert backend.kwargs["temperature"] == 0.0


def test_factory_rejects_non_boolean_fp16() -> None:
    config = _valid_config()
    config["fp16"] = "true"
    with pytest.raises(BackendConfigError, match="fp16"):
        create_asr_backend(config)


def test_factory_rejects_empty_model() -> None:
    config = _valid_config()
    config["model"] = ""
    with pytest.raises(BackendConfigError, match="model"):
        create_asr_backend(config)
