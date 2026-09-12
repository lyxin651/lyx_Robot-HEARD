from __future__ import annotations

from typing import Any, Mapping

from .base import ASRBackend
from .openai_whisper import OpenAIWhisperBackend


class BackendConfigError(ValueError):
    """Raised when an ASR backend configuration is incomplete or invalid."""


def _require(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise BackendConfigError(f"missing required config key: {key}")
    return config[key]


def create_asr_backend(config: Mapping[str, Any]) -> ASRBackend:
    """Create an ASR backend from an explicit configuration mapping."""

    backend = _require(config, "backend")
    if backend != "openai_whisper":
        raise BackendConfigError(f"unsupported ASR backend for V0: {backend!r}")

    decode = config.get("decode")
    if not isinstance(decode, Mapping):
        raise BackendConfigError("config key 'decode' must be a mapping")
    if "temperature" not in decode:
        raise BackendConfigError("missing required config key: decode.temperature")

    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise BackendConfigError("config key 'runtime' must be a mapping when provided")

    fp16 = _require(config, "fp16")
    if not isinstance(fp16, bool):
        raise BackendConfigError("config key 'fp16' must be a boolean")

    return OpenAIWhisperBackend(
        model_name=str(_require(config, "model")),
        language=str(_require(config, "language")),
        task=str(_require(config, "task")),
        device=str(_require(config, "device")),
        fp16=fp16,
        temperature=decode["temperature"],
        download_root=runtime.get("download_root"),
    )
