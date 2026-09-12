from __future__ import annotations

from typing import Any, Mapping, Optional

from .base import ASRBackend
from .openai_whisper import OpenAIWhisperBackend


class BackendConfigError(ValueError):
    """Raised when an ASR backend configuration is incomplete or invalid."""


def _require(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise BackendConfigError(f"missing required config key: {key}")
    return config[key]


def _require_non_empty_str(config: Mapping[str, Any], key: str) -> str:
    value = _require(config, key)
    if not isinstance(value, str) or not value.strip():
        raise BackendConfigError(f"config key {key!r} must be a non-empty string")
    return value


def _optional_str(mapping: Mapping[str, Any], key: str) -> Optional[str]:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise BackendConfigError(f"config key {key!r} must be null or a non-empty string")
    return value


def create_asr_backend(config: Mapping[str, Any]) -> ASRBackend:
    """Create an ASR backend from an explicit configuration mapping."""

    backend = _require_non_empty_str(config, "backend")
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
        model_name=_require_non_empty_str(config, "model"),
        language=_require_non_empty_str(config, "language"),
        task=_require_non_empty_str(config, "task"),
        device=_require_non_empty_str(config, "device"),
        fp16=fp16,
        temperature=decode["temperature"],
        download_root=_optional_str(runtime, "download_root"),
    )
