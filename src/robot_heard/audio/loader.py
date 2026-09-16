from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

PathLike = Union[str, Path]


class AudioInputError(ValueError):
    """Raised when an audio input violates the V0 ASR input contract."""


class AudioToolError(RuntimeError):
    """Raised when ffprobe/ffmpeg is unavailable or fails."""


@dataclass(frozen=True)
class AudioMetadata:
    """Self-describing metadata for one audio file."""

    path: Path
    sample_rate: int
    channels: int
    duration_sec: float
    codec_name: Optional[str]


@dataclass(frozen=True)
class AudioInputPolicy:
    """Frozen V0 policy for audio passed to the ASR backend."""

    target_sample_rate: int = 16000
    require_mono: bool = True


@dataclass(frozen=True)
class PreparedAudio:
    """Result of applying the V0 audio-input policy."""

    source: AudioMetadata
    prepared: AudioMetadata
    converted: bool

    @property
    def path(self) -> Path:
        return self.prepared.path


def audio_policy_from_config(config: Mapping[str, Any]) -> AudioInputPolicy:
    """Read and strictly validate the V0 audio policy from a full config mapping."""

    audio = config.get("audio")
    if not isinstance(audio, Mapping):
        raise AudioInputError("config key 'audio' must be a mapping")

    target_sample_rate = audio.get("target_sample_rate")
    if isinstance(target_sample_rate, bool) or not isinstance(target_sample_rate, int):
        raise AudioInputError("audio.target_sample_rate must be an integer")
    if target_sample_rate != 16000:
        raise AudioInputError("Whisper V0 requires audio.target_sample_rate=16000")

    require_mono = audio.get("require_mono")
    if require_mono is not True:
        raise AudioInputError("Whisper V0 requires audio.require_mono=true")

    return AudioInputPolicy(
        target_sample_rate=target_sample_rate,
        require_mono=require_mono,
    )


def _require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise AudioToolError(f"required executable not found on PATH: {name}")
    return executable


def _run(command: list[str], *, tool_name: str) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise AudioToolError(f"failed to execute {tool_name}: {exc}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "<no stderr>"
        raise AudioToolError(f"{tool_name} failed with exit code {completed.returncode}: {stderr}")
    return completed


def _positive_int(value: Any, *, field: str, path: Path) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AudioInputError(f"{path}: invalid {field}: {value!r}") from exc
    if parsed <= 0:
        raise AudioInputError(f"{path}: {field} must be positive, got {parsed}")
    return parsed


def _positive_float(value: Any, *, field: str, path: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AudioInputError(f"{path}: invalid {field}: {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise AudioInputError(f"{path}: {field} must be a finite positive value, got {parsed!r}")
    return parsed


def probe_audio(path: PathLike) -> AudioMetadata:
    """Inspect one self-describing audio file with ffprobe.

    Headerless raw PCM is intentionally unsupported because sample rate and channel
    count cannot be inferred safely. Raw MISP CSOBx3 data must first be converted
    or channel-selected by an explicit upstream adapter/front-end.
    """

    audio_path = Path(path).expanduser().resolve(strict=False)
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio file does not exist: {audio_path}")

    ffprobe = _require_executable("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,codec_name,duration:format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    completed = _run(command, tool_name="ffprobe")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AudioToolError(f"ffprobe returned invalid JSON for {audio_path}") from exc

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise AudioInputError(f"{audio_path}: no audio stream found")
    stream = streams[0]
    if not isinstance(stream, Mapping):
        raise AudioInputError(f"{audio_path}: invalid ffprobe audio stream metadata")

    sample_rate = _positive_int(stream.get("sample_rate"), field="sample_rate", path=audio_path)
    channels = _positive_int(stream.get("channels"), field="channels", path=audio_path)

    duration_value = stream.get("duration")
    if duration_value in (None, "N/A"):
        format_info = payload.get("format")
        if isinstance(format_info, Mapping):
            duration_value = format_info.get("duration")
    duration_sec = _positive_float(duration_value, field="duration", path=audio_path)

    codec_name = stream.get("codec_name")
    if codec_name is not None and not isinstance(codec_name, str):
        codec_name = str(codec_name)

    return AudioMetadata(
        path=audio_path,
        sample_rate=sample_rate,
        channels=channels,
        duration_sec=duration_sec,
        codec_name=codec_name,
    )


def _validate_source(metadata: AudioMetadata, policy: AudioInputPolicy) -> None:
    if policy.require_mono and metadata.channels != 1:
        raise AudioInputError(
            f"{metadata.path}: multichannel input rejected ({metadata.channels} channels). "
            "Whisper V0 never performs implicit averaging/downmix; select/beamform/separate "
            "an explicit single-channel target waveform upstream."
        )


def prepare_audio(
    path: PathLike,
    *,
    policy: Optional[AudioInputPolicy] = None,
    output_path: Optional[PathLike] = None,
    overwrite: bool = False,
) -> PreparedAudio:
    """Apply the V0 mono/16-kHz policy without hidden multichannel downmix.

    A mono 16-kHz input is passed through unchanged. A mono input at another
    sample rate is resampled to a PCM-s16le WAV at ``output_path`` using ffmpeg.
    Multichannel inputs are rejected before ffmpeg is invoked.
    """

    active_policy = policy or AudioInputPolicy()
    source = probe_audio(path)
    _validate_source(source, active_policy)

    if source.sample_rate == active_policy.target_sample_rate:
        return PreparedAudio(source=source, prepared=source, converted=False)

    if output_path is None:
        raise AudioInputError(
            f"{source.path}: resampling {source.sample_rate} -> "
            f"{active_policy.target_sample_rate} Hz requires output_path"
        )

    destination = Path(output_path).expanduser().resolve(strict=False)
    if destination == source.path:
        raise AudioInputError("output_path must differ from the source path when conversion is required")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"prepared audio already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _require_executable("ffmpeg")
    temporary = destination.with_name(destination.name + ".tmp.wav")
    if temporary.exists():
        temporary.unlink()

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source.path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(active_policy.target_sample_rate),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(temporary),
    ]

    try:
        _run(command, tool_name="ffmpeg")
        prepared = probe_audio(temporary)
        if prepared.sample_rate != active_policy.target_sample_rate:
            raise AudioInputError(
                f"ffmpeg output sample rate mismatch: expected "
                f"{active_policy.target_sample_rate}, got {prepared.sample_rate}"
            )
        if active_policy.require_mono and prepared.channels != 1:
            raise AudioInputError(
                f"ffmpeg output channel mismatch: expected mono, got {prepared.channels} channels"
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    final_metadata = probe_audio(destination)
    return PreparedAudio(source=source, prepared=final_metadata, converted=True)
