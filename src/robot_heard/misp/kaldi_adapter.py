from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from robot_heard.audio import AudioMetadata, probe_audio
from robot_heard.io.manifest import read_manifest, write_jsonl

PathLike = Union[str, Path]

MISP2025_KALDI_ADAPTER_POLICY = "misp2025_avsr_stage2_raw_channel_v1"
MISP2025_KALDI_CHANNELS = tuple(range(8))
TARGET_SAMPLE_RATE = 16000


class MISPKaldiAdapterError(ValueError):
    """Raised when real MISP Kaldi artifacts violate the S6I adapter contract."""


@dataclass(frozen=True)
class KaldiSegment:
    segment_id: str
    recording_id: str
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return round(self.end_sec - self.start_sec, 9)


@dataclass(frozen=True)
class MISPKaldiAdapterSummary:
    recording_id: str
    channel_id: int
    frontend: str
    authoritative_segments: int
    recording_segments: int
    emitted_segments: int
    manifest_path: str
    metadata_path: str
    output_dir: str
    source_audio_path: str
    limited: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256_file(path: PathLike) -> str:
    target = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_regular_file(path: PathLike, *, label: str) -> Path:
    target = Path(path).expanduser().resolve(strict=False)
    if not target.is_file():
        raise FileNotFoundError(f"{label} does not exist: {target}")
    return target


def _parse_finite_float(
    value: str,
    *,
    field: str,
    source: Path,
    line_number: int,
) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise MISPKaldiAdapterError(
            f"{source}: line {line_number}: invalid {field}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise MISPKaldiAdapterError(
            f"{source}: line {line_number}: {field} must be finite"
        )
    return parsed


def read_kaldi_segments(path: PathLike) -> List[KaldiSegment]:
    source = _require_regular_file(path, label="segments file")
    segments: List[KaldiSegment] = []
    seen = set()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise MISPKaldiAdapterError(
                    f"{source}: line {line_number}: expected 4 fields "
                    "'segment_id recording_id start_sec end_sec'"
                )

            segment_id, recording_id, start_raw, end_raw = parts
            if segment_id in seen:
                raise MISPKaldiAdapterError(
                    f"{source}: line {line_number}, segment_id={segment_id!r}: "
                    "duplicate segment"
                )
            if "/" in segment_id or "\\" in segment_id:
                raise MISPKaldiAdapterError(
                    f"{source}: line {line_number}, segment_id={segment_id!r}: "
                    "path separators are not allowed"
                )

            start_sec = _parse_finite_float(
                start_raw,
                field="start_sec",
                source=source,
                line_number=line_number,
            )
            end_sec = _parse_finite_float(
                end_raw,
                field="end_sec",
                source=source,
                line_number=line_number,
            )
            if start_sec < 0:
                raise MISPKaldiAdapterError(
                    f"{source}: line {line_number}, segment_id={segment_id!r}: "
                    "start_sec must be >= 0"
                )
            if end_sec <= start_sec:
                raise MISPKaldiAdapterError(
                    f"{source}: line {line_number}, segment_id={segment_id!r}: "
                    "end_sec must be > start_sec"
                )

            segments.append(
                KaldiSegment(
                    segment_id=segment_id,
                    recording_id=recording_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                )
            )
            seen.add(segment_id)

    if not segments:
        raise MISPKaldiAdapterError(f"{source}: no segments found")
    return segments


def read_kaldi_text(path: PathLike) -> Dict[str, str]:
    source = _require_regular_file(path, label="text file")
    records: Dict[str, str] = {}

    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                raise MISPKaldiAdapterError(
                    f"{source}: line {line_number}: expected "
                    "'segment_id non-empty-reference'"
                )

            segment_id, reference = parts[0], parts[1].strip()
            if segment_id in records:
                raise MISPKaldiAdapterError(
                    f"{source}: line {line_number}, segment_id={segment_id!r}: "
                    "duplicate reference"
                )
            records[segment_id] = reference

    if not records:
        raise MISPKaldiAdapterError(f"{source}: no text records found")
    return records


def read_kaldi_scp(path: PathLike) -> Dict[str, str]:
    source = _require_regular_file(path, label="scp file")
    records: Dict[str, str] = {}

    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                raise MISPKaldiAdapterError(
                    f"{source}: line {line_number}: expected "
                    "'recording_id path-or-stem'"
                )

            recording_id, value = parts[0], parts[1].strip()
            if value.endswith("|"):
                raise MISPKaldiAdapterError(
                    f"{source}: line {line_number}, recording_id={recording_id!r}: "
                    "Kaldi pipe commands are not supported by this adapter"
                )
            if recording_id in records:
                raise MISPKaldiAdapterError(
                    f"{source}: line {line_number}, recording_id={recording_id!r}: "
                    "duplicate mapping"
                )
            records[recording_id] = value

    if not records:
        raise MISPKaldiAdapterError(f"{source}: no scp mappings found")
    return records


def validate_segment_text_ids(
    segments: Sequence[KaldiSegment],
    references: Mapping[str, str],
) -> None:
    segment_ids = {segment.segment_id for segment in segments}
    reference_ids = set(references)
    missing = sorted(segment_ids - reference_ids)
    extra = sorted(reference_ids - segment_ids)
    if not missing and not extra:
        return

    details = []
    if missing:
        details.append(f"missing_reference={missing[:10]!r}")
    if extra:
        details.append(f"extra_reference={extra[:10]!r}")
    raise MISPKaldiAdapterError(
        "segments/text ID sets differ: " + "; ".join(details)
    )


def _resolve_stem(value: str, *, scp_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = scp_path.parent / path
    return path.resolve(strict=False)


def resolve_misp_channel_path(stem: PathLike, channel_id: int) -> Path:
    if isinstance(channel_id, bool) or not isinstance(channel_id, int):
        raise MISPKaldiAdapterError("channel_id must be an integer")
    if channel_id not in MISP2025_KALDI_CHANNELS:
        raise MISPKaldiAdapterError("MISP AVSR raw channel_id must be in 0..7")
    base = Path(stem).expanduser().resolve(strict=False)
    return Path(f"{base}_{channel_id}.wav")


def _validate_recording_sources(
    *,
    stem: Path,
    recording_id: str,
) -> Tuple[List[Path], List[AudioMetadata]]:
    paths: List[Path] = []
    metadata: List[AudioMetadata] = []

    for channel_id in MISP2025_KALDI_CHANNELS:
        path = resolve_misp_channel_path(stem, channel_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"recording_id={recording_id!r}: missing channel {channel_id}: {path}"
            )

        info = probe_audio(path)
        if info.sample_rate != TARGET_SAMPLE_RATE:
            raise MISPKaldiAdapterError(
                f"recording_id={recording_id!r}, channel={channel_id}: "
                f"expected 16000 Hz, got {info.sample_rate}"
            )
        if info.channels != 1:
            raise MISPKaldiAdapterError(
                f"recording_id={recording_id!r}, channel={channel_id}: "
                "expected an already-separated mono channel file, "
                f"got {info.channels} channels"
            )

        paths.append(path.resolve())
        metadata.append(info)

    reference_duration = metadata[0].duration_sec
    for channel_id, info in enumerate(metadata[1:], start=1):
        if abs(info.duration_sec - reference_duration) > 0.001:
            raise MISPKaldiAdapterError(
                f"recording_id={recording_id!r}: channel durations differ: "
                f"ch0={reference_duration}, ch{channel_id}={info.duration_sec}"
            )

    return paths, metadata


def _run_ffmpeg_extract(
    source: Path,
    destination: Path,
    *,
    start_sec: float,
    duration_sec: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("required executable not found on PATH: ffmpeg")

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.9f}",
        "-i",
        str(source),
        "-t",
        f"{duration_sec:.9f}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s32le",
        str(destination),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"failed to execute ffmpeg: {exc}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "<no stderr>"
        raise RuntimeError(
            f"ffmpeg failed with exit code {completed.returncode}: {stderr}"
        )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def prepare_misp_kaldi_recording(
    kaldi_dir: PathLike,
    recording_id: str,
    output_dir: PathLike,
    *,
    channel_id: int,
    limit: Optional[int] = None,
    overwrite: bool = False,
    code_commit: Optional[str] = None,
) -> MISPKaldiAdapterSummary:
    """Materialize one MISP Stage-2 recording as explicit mono ASR segments.

    In the real AVSR baseline artifacts validated for S6I-A, `wav.scp` and
    `channels.scp` contain a shared recording stem. Physical channels are
    `<stem>_0.wav` ... `<stem>_7.wav`. Exactly one caller-selected channel is
    cut into segment WAVs; channels are never averaged or implicitly downmixed.
    """

    root = Path(kaldi_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"kaldi_dir is not a directory: {root}")
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise MISPKaldiAdapterError("recording_id must be a non-empty string")
    if (
        isinstance(channel_id, bool)
        or not isinstance(channel_id, int)
        or channel_id not in MISP2025_KALDI_CHANNELS
    ):
        raise MISPKaldiAdapterError("channel_id must be an integer in 0..7")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise MISPKaldiAdapterError("limit must be a positive integer or null")

    segments_path = root / "segments"
    text_path = root / "text"
    wav_scp_path = root / "wav.scp"
    channels_scp_path = root / "channels.scp"

    segments = read_kaldi_segments(segments_path)
    references = read_kaldi_text(text_path)
    validate_segment_text_ids(segments, references)
    wav_scp = read_kaldi_scp(wav_scp_path)
    channels_scp = read_kaldi_scp(channels_scp_path)

    used_recordings = {segment.recording_id for segment in segments}
    missing_wav = sorted(used_recordings - set(wav_scp))
    missing_channels = sorted(used_recordings - set(channels_scp))
    if missing_wav or missing_channels:
        raise MISPKaldiAdapterError(
            "segments reference recordings missing from scp mappings: "
            f"wav.scp_missing={missing_wav[:10]!r}; "
            f"channels.scp_missing={missing_channels[:10]!r}"
        )
    if recording_id not in used_recordings:
        raise MISPKaldiAdapterError(
            f"recording_id={recording_id!r} is not referenced by segments"
        )

    wav_stem = _resolve_stem(wav_scp[recording_id], scp_path=wav_scp_path)
    channel_stem = _resolve_stem(
        channels_scp[recording_id],
        scp_path=channels_scp_path,
    )
    if wav_stem != channel_stem:
        raise MISPKaldiAdapterError(
            f"recording_id={recording_id!r}: wav.scp/channels.scp stems differ: "
            f"{wav_stem} != {channel_stem}"
        )

    channel_paths, channel_metadata = _validate_recording_sources(
        stem=channel_stem,
        recording_id=recording_id,
    )
    source_audio = channel_paths[channel_id]
    source_metadata = channel_metadata[channel_id]

    recording_segments = [
        segment for segment in segments if segment.recording_id == recording_id
    ]
    for segment in recording_segments:
        if segment.end_sec > source_metadata.duration_sec + 0.001:
            raise MISPKaldiAdapterError(
                f"segment_id={segment.segment_id!r}: end_sec={segment.end_sec} "
                f"exceeds source duration={source_metadata.duration_sec}"
            )

    selected = recording_segments if limit is None else recording_segments[:limit]
    destination = Path(output_dir).expanduser().resolve(strict=False)
    source_parent = channel_stem.parent.resolve(strict=False)

    if destination == root or _is_within(destination, root):
        raise MISPKaldiAdapterError(
            "output_dir must stay outside the authoritative Kaldi data directory"
        )
    if destination == source_parent or _is_within(destination, source_parent):
        raise MISPKaldiAdapterError(
            "output_dir must stay outside the Stage-1 source-audio directory"
        )
    if _is_within(root, destination) or _is_within(source_audio, destination):
        raise MISPKaldiAdapterError(
            "output_dir must not be an ancestor of authoritative inputs"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(
            "adapter output directory already exists; choose a new path or use "
            f"overwrite=True: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=destination.name + ".tmp.",
            dir=str(destination.parent),
        )
    )

    try:
        audio_dir = temporary / "audio"
        audio_dir.mkdir(parents=True, exist_ok=False)
        manifest_records: List[Dict[str, Any]] = []

        for segment in selected:
            segment_audio = audio_dir / f"{segment.segment_id}.wav"
            _run_ffmpeg_extract(
                source_audio,
                segment_audio,
                start_sec=segment.start_sec,
                duration_sec=segment.duration_sec,
            )
            info = probe_audio(segment_audio)
            if info.sample_rate != TARGET_SAMPLE_RATE or info.channels != 1:
                raise MISPKaldiAdapterError(
                    f"segment_id={segment.segment_id!r}: prepared segment is not "
                    "mono 16 kHz"
                )
            if abs(info.duration_sec - segment.duration_sec) > 0.02:
                raise MISPKaldiAdapterError(
                    f"segment_id={segment.segment_id!r}: prepared duration mismatch: "
                    f"expected {segment.duration_sec:.6f}, "
                    f"got {info.duration_sec:.6f}"
                )

            manifest_records.append(
                {
                    "segment_id": segment.segment_id,
                    "audio_path": f"audio/{segment.segment_id}.wav",
                    "reference": references[segment.segment_id],
                    "frontend": f"raw_ch{channel_id}",
                    "start": segment.start_sec,
                    "duration": segment.duration_sec,
                    "recording_id": recording_id,
                    "channel_id": channel_id,
                    "source_audio_path": str(source_audio),
                    "source_recording_stem": str(channel_stem),
                    "source_end": segment.end_sec,
                    "adapter_policy": MISP2025_KALDI_ADAPTER_POLICY,
                }
            )

        manifest_tmp = temporary / "manifest.jsonl"
        write_jsonl(manifest_records, manifest_tmp)
        read_manifest(manifest_tmp, require_audio_exists=True)

        final_manifest = destination / "manifest.jsonl"
        metadata_payload = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "adapter_policy": MISP2025_KALDI_ADAPTER_POLICY,
            "kaldi_dir": str(root),
            "segments_path": str(segments_path.resolve()),
            "segments_sha256": _sha256_file(segments_path),
            "text_path": str(text_path.resolve()),
            "text_sha256": _sha256_file(text_path),
            "wav_scp_path": str(wav_scp_path.resolve()),
            "wav_scp_sha256": _sha256_file(wav_scp_path),
            "channels_scp_path": str(channels_scp_path.resolve()),
            "channels_scp_sha256": _sha256_file(channels_scp_path),
            "authoritative_id_set_equal": True,
            "authoritative_segments": len(segments),
            "recording_id": recording_id,
            "recording_segments": len(recording_segments),
            "channel_id": channel_id,
            "frontend": f"raw_ch{channel_id}",
            "source_recording_stem": str(channel_stem),
            "source_audio_path": str(source_audio),
            "source_audio_sha256": _sha256_file(source_audio),
            "source_audio": {
                "sample_rate": source_metadata.sample_rate,
                "channels": source_metadata.channels,
                "duration_sec": source_metadata.duration_sec,
                "codec_name": source_metadata.codec_name,
            },
            "all_channel_paths": [str(path) for path in channel_paths],
            "all_channel_durations_sec": [
                info.duration_sec for info in channel_metadata
            ],
            "output_audio_codec": "pcm_s32le",
            "output_sample_rate": TARGET_SAMPLE_RATE,
            "output_channels": 1,
            "selection_limit": limit,
            "emitted_segments": len(selected),
            "manifest_path": str(final_manifest),
            "manifest_sha256": _sha256_file(manifest_tmp),
            "code_commit": code_commit,
        }
        _atomic_write_json(metadata_payload, temporary / "adapter.json")

        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    return MISPKaldiAdapterSummary(
        recording_id=recording_id,
        channel_id=channel_id,
        frontend=f"raw_ch{channel_id}",
        authoritative_segments=len(segments),
        recording_segments=len(recording_segments),
        emitted_segments=len(selected),
        manifest_path=str(destination / "manifest.jsonl"),
        metadata_path=str(destination / "adapter.json"),
        output_dir=str(destination),
        source_audio_path=str(source_audio),
        limited=limit is not None,
    )
