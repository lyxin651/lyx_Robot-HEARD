"""Maintained, CPU-testable handoff from MISP baseline artifacts to Whisper.

The baseline owns segmentation and frontend generation.  This module only
validates those artifacts, materializes an explicit target waveform when the
caller selects a raw channel, or consumes a pre-materialized segment waveform
map whose keys already equal the authoritative baseline segment IDs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from robot_heard.audio import probe_audio
from robot_heard.io.manifest import read_manifest, write_jsonl

from .kaldi_adapter import (
    MISP2025_KALDI_ADAPTER_POLICY,
    MISPKaldiAdapterError,
    MISPKaldiAdapterSummary,
    prepare_misp_kaldi_recording,
    read_kaldi_scp,
    read_kaldi_segments,
    read_kaldi_text,
    validate_segment_text_ids,
)

PathLike = Union[str, Path]

MISP_BASELINE_INTEGRATION_POLICY = "misp2025_baseline_whisper_handoff_v1"
REFERENCE_MODES = ("auto", "required", "off")


class BaselineIntegrationError(ValueError):
    """Raised when a baseline-to-Whisper artifact contract is unsafe."""


@dataclass(frozen=True)
class BaselineManifestSummary:
    frontend: str
    authoritative_segments: int
    emitted_segments: int
    manifest_path: str
    metadata_path: str
    output_dir: str
    reference_available: bool
    waveform_scp_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256_file(path: PathLike) -> str:
    target = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_reference_mode(reference_mode: str) -> str:
    if reference_mode not in REFERENCE_MODES:
        raise BaselineIntegrationError(
            f"reference_mode must be one of {REFERENCE_MODES!r}, got {reference_mode!r}"
        )
    return reference_mode


def _read_authoritative(
    baseline_dir: PathLike,
    *,
    reference_mode: str,
) -> Tuple[Path, List[Any], Dict[str, str], Set[str]]:
    root = Path(baseline_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"baseline artifact directory is not a directory: {root}")
    mode = _validate_reference_mode(reference_mode)
    segments_path = root / "segments"
    text_path = root / "text"
    segments = read_kaldi_segments(segments_path)
    references = read_kaldi_text(text_path, required=mode == "required")
    if references:
        validate_segment_text_ids(segments, references)
    ids = {segment.segment_id for segment in segments}
    return root, segments, references, ids


def _select_segments(segments: Sequence[Any], limit: Optional[int]) -> List[Any]:
    if limit is None:
        return list(segments)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise BaselineIntegrationError("limit must be a positive integer or null")
    return list(segments[:limit])


def _prepare_pre_materialized_frontend(
    baseline_dir: PathLike,
    output_dir: PathLike,
    *,
    frontend: str,
    waveform_scp: PathLike,
    reference_mode: str,
    limit: Optional[int],
    overwrite: bool,
    code_commit: Optional[str],
) -> BaselineManifestSummary:
    if not isinstance(frontend, str) or not frontend.strip():
        raise BaselineIntegrationError("frontend must be a non-empty string")
    root, segments, references, authoritative_ids = _read_authoritative(
        baseline_dir, reference_mode=reference_mode
    )
    scp_path = Path(waveform_scp).expanduser().resolve(strict=True)
    waveform_map = read_kaldi_scp(scp_path)
    waveform_ids = set(waveform_map)
    missing = sorted(authoritative_ids - waveform_ids)
    extra = sorted(waveform_ids - authoritative_ids)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing_waveforms={missing[:10]!r}")
        if extra:
            detail.append(f"extra_waveforms={extra[:10]!r}")
        raise BaselineIntegrationError(
            "pre-materialized frontend waveform IDs do not exactly match "
            "authoritative segments: " + "; ".join(detail)
        )

    selected = _select_segments(segments, limit)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    if destination == root or _is_within(destination, root):
        raise BaselineIntegrationError(
            "output_dir must stay outside the authoritative baseline directory"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(
            "integration output directory already exists; choose a new path or "
            f"use overwrite=True: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=destination.name + ".tmp.", dir=str(destination.parent))
    )
    try:
        manifest_records: List[Dict[str, Any]] = []
        for segment in selected:
            value = waveform_map[segment.segment_id]
            audio_path = Path(value).expanduser()
            if not audio_path.is_absolute():
                audio_path = scp_path.parent / audio_path
            audio_path = audio_path.resolve(strict=False)
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"segment_id={segment.segment_id!r}: waveform does not exist: {audio_path}"
                )
            info = probe_audio(audio_path)
            if info.sample_rate != 16000 or info.channels != 1:
                raise BaselineIntegrationError(
                    f"segment_id={segment.segment_id!r}: frontend waveform must be "
                    f"mono 16 kHz, got {info.channels} channel(s) at {info.sample_rate} Hz"
                )

            record: Dict[str, Any] = {
                "segment_id": segment.segment_id,
                "audio_path": str(audio_path),
                "frontend": frontend,
                "start": segment.start_sec,
                "duration": segment.duration_sec,
                "recording_id": segment.recording_id,
                "source_waveform_path": str(audio_path),
                "source_waveform_sha256": _sha256_file(audio_path),
                "adapter_policy": MISP_BASELINE_INTEGRATION_POLICY,
            }
            if references:
                record["reference"] = references[segment.segment_id]
            manifest_records.append(record)

        manifest_tmp = temporary / "manifest.jsonl"
        write_jsonl(manifest_records, manifest_tmp)
        read_manifest(manifest_tmp, require_audio_exists=True)
        final_manifest = destination / "manifest.jsonl"
        metadata = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "integration_policy": MISP_BASELINE_INTEGRATION_POLICY,
            "frontend": frontend,
            "baseline_dir": str(root),
            "segments_path": str((root / "segments").resolve()),
            "segments_sha256": _sha256_file(root / "segments"),
            "text_path": str((root / "text").resolve()),
            "text_sha256": (
                _sha256_file(root / "text") if (root / "text").is_file() else None
            ),
            "waveform_scp_path": str(scp_path),
            "waveform_scp_sha256": _sha256_file(scp_path),
            "authoritative_segments": len(segments),
            "emitted_segments": len(selected),
            "authoritative_id_sha256": _sha256_ids(authoritative_ids),
            "emitted_id_sha256": _sha256_ids(row["segment_id"] for row in manifest_records),
            "authoritative_id_set_equal": set(row["segment_id"] for row in manifest_records)
            == {row.segment_id for row in selected},
            "reference_available": bool(references),
            "reference_mode": reference_mode,
            "selection_limit": limit,
            "manifest_path": str(final_manifest),
            "manifest_sha256": _sha256_file(manifest_tmp),
            "code_commit": code_commit,
        }
        _atomic_write_json(metadata, temporary / "adapter.json")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    return BaselineManifestSummary(
        frontend=frontend,
        authoritative_segments=len(segments),
        emitted_segments=len(selected),
        manifest_path=str(destination / "manifest.jsonl"),
        metadata_path=str(destination / "adapter.json"),
        output_dir=str(destination),
        reference_available=bool(references),
        waveform_scp_path=str(scp_path),
    )


def prepare_baseline_manifest(
    baseline_dir: PathLike,
    output_dir: PathLike,
    *,
    frontend: str,
    recording_id: Optional[str] = None,
    channel_id: Optional[int] = None,
    waveform_scp: Optional[PathLike] = None,
    reference_mode: str = "auto",
    limit: Optional[int] = None,
    overwrite: bool = False,
    code_commit: Optional[str] = None,
) -> Union[BaselineManifestSummary, MISPKaldiAdapterSummary]:
    """Prepare a manifest from real baseline artifacts.

    ``raw_ch0`` through ``raw_ch7`` use the existing raw-channel adapter and
    require ``recording_id``.  Other frontend names consume ``waveform_scp``;
    that SCP must be keyed by the exact authoritative segment IDs.
    """

    if not isinstance(frontend, str) or not frontend.strip():
        raise BaselineIntegrationError("frontend must be a non-empty string")
    _validate_reference_mode(reference_mode)

    if frontend.startswith("raw_ch"):
        suffix = frontend[len("raw_ch") :]
        if not suffix.isdigit() or int(suffix) not in range(8):
            raise BaselineIntegrationError("raw frontend must be raw_ch0 through raw_ch7")
        if recording_id is None:
            raise BaselineIntegrationError(
                "recording_id is required for raw_chX frontend selection"
            )
        if channel_id is not None and channel_id != int(suffix):
            raise BaselineIntegrationError(
                "channel_id disagrees with raw frontend selection"
            )
        return prepare_misp_kaldi_recording(
            baseline_dir,
            recording_id,
            output_dir,
            channel_id=int(suffix),
            limit=limit,
            overwrite=overwrite,
            code_commit=code_commit,
            require_reference=reference_mode == "required",
        )

    if waveform_scp is None:
        raise BaselineIntegrationError(
            "waveform_scp is required for pre-materialized frontend selection"
        )
    if recording_id is not None or channel_id is not None:
        raise BaselineIntegrationError(
            "recording_id/channel_id are only valid for raw_chX frontends"
        )
    return _prepare_pre_materialized_frontend(
        baseline_dir,
        output_dir,
        frontend=frontend,
        waveform_scp=waveform_scp,
        reference_mode=reference_mode,
        limit=limit,
        overwrite=overwrite,
        code_commit=code_commit,
    )


def _read_jsonl_ids(path: PathLike, *, allow_duplicate: bool = False) -> Tuple[List[str], Set[str]]:
    source = Path(path).expanduser().resolve(strict=True)
    ids: List[str] = []
    seen: Set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BaselineIntegrationError(
                    f"{source}: line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, Mapping):
                raise BaselineIntegrationError(
                    f"{source}: line {line_number}: record must be an object"
                )
            segment_id = record.get("segment_id")
            if not isinstance(segment_id, str) or not segment_id.strip():
                raise BaselineIntegrationError(
                    f"{source}: line {line_number}: missing/non-string segment_id"
                )
            if segment_id in seen and not allow_duplicate:
                raise BaselineIntegrationError(
                    f"{source}: line {line_number}: duplicate segment_id={segment_id!r}"
                )
            ids.append(segment_id)
            seen.add(segment_id)
    return ids, seen


def reconcile_ids(
    authoritative_ids: Iterable[str],
    *,
    manifest_ids: Iterable[str],
    result_ids: Optional[Iterable[str]] = None,
    failure_ids: Optional[Iterable[str]] = None,
    scored_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Require exact ID coverage for each supplied integration artifact."""

    expected = set(authoritative_ids)
    if not expected:
        raise BaselineIntegrationError("authoritative ID set must not be empty")
    stages: Dict[str, Any] = {}
    paired_decode_evidence = result_ids is not None and failure_ids is not None
    for name, values in (
        ("manifest", manifest_ids),
        ("results", result_ids),
        ("failures", failure_ids),
        ("scored", scored_ids),
    ):
        if values is None:
            continue
        sequence = list(values)
        actual = set(sequence)
        duplicates = sorted({item for item in sequence if sequence.count(item) > 1})
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        stages[name] = {
            "count": len(sequence),
            "unique": len(actual),
            "missing": missing,
            "extra": extra,
            "duplicates": duplicates,
        }
        # A successful result set and an explicit failure set jointly cover a
        # decode run. Neither side is required to contain every ID alone.
        if paired_decode_evidence and name in {"results", "failures"}:
            missing = []
        duplicate_error = bool(duplicates) and name != "failures"
        if missing or extra or duplicate_error:
            raise BaselineIntegrationError(
                f"{name} ID reconciliation failed: missing={missing[:10]!r}; "
                f"extra={extra[:10]!r}; duplicates={duplicates[:10]!r}"
            )

    if result_ids is not None and failure_ids is not None:
        results = set(result_ids)
        failures = set(failure_ids)
        overlap = sorted(results & failures)
        if results | failures != expected:
            raise BaselineIntegrationError(
                "results + explicit failures do not cover authoritative IDs: "
                f"missing={sorted(expected - (results | failures))[:10]!r}; "
                f"extra={sorted((results | failures) - expected)[:10]!r}"
            )
        stages["results_plus_failures"] = {
            "count": len(results | failures),
            "unique": len(results | failures),
            "overlap": overlap,
        }

    return {
        "authoritative_count": len(expected),
        "authoritative_id_sha256": _sha256_ids(expected),
        "stages": stages,
        "exact": True,
    }


def reconcile_manifest_file(
    baseline_dir: PathLike,
    manifest_path: PathLike,
    *,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Reconcile a prepared manifest against baseline `segments` without GPU."""

    _, segments, _, authoritative_ids = _read_authoritative(
        baseline_dir, reference_mode="off"
    )
    selected_ids = {segment.segment_id for segment in _select_segments(segments, limit)}
    _, manifest_ids = _read_jsonl_ids(manifest_path)
    return reconcile_ids(selected_ids, manifest_ids=manifest_ids)


def read_artifact_ids(path: PathLike, *, allow_duplicate: bool = False) -> List[str]:
    """Read JSONL segment IDs for orchestration/reconciliation checks."""

    return _read_jsonl_ids(path, allow_duplicate=allow_duplicate)[0]
