from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple, Union

from robot_heard.audio import probe_audio
from robot_heard.io.manifest import read_manifest, resolve_audio_path

from .base import ASRBackend

PathLike = Union[str, Path]
ProgressCallback = Callable[[str, str], None]


class BatchResumeError(RuntimeError):
    """Raised when an existing batch run cannot be resumed safely."""


@dataclass(frozen=True)
class BatchProvenance:
    """Inputs that must stay identical across interrupted/resumed runs."""

    manifest_path: str
    manifest_sha256: str
    config_path: str
    config_sha256: str
    config: Dict[str, Any]
    backend: str
    model: str
    code_commit: Optional[str]


@dataclass(frozen=True)
class BatchSummary:
    total: int
    attempted: int
    success: int
    failure: int
    skipped: int
    output_path: str
    error_path: str
    resume: bool
    repaired_trailing_partial: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: PathLike) -> str:
    target = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_error_path(output_path: PathLike) -> Path:
    output = Path(output_path).expanduser().resolve(strict=False)
    if output.suffix:
        return output.with_suffix(".errors.jsonl")
    return output.with_name(output.name + ".errors.jsonl")


def run_metadata_path(output_path: PathLike) -> Path:
    output = Path(output_path).expanduser().resolve(strict=False)
    if output.suffix:
        return output.with_suffix(".run.json")
    return output.with_name(output.name + ".run.json")


def build_batch_provenance(
    *,
    manifest_path: PathLike,
    config_path: PathLike,
    config: Mapping[str, Any],
    backend: str,
    model: str,
    code_commit: Optional[str] = None,
) -> BatchProvenance:
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    config_file = Path(config_path).expanduser().resolve(strict=True)
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("backend must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if code_commit is not None and (not isinstance(code_commit, str) or not code_commit.strip()):
        raise ValueError("code_commit must be null or a non-empty string")

    return BatchProvenance(
        manifest_path=str(manifest),
        manifest_sha256=_sha256_file(manifest),
        config_path=str(config_file),
        config_sha256=_sha256_file(config_file),
        config=dict(config),
        backend=backend,
        model=model,
        code_commit=code_commit,
    )


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

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


def _expected_run_metadata(
    provenance: BatchProvenance,
    *,
    output_path: Path,
    error_path: Path,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_path": provenance.manifest_path,
        "manifest_sha256": provenance.manifest_sha256,
        "config_path": provenance.config_path,
        "config_sha256": provenance.config_sha256,
        "config": provenance.config,
        "backend": provenance.backend,
        "model": provenance.model,
        "code_commit": provenance.code_commit,
        "output_path": str(output_path),
        "error_path": str(error_path),
    }


def _validate_existing_run_metadata(
    metadata_path: Path,
    expected: Mapping[str, Any],
) -> None:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BatchResumeError(
            f"run metadata is invalid JSON: {metadata_path}: {exc.msg}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise BatchResumeError(f"run metadata must be a JSON object: {metadata_path}")

    for key, expected_value in expected.items():
        actual_value = payload.get(key)
        if actual_value != expected_value:
            raise BatchResumeError(
                f"resume metadata mismatch for {key!r}: "
                f"expected {expected_value!r}, found {actual_value!r}"
            )


def _initialize_run_artifacts(
    *,
    output_path: Path,
    error_path: Path,
    provenance: BatchProvenance,
    resume: bool,
) -> Path:
    metadata_path = run_metadata_path(output_path)
    expected = _expected_run_metadata(
        provenance,
        output_path=output_path,
        error_path=error_path,
    )

    existing = [path for path in (output_path, error_path, metadata_path) if path.exists()]

    if not resume:
        if existing:
            rendered = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                "batch output artifacts already exist; use --resume for the same run "
                f"or choose a new output path: {rendered}"
            )
        _atomic_write_json(
            {**expected, "created_at_utc": _utc_now()},
            metadata_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=False)
        return metadata_path

    if not existing:
        _atomic_write_json(
            {**expected, "created_at_utc": _utc_now()},
            metadata_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=False)
        return metadata_path

    if not metadata_path.is_file():
        raise BatchResumeError(
            "cannot safely resume because the run metadata sidecar is missing: "
            f"{metadata_path}"
        )

    _validate_existing_run_metadata(metadata_path, expected)
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=False)
    elif not output_path.is_file():
        raise BatchResumeError(f"batch output is not a regular file: {output_path}")

    return metadata_path


def _repair_trailing_partial_jsonl(path: Path) -> bool:
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False

    last_newline = data.rfind(b"\n")
    safe = data[: last_newline + 1] if last_newline >= 0 else b""
    with path.open("wb") as handle:
        handle.write(safe)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _load_completed_ids(
    output_path: Path,
    *,
    manifest_ids: Set[str],
    allow_repair: bool,
) -> Tuple[Set[str], bool]:
    repaired = False
    if allow_repair:
        repaired = _repair_trailing_partial_jsonl(output_path)

    completed: Set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchResumeError(
                    f"{output_path}: line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, Mapping):
                raise BatchResumeError(
                    f"{output_path}: line {line_number}: result record must be a JSON object"
                )

            segment_id = record.get("segment_id")
            if not isinstance(segment_id, str) or not segment_id.strip():
                raise BatchResumeError(
                    f"{output_path}: line {line_number}: missing/non-string segment_id"
                )
            if record.get("status") != "success":
                raise BatchResumeError(
                    f"{output_path}: line {line_number}, segment_id={segment_id!r}: "
                    "success output may contain only status='success' records"
                )
            if segment_id not in manifest_ids:
                raise BatchResumeError(
                    f"{output_path}: line {line_number}, segment_id={segment_id!r}: "
                    "segment is not present in the current manifest"
                )
            if segment_id in completed:
                raise BatchResumeError(
                    f"{output_path}: line {line_number}, segment_id={segment_id!r}: "
                    "duplicate completed segment"
                )
            completed.add(segment_id)

    return completed, repaired


def _append_jsonl_record(handle, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _success_record(
    *,
    item: Mapping[str, Any],
    audio_path: Path,
    audio_sec: float,
    sample_rate: int,
    channels: int,
    text_raw: str,
    language: Optional[str],
    decode_sec: float,
    provenance: BatchProvenance,
) -> Dict[str, Any]:
    record = dict(item)
    record.update(
        {
            "result_schema_version": 1,
            "status": "success",
            "segment_id": item["segment_id"],
            "audio_path": str(audio_path),
            "frontend": item.get("frontend"),
            "reference": item.get("reference"),
            "text_raw": text_raw,
            "text_norm": None,
            "reference_norm": None,
            "audio_sec": audio_sec,
            "decode_sec": decode_sec,
            "rtf": None,
            "backend": provenance.backend,
            "model": provenance.model,
            "language": language,
            "source_sample_rate": sample_rate,
            "source_channels": channels,
        }
    )
    return record


def _failure_record(
    *,
    item: Mapping[str, Any],
    audio_path: Path,
    exc: Exception,
    provenance: BatchProvenance,
) -> Dict[str, Any]:
    record = dict(item)
    record.update(
        {
            "result_schema_version": 1,
            "status": "failure",
            "segment_id": item["segment_id"],
            "audio_path": str(audio_path),
            "frontend": item.get("frontend"),
            "backend": provenance.backend,
            "model": provenance.model,
            "stage": "batch_decode",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "timestamp_utc": _utc_now(),
        }
    )
    return record


def run_batch(
    *,
    manifest_path: PathLike,
    output_path: PathLike,
    backend_factory: Callable[[], ASRBackend],
    provenance: BatchProvenance,
    resume: bool = False,
    error_path: Optional[PathLike] = None,
    progress: Optional[ProgressCallback] = None,
) -> BatchSummary:
    """Decode a manifest sequentially with durable per-record writes and resume.

    Structural manifest errors abort before decoding. Per-segment audio/backend
    errors are recorded to the error JSONL and decoding continues. Successful
    records are the only records written to the main output JSONL; therefore
    resume skips prior successes while retrying prior failures.
    """

    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve(strict=False)
    errors = (
        Path(error_path).expanduser().resolve(strict=False)
        if error_path is not None
        else default_error_path(output)
    )
    if output == errors:
        raise ValueError("output_path and error_path must be different")

    items = read_manifest(manifest, require_audio_exists=False)
    manifest_ids = {item["segment_id"] for item in items}

    _initialize_run_artifacts(
        output_path=output,
        error_path=errors,
        provenance=provenance,
        resume=resume,
    )
    completed, repaired = _load_completed_ids(
        output,
        manifest_ids=manifest_ids,
        allow_repair=resume,
    )

    backend: Optional[ASRBackend] = None
    if len(completed) < len(items):
        backend = backend_factory()

    attempted = 0
    success = 0
    failure = 0
    skipped = 0
    error_handle = None

    with output.open("a", encoding="utf-8") as success_handle:
        try:
            for item in items:
                segment_id = item["segment_id"]
                if segment_id in completed:
                    skipped += 1
                    if progress is not None:
                        progress("skip", segment_id)
                    continue

                attempted += 1
                audio_path = resolve_audio_path(item, base_dir=manifest.parent)

                try:
                    metadata = probe_audio(audio_path)
                    assert backend is not None
                    result = backend.transcribe(audio_path)
                    record = _success_record(
                        item=item,
                        audio_path=audio_path,
                        audio_sec=metadata.duration_sec,
                        sample_rate=metadata.sample_rate,
                        channels=metadata.channels,
                        text_raw=result.text,
                        language=result.language,
                        decode_sec=result.decode_sec,
                        provenance=provenance,
                    )
                    _append_jsonl_record(success_handle, record)
                    success += 1
                    if progress is not None:
                        progress("success", segment_id)
                except Exception as exc:
                    if error_handle is None:
                        errors.parent.mkdir(parents=True, exist_ok=True)
                        error_handle = errors.open("a", encoding="utf-8")
                    _append_jsonl_record(
                        error_handle,
                        _failure_record(
                            item=item,
                            audio_path=audio_path,
                            exc=exc,
                            provenance=provenance,
                        ),
                    )
                    failure += 1
                    if progress is not None:
                        progress("failure", segment_id)
        finally:
            if error_handle is not None:
                error_handle.close()

    return BatchSummary(
        total=len(items),
        attempted=attempted,
        success=success,
        failure=failure,
        skipped=skipped,
        output_path=str(output),
        error_path=str(errors),
        resume=resume,
        repaired_trailing_partial=repaired,
    )
