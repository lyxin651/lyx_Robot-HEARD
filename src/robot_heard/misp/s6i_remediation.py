"""Explicit S6I compatibility remediation for frozen Whisper evidence.

This module is deliberately outside :mod:`robot_heard.asr.openai_whisper`.
It consumes already-produced default Whisper records and optional diagnostic
retry records.  The default ``text_raw`` evidence is never changed.

The derived fields are:

``text_recognition``
    The selected recognition hypothesis: the frozen primary hypothesis unless
    and only unless that hypothesis is empty and a matching retry exists.

``text_submission``
    The deterministic MISP Task-2 Chinese-only surface after the narrow
    integer-surface policy.  It is ``None`` when strict surface validation
    fails, so an exporter cannot silently bypass the contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .task2_export import MISPTask2ExportError, normalize_misp2025_task2_text

PathLike = Union[str, Path]

MISP2025_S6I_REMEDIATION_POLICY = "misp2025_s6i_empty_retry_numeric_surface_v1"
MISP2025_S6I_RETRY_POLICY = "gss_default_whisper_then_empty_only_no_speech_none_v1"
MISP2025_S6I_NUMERIC_SURFACE_POLICY = "ascii_integer_0_99_zh_cardinal_v1"

_DIGIT_RUN = re.compile(r"[0-9]+")
_DIGITS = "零一二三四五六七八九"
_NUMERIC_PUNCTUATION = frozenset("+-.,/%:/")


class S6IRemediationError(ValueError):
    """Raised when a derived S6I artifact violates its input contract."""


@dataclass(frozen=True)
class NumericReplacement:
    """One reference-independent ASCII integer replacement."""

    source: str
    replacement: str
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NumericSurfaceResult:
    """Result of deterministic numeric-surface normalization."""

    text: str
    policy: str
    changed: bool
    replacements: Tuple[NumericReplacement, ...]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["replacements"] = [item.to_dict() for item in self.replacements]
        return payload


def _is_ascii_alphanumeric(char: Optional[str]) -> bool:
    return bool(char) and ("A" <= char <= "Z" or "a" <= char <= "z" or "0" <= char <= "9")


def _is_numeric_like_context(text: str, start: int, end: int) -> bool:
    """Reject mixed/numeric expressions without interpreting their meaning."""

    left = text[start - 1] if start else None
    right = text[end] if end < len(text) else None
    if _is_ascii_alphanumeric(left) or _is_ascii_alphanumeric(right):
        return True
    if left in _NUMERIC_PUNCTUATION or right in _NUMERIC_PUNCTUATION:
        return True
    return False


def _integer_to_zh(value: int) -> str:
    if value < 0 or value > 99:
        raise ValueError("value must be in 0..99")
    if value < 10:
        return _DIGITS[value]
    if value < 20:
        return "十" if value == 10 else "十一" if value == 11 else "十" + _DIGITS[value % 10]
    tens, ones = divmod(value, 10)
    return _DIGITS[tens] + "十" + (_DIGITS[ones] if ones else "")


def normalize_misp2025_s6i_numeric_surface(text: str) -> NumericSurfaceResult:
    """Convert only safe standalone ASCII integers in ``0..99``.

    This function does not use references or ASR semantics.  It preserves all
    non-digit characters byte-for-byte at the Python string level.  A digit
    run is left untouched when it is mixed with ASCII letters/digits, adjacent
    to numeric punctuation, has a leading zero in a multi-digit run, or is
    outside ``0..99``.  The strict Task-2 validator subsequently rejects any
    unsupported digits that remain.
    """

    if not isinstance(text, str):
        raise S6IRemediationError("numeric surface input must be a string")

    replacements: List[NumericReplacement] = []
    pieces: List[str] = []
    cursor = 0
    for match in _DIGIT_RUN.finditer(text):
        pieces.append(text[cursor : match.start()])
        token = match.group(0)
        replacement = token
        if not _is_numeric_like_context(text, match.start(), match.end()):
            if not (len(token) > 1 and token.startswith("0")):
                value = int(token)
                if 0 <= value <= 99:
                    replacement = _integer_to_zh(value)
                    replacements.append(
                        NumericReplacement(
                            source=token,
                            replacement=replacement,
                            start=match.start(),
                            end=match.end(),
                        )
                    )
        pieces.append(replacement)
        cursor = match.end()
    pieces.append(text[cursor:])
    normalized = "".join(pieces)
    return NumericSurfaceResult(
        text=normalized,
        policy=MISP2025_S6I_NUMERIC_SURFACE_POLICY,
        changed=normalized != text,
        replacements=tuple(replacements),
    )


def _require_segment_id(record: Mapping[str, Any], *, label: str) -> str:
    segment_id = record.get("segment_id")
    if not isinstance(segment_id, str) or not segment_id.strip():
        raise S6IRemediationError(f"{label}: segment_id must be a non-empty string")
    return segment_id


def _require_primary_record(record: Mapping[str, Any]) -> str:
    segment_id = _require_segment_id(record, label="primary result")
    if record.get("status") != "success":
        raise S6IRemediationError(
            f"segment_id={segment_id!r}: primary result must have status='success'"
        )
    text_raw = record.get("text_raw")
    if not isinstance(text_raw, str):
        raise S6IRemediationError(
            f"segment_id={segment_id!r}: primary text_raw must be a string"
        )
    return segment_id


def _retry_text(retry: Mapping[str, Any], *, segment_id: str) -> str:
    retry_id = _require_segment_id(retry, label="retry result")
    if retry_id != segment_id:
        raise S6IRemediationError(
            f"retry segment_id={retry_id!r} does not match primary {segment_id!r}"
        )
    text = retry.get("retry_text_raw")
    if not isinstance(text, str):
        text = retry.get("text_raw")
    if not isinstance(text, str):
        raise S6IRemediationError(
            f"segment_id={segment_id!r}: retry text must be a string"
        )
    return text


def _submission_surface(text: str, *, segment_id: str) -> Tuple[Optional[str], Optional[str], NumericSurfaceResult]:
    numeric = normalize_misp2025_s6i_numeric_surface(text)
    try:
        submission = normalize_misp2025_task2_text(
            numeric.text,
            segment_id=segment_id,
        )
    except MISPTask2ExportError as exc:
        return None, str(exc), numeric
    return submission, None, numeric


def remediate_s6i_record(
    primary: Mapping[str, Any],
    *,
    retry: Optional[Mapping[str, Any]] = None,
    code_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Derive one S6I record without changing primary ASR evidence."""

    segment_id = _require_primary_record(primary)
    primary_text = primary["text_raw"]
    primary_empty = not primary_text.strip()
    if retry is not None and not primary_empty:
        raise S6IRemediationError(
            f"segment_id={segment_id!r}: non-empty primary is not eligible for retry"
        )

    retry_text: Optional[str] = None
    retry_triggered = primary_empty and retry is not None
    if retry is not None:
        retry_text = _retry_text(retry, segment_id=segment_id)

    if retry_triggered:
        recognition_text = retry_text or ""
        recognition_source = "gss_no_speech_retry"
    else:
        recognition_text = primary_text
        recognition_source = "gss_default"

    submission_text, invalid_reason, numeric = _submission_surface(
        recognition_text,
        segment_id=segment_id,
    )

    derived = dict(primary)
    derived.update(
        {
            "s6i_remediation_schema_version": 1,
            "text_raw": primary_text,
            "text_recognition": recognition_text,
            "text_submission": submission_text,
            "submission_valid": submission_text is not None,
            "submission_invalid_reason": invalid_reason,
            "s6i_remediation": {
                "policy": MISP2025_S6I_REMEDIATION_POLICY,
                "retry_policy": MISP2025_S6I_RETRY_POLICY,
                "retry_triggered": retry_triggered,
                "primary_empty": primary_empty,
                "recognition_source": recognition_source,
                "primary_text_raw": primary_text,
                "retry_text_raw": retry_text,
                "numeric_surface": numeric.to_dict(),
                "code_commit": code_commit,
                "reference_used_for_decision": False,
                "reference_role": "scoring_only",
                "retry_provenance": dict(retry) if retry is not None else None,
            },
        }
    )
    return derived


def build_s6i_remediated_records(
    primary_records: Iterable[Mapping[str, Any]],
    retry_records: Iterable[Mapping[str, Any]] = (),
    *,
    code_commit: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build an exact-order derived artifact from primary and retry records."""

    primary_list = list(primary_records)
    retry_list = list(retry_records)
    primary_map: Dict[str, Mapping[str, Any]] = {}
    for record in primary_list:
        segment_id = _require_primary_record(record)
        if segment_id in primary_map:
            raise S6IRemediationError(f"duplicate primary segment_id={segment_id!r}")
        primary_map[segment_id] = record

    retry_map: Dict[str, Mapping[str, Any]] = {}
    for record in retry_list:
        segment_id = _require_segment_id(record, label="retry result")
        if segment_id in retry_map:
            raise S6IRemediationError(f"duplicate retry segment_id={segment_id!r}")
        if segment_id not in primary_map:
            raise S6IRemediationError(
                f"retry segment_id={segment_id!r} is not in primary results"
            )
        retry_map[segment_id] = record

    output = [
        remediate_s6i_record(
            primary,
            retry=retry_map.get(primary["segment_id"]),
            code_commit=code_commit,
        )
        for primary in primary_list
    ]
    return output


def _sha256_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    records: List[Dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise S6IRemediationError(
                    f"{source}: line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, Mapping):
                raise S6IRemediationError(
                    f"{source}: line {line_number}: record must be an object"
                )
            records.append(dict(record))
    return records


def write_s6i_remediated_jsonl(
    primary_path: PathLike,
    output_path: PathLike,
    *,
    retry_path: Optional[PathLike] = None,
    metadata_path: Optional[PathLike] = None,
    code_commit: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Materialize derived records and provenance from JSONL artifacts."""

    primary = Path(primary_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve(strict=False)
    metadata_target = (
        Path(metadata_path).expanduser().resolve(strict=False)
        if metadata_path is not None
        else None
    )
    if primary == output:
        raise S6IRemediationError("primary and output paths must differ")
    if output.exists() and not overwrite:
        raise FileExistsError(f"remediated output already exists: {output}")
    if metadata_target is not None and metadata_target.exists() and not overwrite:
        raise FileExistsError(
            f"remediated metadata already exists: {metadata_target}"
        )
    retry = _read_jsonl(retry_path) if retry_path is not None else []
    records = build_s6i_remediated_records(
        _read_jsonl(primary), retry, code_commit=code_commit
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": MISP2025_S6I_REMEDIATION_POLICY,
        "retry_policy": MISP2025_S6I_RETRY_POLICY,
        "numeric_surface_policy": MISP2025_S6I_NUMERIC_SURFACE_POLICY,
        "primary_path": str(primary),
        "primary_sha256": _sha256_file(primary),
        "retry_path": str(Path(retry_path).expanduser().resolve()) if retry_path is not None else None,
        "retry_sha256": _sha256_file(retry_path) if retry_path is not None else None,
        "output_path": str(output),
        "output_sha256": _sha256_file(output),
        "records": len(records),
        "retry_records": len(retry),
        "empty_primary": sum(not row["text_raw"].strip() for row in records),
        "submission_valid": sum(bool(row["submission_valid"]) for row in records),
        "submission_invalid": sum(not row["submission_valid"] for row in records),
        "code_commit": code_commit,
        "reference_used_for_decision": False,
    }
    if metadata_target is not None:
        metadata_target.parent.mkdir(parents=True, exist_ok=True)
        metadata_target.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return metadata
