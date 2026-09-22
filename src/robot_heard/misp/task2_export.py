from __future__ import annotations

import hashlib
import json
import os
import unicodedata
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

PathLike = Union[str, Path]

MISP2025_TASK2_ARCHIVE_NAME = "summission.zip"
MISP2025_TASK2_TRANSCRIPT_NAME = "summission.txt"
MISP2025_TASK2_EXPORT_POLICY = "misp2025_task2_chinese_only_v1"
MISP2025_TASK2_FORMAT_URL = (
    "https://mispchallenge.github.io/mispchallenge2025/task2_software.html"
)

_HAN_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
    (0x30000, 0x323AF),
)


class MISPTask2ExportError(ValueError):
    """Raised when ASR results cannot be exported safely to MISP 2025 Task 2."""


@dataclass(frozen=True)
class MISPTask2ExportSummary:
    records: int
    input_path: str
    transcript_path: str
    archive_path: str
    metadata_path: str
    expected_segments_validated: bool
    expected_segments_path: Optional[str]
    text_policy: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: PathLike) -> str:
    target = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_han_character(char: str) -> bool:
    if len(char) != 1:
        return False
    codepoint = ord(char)
    if codepoint == 0x3007:
        return True
    return any(start <= codepoint <= end for start, end in _HAN_RANGES)


def normalize_misp2025_task2_text(
    text: str,
    *,
    segment_id: Optional[str] = None,
) -> str:
    """Normalize one transcript to the documented MISP 2025 Task 2 text surface.

    The official submission page requires Chinese characters only and no
    punctuation. This exporter removes Unicode whitespace/punctuation after
    NFKC normalization, but deliberately rejects any remaining non-Han
    character instead of silently deleting semantic content such as Latin
    letters or Arabic digits.
    """

    if not isinstance(text, str):
        label = f"segment_id={segment_id!r}: " if segment_id is not None else ""
        raise MISPTask2ExportError(f"{label}text_raw must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    retained: List[str] = []
    unsupported: List[str] = []

    for char in normalized:
        if char.isspace() or unicodedata.category(char).startswith("P"):
            continue
        if _is_han_character(char):
            retained.append(char)
        else:
            unsupported.append(char)

    label = f"segment_id={segment_id!r}: " if segment_id is not None else ""
    if unsupported:
        rendered = "".join(unsupported[:16])
        if len(unsupported) > 16:
            rendered += "..."
        raise MISPTask2ExportError(
            f"{label}MISP 2025 Task 2 submission permits Chinese characters only; "
            f"unsupported non-Han content remains after whitespace/punctuation removal: "
            f"{rendered!r}"
        )

    result = "".join(retained)
    if not result:
        raise MISPTask2ExportError(
            f"{label}submission transcript is empty after MISP Task 2 normalization"
        )
    return result


def _parse_json_result_line(
    raw_line: str,
    *,
    source: Path,
    line_number: int,
) -> Mapping[str, Any]:
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise MISPTask2ExportError(
            f"{source}: line {line_number}: invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(record, Mapping):
        raise MISPTask2ExportError(
            f"{source}: line {line_number}: result record must be a JSON object"
        )
    return record


def read_export_records(input_path: PathLike) -> List[Dict[str, Any]]:
    """Read validated S4/S5 success results for challenge export."""

    source = Path(input_path).expanduser().resolve(strict=True)
    records: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = _parse_json_result_line(
                line,
                source=source,
                line_number=line_number,
            )
            segment_id = record.get("segment_id")
            if not isinstance(segment_id, str) or not segment_id.strip():
                raise MISPTask2ExportError(
                    f"{source}: line {line_number}: segment_id must be a non-empty string"
                )
            if segment_id in seen:
                raise MISPTask2ExportError(
                    f"{source}: line {line_number}, segment_id={segment_id!r}: "
                    "duplicate result segment"
                )
            if record.get("status") != "success":
                raise MISPTask2ExportError(
                    f"{source}: line {line_number}, segment_id={segment_id!r}: "
                    "export accepts only status='success' result records"
                )
            text_raw = record.get("text_raw")
            if not isinstance(text_raw, str):
                raise MISPTask2ExportError(
                    f"{source}: line {line_number}, segment_id={segment_id!r}: "
                    "text_raw must be a string"
                )

            normalized_text = normalize_misp2025_task2_text(
                text_raw,
                segment_id=segment_id,
            )
            records.append(
                {
                    "segment_id": segment_id,
                    "text_raw": text_raw,
                    "submission_text": normalized_text,
                }
            )
            seen.add(segment_id)

    if not records:
        raise MISPTask2ExportError(f"{source}: no exportable result records")
    return records


def read_expected_segment_ids(path: PathLike) -> List[str]:
    """Read an authoritative expected-ID list.

    Supported line forms:
    - `segment_id`
    - `segment_id transcript...` (official-style transcript/example list)
    - JSON object containing `segment_id` (e.g. an integration manifest)
    """

    source = Path(path).expanduser().resolve(strict=True)
    ids: List[str] = []
    seen: Set[str] = set()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            segment_id: Any
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MISPTask2ExportError(
                        f"{source}: line {line_number}: invalid JSON expected-ID record: "
                        f"{exc.msg}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise MISPTask2ExportError(
                        f"{source}: line {line_number}: expected-ID JSON must be an object"
                    )
                segment_id = payload.get("segment_id")
            else:
                segment_id = line.split(maxsplit=1)[0]

            if not isinstance(segment_id, str) or not segment_id.strip():
                raise MISPTask2ExportError(
                    f"{source}: line {line_number}: expected segment_id must be non-empty"
                )
            if segment_id in seen:
                raise MISPTask2ExportError(
                    f"{source}: line {line_number}, segment_id={segment_id!r}: "
                    "duplicate expected segment"
                )
            seen.add(segment_id)
            ids.append(segment_id)

    if not ids:
        raise MISPTask2ExportError(f"{source}: expected segment list is empty")
    return ids


def validate_expected_segment_ids(
    records: Sequence[Mapping[str, Any]],
    expected_ids: Sequence[str],
) -> None:
    actual = {str(record["segment_id"]) for record in records}
    expected = set(expected_ids)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(
                f"missing={missing[:10]!r}" + ("..." if len(missing) > 10 else "")
            )
        if extra:
            details.append(
                f"extra={extra[:10]!r}" + ("..." if len(extra) > 10 else "")
            )
        raise MISPTask2ExportError(
            "result segment IDs do not match authoritative expected segments: "
            + "; ".join(details)
        )


def render_misp2025_task2_transcript(
    records: Sequence[Mapping[str, Any]],
) -> str:
    if not records:
        raise MISPTask2ExportError("cannot render an empty submission")
    lines = []
    for record in records:
        segment_id = record.get("segment_id")
        submission_text = record.get("submission_text")
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise MISPTask2ExportError("render record has invalid segment_id")
        if not isinstance(submission_text, str) or not submission_text:
            raise MISPTask2ExportError(
                f"segment_id={segment_id!r}: render record has invalid submission_text"
            )
        lines.append(f"{segment_id} {submission_text}\n")
    return "".join(lines)


def _deterministic_zip_bytes(transcript_bytes: bytes) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        info = zipfile.ZipInfo(
            filename=MISP2025_TASK2_TRANSCRIPT_NAME,
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, transcript_bytes)
    return buffer.getvalue()


def _atomic_write_bytes(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    data = (
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(data, path)


def export_misp2025_task2_submission(
    input_path: PathLike,
    output_dir: PathLike,
    *,
    expected_segments_path: Optional[PathLike] = None,
    overwrite: bool = False,
    code_commit: Optional[str] = None,
) -> MISPTask2ExportSummary:
    """Create the documented MISP 2025 Task 2 transcript + zip package.

    The submission zip contains exactly one root member named `summission.txt`,
    matching the official challenge page spelling. Export is derived solely
    from existing S4/S5 result JSONL and never invokes ASR inference.
    """

    source = Path(input_path).expanduser().resolve(strict=True)
    directory = Path(output_dir).expanduser().resolve(strict=False)
    transcript_path = directory / MISP2025_TASK2_TRANSCRIPT_NAME
    archive_path = directory / MISP2025_TASK2_ARCHIVE_NAME
    metadata_path = directory / "summission.export.json"

    existing = [
        path
        for path in (transcript_path, archive_path, metadata_path)
        if path.exists()
    ]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "MISP export artifacts already exist; choose a new output directory "
            f"or use overwrite=True: {rendered}"
        )

    records = read_export_records(source)

    expected_validated = False
    expected_path: Optional[Path] = None
    expected_sha256: Optional[str] = None
    if expected_segments_path is not None:
        expected_path = Path(expected_segments_path).expanduser().resolve(strict=True)
        expected_ids = read_expected_segment_ids(expected_path)
        validate_expected_segment_ids(records, expected_ids)
        expected_validated = True
        expected_sha256 = _sha256_file(expected_path)

    transcript_text = render_misp2025_task2_transcript(records)
    transcript_bytes = transcript_text.encode("utf-8")
    archive_bytes = _deterministic_zip_bytes(transcript_bytes)

    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(transcript_bytes, transcript_path)
    _atomic_write_bytes(archive_bytes, archive_path)

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "challenge": "MISP 2025",
        "task": "Task 2 AVSR",
        "official_format_url": MISP2025_TASK2_FORMAT_URL,
        "input_path": str(source),
        "input_sha256": _sha256_file(source),
        "records": len(records),
        "text_policy": MISP2025_TASK2_EXPORT_POLICY,
        "transcript_path": str(transcript_path),
        "transcript_member_name": MISP2025_TASK2_TRANSCRIPT_NAME,
        "transcript_sha256": _sha256_bytes(transcript_bytes),
        "archive_path": str(archive_path),
        "archive_sha256": _sha256_bytes(archive_bytes),
        "expected_segments_validated": expected_validated,
        "expected_segments_path": str(expected_path) if expected_path is not None else None,
        "expected_segments_sha256": expected_sha256,
        "code_commit": code_commit,
    }
    _atomic_write_json(payload, metadata_path)

    return MISPTask2ExportSummary(
        records=len(records),
        input_path=str(source),
        transcript_path=str(transcript_path),
        archive_path=str(archive_path),
        metadata_path=str(metadata_path),
        expected_segments_validated=expected_validated,
        expected_segments_path=(
            str(expected_path) if expected_path is not None else None
        ),
        text_policy=MISP2025_TASK2_EXPORT_POLICY,
    )
