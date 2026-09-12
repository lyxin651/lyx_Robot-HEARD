from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union


class ManifestValidationError(ValueError):
    """Raised when a manifest record violates the V0 contract."""


REQUIRED_FIELDS = ("segment_id", "audio_path")
OPTIONAL_STRING_FIELDS = ("session_id", "speaker_id", "frontend", "reference")


def resolve_audio_path(
    item: Mapping[str, Any],
    *,
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    raw_path = item.get("audio_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ManifestValidationError("audio_path must be a non-empty string")

    path = Path(raw_path).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path.resolve(strict=False)


def _record_label(item: Mapping[str, Any], line_number: Optional[int]) -> str:
    segment_id = item.get("segment_id", "<missing-segment_id>")
    if line_number is None:
        return f"segment_id={segment_id!r}"
    return f"line={line_number}, segment_id={segment_id!r}"


def validate_item(
    item: Mapping[str, Any],
    *,
    line_number: Optional[int] = None,
    base_dir: Optional[Union[str, Path]] = None,
    require_audio_exists: bool = True,
) -> None:
    if not isinstance(item, Mapping):
        raise ManifestValidationError(f"line={line_number}: manifest record must be a JSON object")

    label = _record_label(item, line_number)

    for field in REQUIRED_FIELDS:
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ManifestValidationError(f"{label}: {field} must be a non-empty string")

    for field in OPTIONAL_STRING_FIELDS:
        if field in item and item[field] is not None and not isinstance(item[field], str):
            raise ManifestValidationError(f"{label}: optional field {field} must be a string or null")

    if "start" in item and item["start"] is not None:
        start = item["start"]
        if isinstance(start, bool) or not isinstance(start, (int, float)) or start < 0:
            raise ManifestValidationError(f"{label}: start must be a non-negative number")

    if "duration" in item and item["duration"] is not None:
        duration = item["duration"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            raise ManifestValidationError(f"{label}: duration must be a positive number")

    if require_audio_exists:
        audio_path = resolve_audio_path(item, base_dir=base_dir)
        if not audio_path.is_file():
            raise ManifestValidationError(f"{label}: audio file does not exist: {audio_path}")


def validate_manifest(
    items: Iterable[Mapping[str, Any]],
    *,
    base_dir: Optional[Union[str, Path]] = None,
    require_audio_exists: bool = True,
) -> None:
    seen = set()
    for index, item in enumerate(items, start=1):
        validate_item(
            item,
            line_number=index,
            base_dir=base_dir,
            require_audio_exists=require_audio_exists,
        )
        segment_id = item["segment_id"]
        if segment_id in seen:
            raise ManifestValidationError(
                f"line={index}, segment_id={segment_id!r}: duplicate segment_id"
            )
        seen.add(segment_id)


def read_manifest(
    path: Union[str, Path],
    *,
    require_audio_exists: bool = True,
) -> List[Dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve(strict=False)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")

    items: List[Dict[str, Any]] = []
    seen = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestValidationError(
                    f"line={line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(item, dict):
                raise ManifestValidationError(
                    f"line={line_number}: manifest record must be a JSON object"
                )

            validate_item(
                item,
                line_number=line_number,
                base_dir=manifest_path.parent,
                require_audio_exists=require_audio_exists,
            )
            segment_id = item["segment_id"]
            if segment_id in seen:
                raise ManifestValidationError(
                    f"line={line_number}, segment_id={segment_id!r}: duplicate segment_id"
                )
            seen.add(segment_id)
            items.append(item)

    return items


def write_jsonl(items: Iterable[Mapping[str, Any]], path: Union[str, Path]) -> None:
    """Atomically write JSONL while preserving unknown metadata fields."""

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
