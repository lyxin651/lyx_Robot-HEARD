#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from robot_heard.asr.scoring import (
    score_results_jsonl,
    validate_normalization_policy,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"scoring config must contain a YAML mapping: {path}")
    unknown = set(config) - {"normalization_policy"}
    if unknown:
        raise ValueError(f"unsupported scoring config keys: {sorted(unknown)}")
    policy = config.get("normalization_policy")
    validate_normalization_policy(policy)
    return config


def _git_head(repo_root: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _default_summary_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(".metrics.json")
    return output_path.with_name(output_path.name + ".metrics.json")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize S4 ASR results, compute character-error counts/CER when a "
            "reference is present, and compute per-segment plus global RTF."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Metrics/provenance JSON path (default: derived from --output).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing scored output and summary atomically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve(strict=False)
    summary_path = (
        args.summary.expanduser().resolve(strict=False)
        if args.summary is not None
        else _default_summary_path(output_path)
    )

    if input_path == output_path:
        raise ValueError("--input and --output must differ")
    if output_path == summary_path:
        raise ValueError("--output and --summary must differ")
    if not args.overwrite:
        existing = [path for path in (output_path, summary_path) if path.exists()]
        if existing:
            rendered = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                "S5 output artifacts already exist; choose new paths or use "
                f"--overwrite: {rendered}"
            )

    config = _load_config(config_path)
    policy = config["normalization_policy"]
    summary = score_results_jsonl(
        input_path,
        output_path,
        normalization_policy=policy,
        overwrite=args.overwrite,
    )

    repo_root = Path(__file__).resolve().parents[1]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "output_path": str(output_path),
        "scoring_config_path": str(config_path),
        "scoring_config_sha256": _sha256_file(config_path),
        "scoring_config": config,
        "code_commit": _git_head(repo_root),
        **summary.to_dict(),
    }
    _atomic_write_json(payload, summary_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
