#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from robot_heard.asr.batch import (
    build_batch_provenance,
    default_error_path,
    run_batch,
)
from robot_heard.asr.factory import create_asr_backend


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a YAML mapping: {path}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode a validated manifest sequentially with one ASR backend instance, "
            "durable JSONL writes, per-segment failure logging, and safe resume."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--errors",
        type=Path,
        default=None,
        help="Failure JSONL path (default: derived from --output).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the exact same manifest/config/code run and skip prior successes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve(strict=False)
    error_path = (
        args.errors.expanduser().resolve(strict=False)
        if args.errors is not None
        else default_error_path(output_path)
    )

    config = _load_config(config_path)
    backend_name = config.get("backend")
    model_name = config.get("model")
    if not isinstance(backend_name, str) or not backend_name.strip():
        raise ValueError("config key 'backend' must be a non-empty string")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("config key 'model' must be a non-empty string")

    repo_root = Path(__file__).resolve().parents[1]
    provenance = build_batch_provenance(
        manifest_path=manifest_path,
        config_path=config_path,
        config=config,
        backend=backend_name,
        model=model_name,
        code_commit=_git_head(repo_root),
    )

    def progress(event: str, segment_id: str) -> None:
        print(f"[{event}] {segment_id}", file=sys.stderr, flush=True)

    summary = run_batch(
        manifest_path=manifest_path,
        output_path=output_path,
        error_path=error_path,
        backend_factory=lambda: create_asr_backend(config),
        provenance=provenance,
        resume=args.resume,
        progress=progress,
    )

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    if summary.failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
