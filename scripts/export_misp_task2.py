#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional

from robot_heard.misp.task2_export import export_misp2025_task2_submission


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
            "Export existing ASR result JSONL to the documented MISP 2025 "
            "Task 2 / AVSR submission package."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-segments",
        type=Path,
        default=None,
        help=(
            "Authoritative expected segment list. Supports one ID per line, "
            "official-style 'segment_id transcript' lines, or manifest JSONL."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing S6 export artifacts in --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    summary = export_misp2025_task2_submission(
        args.input,
        args.output_dir,
        expected_segments_path=args.expected_segments,
        overwrite=args.overwrite,
        code_commit=_git_head(repo_root),
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
