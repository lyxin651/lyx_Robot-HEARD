#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional

from robot_heard.misp.s6i_remediation import write_s6i_remediated_jsonl


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
            "Build an auditable S6I-derived artifact from frozen primary results "
            "and optional empty-only retry diagnostics."
        )
    )
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--retry", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    metadata = write_s6i_remediated_jsonl(
        args.primary,
        args.output,
        retry_path=args.retry,
        metadata_path=args.metadata,
        code_commit=_git_head(repo_root),
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
