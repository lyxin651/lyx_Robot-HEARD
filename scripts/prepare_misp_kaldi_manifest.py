#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional

from robot_heard.misp.kaldi_adapter import prepare_misp_kaldi_recording


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
            "Materialize one real MISP 2025 AVSR Stage-2 Kaldi recording as "
            "explicit mono segment WAVs plus a Whisper manifest."
        )
    )
    parser.add_argument("--kaldi-dir", required=True, type=Path)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument(
        "--channel",
        required=True,
        type=int,
        dest="channel_id",
        help="Explicit AVSR raw channel ID in 0..7; channels are never averaged.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Deterministically emit only the first N segments for a smoke run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing adapter output only after a complete new build "
            "succeeds."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    summary = prepare_misp_kaldi_recording(
        args.kaldi_dir,
        args.recording_id,
        args.output_dir,
        channel_id=args.channel_id,
        limit=args.limit,
        overwrite=args.overwrite,
        code_commit=_git_head(repo_root),
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
