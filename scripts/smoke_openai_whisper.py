#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from robot_heard.asr.factory import create_asr_backend
from robot_heard.io.manifest import read_manifest, resolve_audio_path


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a YAML mapping: {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one OpenAI Whisper V0 transcription from a validated manifest."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based manifest record index to transcribe (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve(strict=False)
    config_path = args.config.expanduser().resolve(strict=False)

    items = read_manifest(manifest_path)
    if not items:
        raise ValueError(f"manifest contains no records: {manifest_path}")
    if args.index < 0 or args.index >= len(items):
        raise IndexError(
            f"--index {args.index} is out of range for manifest with {len(items)} records"
        )

    item = items[args.index]
    audio_path = resolve_audio_path(item, base_dir=manifest_path.parent)
    config = _load_config(config_path)

    backend = create_asr_backend(config)
    result = backend.transcribe(audio_path)

    payload = {
        "segment_id": item["segment_id"],
        "audio_path": str(audio_path),
        "text": result.text,
        "language": result.language,
        "decode_sec": result.decode_sec,
        "backend": config["backend"],
        "model": config["model"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
