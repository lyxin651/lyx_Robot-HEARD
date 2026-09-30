#!/usr/bin/env python3
"""Replay one mono PCM16 WAV through the qualified WeNet native API."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# The repository's package initializers have a legacy replay/streaming import
# cycle when this standalone script is the first importer.  Initialize replay
# first, as the harness runner does, without changing either package's runtime
# semantics.
import robot_heard.replay as _replay_package  # noqa: E402
del _replay_package

from robot_heard.streaming.adapters.wenet_u2pp import CtypesWeNetDecoderFactory


DEFAULT_LIBRARY = Path(
    "/home/gc6_challenge/code/native_backend_qualification/wenet/"
    "runtime/libtorch/build2/api/libwenet_api.so"
)
DEFAULT_MODEL_DIR = Path(
    "/home/gc6_challenge/models/native_streaming/wenet/"
    "20210601_u2++_conformer_libtorch"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--library-path", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--chunk-samples", type=int, default=320)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_samples <= 0:
        raise ValueError("chunk-samples must be positive")
    wav_path = args.wav.expanduser().resolve(strict=True)
    factory = CtypesWeNetDecoderFactory(
        args.library_path.expanduser().resolve(strict=True),
        args.model_dir.expanduser().resolve(strict=True),
        wenet_chunk_size=16,
        language="chs",
    )
    decoder = factory()
    partial_count = final_count = non_empty_partial = changed = repeated = 0
    previous_partial: Optional[str] = None
    previous_seen = False
    final_text = ""
    call_count = 0
    start = time.perf_counter()
    with wave.open(str(wav_path), "rb") as reader:
        if (
            reader.getnchannels(),
            reader.getframerate(),
            reader.getsampwidth(),
            reader.getcomptype(),
        ) != (1, 16000, 2, "NONE"):
            raise ValueError("smoke input must be mono 16 kHz PCM16LE WAV")
        total_samples = reader.getnframes()
        for offset in range(0, total_samples, args.chunk_samples):
            count = min(args.chunk_samples, total_samples - offset)
            pcm16le = reader.readframes(count)
            last = offset + count == total_samples
            result = decoder.decode(pcm16le, last=last)
            call_count += 1
            if result.result_type == "partial_result":
                partial_count += 1
                if result.text:
                    non_empty_partial += 1
                if previous_seen:
                    if result.text == previous_partial:
                        repeated += 1
                    else:
                        changed += 1
                previous_partial = result.text
                previous_seen = True
            else:
                final_count += 1
                final_text = result.text
    wall_sec = time.perf_counter() - start
    decoder.close()
    report = {
        "wav": str(wav_path),
        "audio_sec": total_samples / 16000,
        "chunk_samples": args.chunk_samples,
        "decoder_call_count": call_count,
        "partial_return_count": partial_count,
        "final_return_count": final_count,
        "non_empty_partial_count": non_empty_partial,
        "changed_text_count": changed,
        "repeated_identical_count": repeated,
        "wall_sec": wall_sec,
        "cpu_rtf": wall_sec / (total_samples / 16000),
        "final_text": final_text,
        "native_api": "official WeNet LibTorch C API",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if final_count == 1 and partial_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
