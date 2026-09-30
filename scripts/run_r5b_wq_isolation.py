#!/usr/bin/env python3
"""Run the R5B-WQ WeNet quality-isolation evidence gates.

This is an evidence script, not a replay-runtime variant.  It reads the
canonical PCM32LE raw_ch0 source independently, compares it with the existing
source packet path, then exercises the same qualified native API on the exact
segment PCM16 payloads.  References are read only for the Q2/Q1 scoring steps.
"""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
import resource
import struct
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_heard.asr.scoring import score_result_record
from robot_heard.misp.kaldi_adapter import read_kaldi_text
from robot_heard.misp.timeline import read_stage2_segment_samples
from robot_heard.replay.wav_source import MultiChannelPcmWavSource
from robot_heard.streaming.adapters.wenet_u2pp import (
    CtypesWeNetDecoderFactory,
    DEFAULT_MODEL_CHUNK_SAMPLES,
    pcm32_to_pcm16le,
)


RECORDING_ID = "M028_S197199201241_F8N_Far"
SOURCE_WAV = Path(
    "/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename/"
    "M028_S197199201241_F8N_Far_0.wav"
)
SEGMENTS_PATH = Path(
    "/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/segments"
)
TEXT_PATH = Path(
    "/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/text"
)
R5B_SCORED_PATH = Path(
    "/home/gc6_challenge/work/realtime_replay/r5b_wenet_full/r5b_scored.jsonl"
)
SNAPSHOT_PATH = Path(
    "/home/gc6_challenge/work/realtime_replay/r5b_wenet_full/r5b_snapshots.jsonl"
)
LIBRARY_PATH = Path(
    "/home/gc6_challenge/code/native_backend_qualification/wenet/runtime/"
    "libtorch/build2/api/libwenet_api.so"
)
MODEL_DIR = Path(
    "/home/gc6_challenge/models/native_streaming/wenet/"
    "20210601_u2++_conformer_libtorch"
)
OUTPUT_ROOT = Path("/home/gc6_challenge/work/realtime_replay/r5b_wq")
WENET_REVISION = "d17059667d6afe0680d19b3a4948ab825ef25105"
MODEL_ARCHIVE_SHA256 = "ace400d74a0eb073b0ee284e0c21be18796a7ab095a8b013d0e703e0180b1968"
LIBTORCH_SHA256 = "e1f6bc48403022ff4680c7299cc8b160df146892c414b8a6b6f7d5aff65bcbce"
SAMPLE_RATE = 16000
SOURCE_PACKET_SAMPLES = 320
MODEL_CHUNK_SAMPLES = DEFAULT_MODEL_CHUNK_SAMPLES


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _pack_pcm32(values: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def _read_independent_pcm32(start: int, end: int) -> Tuple[bytes, Tuple[int, ...]]:
    with wave.open(str(SOURCE_WAV), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != SAMPLE_RATE:
            raise RuntimeError("raw_ch0 independent reader saw unexpected WAV metadata")
        if handle.getsampwidth() != 4 or handle.getcomptype() != "NONE":
            raise RuntimeError("raw_ch0 independent reader requires PCM32LE")
        handle.setpos(start)
        payload = handle.readframes(end - start)
    values = struct.unpack(f"<{end - start}i", payload)
    return payload, values


def _reference_pcm16(start: int, end: int) -> bytes:
    values, rate = sf.read(
        str(SOURCE_WAV),
        start=start,
        stop=end,
        dtype="int16",
        always_2d=False,
    )
    if rate != SAMPLE_RATE:
        raise RuntimeError(f"soundfile reference rate mismatch: {rate}")
    array = np.asarray(values, dtype="<i2")
    if array.ndim != 1 or len(array) != end - start:
        raise RuntimeError("soundfile reference conversion returned wrong frame count")
    return array.tobytes()


def _stats(raw: np.ndarray, converted: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
    raw64 = np.asarray(raw, dtype=np.int64)
    pcm16 = np.asarray(converted, dtype=np.int64)
    ref16 = np.asarray(reference, dtype=np.int64)

    def basic(values: np.ndarray) -> Dict[str, Any]:
        nonzero = values[values != 0]
        return {
            "count": int(values.size),
            "min": int(values.min()) if values.size else None,
            "max": int(values.max()) if values.size else None,
            "peak_abs": int(np.max(np.abs(values))) if values.size else None,
            "rms": float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
            if values.size
            else None,
            "zero_sample_ratio": float(np.mean(values == 0)) if values.size else None,
            "nonzero_count": int(nonzero.size),
            "max_abs_bit_length": (
                int(int(np.max(np.abs(nonzero))).bit_length()) if nonzero.size else 0
            ),
        }

    raw_low16 = np.bitwise_and(raw64, 0xFFFF)
    pcm16_lsb = np.bitwise_and(pcm16, 1)
    result = {
        "pcm32": basic(raw64),
        "pcm16_shift": basic(pcm16),
        "pcm16_soundfile_reference": basic(ref16),
        "raw_nonzero_low16_ratio": float(np.mean(raw_low16 != 0)) if raw64.size else None,
        "pcm16_nonzero_lsb_ratio": float(np.mean(pcm16_lsb != 0)) if pcm16.size else None,
        "reference_max_abs_difference": int(np.max(np.abs(pcm16 - ref16))) if pcm16.size else 0,
        "reference_rms_difference": float(
            np.sqrt(np.mean(np.square(pcm16 - ref16, dtype=np.float64)))
        )
        if pcm16.size
        else 0.0,
        "reference_exact_equal": bool(np.array_equal(pcm16, ref16)),
    }
    return result


class _SegmentHashState:
    def __init__(self) -> None:
        self.raw = hashlib.sha256()
        self.pcm16 = hashlib.sha256()
        self.count = 0
        self.first = None
        self.last = None

    def update(self, values: Sequence[int]) -> None:
        if not values:
            return
        raw = _pack_pcm32(values)
        pcm16 = pcm32_to_pcm16le(values)
        self.raw.update(raw)
        self.pcm16.update(pcm16)
        self.count += len(values)
        if self.first is None:
            self.first = int(values[0])
        self.last = int(values[-1])


def _continuous_segment_hashes(segments: Sequence[Any]) -> Dict[str, _SegmentHashState]:
    source = MultiChannelPcmWavSource(
        [SOURCE_WAV],
        channel_ids=["ch0"],
        stream_id=RECORDING_ID,
        packet_samples=SOURCE_PACKET_SAMPLES,
    )
    states = {row.segment_id: _SegmentHashState() for row in segments}
    for packet in source:
        values = packet.samples[0]
        for row in segments:
            overlap_start = max(packet.start_sample, row.start_sample)
            overlap_end = min(packet.end_sample, row.end_sample)
            if overlap_start >= overlap_end:
                continue
            states[row.segment_id].update(
                values[overlap_start - packet.start_sample : overlap_end - packet.start_sample]
            )
    return states


def _q0_parity_and_conversion(segments: Sequence[Any]) -> Dict[str, Any]:
    continuous = _continuous_segment_hashes(segments)
    rows = []
    for row in segments:
        pcm32_payload, values = _read_independent_pcm32(row.start_sample, row.end_sample)
        pcm16_payload = pcm32_to_pcm16le(values)
        state = continuous[row.segment_id]
        independent_reference = _reference_pcm16(row.start_sample, row.end_sample)
        rows.append(
            {
                "segment_id": row.segment_id,
                "start_sample": row.start_sample,
                "end_sample": row.end_sample,
                "sample_count": row.frame_count,
                "continuous_sample_count": state.count,
                "independent_first_sample": int(values[0]),
                "independent_last_sample": int(values[-1]),
                "continuous_first_sample": state.first,
                "continuous_last_sample": state.last,
                "independent_pcm32_sha256": _sha256(pcm32_payload),
                "continuous_pcm32_sha256": state.raw.hexdigest(),
                "independent_pcm16_sha256": _sha256(pcm16_payload),
                "continuous_pcm16_sha256": state.pcm16.hexdigest(),
                "soundfile_reference_pcm16_sha256": _sha256(independent_reference),
                "pcm32_parity": (
                    state.count == row.frame_count
                    and state.first == int(values[0])
                    and state.last == int(values[-1])
                    and state.raw.hexdigest() == _sha256(pcm32_payload)
                ),
                "pcm16_parity": (
                    state.pcm16.hexdigest() == _sha256(pcm16_payload)
                    and _sha256(pcm16_payload) == _sha256(independent_reference)
                ),
            }
        )

    selected = [segments[0], segments[len(segments) // 2], segments[-1]]
    selected.extend(row for row in segments if "005620-005708" in row.segment_id)
    selected_ids = []
    stats = {}
    for row in selected:
        if row.segment_id in selected_ids:
            continue
        selected_ids.append(row.segment_id)
        _, values = _read_independent_pcm32(row.start_sample, row.end_sample)
        raw = np.asarray(values, dtype=np.int64)
        shifted = (raw >> 16).clip(-32768, 32767).astype(np.int16)
        reference = np.frombuffer(
            _reference_pcm16(row.start_sample, row.end_sample), dtype="<i2"
        )
        stats[row.segment_id] = _stats(raw, shifted, reference)

    _, full_values = _read_independent_pcm32(0, 480000)
    full_raw = np.asarray(full_values, dtype=np.int64)
    full_shift = (full_raw >> 16).clip(-32768, 32767).astype(np.int16)
    full_ref = np.frombuffer(_reference_pcm16(0, 480000), dtype="<i2")
    stats["first_30_seconds"] = _stats(full_raw, full_shift, full_ref)

    all_pcm32 = all(row["pcm32_parity"] for row in rows)
    all_pcm16 = all(row["pcm16_parity"] for row in rows)
    return {
        "conversion_policy": "signed PCM32LE arithmetic right shift 16, saturation to signed PCM16LE",
        "independent_reference": {
            "library": "soundfile/libsndfile",
            "operation": "read(dtype=int16), no normalization, no resampling, no downmix",
            "sample_rate": SAMPLE_RATE,
        },
        "representative_stats": stats,
        "segments": rows,
        "all_144_pcm32_parity": all_pcm32,
        "all_144_pcm16_parity": all_pcm16,
        "all_144_exact_parity": all_pcm32 and all_pcm16,
        "input_conversion_defect": not all_pcm16,
    }


def _load_harness_finals() -> Dict[str, Dict[str, Any]]:
    return {
        row["segment_id"]: row
        for row in _read_jsonl(SNAPSHOT_PATH)
        if row.get("kind") == "FINAL"
    }


def _run_native_streaming_parity(
    segments: Sequence[Any], pcm16_by_id: Mapping[str, bytes]
) -> Dict[str, Any]:
    harness = _load_harness_finals()
    factory = CtypesWeNetDecoderFactory(
        LIBRARY_PATH, MODEL_DIR, wenet_chunk_size=16, language="chs"
    )
    load_start = time.perf_counter()
    decoder = factory()
    load_sec = time.perf_counter() - load_start
    rows = []
    for segment in segments:
        payload = pcm16_by_id[segment.segment_id]
        result = None
        calls = 0
        start = time.perf_counter()
        for offset in range(0, len(payload), MODEL_CHUNK_SAMPLES * 2):
            chunk = payload[offset : offset + MODEL_CHUNK_SAMPLES * 2]
            last = offset + len(chunk) == len(payload)
            result = decoder.decode(chunk, last=last)
            calls += 1
        elapsed = time.perf_counter() - start
        if result is None or result.result_type != "final_result":
            raise RuntimeError(f"streaming parity did not return FINAL: {segment.segment_id}")
        decoder.reset()
        expected = harness.get(segment.segment_id)
        if expected is None:
            raise RuntimeError(f"missing harness FINAL: {segment.segment_id}")
        rows.append(
            {
                "segment_id": segment.segment_id,
                "harness_text": expected["text_raw"],
                "standalone_streaming_text": result.text,
                "exact_text_equal": expected["text_raw"] == result.text,
                "harness_empty": expected["text_raw"] == "",
                "standalone_empty": result.text == "",
                "empty_equal": (expected["text_raw"] == "") == (result.text == ""),
                "decoder_calls": calls,
                "decode_sec": elapsed,
            }
        )
    decoder.close()
    return {
        "native_api": "official wenet_init/wenet_decode/wenet_reset, chunk_size=16",
        "model_chunk_samples": MODEL_CHUNK_SAMPLES,
        "model_load_sec": load_sec,
        "segments": rows,
        "exact_final_text_equal_count": sum(row["exact_text_equal"] for row in rows),
        "empty_id_equal_count": sum(row["empty_equal"] for row in rows),
        "all_exact_final_text_equal": all(row["exact_text_equal"] for row in rows),
        "all_empty_ids_equal": all(row["empty_equal"] for row in rows),
        "adapter_native_parity_defect": not all(
            row["exact_text_equal"] and row["empty_equal"] for row in rows
        ),
    }


def _q2_decomposition() -> Dict[str, Any]:
    rows = _read_jsonl(R5B_SCORED_PATH)
    if len(rows) != 144:
        raise RuntimeError(f"Q2 expected 144 scored rows, got {len(rows)}")
    per_segment = []
    for row in rows:
        reference = row["reference_norm"]
        hypothesis = row["text_norm"]
        per_segment.append(
            {
                "segment_id": row["segment_id"],
                "duration_sec": row["audio_sec"],
                "reference_length": len(reference),
                "hypothesis_length": len(hypothesis),
                "cer": row["cer"],
                "s": row["cer_s"],
                "d": row["cer_d"],
                "i": row["cer_i"],
                "n": row["cer_n"],
                "empty": hypothesis == "",
                "reference_norm": reference,
                "hypothesis_norm": hypothesis,
            }
        )
    s = sum(row["s"] for row in per_segment)
    d = sum(row["d"] for row in per_segment)
    i = sum(row["i"] for row in per_segment)
    n = sum(row["n"] for row in per_segment)
    cer_values = np.asarray([row["cer"] for row in per_segment], dtype=np.float64)

    def correlation(key: str) -> float | None:
        x = np.asarray([row[key] for row in per_segment], dtype=np.float64)
        y = cer_values
        if np.std(x) == 0 or np.std(y) == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    def rank_quartiles(key: str) -> List[Dict[str, Any]]:
        ordered = sorted(per_segment, key=lambda row: (row[key], row["segment_id"]))
        result = []
        for index in range(4):
            group = ordered[index * 36 : (index + 1) * 36]
            result.append(
                {
                    "quartile": index + 1,
                    "count": len(group),
                    "key_min": min(row[key] for row in group),
                    "key_max": max(row[key] for row in group),
                    "mean_key": float(np.mean([row[key] for row in group])),
                    "mean_cer": float(np.mean([row["cer"] for row in group])),
                    "median_cer": float(np.median([row["cer"] for row in group])),
                }
            )
        return result

    empty_ids = [row["segment_id"] for row in per_segment if row["empty"]]
    worst = sorted(
        per_segment,
        key=lambda row: (-row["cer"], -row["n"], row["segment_id"]),
    )[:20]
    return {
        "source": str(R5B_SCORED_PATH),
        "normalization_policy": rows[0]["normalization_policy"],
        "global": {
            "s": s,
            "d": d,
            "i": i,
            "n": n,
            "errors": s + d + i,
            "cer": (s + d + i) / n if n else None,
            "empty_count": len(empty_ids),
            "non_empty_count": len(per_segment) - len(empty_ids),
        },
        "cer_percentiles": {
            "p50": float(np.percentile(cer_values, 50)),
            "p75": float(np.percentile(cer_values, 75)),
            "p90": float(np.percentile(cer_values, 90)),
            "p95": float(np.percentile(cer_values, 95)),
            "method": "numpy linear percentile over per-segment CER",
        },
        "relationship": {
            "duration_sec_vs_cer_pearson": correlation("duration_sec"),
            "reference_length_vs_cer_pearson": correlation("reference_length"),
            "duration_rank_quartiles": rank_quartiles("duration_sec"),
            "reference_length_rank_quartiles": rank_quartiles("reference_length"),
        },
        "empty_segment_ids": empty_ids,
        "worst_20": worst,
        "per_segment": per_segment,
    }


def _score_full_context(
    segments: Sequence[Any], pcm16_by_id: Mapping[str, bytes], references: Mapping[str, str]
) -> Dict[str, Any]:
    factory = CtypesWeNetDecoderFactory(
        LIBRARY_PATH, MODEL_DIR, wenet_chunk_size=-1, language="chs"
    )
    load_start = time.perf_counter()
    decoder = factory()
    load_sec = time.perf_counter() - load_start
    scored = []
    native_rows = []
    for segment in segments:
        payload = pcm16_by_id[segment.segment_id]
        start = time.perf_counter()
        result = decoder.decode(payload, last=True)
        elapsed = time.perf_counter() - start
        if result.result_type != "final_result":
            raise RuntimeError(f"full-context call did not return FINAL: {segment.segment_id}")
        native_rows.append(
            {
                "segment_id": segment.segment_id,
                "text_raw": result.text,
                "decode_sec": elapsed,
            }
        )
        scored.append(
            score_result_record(
                {
                    "status": "success",
                    "segment_id": segment.segment_id,
                    "text_raw": result.text,
                    "reference": references[segment.segment_id],
                    "audio_sec": segment.frame_count / SAMPLE_RATE,
                    "decode_sec": elapsed,
                }
            )
        )
        decoder.reset()
    decoder.close()
    s = sum(row["cer_s"] for row in scored)
    d = sum(row["cer_d"] for row in scored)
    i = sum(row["cer_i"] for row in scored)
    n = sum(row["cer_n"] for row in scored)
    return {
        "configuration": {
            "same_checkpoint": True,
            "same_pcm16_segment_payloads": True,
            "same_scoring": True,
            "official_wenet_chunk_size": -1,
            "mode": "official full/non-streaming context diagnostic",
            "support_evidence": [
                "runtime/core/api/wenet_api.h: -1 for non-streaming decoding",
                "runtime/libtorch/README.md: decoder_main --chunk_size -1",
                "docs/runtime.md: infinite chunk size is non-streaming",
            ],
        },
        "model_load_sec": load_sec,
        "cpu_memory_peak_bytes_process": _rss_bytes(),
        "global": {
            "s": s,
            "d": d,
            "i": i,
            "n": n,
            "errors": s + d + i,
            "cer": (s + d + i) / n if n else None,
            "empty_count": sum(row["text_raw"] == "" for row in native_rows),
            "non_empty_count": sum(row["text_raw"] != "" for row in native_rows),
            "rtf": sum(row["decode_sec"] for row in native_rows)
            / sum(segment.frame_count for segment in segments)
            * SAMPLE_RATE,
        },
        "per_segment": scored,
    }


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite existing Q0/Q1/Q2 output: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    with wave.open(str(SOURCE_WAV), "rb") as handle:
        total_samples = handle.getnframes()
    segments = read_stage2_segment_samples(
        SEGMENTS_PATH,
        recording_id=RECORDING_ID,
        sample_rate=SAMPLE_RATE,
        source_total_samples=total_samples,
    )
    if len(segments) != 144:
        raise RuntimeError(f"expected 144 Stage-2 segments, got {len(segments)}")
    q0 = _q0_parity_and_conversion(segments)
    if not q0["all_144_exact_parity"]:
        raise SystemExit("HARNESS_INPUT_PARITY_DEFECT")

    pcm16_by_id = {}
    for segment in segments:
        _, values = _read_independent_pcm32(segment.start_sample, segment.end_sample)
        pcm16_by_id[segment.segment_id] = pcm32_to_pcm16le(values)

    native_parity = _run_native_streaming_parity(segments, pcm16_by_id)
    if native_parity["adapter_native_parity_defect"]:
        raise SystemExit("ADAPTER_NATIVE_PARITY_DEFECT")

    q2 = _q2_decomposition()
    references = read_kaldi_text(TEXT_PATH, required=True)
    q1 = _score_full_context(segments, pcm16_by_id, references)
    result = {
        "schema_version": "r5b.wq.isolation.v1",
        "recording_id": RECORDING_ID,
        "source_wav": str(SOURCE_WAV),
        "stage2_segments": str(SEGMENTS_PATH),
        "source_packet_samples": SOURCE_PACKET_SAMPLES,
        "model_chunk_samples": MODEL_CHUNK_SAMPLES,
        "wenet_revision": WENET_REVISION,
        "model_archive_sha256": MODEL_ARCHIVE_SHA256,
        "libtorch_cpu_runtime_sha256": LIBTORCH_SHA256,
        "native_library": str(LIBRARY_PATH),
        "model_dir": str(MODEL_DIR),
        "q0": q0,
        "native_decode_parity": native_parity,
        "q2": q2,
        "q1_full_context": q1,
    }
    (OUTPUT_ROOT / "r5b_wq_isolation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(OUTPUT_ROOT / "r5b_wq_isolation.json"),
        "q0_exact_parity": q0["all_144_exact_parity"],
        "native_parity": native_parity["all_exact_final_text_equal"],
        "streaming_cer": q2["global"]["cer"],
        "streaming_sdi": [q2["global"][key] for key in ("s", "d", "i")],
        "streaming_empty": q2["global"]["empty_count"],
        "full_context_cer": q1["global"]["cer"],
        "full_context_sdi": [q1["global"][key] for key in ("s", "d", "i")],
        "full_context_empty": q1["global"]["empty_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
