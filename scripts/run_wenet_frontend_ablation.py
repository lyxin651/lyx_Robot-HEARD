#!/usr/bin/env python3
"""Run the frozen WeNet frontend ablation on one deterministic MISP meeting.

This script composes the existing replay runtime and native WeNet adapter.  It
does not alter replay-core or adapter semantics.  The streaming frontend arm
is intentionally reported as not run when the server audit finds no qualified
true-streaming frontend.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import resource
import sys
import tempfile
import time
import wave
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_heard.asr.scoring import score_result_record
from robot_heard.misp.kaldi_adapter import read_kaldi_scp, read_kaldi_text, resolve_misp_channel_path
from robot_heard.misp.timeline import read_stage2_segment_samples
from robot_heard.replay.clock import MonotonicWallClock
from robot_heard.replay.metrics import compute_paced_metrics, compute_r5b_metrics, write_metrics_json
from robot_heard.replay.oracle import Stage2ScopeBoundaryReleaseGate
from robot_heard.replay.paced_runtime import PacedReplayRuntime
from robot_heard.replay.trace import TraceDurabilityPolicy, TraceWriter
from robot_heard.replay.wav_source import MultiChannelPcmWavSource
from robot_heard.streaming.adapters.wenet_u2pp import (
    CtypesWeNetDecoderFactory,
    DEFAULT_WENET_CHUNK_SIZE,
    WeNetU2PPStreamingConsumer,
    parse_native_result,
)


DEFAULT_LIBRARY = Path(
    "/home/gc6_challenge/code/native_backend_qualification/wenet/"
    "runtime/libtorch/build2/api/libwenet_api.so"
)
DEFAULT_MODEL_DIR = Path(
    "/home/gc6_challenge/models/native_streaming/wenet/"
    "wenetspeech/20220506_u2pp_conformer_libtorch"
)
MODEL_ARCHIVE_SHA256 = "cd82435483dfbcf8dc7385e10c72051be5488893f010ce7866b94600a1ccbca9"
WENET_REVISION = "d17059667d6afe0680d19b3a4948ab825ef25105"
MISP_RECORDING_RE = re.compile(r"^(M\d+)_")
SEGMENT_RE = re.compile(r"^(S\d+)_.*_(\d+-\d+)$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _git_head() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
    ).strip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _recording_label(recording_id: str) -> str:
    match = MISP_RECORDING_RE.match(recording_id)
    return match.group(1) if match else recording_id


def _read_wav_info(path: Path) -> Mapping[str, Any]:
    with wave.open(str(path), "rb") as handle:
        return {
            "channels": handle.getnchannels(),
            "sample_rate": handle.getframerate(),
            "sample_width_bytes": handle.getsampwidth(),
            "frames": handle.getnframes(),
            "duration_sec": handle.getnframes() / handle.getframerate(),
            "compression_type": handle.getcomptype(),
        }


def _read_pcm16(path: Path) -> tuple[bytes, int, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        if (channels, sample_rate, sample_width, handle.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError(f"offline GSS artifact has unsupported WAV format: {path}")
        raw = handle.readframes(frames)
    return raw, frames, sample_rate


def _read_recording_segments(
    path: Path,
    *,
    recording_id: str,
    sample_rate: int,
    source_total_samples: int,
):
    filtered = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if parts and len(parts) == 4 and parts[1] == recording_id:
                filtered.append(line)
    if not filtered:
        raise ValueError(f"no Stage-2 segments found for {recording_id}")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".segments", delete=False) as handle:
        temporary = Path(handle.name)
        handle.writelines(filtered)
    try:
        return read_stage2_segment_samples(
            temporary,
            recording_id=recording_id,
            sample_rate=sample_rate,
            source_total_samples=source_total_samples,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _union_duration(rows: Sequence[object]) -> float:
    intervals = sorted(
        (float(getattr(row, "start_sample")), float(getattr(row, "end_sample")))
        for row in rows
    )
    total = 0.0
    end = -1.0
    for start, stop in intervals:
        if start >= end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def _candidate_inventory() -> list[Mapping[str, Any]]:
    return [
        {
            "candidate": "standard GSS (WPE + CACGMM + Souden MVDR/BAN)",
            "path": "/home/gc6_challenge/code/gss",
            "git_repo_revision": "local checkout; no usable git revision metadata",
            "upstream_source": "local GSS implementation; MISP offline GSS pipeline",
            "input_channel_count": 8,
            "input_format_sample_rate": "separate mono PCM WAV, 16 kHz",
            "output_format_channel_count": "mono PCM/FLAC/WAV artifacts",
            "runtime_api": "offline GSS enhancement CLI/pipeline",
            "required_model_checkpoint": "none for classical GSS; complete activity/segment inputs",
            "required_environment": "GSS/CuPy environment and full meeting artifacts",
            "already_runnable": True,
            "category": "OFFLINE_ONLY",
            "true_streaming": False,
            "bounded_lookahead": False,
            "future_metadata": True,
            "eight_channel_compatible": True,
            "directly_reusable": False,
            "integration_effort": "high; violates causal contract",
            "decision": "OFFLINE upper-bound diagnostic only",
        },
        {
            "candidate": "OnlineFIVE",
            "path": "/home/gc6_challenge/code/gss/gss/online_five",
            "git_repo_revision": "same local GSS checkout",
            "upstream_source": "FIVE / auxiva-ipa family; local block-online extension",
            "input_channel_count": "2+; tested contract supports 8",
            "input_format_sample_rate": "multichannel sample blocks, default 16 kHz",
            "output_format_channel_count": "mono float stream",
            "runtime_api": "OnlineFIVE.process(block), OnlineFIVE.flush()",
            "required_model_checkpoint": "none",
            "required_environment": "AVDR environment; NumPy",
            "already_runnable": True,
            "category": "BLOCK_ONLINE_OR_PSEUDO_STREAMING",
            "true_streaming": False,
            "bounded_lookahead": "yes; STFT plus 32-frame adaptation block (~560 ms by implementation)",
            "future_metadata": False,
            "eight_channel_compatible": True,
            "directly_reusable": False,
            "integration_effort": "medium; output/target and block semantics need qualification",
            "decision": "exclude from primary B; pseudo-streaming diagnostic candidate",
        },
        {
            "candidate": "FIVE/IVE 2020 batch implementation",
            "path": "/home/gc6_challenge/code/five_2020ICASSP",
            "git_repo_revision": "66ddc55d50bbbc41cb73e375c7a93ac34d45dc43",
            "upstream_source": "https://github.com/onolab-tmu/code_2020ICASSP_five",
            "input_channel_count": "multichannel tensor",
            "input_format_sample_rate": "finite STFT/audio tensor",
            "output_format_channel_count": "separated finite-array output",
            "runtime_api": "batch FIVE/IVE functions",
            "required_model_checkpoint": "none",
            "required_environment": "repository environment.yml",
            "already_runnable": True,
            "category": "OFFLINE_ONLY",
            "true_streaming": False,
            "bounded_lookahead": False,
            "future_metadata": False,
            "eight_channel_compatible": "not established",
            "directly_reusable": False,
            "integration_effort": "high; no incremental state API",
            "decision": "exclude; batch reference only",
        },
        {
            "candidate": "NBSS OnlineSpatialNet",
            "path": "/home/gc6_challenge/gc6_model/NBSS",
            "git_repo_revision": "cc42fc8ad2e6642c09b8f4169a85b4766dc22b7e",
            "upstream_source": "https://github.com/Audio-WestlakeU/NBSS.git",
            "input_channel_count": "8 raw channels plus GSS/reference features in MISP config",
            "input_format_sample_rate": "complex STFT feature tensor; MISP config dim_input=18",
            "output_format_channel_count": "single enhanced output feature stream",
            "runtime_api": "OnlineSpatialNet.forward(x, inference=...)",
            "required_model_checkpoint": "trained checkpoint required",
            "required_environment": "NBSS/PyTorch environment",
            "already_runnable": False,
            "category": "BLOCK_ONLINE_OR_PSEUDO_STREAMING",
            "true_streaming": False,
            "bounded_lookahead": "causal layers exist, but top-level call has no persistent cross-call state contract",
            "future_metadata": "MISP config/runner references GSS-derived inputs",
            "eight_channel_compatible": False,
            "directly_reusable": False,
            "integration_effort": "high; checkpoint, feature adapter, state API, and target contract required",
            "decision": "exclude from primary B",
        },
        {
            "candidate": "AV-SkiM target-speaker extraction",
            "path": "/home/gc6_challenge/gc6_model/ClearerVoice-Studio/train/target_speaker_extraction_online",
            "git_repo_revision": "ClearerVoice-Studio 6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61",
            "upstream_source": "https://github.com/modelscope/ClearerVoice-Studio.git",
            "input_channel_count": "audio plus visual reference",
            "input_format_sample_rate": "LRS3 audio/visual stream, not MISP 8ch contract",
            "output_format_channel_count": "single target-speaker output",
            "runtime_api": "forward_streaming_av(audio, visual, states, reference)",
            "required_model_checkpoint": "LRS3 target-speaker checkpoint",
            "required_environment": "ClearerVoice/PyTorch/visual frontend",
            "already_runnable": False,
            "category": "ORACLE_DIAGNOSTIC_ONLY",
            "true_streaming": False,
            "bounded_lookahead": "causal state API exists",
            "future_metadata": True,
            "eight_channel_compatible": False,
            "directly_reusable": False,
            "integration_effort": "high; requires visual target identity and unsupported checkpoint",
            "decision": "oracle/target-speaker diagnostic only; not generic ASR frontend",
        },
    ]


def _write_frontend_inventory(path: Path) -> None:
    candidates = _candidate_inventory()
    headers = [
        "candidate",
        "path",
        "category",
        "true_streaming",
        "bounded_lookahead",
        "future_metadata",
        "eight_channel_compatible",
        "directly_reusable",
        "integration_effort",
        "decision",
    ]
    lines = [
        "# Server Frontend Inventory",
        "",
        "Audit scope: filename/repository-level search under `/home/gc6_challenge`, "
        "with source inspection for GSS, FIVE/IVE, online beamforming, WPE, "
        "causal enhancement/separation, neural beamforming, and realtime frontend terms.",
        "",
        "`PRIMARY_STREAMING_FRONTEND`: `NO_DIRECTLY_REUSABLE_TRUE_STREAMING_FRONTEND`.",
        "No candidate simultaneously exposed a MISP-compatible 8-channel causal stream, "
        "persistent state, bounded lookahead, sample-aligned output, and no future/oracle metadata.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for candidate in candidates:
        values = [str(candidate.get(key, "")).replace("|", "\\|") for key in headers]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Candidate Details",
            "",
            "- Standard GSS was checked for complete-session clustering, activity masks, WPE context, beamformer estimation, and lack of incremental state; it is `OFFLINE_ONLY`.",
            "- `OnlineFIVE` is stateful and runnable on 8-channel audio, but its documented 32-frame adaptation block and one-unit source ambiguity make it block-online/pseudo-streaming rather than the strict primary B condition.",
            "- Batch FIVE/IVE consumes finite tensors and has no streaming state API.",
            "- NBSS exposes causal internal layers but not a directly callable persistent streaming frontend for this MISP setup; its local MISP configuration also expects GSS/reference-derived features.",
            "- AV-SkiM has a streaming state API but requires visual target-speaker information and LRS3-oriented checkpoints; it is not a generic 8-channel MISP frontend.",
            "",
            "## GSS Audit Conclusion",
            "",
            "`OFFLINE_ONLY`: full-session/segment GSS with WPE, CACGMM/activity guidance, and beamforming. Existing enhanced WAVs are therefore eligible only for Condition C offline upper-bound diagnostics.",
            "",
            "## Causality Gate",
            "",
            "A candidate is not promoted merely because it accepts chunks. The primary B arm requires persistent state across chunks, explicit finite lookahead, no future Stage-2/RTTM/reference/oracle speaker information, and strict input-to-output sample alignment. None passed all gates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _inventory_dataset(args: argparse.Namespace, output_root: Path) -> Mapping[str, Any]:
    data_dir = args.data_dir.resolve(strict=True)
    source_dir = args.source_dir.resolve(strict=True)
    wav_scp = read_kaldi_scp(data_dir / "wav.scp")
    references = read_kaldi_text(data_dir / "text", required=True)
    all_segments = []
    with (data_dir / "segments").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                all_segments.append(line.strip().split())
    utt2spk = {}
    with (data_dir / "utt2spk").open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if parts:
                utt2spk[parts[0]] = parts[1]

    by_recording: dict[str, list[list[str]]] = defaultdict(list)
    for row in all_segments:
        by_recording[row[1]].append(row)
    recordings = []
    for recording_id in sorted(wav_scp):
        stem = Path(wav_scp[recording_id])
        if not stem.is_absolute():
            stem = data_dir / stem
        channels = [resolve_misp_channel_path(stem, index) for index in range(8)]
        available = all(path.is_file() for path in channels)
        info = _read_wav_info(channels[0]) if available else {}
        rows = by_recording.get(recording_id, [])
        source_samples = int(info.get("frames", 0))
        ref_ids = {row[0] for row in rows}
        complete = available and ref_ids <= set(references) and all(
            float(row[2]) >= 0.0
            and float(row[3]) > float(row[2])
            and float(row[3]) * float(info["sample_rate"]) <= source_samples
            for row in rows
        )
        recordings.append(
            {
                "meeting_id": _recording_label(recording_id),
                "recording_id": recording_id,
                "source_stem": str(stem),
                "channel_paths": [str(path) for path in channels],
                "available": available,
                "reference_complete": complete,
                "channels": info.get("channels"),
                "sample_rate": info.get("sample_rate"),
                "sample_width_bytes": info.get("sample_width_bytes"),
                "duration_sec": info.get("duration_sec"),
                "scope_count": len(rows),
                "speaker_count": len({utt2spk[row[0]] for row in rows if row[0] in utt2spk}),
                "reference_chars": sum(len(references[row[0]]) for row in rows if row[0] in references),
                "scope_audio_sec_sum": sum(float(row[3]) - float(row[2]) for row in rows),
                "status": "AVAILABLE" if complete else ("INCOMPLETE" if available else "MISSING"),
            }
        )

    selected = next(
        row for row in recordings if row["recording_id"] == args.recording_id
    )
    if selected["status"] != "AVAILABLE":
        raise ValueError(f"selected recording is not complete: {selected}")
    available = [row for row in recordings if row["status"] == "AVAILABLE"]
    inventory = {
        "schema_version": "wf.dataset_inventory.v1",
        "dataset": "MISP 2025",
        "split": "dev_far",
        "authority": {
            "kaldi_data_dir": str(data_dir),
            "continuous_source_dir": str(source_dir),
            "files": ["wav.scp", "segments", "text", "utt2spk", "spk2utt"],
        },
        "full_available_inventory": {
            "recording_count": len(available),
            "scope_count": sum(row["scope_count"] for row in available),
            "duration_sec": sum(row["duration_sec"] or 0.0 for row in available),
            "reference_chars": sum(row["reference_chars"] for row in available),
            "speakers": len(set(utt2spk.values())),
        },
        "selected_evaluation": {
            "selection_rule": "Latest user instruction explicitly selects the complete M028 recording for the final streaming experiment; selection was frozen before decode.",
            "recording_id": args.recording_id,
            "meeting_id": selected["meeting_id"],
            "scope_count": selected["scope_count"],
            "duration_sec": selected["duration_sec"],
            "reference_chars": selected["reference_chars"],
            "status": selected["status"],
        },
        "recordings": recordings,
    }
    _write_json(output_root / "dataset_inventory.json", inventory)
    lines = [
        "# Dataset Inventory",
        "",
        "- Dataset: `MISP 2025`",
        "- Split: `dev_far`",
        f"- Stage-2 authority: `{data_dir}`",
        f"- Continuous source: `{source_dir}`",
        f"- Full available inventory: {len(available)} recordings, {sum(row['scope_count'] for row in available)} scopes, {sum(row['duration_sec'] or 0.0 for row in available) / 3600:.3f} hours, {sum(row['reference_chars'] for row in available)} reference characters.",
        "- All complete recordings were enumerated before decoding; the main evaluation uses one frozen meeting per the latest user instruction.",
        f"- Selected meeting: `{selected['recording_id']}` (`{selected['scope_count']}` scopes, {selected['duration_sec']:.3f} seconds, {selected['reference_chars']} reference characters).",
        "",
        "| meeting | recording | status | scopes | speakers | duration_sec | scope_audio_sec_sum | reference_chars | channels | sample_rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in recordings:
        lines.append(
            f"| {row['meeting_id']} | {row['recording_id']} | {row['status']} | {row['scope_count']} | {row['speaker_count']} | {row['duration_sec'] or 0:.3f} | {row['scope_audio_sec_sum']:.3f} | {row['reference_chars']} | {row['channels']} | {row['sample_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Selection Freeze",
            "",
            "The selected meeting is fixed before decode and was not changed using recognition results. The source is continuous 8-channel far-field PCM32 WAV; references are Stage-2 `segments`/`text` authority.",
            "",
            "## Missingness",
            "",
            "`AVAILABLE` means all eight source channels and all Stage-2 reference IDs are present. `MISSING` and `INCOMPLETE` rows would be excluded only from a pre-declared subset; no post-result filtering is performed.",
        ]
    )
    (output_root / "dataset_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inventory


def _write_contract(args: argparse.Namespace, output_root: Path, inventory: Mapping[str, Any]) -> None:
    selected = inventory["selected_evaluation"]
    contract = {
        "schema_version": "wf.experiment_contract.v1",
        "status": "FROZEN_BEFORE_DECODE",
        "repository": "lyxin651/lyx_Robot-HEARD",
        "base_commit": "0e9676da6635232eaf37e46ec543d2ff71281b9c",
        "branch": "feat/wenet-streaming-frontend-ablation",
        "dataset": inventory["dataset"],
        "split": inventory["split"],
        "selected_recording_id": selected["recording_id"],
        "selected_scopes": selected["scope_count"],
        "conditions": {
            "A": {
                "label": "raw_ch0 + WenetSpeech U2++",
                "input": "continuous 8-channel far-field source; adapter selects raw_ch0",
                "frontend": "none; only frozen PCM32LE to PCM16LE conversion",
                "online_status": "native streaming ASR with frozen PR#8 oracle scope boundary semantics",
            },
            "B": {
                "label": "qualified streaming frontend + WenetSpeech U2++",
                "status": "NOT_RUN",
                "reason": "NO_DIRECTLY_REUSABLE_TRUE_STREAMING_FRONTEND",
            },
            "C": {
                "label": "offline GSS + WenetSpeech U2++",
                "status": "NOT_RUN_BY_USER_INSTRUCTION",
                "input": "not evaluated in this run",
                "latency_comparability": "NOT_APPLICABLE",
            },
        },
        "frozen_shared": {
            "model_dir": str(args.model_dir.resolve()),
            "model_archive_sha256": MODEL_ARCHIVE_SHA256,
            "wenet_revision": WENET_REVISION,
            "library_path": str(args.library_path.resolve()),
            "language": "chs",
            "sample_rate": 16000,
            "channel_count": 8,
            "packet_samples": args.packet_samples,
            "model_chunk_samples": args.model_chunk_samples,
            "wenet_chunk_size": args.wenet_chunk_size,
            "partial_emission_policy": "suppress initial empty/repeated visible text; emit changed visible text; FINAL once",
            "scoring_normalization": "v0_nfkc_casefold_strip_space_punct",
            "scope_policy": "Stage-2 START/END gate, frozen from accepted PR#8 semantics",
        },
        "leakage_policy": {
            "future_rttm": False,
            "future_reference": False,
            "oracle_speaker_for_A": False,
            "condition_C_uses_offline_stage2_guidance": "NOT_RUN",
            "condition_B_future_perturbation_test": "NOT_APPLICABLE; no qualified B frontend",
        },
    }
    _write_json(output_root / "experiment_contract.json", contract)


def _score_snapshot_rows(
    snapshots: Sequence[Mapping[str, Any]],
    references: Mapping[str, str],
    sample_rate: int,
    output_root: Path,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    scored_rows = []
    evaluator_path = output_root / "evaluator_results.jsonl"
    scored_path = output_root / "scored.jsonl"
    with evaluator_path.open("w", encoding="utf-8") as evaluator, scored_path.open(
        "w", encoding="utf-8"
    ) as scored_handle:
        for row in snapshots:
            if row.get("kind") != "FINAL":
                continue
            segment_id = str(row["segment_id"])
            result_row = {
                "status": "success",
                "segment_id": segment_id,
                "text_raw": row["text_raw"],
                "reference": references[segment_id],
                "audio_sec": (row["end_sample"] - row["start_sample"]) / sample_rate,
                "decode_sec": (
                    None
                    if row["native_call_start_wall_offset_sec"] is None
                    or row["native_call_return_wall_offset_sec"] is None
                    else row["native_call_return_wall_offset_sec"]
                    - row["native_call_start_wall_offset_sec"]
                ),
            }
            evaluator.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            scored = score_result_record(result_row)
            scored_handle.write(json.dumps(scored, ensure_ascii=False) + "\n")
            scored_rows.append(scored)
    summary = _score_summary(scored_rows)
    return scored_rows, summary


def _score_summary(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    errors = sum(int(row["cer_s"]) + int(row["cer_d"]) + int(row["cer_i"]) for row in rows)
    reference_chars = sum(int(row["cer_n"]) for row in rows)
    return {
        "scope_count": len(rows),
        "empty_count": sum(row["text_raw"] == "" for row in rows),
        "non_empty_count": sum(row["text_raw"] != "" for row in rows),
        "cer_s": sum(int(row["cer_s"]) for row in rows),
        "cer_d": sum(int(row["cer_d"]) for row in rows),
        "cer_i": sum(int(row["cer_i"]) for row in rows),
        "cer_n": reference_chars,
        "errors": errors,
        "cer": errors / reference_chars if reference_chars else None,
        "scope_cer_mean": (
            sum(float(row["cer"]) for row in rows if row["cer"] is not None)
            / sum(row["cer"] is not None for row in rows)
            if rows
            else None
        ),
    }


def _run_raw(args: argparse.Namespace, output_root: Path, inventory: Mapping[str, Any]) -> Mapping[str, Any]:
    data_dir = args.data_dir.resolve(strict=True)
    source_dir = args.source_dir.resolve(strict=True)
    selected = inventory["selected_evaluation"]
    recording_id = selected["recording_id"]
    wav_scp = read_kaldi_scp(data_dir / "wav.scp")
    stem = Path(wav_scp[recording_id])
    if not stem.is_absolute():
        stem = data_dir / stem
    channel_paths = tuple(resolve_misp_channel_path(stem, index) for index in range(8))
    source = MultiChannelPcmWavSource(
        channel_paths,
        channel_ids=tuple(f"ch{index}" for index in range(8)),
        stream_id=recording_id,
        packet_samples=args.packet_samples,
    )
    if (source.metadata.sample_rate, source.sample_width, source.compression_type, source.metadata.num_channels) != (16000, 4, "NONE", 8):
        raise ValueError("raw condition requires 8-channel 16 kHz PCM32 WAV")
    segments = _read_recording_segments(
        data_dir / "segments",
        recording_id=recording_id,
        sample_rate=source.metadata.sample_rate,
        source_total_samples=source.total_samples,
    )
    references_all = read_kaldi_text(data_dir / "text", required=True)
    references = {row.segment_id: references_all[row.segment_id] for row in segments}
    run_root = output_root / "raw_baseline" / recording_id
    run_root.mkdir(parents=True)
    factory = CtypesWeNetDecoderFactory(
        args.library_path.resolve(),
        args.model_dir.resolve(),
        wenet_chunk_size=args.wenet_chunk_size,
        language="chs",
    )
    consumer = WeNetU2PPStreamingConsumer(
        decoder_factory=factory,
        model_chunk_samples=args.model_chunk_samples,
        selected_channel_id="ch0",
        frontend_condition="raw_ch0",
        warmup_pcm16le=bytes(args.warmup_samples * 2) if args.warmup_samples else None,
        snapshot_evidence_path=run_root / "snapshots.jsonl",
        wall_clock=MonotonicWallClock(),
    )
    gate = Stage2ScopeBoundaryReleaseGate(
        segments,
        sample_rate=16000,
        condition="wf_raw_ch0_stage2_scope_boundaries",
        packet_duration_sec=args.packet_samples / 16000,
    )
    trace_path = run_root / "trace.jsonl"
    run_path = run_root / "trace.run.json"
    provenance = {
        "schema_version": "wf.wenet.raw.trace.v1",
        "run_id": f"wf-raw-{recording_id}",
        "replay_mode": "COMPUTE_AWARE",
        "source_identity": source.identity(),
        "source_sha256": source.content_sha256,
        "source_sample_rate": source.metadata.sample_rate,
        "channel_ids": list(source.metadata.channel_ids),
        "packet_samples": source.packet_samples,
        "queue_policy": "lossless_unbounded_fifo",
        "timeline_authority": "integer_sample_index",
        "consumer_identity": type(consumer).__name__,
        "consumer_config": {
            "condition": "raw_ch0",
            "selected_channel_id": "ch0",
            "model_chunk_samples": args.model_chunk_samples,
            "pcm_conversion_policy": "signed_pcm32le_arithmetic_right_shift_16_to_pcm16le_with_saturation",
            "partial_event_policy": "suppress initial empty and repeated visible text; emit changed text; FINAL once",
        },
        "consumer_capabilities": dict(consumer.capabilities().to_dict()),
        "code_commit": _git_head(),
        "oracle_condition": "wf_raw_ch0_stage2_scope_boundaries",
        "warmup_load_policy": {
            "warmup_samples": args.warmup_samples,
            "model_load_excluded_from_measured_stream": True,
        },
        "clock_policy": "monotonic_perf_counter_wall_clock",
        "clock_origin": {"kind": "pending_until_consumer_start", "run_wall_origin": "pending"},
        "metric_window": {"start_sample": 0, "end_sample": source.total_samples},
        "trace_path": str(trace_path.resolve()),
        "native_runtime": {
            "api": "official WeNet LibTorch C API",
            "wenet_revision": WENET_REVISION,
            "library_path": str(args.library_path.resolve()),
            "library_sha256": _sha256_file(args.library_path.resolve()),
            "model_dir": str(args.model_dir.resolve()),
            "model_archive_sha256": MODEL_ARCHIVE_SHA256,
        },
        "device_provenance": {"device": "cpu"},
        "trace_durability_policy": TraceDurabilityPolicy.BUFFERED_CLOSE_FSYNC.value,
    }
    writer = TraceWriter(
        trace_path,
        run_path,
        provenance,
        schema_version="wf.wenet.raw.trace.v1",
        durability_policy=TraceDurabilityPolicy.BUFFERED_CLOSE_FSYNC,
    )
    result = PacedReplayRuntime(
        source,
        consumer,
        writer,
        wall_clock=consumer.wall_clock,
        oracle_gate=gate,
    ).run()
    run_provenance = json.loads(run_path.read_text(encoding="utf-8"))
    paced = compute_paced_metrics(
        result.trace_events,
        sample_rate=16000,
        source_total_samples=source.total_samples,
        run_id=f"wf-raw-{recording_id}",
        run_provenance=run_provenance,
        measured_wall_interval_sec=result.finish_return_wall_offset_sec,
    )
    r5b = compute_r5b_metrics(
        consumer.snapshots,
        sample_rate=16000,
        native_audit=consumer.native_audit,
        global_metrics=paced,
    )
    write_metrics_json(run_root / "paced_metrics.json", paced)
    write_metrics_json(run_root / "r5b_metrics.json", r5b)
    scored_rows, score_summary = _score_snapshot_rows(
        consumer.snapshots, references, 16000, run_root
    )
    per_scope = [
        {
            "segment_id": row["segment_id"],
            "cer": row["cer"],
            "cer_s": row["cer_s"],
            "cer_d": row["cer_d"],
            "cer_i": row["cer_i"],
            "cer_n": row["cer_n"],
            "empty": row["text_raw"] == "",
        }
        for row in scored_rows
    ]
    report = {
        "schema_version": "wf.raw_baseline_report.v1",
        "condition": "A_RAW_CH0_WENET_U2PP",
        "status": "PASS",
        "dataset": "MISP 2025",
        "split": "dev_far",
        "meeting_id": _recording_label(recording_id),
        "recording_id": recording_id,
        "source": {
            "continuous_far_field_multichannel_wav": str(source_dir),
            "channel_count": 8,
            "sample_rate": 16000,
            "duration_sec": source.total_samples / 16000,
            "selected_channel": "ch0",
        },
        "scoring": score_summary,
        "per_scope": per_scope,
        "paced_metrics": dict(paced),
        "r5b_metrics": dict(r5b),
        "latency": {
            "frontend_RTF": 0.0,
            "ASR_RTF": paced["RTF_consume"],
            "total_compute_RTF": paced["RTF_total_compute"],
            "TTFT": r5b["scope_TTFT"],
            "segment_end_to_FINAL": r5b["segment_end_to_final"],
            "P50_backlog_sec": paced["queue_backlog_sec"]["p50"],
            "P95_backlog_sec": paced["queue_backlog_sec"]["p95"],
            "max_backlog_sec": paced["queue_backlog_sec"]["max"],
            "completion_overrun_sec": paced["completion_overrun_sec"],
        },
        "memory": {"peak_rss_bytes": _rss_bytes()},
        "leakage": {
            "reference_leakage": False,
            "future_segment_leakage": False,
            "scope_boundary_oracle_semantics": True,
            "oracle_speaker_dependency": False,
        },
        "native_runtime": provenance["native_runtime"],
        "artifacts": {
            "trace": str(trace_path.resolve()),
            "snapshots": str((run_root / "snapshots.jsonl").resolve()),
            "paced_metrics": str((run_root / "paced_metrics.json").resolve()),
            "r5b_metrics": str((run_root / "r5b_metrics.json").resolve()),
            "scored": str((run_root / "scored.jsonl").resolve()),
        },
    }
    _write_json(output_root / "raw_baseline_report.json", report)
    return report


def _offline_artifact_path(enhanced_root: Path, recording_id: str, segment_id: str) -> Path:
    match = SEGMENT_RE.match(segment_id)
    if match is None:
        raise ValueError(f"cannot map Stage-2 segment to GSS artifact: {segment_id}")
    speaker, interval = match.groups()
    return enhanced_root / recording_id / f"{recording_id}-{speaker}-{interval.replace('-', '_')}.wav"


def _run_offline(args: argparse.Namespace, output_root: Path, inventory: Mapping[str, Any]) -> Mapping[str, Any]:
    data_dir = args.data_dir.resolve(strict=True)
    enhanced_root = args.enhanced_root.resolve(strict=True)
    recording_id = inventory["selected_evaluation"]["recording_id"]
    rows = _read_recording_segments(
        data_dir / "segments",
        recording_id=recording_id,
        sample_rate=16000,
        source_total_samples=int(inventory["selected_evaluation"]["duration_sec"] * 16000),
    )
    references_all = read_kaldi_text(data_dir / "text", required=True)
    run_root = output_root / "offline_upper_bound" / recording_id
    run_root.mkdir(parents=True)
    factory = CtypesWeNetDecoderFactory(
        args.library_path.resolve(),
        args.model_dir.resolve(),
        wenet_chunk_size=args.wenet_chunk_size,
        language="chs",
    )
    load_start = time.perf_counter()
    decoder = factory()
    model_load_sec = time.perf_counter() - load_start
    scored_rows = []
    missing = []
    decode_sec_total = 0.0
    try:
        for row in rows:
            artifact = _offline_artifact_path(enhanced_root, recording_id, row.segment_id)
            if not artifact.is_file():
                missing.append(row.segment_id)
                continue
            raw, frame_count, sample_rate = _read_pcm16(artifact)
            if frame_count <= 0:
                raise ValueError(f"empty offline GSS artifact: {artifact}")
            final_text = ""
            offset = 0
            while offset < frame_count:
                count = min(args.model_chunk_samples, frame_count - offset)
                last = offset + count == frame_count
                call_start = time.perf_counter()
                native = parse_native_result(
                    decoder.decode(raw[offset * 2 : (offset + count) * 2], last=last)
                )
                decode_sec_total += time.perf_counter() - call_start
                final_text = native.text
                offset += count
            decoder.reset()
            result_row = {
                "status": "success",
                "segment_id": row.segment_id,
                "text_raw": final_text,
                "reference": references_all[row.segment_id],
                "audio_sec": frame_count / sample_rate,
                "decode_sec": None,
            }
            scored_rows.append(result_row)
    finally:
        decoder.close()
    if missing:
        raise FileNotFoundError(f"missing offline GSS artifacts: {missing[:5]} (total={len(missing)})")
    # Allocate the measured native time across records only for a diagnostic
    # aggregate; frontend artifact generation time is intentionally unavailable.
    total_audio = sum(float(row["audio_sec"]) for row in scored_rows)
    per_scope = []
    for row in scored_rows:
        scored = score_result_record(
            {**row, "decode_sec": 0.0 if total_audio == 0 else float(row["audio_sec"]) * decode_sec_total / total_audio}
        )
        per_scope.append(scored)
    score_summary = _score_summary(per_scope)
    report = {
        "schema_version": "wf.offline_upper_bound_report.v1",
        "condition": "C_OFFLINE_GSS_UPPER_BOUND_WENET_U2PP",
        "status": "PASS_OFFLINE_DIAGNOSTIC",
        "label": "OFFLINE_FRONTEND_UPPER_BOUND",
        "online_comparable": False,
        "latency_comparability": "OFFLINE / NOT ONLINE-COMPARABLE FOR LATENCY",
        "dataset": "MISP 2025",
        "split": "dev_far",
        "meeting_id": _recording_label(recording_id),
        "recording_id": recording_id,
        "frontend": {
            "artifact_root": str(enhanced_root),
            "artifact_count": len(per_scope),
            "artifact_format": "mono PCM16 WAV, per Stage-2 scope",
            "algorithm": "existing offline GSS: WPE + CACGMM + Souden MVDR/BAN",
            "uses_future_context": True,
            "uses_stage2_activity_or_segments": True,
            "algorithmic_lookahead": "full offline scope/context; unbounded for online comparison",
            "frontend_RTF": None,
        },
        "scoring": score_summary,
        "per_scope": [
            {
                "segment_id": row["segment_id"],
                "cer": row["cer"],
                "cer_s": row["cer_s"],
                "cer_d": row["cer_d"],
                "cer_i": row["cer_i"],
                "cer_n": row["cer_n"],
                "empty": row["text_raw"] == "",
            }
            for row in per_scope
        ],
        "asr_diagnostic": {
            "ASR_RTF": decode_sec_total / total_audio,
            "asr_compute_sec": decode_sec_total,
            "model_load_sec": model_load_sec,
            "total_compute_RTF": None,
        },
        "latency": {
            "frontend_algorithmic_lookahead": "OFFLINE",
            "TTFT": None,
            "segment_end_to_FINAL": None,
            "backlog": None,
            "completion_overrun": None,
        },
        "memory": {"peak_rss_bytes": _rss_bytes()},
        "leakage": {
            "reference_leakage": False,
            "future_segment_leakage": True,
            "oracle_speaker_dependency": True,
            "classification": "OFFLINE_ONLY",
        },
        "native_runtime": {
            "wenet_revision": WENET_REVISION,
            "library_path": str(args.library_path.resolve()),
            "library_sha256": _sha256_file(args.library_path.resolve()),
            "model_dir": str(args.model_dir.resolve()),
            "model_archive_sha256": MODEL_ARCHIVE_SHA256,
            "model_chunk_samples": args.model_chunk_samples,
        },
        "artifacts": {"artifact_root": str(enhanced_root)},
    }
    _write_json(output_root / "offline_upper_bound_report.json", report)
    return report


def _write_not_run_streaming_report(output_root: Path) -> Mapping[str, Any]:
    report = {
        "schema_version": "wf.streaming_frontend_report.v1",
        "condition": "B_QUALIFIED_STREAMING_FRONTEND_WENET_U2PP",
        "status": "NOT_RUN",
        "decision": "NO_DIRECTLY_REUSABLE_TRUE_STREAMING_FRONTEND",
        "reason": "No audited candidate met all strict TRUE_STREAMING_REUSABLE requirements.",
        "primary_streaming_frontend": None,
        "future_perturbation_test": {
            "status": "NOT_APPLICABLE",
            "reason": "No qualified frontend was available for Condition B.",
        },
        "latency": None,
        "scoring": None,
    }
    _write_json(output_root / "streaming_frontend_report.json", report)
    return report


def _write_not_run_offline_report(output_root: Path) -> Mapping[str, Any]:
    report = {
        "schema_version": "wf.offline_upper_bound_report.v1",
        "condition": "C_OFFLINE_GSS_UPPER_BOUND_WENET_U2PP",
        "status": "NOT_RUN_BY_USER_INSTRUCTION",
        "label": "OFFLINE_FRONTEND_UPPER_BOUND",
        "online_comparable": False,
        "reason": "User explicitly requested no offline experiment in the final run.",
        "scoring": None,
        "latency": None,
    }
    _write_json(output_root / "offline_upper_bound_report.json", report)
    return report


def _write_current_summary(
    output_root: Path,
    inventory: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> Mapping[str, Any]:
    scope_cers = [float(row["cer"]) for row in raw["per_scope"] if row["cer"] is not None]
    robustness = {
        "paired_unit": None,
        "scope_count": len(scope_cers),
        "meeting_count": 1,
        "raw_scope_cer": {
            "median": median(scope_cers) if scope_cers else None,
            "best": min(scope_cers) if scope_cers else None,
            "worst": max(scope_cers) if scope_cers else None,
        },
        "bootstrap_95ci": None,
        "note": "No frontend comparison was run: B has no qualified candidate and C was explicitly skipped.",
    }
    _write_json(output_root / "robustness_summary.json", robustness)
    rows = [
        {
            "condition": "A_RAW_CH0_WENET_U2PP",
            "meeting_id": raw["meeting_id"],
            "status": raw["status"],
            "num_scopes": raw["scoring"]["scope_count"],
            "audio_duration_sec": raw["source"]["duration_sec"],
            "reference_chars": raw["scoring"]["cer_n"],
            "cer": raw["scoring"]["cer"],
            "S": raw["scoring"]["cer_s"],
            "D": raw["scoring"]["cer_d"],
            "I": raw["scoring"]["cer_i"],
            "empty_count": raw["scoring"]["empty_count"],
            "ASR_RTF": raw["latency"]["ASR_RTF"],
            "frontend_RTF": raw["latency"]["frontend_RTF"],
            "total_compute_RTF": raw["latency"]["total_compute_RTF"],
            "TTFT_P50_sec": raw["latency"]["TTFT"]["p50"],
            "FINAL_P50_sec": raw["latency"]["segment_end_to_FINAL"]["p50"],
            "P95_backlog_sec": raw["latency"]["P95_backlog_sec"],
            "max_backlog_sec": raw["latency"]["max_backlog_sec"],
            "memory_peak_bytes": raw["memory"]["peak_rss_bytes"],
        },
        {
            "condition": "B_STREAMING_FRONTEND_WENET_U2PP",
            "meeting_id": raw["meeting_id"],
            "status": "NOT_RUN",
            "num_scopes": "",
            "audio_duration_sec": "",
            "reference_chars": "",
            "cer": "",
            "S": "",
            "D": "",
            "I": "",
            "empty_count": "",
            "ASR_RTF": "",
            "frontend_RTF": "",
            "total_compute_RTF": "",
            "TTFT_P50_sec": "",
            "FINAL_P50_sec": "",
            "P95_backlog_sec": "",
            "max_backlog_sec": "",
            "memory_peak_bytes": "",
        },
        {
            "condition": "C_OFFLINE_GSS_UPPER_BOUND_WENET_U2PP",
            "meeting_id": raw["meeting_id"],
            "status": "NOT_RUN_BY_USER_INSTRUCTION",
            "num_scopes": "",
            "audio_duration_sec": "",
            "reference_chars": "",
            "cer": "",
            "S": "",
            "D": "",
            "I": "",
            "empty_count": "",
            "ASR_RTF": "",
            "frontend_RTF": "",
            "total_compute_RTF": "",
            "TTFT_P50_sec": "",
            "FINAL_P50_sec": "",
            "P95_backlog_sec": "",
            "max_backlog_sec": "",
            "memory_peak_bytes": "",
        },
    ]
    fields = list(rows[0])
    with (output_root / "meeting_level_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_lines = [
        "# Ablation Summary",
        "",
        "| condition | status | CER | ASR_RTF | total_compute_RTF | latency note |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        f"| A raw_ch0 + WenetSpeech U2++ | PASS | {raw['scoring']['cer']:.6f} | {raw['latency']['ASR_RTF']:.6f} | {raw['latency']['total_compute_RTF']:.6f} | M028 continuous replay |",
        "| B qualified streaming frontend + WenetSpeech U2++ | NOT RUN | — | — | — | no reusable true-streaming frontend |",
        "| C offline GSS + WenetSpeech U2++ | NOT RUN | — | — | — | explicitly skipped |",
        "",
        "M028 is the final requested meeting. It contains 144 Stage-2 scopes and is therefore qualification-sized, not the previously requested >144 expanded evaluation.",
        "No frontend gain claim is made because no qualified streaming frontend was available and offline C was not run.",
    ]
    (output_root / "ablation_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return robustness


def _write_current_final_report(
    output_root: Path,
    inventory: Mapping[str, Any],
    raw: Mapping[str, Any],
    robustness: Mapping[str, Any],
) -> None:
    status = "WF_NO_REUSABLE_TRUE_STREAMING_FRONTEND"
    lines = [
        "# WF Final Report",
        "",
        f"## Final Status: `{status}`",
        "",
        "The latest user instruction requested no offline experiment and selected M028 for the final streaming run. The accepted PR #8 WeNet baseline semantics, replay core, Clock, and WeNet adapter were not changed.",
        "",
        "## Frontend Audit",
        "",
        "- Found standard offline GSS, block-online OnlineFIVE, batch FIVE/IVE, NBSS OnlineSpatialNet, and AV-SkiM target-speaker streaming code.",
        "- No candidate met the strict generic MISP 8-channel `TRUE_STREAMING_REUSABLE` contract.",
        "- Standard GSS is `OFFLINE_ONLY`; OnlineFIVE is `BLOCK_ONLINE_OR_PSEUDO_STREAMING`; batch FIVE/IVE is `OFFLINE_ONLY`; NBSS is not directly runnable with persistent state; AV-SkiM is `ORACLE_DIAGNOSTIC_ONLY`.",
        "- Condition B is not run; no new frontend is invented.",
        "",
        "## Dataset and Selection",
        "",
        f"- Dataset/split: `{inventory['dataset']}` / `{inventory['split']}`.",
        f"- Full available authority inventory: `{inventory['full_available_inventory']['recording_count']}` recordings, `{inventory['full_available_inventory']['scope_count']}` scopes, `{inventory['full_available_inventory']['duration_sec']:.3f}` seconds, `{inventory['full_available_inventory']['reference_chars']}` reference characters.",
        f"- Final run: `{inventory['selected_evaluation']['recording_id']}` (`{inventory['selected_evaluation']['scope_count']}` scopes, `{inventory['selected_evaluation']['duration_sec']:.3f}` seconds, `{inventory['selected_evaluation']['reference_chars']}` reference characters).",
        "- Source: continuous 8-channel far-field PCM32 WAV at 16 kHz; references: Stage-2 `segments`/`text` authority.",
        "",
        "## M028 Raw Result",
        "",
        f"- A raw_ch0 + WenetSpeech U2++: CER `{raw['scoring']['cer']:.6f}`; S/D/I `{raw['scoring']['cer_s']}/{raw['scoring']['cer_d']}/{raw['scoring']['cer_i']}`; N `{raw['scoring']['cer_n']}`; empty/non-empty `{raw['scoring']['empty_count']}/{raw['scoring']['non_empty_count']}`.",
        f"- ASR RTF `{raw['latency']['ASR_RTF']:.6f}`; total compute RTF `{raw['latency']['total_compute_RTF']:.6f}`; TTFT P50/P95 `{raw['latency']['TTFT']['p50']:.3f}/{raw['latency']['TTFT']['p95']:.3f}` sec; segment-end-to-FINAL P50/P95 `{raw['latency']['segment_end_to_FINAL']['p50']:.3f}/{raw['latency']['segment_end_to_FINAL']['p95']:.3f}` sec; backlog P95/max `{raw['latency']['P95_backlog_sec']:.3f}/{raw['latency']['max_backlog_sec']:.3f}` sec.",
        "- Future/reference leakage in A: no transcript/reference or future RTTM enters the decoder; accepted PR #8 Stage-2 scope-boundary oracle semantics remain in force.",
        "",
        "## Conditions B/C",
        "",
        "- B qualified streaming frontend: `NOT_RUN`; no qualified reusable true-streaming candidate exists, so no CER gain or frontend latency claim is made.",
        "- C offline GSS upper bound: `NOT_RUN_BY_USER_INSTRUCTION`; no offline experiment was performed.",
        "- Future perturbation test: not applicable because B was not available.",
        "",
        "## Limitations",
        "",
        "M028 has exactly 144 Stage-2 scopes. This final run follows the latest request for M028 but does not satisfy the earlier >144 expanded-evaluation criterion; it should be treated as a final streaming qualification/closure run, not as the expanded multi-meeting main result.",
        "",
        "## Recommendation",
        "",
        "Keep the accepted raw_ch0 + WenetSpeech U2++ native-streaming baseline for the RoboHEARD online track. Do not promote OnlineFIVE, offline GSS, NBSS, or AV-SkiM as a generic causal frontend without a separate contract review and qualification.",
        "",
        "## Artifacts",
        "",
        "- `frontend_inventory.md`",
        "- `dataset_inventory.md` / `dataset_inventory.json`",
        "- `experiment_contract.json`",
        "- `raw_baseline_report.json`",
        "- `streaming_frontend_report.json`",
        "- `offline_upper_bound_report.json` (not-run marker only)",
        "- `meeting_level_results.csv`",
        "- `robustness_summary.json`",
        "- `ablation_summary.md`",
    ]
    (output_root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(
    output_root: Path,
    inventory: Mapping[str, Any],
    raw: Mapping[str, Any],
    offline: Mapping[str, Any],
) -> None:
    raw_by_id = {row["segment_id"]: row for row in raw["per_scope"]}
    off_by_id = {row["segment_id"]: row for row in offline["per_scope"]}
    paired = []
    for segment_id in sorted(set(raw_by_id) & set(off_by_id)):
        delta = float(off_by_id[segment_id]["cer"]) - float(raw_by_id[segment_id]["cer"])
        paired.append((segment_id, delta))
    improved = sum(delta < -1e-12 for _, delta in paired)
    worse = sum(delta > 1e-12 for _, delta in paired)
    ties = len(paired) - improved - worse
    best = min(paired, key=lambda item: item[1]) if paired else (None, None)
    worst = max(paired, key=lambda item: item[1]) if paired else (None, None)
    robustness = {
        "paired_unit": "Stage-2 scope",
        "paired_scope_count": len(paired),
        "offline_minus_raw_cer_delta": {
            "improved_count": improved,
            "worse_count": worse,
            "tie_count": ties,
            "median": median(delta for _, delta in paired) if paired else None,
            "best_scope": {"segment_id": best[0], "delta": best[1]},
            "worst_scope": {"segment_id": worst[0], "delta": worst[1]},
        },
        "bootstrap_95ci": None,
        "note": "One meeting was authorized by the latest user instruction; paired scope comparison is reported instead of a multi-meeting bootstrap.",
    }
    _write_json(output_root / "robustness_summary.json", robustness)

    rows = [
        {
            "condition": "A_RAW_CH0_WENET_U2PP",
            "meeting_id": raw["meeting_id"],
            "status": raw["status"],
            "num_scopes": raw["scoring"]["scope_count"],
            "audio_duration_sec": raw["source"]["duration_sec"],
            "reference_chars": raw["scoring"]["cer_n"],
            "cer": raw["scoring"]["cer"],
            "S": raw["scoring"]["cer_s"],
            "D": raw["scoring"]["cer_d"],
            "I": raw["scoring"]["cer_i"],
            "empty_count": raw["scoring"]["empty_count"],
            "ASR_RTF": raw["latency"]["ASR_RTF"],
            "frontend_RTF": raw["latency"]["frontend_RTF"],
            "total_compute_RTF": raw["latency"]["total_compute_RTF"],
            "TTFT_P50_sec": raw["latency"]["TTFT"]["p50"],
            "FINAL_P50_sec": raw["latency"]["segment_end_to_FINAL"]["p50"],
            "P95_backlog_sec": raw["latency"]["P95_backlog_sec"],
            "max_backlog_sec": raw["latency"]["max_backlog_sec"],
            "memory_peak_bytes": raw["memory"]["peak_rss_bytes"],
        },
        {
            "condition": "B_STREAMING_FRONTEND_WENET_U2PP",
            "meeting_id": raw["meeting_id"],
            "status": "NOT_RUN",
            "num_scopes": "",
            "audio_duration_sec": "",
            "reference_chars": "",
            "cer": "",
            "S": "",
            "D": "",
            "I": "",
            "empty_count": "",
            "ASR_RTF": "",
            "frontend_RTF": "",
            "total_compute_RTF": "",
            "TTFT_P50_sec": "",
            "FINAL_P50_sec": "",
            "P95_backlog_sec": "",
            "max_backlog_sec": "",
            "memory_peak_bytes": "",
        },
        {
            "condition": "C_OFFLINE_GSS_UPPER_BOUND_WENET_U2PP",
            "meeting_id": offline["meeting_id"],
            "status": offline["status"],
            "num_scopes": offline["scoring"]["scope_count"],
            "audio_duration_sec": inventory["selected_evaluation"]["duration_sec"],
            "reference_chars": offline["scoring"]["cer_n"],
            "cer": offline["scoring"]["cer"],
            "S": offline["scoring"]["cer_s"],
            "D": offline["scoring"]["cer_d"],
            "I": offline["scoring"]["cer_i"],
            "empty_count": offline["scoring"]["empty_count"],
            "ASR_RTF": offline["asr_diagnostic"]["ASR_RTF"],
            "frontend_RTF": "OFFLINE",
            "total_compute_RTF": "OFFLINE",
            "TTFT_P50_sec": "OFFLINE",
            "FINAL_P50_sec": "OFFLINE",
            "P95_backlog_sec": "OFFLINE",
            "max_backlog_sec": "OFFLINE",
            "memory_peak_bytes": offline["memory"]["peak_rss_bytes"],
        },
    ]
    fields = list(rows[0])
    with (output_root / "meeting_level_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    raw_cer = raw["scoring"]["cer"]
    off_cer = offline["scoring"]["cer"]
    absolute_gain = raw_cer - off_cer
    relative_gain = absolute_gain / raw_cer if raw_cer else None
    summary_lines = [
        "# Ablation Summary",
        "",
        "| condition | status | CER | ΔCER vs raw | relative CER reduction | ASR_RTF | total_compute_RTF | latency note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| A raw_ch0 + WenetSpeech U2++ | PASS | {raw_cer:.6f} | 0 | 0 | {raw['latency']['ASR_RTF']:.6f} | {raw['latency']['total_compute_RTF']:.6f} | online replay; raw frontend cost 0 |",
        "| B qualified streaming frontend + WenetSpeech U2++ | NOT RUN | — | — | — | — | — | no reusable true-streaming frontend |",
        f"| C offline GSS + WenetSpeech U2++ | OFFLINE | {off_cer:.6f} | {off_cer - raw_cer:.6f} | {-relative_gain if relative_gain is not None else 0:.3%} | {offline['asr_diagnostic']['ASR_RTF']:.6f} | OFFLINE | not online-comparable |",
        "",
        f"Offline upper-bound absolute CER gain vs raw: `{absolute_gain:.6f}`; relative reduction: `{relative_gain:.3%}`." if relative_gain is not None else "Offline upper-bound relative gain is unavailable.",
        f"Paired scope robustness: `{robustness['offline_minus_raw_cer_delta']['improved_count']}` improved, `{robustness['offline_minus_raw_cer_delta']['worse_count']}` worse, `{robustness['offline_minus_raw_cer_delta']['tie_count']}` tied; median offline-minus-raw scope CER delta `{robustness['offline_minus_raw_cer_delta']['median']:.6f}`.",
        "Condition C uses existing per-scope offline GSS artifacts with future/session Stage-2 guidance and is excluded from online latency/RTF feasibility conclusions.",
    ]
    (output_root / "ablation_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return robustness


def _write_final_report(
    output_root: Path,
    inventory: Mapping[str, Any],
    raw: Mapping[str, Any],
    offline: Mapping[str, Any],
    robustness: Mapping[str, Any],
) -> None:
    raw_cer = raw["scoring"]["cer"]
    offline_cer = offline["scoring"]["cer"]
    absolute_gain = raw_cer - offline_cer
    relative_gain = absolute_gain / raw_cer if raw_cer else None
    status = "WF_NO_REUSABLE_TRUE_STREAMING_FRONTEND"
    lines = [
        "# WF Final Report",
        "",
        f"## Final Status: `{status}`",
        "",
        "This is an external-review handoff artifact. The accepted PR #8 WeNet baseline semantics, ReplaySource, PacedReplayRuntime, Clock, and WeNet adapter were not changed.",
        "",
        "## 1. Frontend Audit",
        "",
        "- Found standard offline GSS, block-online OnlineFIVE, batch FIVE/IVE, NBSS OnlineSpatialNet, and AV-SkiM target-speaker streaming code.",
        "- No candidate is promoted as `TRUE_STREAMING_REUSABLE` for the current generic MISP 8-channel frontend contract.",
        "- GSS is `OFFLINE_ONLY`; OnlineFIVE is `BLOCK_ONLINE_OR_PSEUDO_STREAMING`; batch FIVE/IVE is `OFFLINE_ONLY`; NBSS is not directly runnable with a persistent state adapter; AV-SkiM is `ORACLE_DIAGNOSTIC_ONLY` because it needs visual target identity and unsupported LRS3-oriented artifacts.",
        "- Primary frontend: `NO_DIRECTLY_REUSABLE_TRUE_STREAMING_FRONTEND`; Condition B is not run and no new frontend is invented.",
        "",
        "## 2. Expanded Dataset",
        "",
        f"- Dataset/split: `{inventory['dataset']}` / `{inventory['split']}`.",
        f"- Full available authority: `{inventory['full_available_inventory']['recording_count']}` recordings, `{inventory['full_available_inventory']['scope_count']}` scopes, `{inventory['full_available_inventory']['duration_sec']:.3f}` seconds, `{inventory['full_available_inventory']['reference_chars']}` reference characters.",
        f"- Main evaluation selection: `{inventory['selected_evaluation']['recording_id']}` (`{inventory['selected_evaluation']['scope_count']}` scopes, `{inventory['selected_evaluation']['duration_sec']:.3f}` seconds, `{inventory['selected_evaluation']['reference_chars']}` reference characters). Selection was frozen before decode under the latest instruction to use one meeting.",
        "- Source: prepared continuous 8-channel far-field PCM32 WAV at 16 kHz; references: Stage-2 `segments`/`text` authority.",
        "",
        "## 3. Primary Results",
        "",
        f"- A raw_ch0 + WenetSpeech U2++: CER `{raw_cer:.6f}`; S/D/I `{raw['scoring']['cer_s']}/{raw['scoring']['cer_d']}/{raw['scoring']['cer_i']}`; N `{raw['scoring']['cer_n']}`; empty/non-empty `{raw['scoring']['empty_count']}/{raw['scoring']['non_empty_count']}`.",
        f"- A latency: ASR RTF `{raw['latency']['ASR_RTF']:.6f}`, total compute RTF `{raw['latency']['total_compute_RTF']:.6f}`, TTFT P50/P95 `{raw['latency']['TTFT']['p50']:.3f}/{raw['latency']['TTFT']['p95']:.3f}` sec, segment-end-to-FINAL P50/P95 `{raw['latency']['segment_end_to_FINAL']['p50']:.3f}/{raw['latency']['segment_end_to_FINAL']['p95']:.3f}` sec, backlog P95/max `{raw['latency']['P95_backlog_sec']:.3f}/{raw['latency']['max_backlog_sec']:.3f}` sec.",
        "- B qualified streaming frontend: not run because no reusable true-streaming frontend was found; therefore no streaming CER gain or latency claim is made.",
        f"- C offline GSS upper bound: CER `{offline_cer:.6f}`; S/D/I `{offline['scoring']['cer_s']}/{offline['scoring']['cer_d']}/{offline['scoring']['cer_i']}`; absolute gain vs raw `{absolute_gain:.6f}`, relative reduction `{relative_gain:.3%}`." if relative_gain is not None else "- C offline GSS upper bound: relative gain unavailable.",
        "- C is explicitly `OFFLINE / NOT ONLINE-COMPARABLE FOR LATENCY`; its frontend RTF, TTFT, backlog, and sustained realtime status are not used in the online decision.",
        "",
        "## 4. Robustness and Leakage",
        "",
        f"- Paired scope comparison C minus A: `{robustness['offline_minus_raw_cer_delta']['improved_count']}` improved, `{robustness['offline_minus_raw_cer_delta']['worse_count']}` worse, `{robustness['offline_minus_raw_cer_delta']['tie_count']}` tied; median delta `{robustness['offline_minus_raw_cer_delta']['median']:.6f}`.",
        "- One meeting was used as authorized; a multi-meeting bootstrap is not claimed.",
        "- A uses no reference transcript or future RTTM in the decoder path, but retains accepted PR #8 Stage-2 scope-boundary oracle semantics.",
        "- B future perturbation test is not applicable because B was not available. C has explicit future/session Stage-2 guidance and is offline only.",
        "",
        "## 5. Recommendation",
        "",
        "For the RoboHEARD online track, keep the accepted raw_ch0 + WenetSpeech U2++ native-streaming baseline as the qualified online configuration. Do not promote OnlineFIVE, offline GSS, NBSS, or AV-SkiM as a generic causal frontend without a separate contract review and qualification.",
        "",
        "## Artifacts",
        "",
        "- `frontend_inventory.md`",
        "- `dataset_inventory.md` / `dataset_inventory.json`",
        "- `experiment_contract.json`",
        "- `raw_baseline_report.json`",
        "- `streaming_frontend_report.json`",
        "- `offline_upper_bound_report.json`",
        "- `meeting_level_results.csv`",
        "- `robustness_summary.json`",
        "- `ablation_summary.md`",
    ]
    (output_root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far"))
    parser.add_argument("--source-dir", type=Path, default=Path("/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename"))
    parser.add_argument("--enhanced-root", type=Path, default=Path("/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/outputs/dev_far_audio_segment/enhanced"))
    parser.add_argument("--recording-id", default="M028_S197199201241_F8N_Far")
    parser.add_argument("--output-root", type=Path, default=Path("/home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation"))
    parser.add_argument("--library-path", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--packet-samples", type=int, default=320)
    parser.add_argument("--model-chunk-samples", type=int, default=320)
    parser.add_argument("--wenet-chunk-size", type=int, default=DEFAULT_WENET_CHUNK_SIZE)
    parser.add_argument("--warmup-samples", type=int, default=16000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.expanduser().resolve(strict=False)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_frontend_inventory(output_root / "frontend_inventory.md")
    inventory = _inventory_dataset(args, output_root)
    _write_contract(args, output_root, inventory)
    raw = _run_raw(args, output_root, inventory)
    _write_not_run_offline_report(output_root)
    _write_not_run_streaming_report(output_root)
    robustness = _write_current_summary(output_root, inventory, raw)
    _write_current_final_report(output_root, inventory, raw, robustness)
    print(json.dumps({"status": "complete", "output_root": str(output_root), "condition_A": raw["scoring"], "condition_B": "NOT_RUN", "condition_C": "NOT_RUN_BY_USER_INSTRUCTION"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
