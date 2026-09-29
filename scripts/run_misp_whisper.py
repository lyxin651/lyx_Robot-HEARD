#!/usr/bin/env python3
"""Run the maintained MISP-baseline -> Whisper integration path.

The command consumes already-produced baseline artifacts. It does not run or
modify baseline segmentation, GSS, diarization, or WeNet stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from robot_heard.asr.batch import build_batch_provenance, default_error_path, run_batch
from robot_heard.asr.factory import create_asr_backend
from robot_heard.asr.scoring import score_results_jsonl
from robot_heard.misp.baseline_integration import (
    BaselineIntegrationError,
    prepare_baseline_manifest,
    read_artifact_ids,
    reconcile_ids,
    reconcile_manifest_file,
)
from robot_heard.misp.kaldi_adapter import read_kaldi_segments
from robot_heard.misp.task2_export import export_misp2025_task2_submission


STAGES = ("prepare", "decode", "score", "export", "all")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a YAML mapping: {path}")
    return value


def _expected_ids(baseline_dir: Path, limit: Optional[int]) -> Sequence[str]:
    segments = read_kaldi_segments(baseline_dir / "segments")
    selected = segments if limit is None else segments[:limit]
    return [segment.segment_id for segment in selected]


def _existing_ids(path: Path, *, allow_duplicate: bool = False):
    if not path.is_file():
        return []
    return read_artifact_ids(path, allow_duplicate=allow_duplicate)


def _prepare(args: argparse.Namespace, *, repo_root: Path) -> Dict[str, Any]:
    output_dir: Path = args.output_dir
    prepared_dir = output_dir / "prepared"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = prepared_dir / "manifest.jsonl"
    expected_ids = _expected_ids(args.baseline_dir, args.limit)

    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"prepared manifest already exists; use --resume or --overwrite: {manifest_path}"
            )
        reconciliation = reconcile_manifest_file(
            args.baseline_dir, manifest_path, limit=args.limit
        )
        metadata_path = prepared_dir / "adapter.json"
        return {
            "status": "reused",
            "manifest_path": str(manifest_path),
            "metadata_path": str(metadata_path),
            "reconciliation": reconciliation,
        }

    summary = prepare_baseline_manifest(
        args.baseline_dir,
        prepared_dir,
        frontend=args.frontend,
        recording_id=args.recording_id,
        channel_id=args.channel_id,
        waveform_scp=args.waveform_scp,
        reference_mode=args.reference_mode,
        limit=args.limit,
        overwrite=args.overwrite,
        code_commit=_git_head(repo_root),
    )
    manifest_ids = read_artifact_ids(summary.manifest_path)
    reconciliation = reconcile_ids(expected_ids, manifest_ids=manifest_ids)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "prepare",
        "baseline_dir": str(args.baseline_dir),
        "baseline_segments_sha256": _sha256_file(args.baseline_dir / "segments"),
        "frontend": args.frontend,
        "recording_id": args.recording_id,
        "channel_id": args.channel_id,
        "waveform_scp": str(args.waveform_scp) if args.waveform_scp else None,
        "reference_mode": args.reference_mode,
        "limit": args.limit,
        "manifest_path": summary.manifest_path,
        "manifest_sha256": _sha256_file(Path(summary.manifest_path)),
        "expected_ids": reconciliation,
        "code_commit": _git_head(repo_root),
    }
    _atomic_write_json(payload, output_dir / "integration.run.json")
    return {"status": "created", **summary.to_dict(), "reconciliation": reconciliation}


def _decode(args: argparse.Namespace, *, repo_root: Path) -> Dict[str, Any]:
    if args.config is None:
        raise ValueError("--config is required for decode/score/export/all stages")
    manifest_path = args.output_dir / "prepared" / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepared manifest does not exist: {manifest_path}")
    config = _load_yaml(args.config)
    backend_name = config.get("backend")
    model_name = config.get("model")
    if not isinstance(backend_name, str) or not backend_name.strip():
        raise ValueError("Whisper config key 'backend' must be a non-empty string")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Whisper config key 'model' must be a non-empty string")

    output_path = args.output_dir / "results.jsonl"
    error_path = default_error_path(output_path)
    provenance = build_batch_provenance(
        manifest_path=manifest_path,
        config_path=args.config,
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
    expected_ids = _expected_ids(args.baseline_dir, args.limit)
    result_ids = _existing_ids(output_path)
    failure_ids = _existing_ids(error_path, allow_duplicate=True)
    reconciliation = reconcile_ids(
        expected_ids,
        manifest_ids=read_artifact_ids(manifest_path),
        result_ids=result_ids,
        failure_ids=failure_ids,
    )
    return {"summary": summary.to_dict(), "reconciliation": reconciliation}


def _score(args: argparse.Namespace, *, repo_root: Path) -> Dict[str, Any]:
    input_path = args.output_dir / "results.jsonl"
    if not input_path.is_file():
        raise FileNotFoundError(f"decode result does not exist: {input_path}")
    error_path = default_error_path(input_path)
    failure_ids = _existing_ids(error_path, allow_duplicate=True)
    if failure_ids:
        raise BaselineIntegrationError(
            "cannot produce a complete S5 artifact while explicit decode failures "
            f"remain ({len(set(failure_ids))} segment IDs); resolve/resume first"
        )
    output_path = args.output_dir / "results.scored.jsonl"
    summary = score_results_jsonl(
        input_path,
        output_path,
        normalization_policy=_load_yaml(args.scoring_config)["normalization_policy"],
        text_field=args.text_field,
        overwrite=args.overwrite,
    )
    expected_ids = _expected_ids(args.baseline_dir, args.limit)
    scored_ids = read_artifact_ids(output_path)
    reconciliation = reconcile_ids(expected_ids, manifest_ids=read_artifact_ids(
        args.output_dir / "prepared" / "manifest.jsonl"
    ), scored_ids=scored_ids)
    metrics = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "scoring_config_path": str(args.scoring_config),
        "scoring_config_sha256": _sha256_file(args.scoring_config),
        "text_field": args.text_field,
        "code_commit": _git_head(repo_root),
        "summary": summary.to_dict(),
        "reconciliation": reconciliation,
    }
    _atomic_write_json(metrics, args.output_dir / "results.scored.metrics.json")
    return metrics


def _export(args: argparse.Namespace, *, repo_root: Path) -> Dict[str, Any]:
    input_path = args.output_dir / (
        "results.scored.jsonl" if (args.output_dir / "results.scored.jsonl").is_file()
        else "results.jsonl"
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"result artifact does not exist: {input_path}")
    expected_path = args.output_dir / "prepared" / "manifest.jsonl"
    summary = export_misp2025_task2_submission(
        input_path,
        args.output_dir / "submission",
        expected_segments_path=expected_path,
        text_field=args.text_field,
        overwrite=args.overwrite,
        code_commit=_git_head(repo_root),
    )
    return summary.to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the maintained MISP baseline artifact -> Whisper V0 -> S5/S6 "
            "integration path. Baseline frontend stages are upstream inputs."
        )
    )
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frontend", required=True)
    parser.add_argument("--recording-id")
    parser.add_argument("--channel-id", type=int)
    parser.add_argument(
        "--waveform-scp",
        type=Path,
        help="Segment-ID keyed waveform SCP for a pre-materialized frontend.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("auto", "required", "off"),
        default="auto",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--scoring-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "scoring_v0.yaml",
    )
    parser.add_argument("--text-field", default="text_raw")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and reconcile artifacts only; never constructs a Whisper backend.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only explicitly selected output artifacts; default is fail-closed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.baseline_dir = args.baseline_dir.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve(strict=False)
    if args.waveform_scp is not None:
        args.waveform_scp = args.waveform_scp.expanduser().resolve(strict=True)
    if args.config is not None:
        args.config = args.config.expanduser().resolve(strict=True)
    args.scoring_config = args.scoring_config.expanduser().resolve(strict=True)

    if args.prepare_only or args.dry_run:
        args.stage = "prepare"
    repo_root = Path(__file__).resolve().parents[1]
    result: Dict[str, Any] = {"stage": args.stage, "dry_run": args.dry_run}
    if args.stage in {"prepare", "all"}:
        result["prepare"] = _prepare(args, repo_root=repo_root)
    if args.dry_run or args.stage == "prepare":
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.stage in {"decode", "all"}:
        result["decode"] = _decode(args, repo_root=repo_root)
    if args.stage in {"score", "all"}:
        result["score"] = _score(args, repo_root=repo_root)
    if args.stage in {"export", "all"}:
        result["export"] = _export(args, repo_root=repo_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
