import json
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from robot_heard.misp.baseline_integration import (
    BaselineIntegrationError,
    prepare_baseline_manifest,
    reconcile_ids,
    reconcile_manifest_file,
)


FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE,
    reason="ffmpeg/ffprobe required for baseline integration tests",
)


def _write_wav(path: Path, *, seconds: float = 1.0, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * 16000)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack("<h", 0) * channels * frames)


def _raw_fixture(tmp_path: Path, *, with_text: bool = True):
    baseline = tmp_path / "kaldi"
    baseline.mkdir()
    stem = tmp_path / "source" / "rec"
    for channel_id in range(8):
        _write_wav(Path(f"{stem}_{channel_id}.wav"))
    (baseline / "segments").write_text(
        "seg-a rec 0.10 0.30\nseg-b rec 0.40 0.70\n", encoding="utf-8"
    )
    if with_text:
        (baseline / "text").write_text("seg-a 你好\nseg-b 世界\n", encoding="utf-8")
    (baseline / "wav.scp").write_text(f"rec {stem}\n", encoding="utf-8")
    (baseline / "channels.scp").write_text(f"rec {stem}\n", encoding="utf-8")
    return baseline


def _materialized_fixture(tmp_path: Path, *, ids=("seg-a", "seg-b")):
    baseline = _raw_fixture(tmp_path)
    source = tmp_path / "prepared"
    rows = []
    for index, segment_id in enumerate(ids):
        audio = source / f"{segment_id}.wav"
        _write_wav(audio)
        rows.append(f"{segment_id} {audio}\n")
    waveform_scp = tmp_path / "frontend.wav.scp"
    waveform_scp.write_text("".join(rows), encoding="utf-8")
    return baseline, waveform_scp


@requires_ffmpeg
def test_raw_frontend_has_explicit_channel_and_optional_reference(tmp_path):
    baseline = _raw_fixture(tmp_path, with_text=False)
    summary = prepare_baseline_manifest(
        baseline,
        tmp_path / "out",
        frontend="raw_ch3",
        recording_id="rec",
        reference_mode="auto",
        limit=1,
    )

    assert summary.frontend == "raw_ch3"
    assert summary.reference_available is False
    row = json.loads((tmp_path / "out" / "manifest.jsonl").read_text().strip())
    assert row["frontend"] == "raw_ch3"
    assert row["channel_id"] == 3
    assert "reference" not in row
    assert json.loads((tmp_path / "out" / "adapter.json").read_text())["reference_available"] is False


@requires_ffmpeg
def test_required_reference_is_enforced(tmp_path):
    baseline = _raw_fixture(tmp_path, with_text=False)
    with pytest.raises(FileNotFoundError, match="text file does not exist"):
        prepare_baseline_manifest(
            baseline,
            tmp_path / "out",
            frontend="raw_ch0",
            recording_id="rec",
            reference_mode="required",
        )


@requires_ffmpeg
def test_pre_materialized_frontend_requires_exact_authoritative_ids(tmp_path):
    baseline, waveform_scp = _materialized_fixture(tmp_path)
    summary = prepare_baseline_manifest(
        baseline,
        tmp_path / "out",
        frontend="gss",
        waveform_scp=waveform_scp,
        reference_mode="auto",
    )
    assert summary.reference_available is True
    rows = [json.loads(line) for line in (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()]
    assert [row["segment_id"] for row in rows] == ["seg-a", "seg-b"]
    assert all(row["frontend"] == "gss" for row in rows)

    bad_scp = tmp_path / "bad.scp"
    bad_scp.write_text(f"seg-a {tmp_path / 'prepared' / 'seg-a.wav'}\n", encoding="utf-8")
    with pytest.raises(BaselineIntegrationError, match="exactly match"):
        prepare_baseline_manifest(
            baseline,
            tmp_path / "bad-out",
            frontend="gss",
            waveform_scp=bad_scp,
        )


@requires_ffmpeg
def test_invalid_multichannel_frontend_is_rejected(tmp_path):
    baseline = _raw_fixture(tmp_path)
    multichannel = tmp_path / "multi.wav"
    _write_wav(multichannel, channels=2)
    waveform_scp = tmp_path / "frontend.scp"
    waveform_scp.write_text(f"seg-a {multichannel}\nseg-b {multichannel}\n", encoding="utf-8")

    with pytest.raises(BaselineIntegrationError, match="mono 16 kHz"):
        prepare_baseline_manifest(
            baseline,
            tmp_path / "out",
            frontend="beamformed",
            waveform_scp=waveform_scp,
        )


@requires_ffmpeg
def test_manifest_reconciliation_and_missing_segment_fail_closed(tmp_path):
    baseline, waveform_scp = _materialized_fixture(tmp_path)
    prepare_baseline_manifest(
        baseline,
        tmp_path / "out",
        frontend="gss",
        waveform_scp=waveform_scp,
    )
    evidence = reconcile_manifest_file(baseline, tmp_path / "out" / "manifest.jsonl")
    assert evidence["exact"] is True
    assert evidence["authoritative_count"] == 2

    with pytest.raises(BaselineIntegrationError, match="reconciliation failed"):
        reconcile_ids(["seg-a", "seg-b"], manifest_ids=["seg-a", "seg-a"])


@requires_ffmpeg
def test_prepared_manifest_generation_is_deterministic(tmp_path):
    baseline, waveform_scp = _materialized_fixture(tmp_path)
    prepare_baseline_manifest(
        baseline,
        tmp_path / "out-a",
        frontend="gss",
        waveform_scp=waveform_scp,
    )
    prepare_baseline_manifest(
        baseline,
        tmp_path / "out-b",
        frontend="gss",
        waveform_scp=waveform_scp,
    )
    assert (tmp_path / "out-a" / "manifest.jsonl").read_bytes() == (
        tmp_path / "out-b" / "manifest.jsonl"
    ).read_bytes()


def test_id_reconciliation_rejects_extra_and_covers_results_plus_failures():
    with pytest.raises(BaselineIntegrationError, match="extra"):
        reconcile_ids(["a", "b"], manifest_ids=["a", "b", "c"])

    evidence = reconcile_ids(
        ["a", "b"],
        manifest_ids=["a", "b"],
        result_ids=["a"],
        failure_ids=["b"],
    )
    assert evidence["exact"] is True
    assert evidence["stages"]["results_plus_failures"]["count"] == 2


@requires_ffmpeg
def test_maintained_cli_dry_run_does_not_load_backend(tmp_path):
    baseline, waveform_scp = _materialized_fixture(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_misp_whisper.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline-dir",
            str(baseline),
            "--output-dir",
            str(tmp_path / "run"),
            "--frontend",
            "gss",
            "--waveform-scp",
            str(waveform_scp),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(script.parents[1] / "src")},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["stage"] == "prepare"
    assert payload["prepare"]["reconciliation"]["exact"] is True
    assert "torch" not in result.stderr.lower()
