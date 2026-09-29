import json
from pathlib import Path

import pytest

import robot_heard.asr.batch as batch_module
from robot_heard.asr.base import ASRResult
from robot_heard.asr.batch import (
    BatchResumeError,
    build_batch_provenance,
    run_batch,
    run_metadata_path,
)
from robot_heard.audio import AudioMetadata


class _FakeBackend:
    def __init__(self, *, fail_names=()):
        self.fail_names = set(fail_names)
        self.calls = []

    def transcribe(self, audio_path):
        path = Path(audio_path)
        self.calls.append(path.name)
        if path.name in self.fail_names:
            raise RuntimeError(f"synthetic failure for {path.name}")
        return ASRResult(
            text=f"text:{path.stem}",
            language="zh",
            decode_sec=0.25,
        )


def _write_manifest(tmp_path: Path, names):
    records = []
    for index, name in enumerate(names):
        audio = tmp_path / name
        audio.touch()
        records.append(
            {
                "segment_id": f"seg-{index}",
                "audio_path": audio.name,
                "frontend": "near_field_f8n",
                "custom_metadata": f"m{index}",
            }
        )

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest, records


def _write_config(tmp_path: Path, *, marker="v1"):
    config = {
        "backend": "fake_backend",
        "model": "fake_model",
        "marker": marker,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"backend: fake_backend\nmodel: fake_model\nmarker: {marker}\n",
        encoding="utf-8",
    )
    return config_path, config


def _provenance(manifest: Path, config_path: Path, config):
    return build_batch_provenance(
        manifest_path=manifest,
        config_path=config_path,
        config=config,
        backend=config["backend"],
        model=config["model"],
        code_commit="deadbeef",
    )


def _fake_probe(path):
    resolved = Path(path).resolve()
    return AudioMetadata(
        path=resolved,
        sample_rate=16000,
        channels=1,
        duration_sec=2.0,
        codec_name="pcm_s16le",
    )


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_batch_writes_success_records_and_loads_backend_once(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_module, "probe_audio", _fake_probe)
    manifest, _ = _write_manifest(tmp_path, ["a.wav", "b.wav"])
    config_path, config = _write_config(tmp_path)
    provenance = _provenance(manifest, config_path, config)
    output = tmp_path / "results.jsonl"

    backend = _FakeBackend()
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return backend

    summary = run_batch(
        manifest_path=manifest,
        output_path=output,
        backend_factory=factory,
        provenance=provenance,
    )

    assert summary.total == 2
    assert summary.attempted == 2
    assert summary.success == 2
    assert summary.failure == 0
    assert summary.skipped == 0
    assert factory_calls == [True]
    assert backend.calls == ["a.wav", "b.wav"]

    records = _read_jsonl(output)
    assert [record["segment_id"] for record in records] == ["seg-0", "seg-1"]
    assert all(record["status"] == "success" for record in records)
    assert all(record["text_norm"] is None for record in records)
    assert all(record["rtf"] is None for record in records)
    assert all(record["audio_sec"] == 2.0 for record in records)
    assert records[0]["custom_metadata"] == "m0"

    metadata = json.loads(run_metadata_path(output).read_text(encoding="utf-8"))
    assert metadata["config"] == config
    assert metadata["manifest_sha256"] == provenance.manifest_sha256
    assert metadata["config_sha256"] == provenance.config_sha256
    assert metadata["code_commit"] == "deadbeef"


def test_batch_continues_after_segment_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_module, "probe_audio", _fake_probe)
    manifest, _ = _write_manifest(tmp_path, ["good.wav", "bad.wav"])
    config_path, config = _write_config(tmp_path)
    provenance = _provenance(manifest, config_path, config)
    output = tmp_path / "results.jsonl"

    backend = _FakeBackend(fail_names={"bad.wav"})
    summary = run_batch(
        manifest_path=manifest,
        output_path=output,
        backend_factory=lambda: backend,
        provenance=provenance,
    )

    assert summary.success == 1
    assert summary.failure == 1
    assert backend.calls == ["good.wav", "bad.wav"]

    successes = _read_jsonl(output)
    failures = _read_jsonl(Path(summary.error_path))
    assert [record["segment_id"] for record in successes] == ["seg-0"]
    assert failures[0]["segment_id"] == "seg-1"
    assert failures[0]["status"] == "failure"
    assert failures[0]["error_type"] == "RuntimeError"
    assert "synthetic failure" in failures[0]["error_message"]


def test_resume_skips_successes_and_retries_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_module, "probe_audio", _fake_probe)
    manifest, _ = _write_manifest(tmp_path, ["a.wav", "b.wav", "c.wav"])
    config_path, config = _write_config(tmp_path)
    provenance = _provenance(manifest, config_path, config)
    output = tmp_path / "results.jsonl"

    first_backend = _FakeBackend(fail_names={"b.wav"})
    first = run_batch(
        manifest_path=manifest,
        output_path=output,
        backend_factory=lambda: first_backend,
        provenance=provenance,
    )
    assert first.success == 2
    assert first.failure == 1

    second_backend = _FakeBackend()
    factory_calls = []

    def second_factory():
        factory_calls.append(True)
        return second_backend

    second = run_batch(
        manifest_path=manifest,
        output_path=output,
        backend_factory=second_factory,
        provenance=provenance,
        resume=True,
    )

    assert second.attempted == 1
    assert second.success == 1
    assert second.failure == 0
    assert second.skipped == 2
    assert factory_calls == [True]
    assert second_backend.calls == ["b.wav"]

    results = _read_jsonl(output)
    assert len(results) == 3
    assert {record["segment_id"] for record in results} == {"seg-0", "seg-1", "seg-2"}


def test_resume_repairs_partial_last_line_and_avoids_model_load(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_module, "probe_audio", _fake_probe)
    manifest, _ = _write_manifest(tmp_path, ["a.wav"])
    config_path, config = _write_config(tmp_path)
    provenance = _provenance(manifest, config_path, config)
    output = tmp_path / "results.jsonl"

    backend = _FakeBackend()
    run_batch(
        manifest_path=manifest,
        output_path=output,
        backend_factory=lambda: backend,
        provenance=provenance,
    )

    with output.open("ab") as handle:
        handle.write(b'{"segment_id":"partial"')

    def must_not_load():
        raise AssertionError("backend must not load when all manifest segments are complete")

    summary = run_batch(
        manifest_path=manifest,
        output_path=output,
        backend_factory=must_not_load,
        provenance=provenance,
        resume=True,
    )

    assert summary.repaired_trailing_partial is True
    assert summary.skipped == 1
    assert summary.attempted == 0
    assert len(_read_jsonl(output)) == 1
    assert output.read_bytes().endswith(b"\n")


def test_resume_rejects_changed_config(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_module, "probe_audio", _fake_probe)
    manifest, _ = _write_manifest(tmp_path, ["a.wav"])
    config_path, config = _write_config(tmp_path, marker="v1")
    provenance = _provenance(manifest, config_path, config)
    output = tmp_path / "results.jsonl"

    run_batch(
        manifest_path=manifest,
        output_path=output,
        backend_factory=lambda: _FakeBackend(),
        provenance=provenance,
    )

    config_path, changed_config = _write_config(tmp_path, marker="v2")
    changed_provenance = _provenance(manifest, config_path, changed_config)

    with pytest.raises(BatchResumeError, match="resume metadata mismatch"):
        run_batch(
            manifest_path=manifest,
            output_path=output,
            backend_factory=lambda: _FakeBackend(),
            provenance=changed_provenance,
            resume=True,
        )


def test_fresh_run_refuses_existing_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_module, "probe_audio", _fake_probe)
    manifest, _ = _write_manifest(tmp_path, ["a.wav"])
    config_path, config = _write_config(tmp_path)
    provenance = _provenance(manifest, config_path, config)
    output = tmp_path / "results.jsonl"
    output.write_text("", encoding="utf-8")

    with pytest.raises(FileExistsError, match="use --resume"):
        run_batch(
            manifest_path=manifest,
            output_path=output,
            backend_factory=lambda: _FakeBackend(),
            provenance=provenance,
        )


def test_resume_rejects_changed_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_module, "probe_audio", _fake_probe)
    manifest, records = _write_manifest(tmp_path, ["a.wav"])
    config_path, config = _write_config(tmp_path)
    provenance = _provenance(manifest, config_path, config)
    output = tmp_path / "results.jsonl"

    run_batch(
        manifest_path=manifest,
        output_path=output,
        backend_factory=lambda: _FakeBackend(),
        provenance=provenance,
    )

    manifest.write_text(
        json.dumps(records[0]) + "\n\n",
        encoding="utf-8",
    )
    changed_provenance = _provenance(manifest, config_path, config)

    with pytest.raises(BatchResumeError, match="resume metadata mismatch"):
        run_batch(
            manifest_path=manifest,
            output_path=output,
            backend_factory=lambda: _FakeBackend(),
            provenance=changed_provenance,
            resume=True,
        )
