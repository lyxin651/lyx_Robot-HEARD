# RoboHEARD / MISP ASR Backend

This repository contains the ASR integration work used to prototype MISP AVDR components before RoboHEARD 2027 data and baselines are released.

## Current backend contract

The ASR layer consumes a manifest whose records point to **one target waveform per segment**. Upstream components such as WPE/GSS, SpatialNet, MossFormer2, beamforming, channel selection, and diarization remain outside the Whisper backend.

The backend must never silently average a MISP CSOBx3 multi-channel recording. A raw far-field experiment must explicitly identify its input, for example `raw_ch0`, `beamformed`, or `gss`.

OpenAI Whisper `large-v3` frozen inference is the reference backend. `faster-whisper` will only be added after the reference backend passes the V0 gates and server validation.

## Repository layout

```text
configs/                     Runtime configuration
src/robot_heard/asr/         ASR backend interface and implementations
src/robot_heard/audio/       Audio inspection and explicit mono/16-kHz preparation
src/robot_heard/io/          Manifest contract and JSONL utilities
scripts/                     User-facing smoke/evaluation entry points
tests/                       Unit/integration tests
```

## Environment setup

A CUDA-enabled PyTorch installation appropriate for the 5090 server should be installed/verified separately. Then install the project and Python dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

OpenAI Whisper and the S3 audio layer require `ffmpeg`/`ffprobe` on the system path.

## Manifest contract

Minimum JSONL record:

```json
{"segment_id":"M005_SPK01_000001","audio_path":"/absolute/path/to/segment.wav"}
```

Recommended metadata can include `session_id`, `speaker_id`, `start`, `duration`, `frontend`, and (for development scoring) `reference`. Relative `audio_path` values are resolved relative to the manifest file.

## Audio input contract (S3)

Whisper V0 uses an explicit audio policy:

```yaml
audio:
  target_sample_rate: 16000
  require_mono: true
```

`robot_heard.audio.probe_audio()` records source sample rate, channel count, duration, codec, and resolved path using `ffprobe`.

`robot_heard.audio.prepare_audio()` behaves deliberately:

- mono 16-kHz audio is passed through unchanged;
- mono audio at another sample rate is resampled to a caller-specified PCM-s16le WAV using `ffmpeg`;
- multi-channel input is rejected **before conversion** rather than averaged/downmixed;
- resampling never overwrites the source file and requires an explicit output path;
- headerless raw PCM is not inferred automatically because sample rate/channel layout are not self-describing.

Therefore MISP CSOBx3 raw 8-channel PCM must first go through an explicit upstream channel-selection, beamforming, separation, or conversion step. The ASR layer never guesses that policy.

## Single-file Whisper smoke test

```bash
python scripts/smoke_openai_whisper.py \
  --manifest /path/to/smoke_manifest.jsonl \
  --config configs/whisper_openai_v0.yaml
```

The script validates the manifest before model loading, loads `large-v3` once, transcribes the selected segment, and prints JSON containing the segment id, text, language, and decode time.

## Tests

```bash
python -m pytest -q
```

Audio integration tests use `ffmpeg`/`ffprobe` when available and are skipped when those executables are absent. Real MISP audio and 5090 validation remain separate Gate evidence.

## Project authority

Long-term execution rules, gate definitions, reporting requirements, and server-validation policy are maintained in GitHub Issue #1 (`Whisper Backend V0 — Execution Charter`).
