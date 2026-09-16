# RoboHEARD / MISP ASR Backend

This repository contains the ASR integration work used to prototype MISP AVDR components before RoboHEARD 2027 data and baselines are released.

## Current backend contract

The ASR layer consumes a manifest whose records point to **one target waveform per segment**. Upstream components such as WPE/GSS, SpatialNet, MossFormer2, beamforming, channel selection, and diarization remain outside the Whisper backend.

The backend must never silently average a MISP CSOBx3 multi-channel recording. A raw far-field experiment must explicitly identify its input, for example `raw_ch0`, `beamformed`, or `gss`.

OpenAI Whisper `large-v3` frozen inference is the reference backend. `faster-whisper` will only be added after the reference backend passes the V0 gates and server validation.

## Repository layout

```text
configs/                     Runtime and scoring configuration
src/robot_heard/asr/         ASR backend, batch/resume runner, scoring, result contracts
src/robot_heard/audio/       Audio inspection and explicit mono/16-kHz preparation
src/robot_heard/io/          Manifest contract and JSONL utilities
src/robot_heard/misp/        Challenge-specific MISP export adapters
scripts/                     User-facing smoke, batch, scoring, and export entry points
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

The OpenAI Whisper backend also probes every input before decoding and refuses non-mono or non-16-kHz files. This prevents callers from bypassing the S3 policy and relying on Whisper's internal ffmpeg path to downmix/resample silently.

Therefore MISP CSOBx3 raw 8-channel PCM must first go through an explicit upstream channel-selection, beamforming, separation, or conversion step. The ASR layer never guesses that policy.

## Single-file Whisper smoke test

```bash
python scripts/smoke_openai_whisper.py \
  --manifest /path/to/smoke_manifest.jsonl \
  --config configs/whisper_openai_v0.yaml
```

The script validates the manifest before model loading, loads `large-v3` once, checks the S3 audio boundary, transcribes the selected segment, and prints JSON containing the segment id, text, language, and decode time.

## Batch decode and resume (S4)

Batch decoding uses one backend instance for the whole run and writes each successful segment immediately to the result JSONL with `flush + fsync`. Per-segment failures are appended to a separate error JSONL and do not stop later segments.

```bash
python scripts/decode_manifest.py \
  --manifest /path/to/manifest.jsonl \
  --config configs/whisper_openai_v0.yaml \
  --output /path/to/results.jsonl
```

Default sidecars for `results.jsonl` are:

```text
results.run.json      exact manifest/config hashes, full config, backend/model, code commit
results.errors.jsonl  append-only failure-attempt history (created only after a failure)
```

A fresh run refuses to overwrite existing run artifacts. To continue an interrupted run:

```bash
python scripts/decode_manifest.py \
  --manifest /path/to/manifest.jsonl \
  --config configs/whisper_openai_v0.yaml \
  --output /path/to/results.jsonl \
  --resume
```

Resume semantics are intentionally strict:

- only prior `status="success"` result records are skipped;
- prior failures are retried and their old error entries remain as history;
- manifest/config/code provenance must match the original run metadata;
- a trailing partial JSONL record caused by interruption is truncated before resume;
- malformed completed lines, duplicate completed IDs, or results from another manifest cause a hard resume error;
- if every segment is already complete, `--resume` does not load the Whisper model again.

S4 writes `text_raw`, `audio_sec`, and `decode_sec`. The `text_norm`, `reference_norm`, and `rtf` keys are intentionally left as `null` placeholders in the raw batch output; S5 fills scoring fields in a **separate scored JSONL** without rerunning Whisper or mutating the raw output.

The command prints progress events to stderr and a final JSON summary to stdout. If any segment fails, decoding still finishes the remaining manifest but the process exits non-zero after printing the summary so failures cannot be missed silently.

## Normalization, CER, and RTF (S5)

S5 is a post-processing stage over successful S4 result JSONL. This separation is deliberate: `text_raw` and the original timing records remain auditable, and future official MISP normalization/scoring can be substituted without rerunning Whisper.

The local V0 scoring config is explicit:

```yaml
normalization_policy: v0_nfkc_casefold_strip_space_punct
```

The policy performs Unicode NFKC normalization, Unicode `casefold()`, and removes Unicode whitespace plus punctuation characters. It does **not** perform traditional/simplified Chinese conversion, number verbalization, event-tag removal, or other challenge-specific text rewriting.

This policy is a local diagnostic baseline only. **The official MISP normalization/scoring script is authoritative once available.** Results from this policy must not be described as official MISP CER unless parity with the official scorer has been verified.

Run scoring independently from inference:

```bash
python scripts/score_results.py \
  --input /path/to/results.jsonl \
  --config configs/scoring_v0.yaml \
  --output /path/to/results.scored.jsonl
```

The scored output preserves every raw field and adds/fills:

```text
result_schema_version=2
normalization_policy
text_norm
reference_norm              # null if no reference exists
rtf                         # decode_sec / audio_sec
cer_s / cer_d / cer_i / cer_n
cer                         # null when reference is absent or normalized N=0
```

Character error counts are computed with deterministic Levenshtein alignment. Segment CER is `(S + D + I) / N` when `N > 0`. For records without `reference`, normalization and RTF are still produced while all CER/reference scoring fields remain null.

A metrics sidecar is written next to the scored JSONL by default (for example `results.scored.metrics.json`). It records input/config SHA-256 values, full scoring config, Git commit, record counts, summed audio/decode seconds, **global RTF = sum(decode_sec) / sum(audio_sec)**, aggregate S/D/I/N, and aggregate global CER. Global CER is computed from aggregate edit counts, not by averaging per-segment CER values.

Scoring is atomic: malformed input or a scoring validation failure leaves no partial output. Input and output paths must differ so the original S4 result JSONL remains intact. Existing scored output/summary files are not overwritten unless `--overwrite` is explicitly provided.

## MISP 2025 Task 2 submission export (S6)

S6 is challenge-specific post-processing and never reruns Whisper. The official MISP 2025 Task 2 / AVSR submission page documents a zip package named `summission.zip` containing a single root file named `summission.txt`; each transcript row is `segment_id transcript`, and the transcript part contains Chinese characters only without punctuation. The exporter intentionally preserves the spelling used by the official challenge page.

```bash
python scripts/export_misp_task2.py \
  --input /path/to/results.jsonl \
  --output-dir /path/to/misp_task2_submission \
  --expected-segments /path/to/authoritative_segment_list.txt
```

The output directory contains:

```text
summission.txt          human-inspectable transcript
summission.zip          upload package; zip root contains only summission.txt
summission.export.json  input/output hashes, policy, code commit, expected-ID evidence
```

The exporter always uses `text_raw`; it does not reuse the local S5 normalizer as if it were an official MISP scorer. Its submission-surface policy applies NFKC, removes Unicode whitespace/punctuation, and then **rejects** remaining non-Han semantic content (for example Latin letters or Arabic digits) rather than silently deleting it. Empty post-normalization transcripts are also rejected. This makes policy gaps visible during integration instead of corrupting text silently.

`--expected-segments` is optional for module-level experiments but required by the S6I real-integration Gate. It accepts one segment ID per line, official-style `segment_id transcript` lines, or manifest JSONL containing `segment_id`; the exporter then requires exact ID-set equality before writing submission artifacts.

The MISP 2025 Task 3 / AVDR submission contract is different: it requires per-session text files with `local_speaker_id transcript`. Speaker assignment remains an upstream diarization/AVDR responsibility and is deliberately not guessed inside this ASR exporter.

## MISP Integration Compatibility (S6I)

Passing module tests is not sufficient to claim that the backend is MISP-ready. Before S7, at least one real MISP Dev session must be run from actual baseline segmentation/reference/frontend artifacts through manifest/waveform preparation, Whisper batch inference, S5 scoring, and S6 export without hand-editing intermediate files. Segment counts and exact ID sets must reconcile at each stage. If a runnable official scorer/evaluator is available, parity must also be checked; otherwise the limitation remains explicitly `PENDING_EXTERNAL_SCORER_PARITY`.

## Tests

```bash
python -m pytest -q
```

Audio integration tests use `ffmpeg`/`ffprobe` when available and are skipped when those executables are absent. Real MISP audio and 5090 validation remain separate Gate evidence.

## Project authority

Long-term execution rules, gate definitions, reporting requirements, and server-validation policy are maintained in GitHub Issue #1 (`Whisper Backend V0 — Execution Charter`).
