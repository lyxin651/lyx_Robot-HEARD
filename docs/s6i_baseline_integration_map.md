# S6I baseline-main-flow integration map

Status: B0 audit, based on the local MISP AVSR baseline and persisted M028
artifacts inspected on 2026-09-29.

## Authoritative baseline call path

```text
raw MISP data
  -> AVSR Stage 1 (`scipts/prep_far_avsr.sh`)
  -> renamed 8-channel far-field WAV + renamed TextGrid
  -> AVSR Stage 2 (`scipts/prep_far_avsr_gss.sh`)
  -> Kaldi data dir: segments/text/utt2spk/spk2utt/wav.scp/channels.scp
  -> optional Stage 3 GSS
  -> explicit frontend waveform artifact
  -> Whisper manifest
  -> existing `scripts/decode_manifest.py` (S4)
  -> existing `scripts/score_results.py` (S5, Dev only)
  -> existing `scripts/export_misp_task2.py` (S6, strict)
  -> downstream handoff
```

The baseline's current WeNet path starts after this boundary. In
`s0/run_asr_far.sh`, Stage 4 converts `wav.scp` + `text` (and optionally
`segments`) to `data.list`; later stages train/decode WeNet. There is no
maintained Whisper invocation in that script. The Whisper integration should
remain a sidecar entry point over already-produced baseline artifacts rather
than rewriting the baseline's GSS, diarization, or WeNet stages.

## Ownership and artifact contracts

| Baseline artifact/stage | Owner | Contract used by integration |
| --- | --- | --- |
| Stage-1 PCM conversion, 8-channel WAV, TextGrid slicing/rename | MISP baseline wrapper | Produces explicit source recordings; no Whisper logic |
| Stage-2 segmentation/reference preparation | `prepare_gss_data.py` path | `segments` is authoritative for IDs/times; `text` is optional on Eval and required only for Dev scoring |
| `wav.scp` / `channels.scp` | baseline frontend data dir | Recording stem mappings; channel selection is explicit and never averaged |
| Stage-3 GSS | baseline/GSS path | Provides frontend-specific materialized waveforms; Whisper only consumes them |
| Whisper manifest | integration adapter | One target waveform per authoritative segment, explicit `frontend`, source/session metadata |
| S4/S5/S6 | existing Whisper modules | Reused without duplication; S6 remains fail-closed |

The local M028 Stage-2 contract is
`/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far` with
144 authoritative segments and 144 references. Its source recording stem is
the eight explicit files
`dev_far_audio_multi_channel_rename/M028_S197199201241_F8N_Far_0.wav` through
`_7.wav`. The persisted full-GSS result is a separate historical artifact;
its provenance is not used to relax the adapter contract.

## Recommended integration point

Add one maintained baseline-facing orchestrator under `whisper_asr/scripts/`
and its library adapter under `src/robot_heard/misp/`. It will:

1. validate the selected baseline Kaldi directory and authoritative IDs;
2. select an explicit target frontend (initially `raw_ch0`..`raw_ch7`, with
   pre-materialized frontend support requiring exact ID equality);
3. create/reuse a validated Whisper manifest and reconciliation metadata;
4. optionally call the existing S4 decoder, S5 scorer, and strict S6 exporter;
5. write a run metadata/provenance record and fail closed on any ID/path or
   reference mismatch.

No file in the MISP baseline repository needs to change for this integration.
The baseline remains responsible for segmentation, channel materialization,
GSS, and any diarization. The Whisper repo owns only the adapter/orchestrator
and its CPU-testable contract.

## Important incompatibility found during audit

The current baseline Stage-3 exported `wav.scp` uses IDs such as
`..._014152_015076`, while the Stage-2 authoritative `segments` uses
`..._014152-015076`. This is a real ID-set mismatch, not a cosmetic naming
choice. The integration must not silently regenerate or rewrite IDs. A
pre-materialized frontend is accepted only when its waveform map already uses
the authoritative IDs (or when an explicit, provenance-tracked upstream map is
provided in a future change).

## Files to modify in this repository

- `src/robot_heard/misp/kaldi_adapter.py`: optional Dev reference handling and
  exact artifact validation needed by the maintained entry point.
- `src/robot_heard/misp/baseline_integration.py`: baseline-facing contract,
  frontend selection, manifest/reconciliation/provenance orchestration.
- `scripts/run_misp_whisper.py`: one invocation path for prepare/dry-run,
  decode, optional score, and optional strict export.
- `tests/`: normal/failure/reconciliation/resume-facing contract tests; no GPU
  required.
- `README.md` and this document: operator-facing invocation and audit record.

The baseline repository and frozen OpenAI Whisper decoder policy are outside
the allowed modification scope.
