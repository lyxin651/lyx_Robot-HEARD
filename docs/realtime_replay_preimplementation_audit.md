# Real-Time Replay Harness Pre-Implementation Audit

Audit date: 2026-09-29 (Asia/Shanghai)

Scope: read-only inspection of `/home/gc6_challenge/whisper_asr`, the local MISP
baseline checkout, and existing M028 work artifacts. No Whisper large-v3
inference, GSS/WPE/diarization job, streaming experiment, dependency install,
GitHub mutation, baseline mutation, or source/config/test change was performed.

Conclusion labels used below:

- `CONFIRMED`: directly supported by code, artifact contents, or a recorded command result.
- `INFERRED`: a bounded conclusion derived from confirmed evidence; not a frozen contract.
- `UNRESOLVED`: the inspected material does not establish the answer.

## 1. Executive Summary

`CONFIRMED`: the repository is a synchronous, file-oriented, segment-level ASR
package. The frozen public backend contract is:

```python
ASRBackend.transcribe(audio_path) -> ASRResult
```

`ASRResult` contains only `text`, `language`, and `decode_sec`. There is no
partial/final label, revision identifier, endpoint event, token timestamp,
confidence, persistent utterance state, reset, finalize, or flush operation.

`CONFIRMED`: M028 has a usable continuous replay source after baseline
preparation: eight separate, synchronized mono WAV files at
`/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename/`,
each 16 kHz, PCM 32-bit, 798.299 seconds. The authoritative M028 Stage-2
segments cover that prepared timeline; all 144 segment ends are within the
source and all checked materialized segments have exact expected sample counts.

`CONFIRMED`: the original CSOBx3 interleaved PCM also exists, but it is not on
the same zero point as the prepared/F8N segment timeline. M028 preparation uses
`timestamp.txt` from `1:22.463` to `14:40.762`; the crop offset is 82.463
seconds and the resulting duration is exactly 798.299 seconds. A replay source
that uses the original PCM therefore needs that provenance; the current
manifest/adapter metadata does not carry the raw-PCM crop offset as a field.

`CONFIRMED`: current raw-channel and GSS integration artifacts are already
segment artifacts. The raw adapter cuts one selected channel into one WAV per
segment. The persisted GSS frontend has 144 per-segment FLAC files. Neither is
a continuous frontend stream.

`CONFIRMED`: timing is limited to `time.perf_counter()` around the synchronous
Whisper model call. Model load is outside `decode_sec`; there is no explicit
CUDA synchronization, audio-clock event log, pacing scheduler, queue runtime,
or partial-output timestamping. Scoring accepts only successful final records.

`INFERRED`: the R0 design must preserve the existing offline contract as a
compatibility surface and establish any streaming contract separately. The
current `ASRBackend` cannot itself be treated as a unified streaming-consumer
interface without losing explicit lifecycle and event semantics.

## 2. Repository State

### Local repository

| Item | Evidence | Result |
|---|---|---|
| Path | `pwd` | `/home/gc6_challenge/whisper_asr` |
| Branch | `git branch --show-current` | `feat/whisper-backend-v0` |
| Starting HEAD | `git rev-parse HEAD` | `239cead78c42bf5c549cd5831cf43f99ba349b1e` |
| Worktree at audit start | `git status --short` | clean, empty output |
| Remote | `git remote -v` | `origin=https://github.com/lyxin651/lyx_Robot-HEARD.git` |
| Tracking ref | `git branch -vv` | local branch is two commits ahead of local `origin/feat/whisper-backend-v0` at `87ab962` |

The audit document is the only intended worktree addition. No code, shell,
YAML, config, test, MISP baseline, or source-data file is in scope for change.

### GitHub Issue / PR state

`CONFIRMED` from the public GitHub pages/API inspected on 2026-09-29:

- Issue #1, **Whisper Backend V0 — Execution Charter**, is open and is the
  project authority. It explicitly excludes streaming ASR from V0 and says to
  complete S6I before S7; it also requires feature-branch development,
  provenance, work reports in the PR, and no claim of freeze without server
  validation.
- Issue #3, **S6I-B — Whisper integration into MISP baseline main flow**, is
  open and explicitly a child task under Issue #1. It says PR #2 remains the
  work-report authority and explicitly lists streaming implementation as a
  non-goal.
- PR #2, **Whisper Backend V0: S0-S6 + S6I adapter**, is open and visibly
  `DRAFT`, merging `feat/whisper-backend-v0` into `main`. The page currently
  reports head `a0e19cef3a5a09e79d4de8ddcec03098826f4d5d`, which is not present
  in the local fetched refs; this is a governance/ref synchronization fact,
  not an assumption that the local and displayed PR heads are identical.
- The PR page's current status summary reports S0-S6 `SERVER_VALIDATED`, S6I
  real-input contract `PASS`, S6I adapter `SERVER_VALIDATED`, and M028 full
  session adapter/S4/S5 `PASS`; full-session S6 export remains blocked by
  empty/invalid Task-2 submission surfaces. S6I remains open and S7 is not
  started.
- The latest public PR comment is **S6I inventory — authoritative MISP
  artifacts found** (2026-09-17). It records the first real integration target,
  the available 8-channel source, the absence of a separate challenge
  submission evaluator, and the intended S6I-A/B/C/D split. The PR page itself
  contains later status updates through 2026-09-29.

`UNRESOLVED`: the current public GitHub governance does not define a separate
streaming/replay issue or a precedent for moving this work to a new PR. It does
define Issue #1 as the parent charter and PR #2 as the current V0 report
authority. Whether replay should receive a new child issue/charter is a future
governance decision.

## 3. Current Architecture Map

### Actual repository structure

```text
whisper_asr/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── scoring_v0.yaml
│   └── whisper_openai_v0.yaml
├── docs/
│   ├── s6i_baseline_integration_map.md
│   └── s6i_dev_preparation_audit.md
├── src/robot_heard/
│   ├── __init__.py
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── batch.py
│   │   ├── factory.py
│   │   ├── openai_whisper.py
│   │   └── scoring.py
│   ├── audio/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── io/
│   │   ├── __init__.py
│   │   └── manifest.py
│   └── misp/
│       ├── __init__.py
│       ├── baseline_integration.py
│       ├── kaldi_adapter.py
│       ├── s6i_remediation.py
│       └── task2_export.py
├── scripts/
│   ├── decode_manifest.py
│   ├── export_misp_task2.py
│   ├── prepare_misp_kaldi_manifest.py
│   ├── remediate_s6i.py
│   ├── run_misp_whisper.py
│   ├── score_results.py
│   └── smoke_openai_whisper.py
└── tests/
    ├── test_audio_loader.py
    ├── test_audio_policy.py
    ├── test_baseline_integration.py
    ├── test_batch.py
    ├── test_factory.py
    ├── test_manifest.py
    ├── test_manifest_errors.py
    ├── test_manifest_write.py
    ├── test_misp_kaldi_adapter.py
    ├── test_misp_s6i_remediation.py
    ├── test_misp_task2_export.py
    ├── test_openai_whisper_backend.py
    └── test_scoring.py
```

### Key file responsibilities and reuse boundary

| Path | Class/function(s) | Responsibility | Future replay reuse | Keep untouched for R0 |
|---|---|---|---|---|
| `src/robot_heard/asr/base.py:9-28` | `ASRResult`, `ASRBackend` | Frozen synchronous backend contract | Offline final-result consumer only | `YES` |
| `src/robot_heard/asr/openai_whisper.py:12-105` | `OpenAIWhisperBackend` | Loads OpenAI Whisper once; validates input; decodes one file; measures decode time | Offline reference backend; not a streaming runtime | `YES` |
| `src/robot_heard/audio/loader.py:22-260` | `AudioMetadata`, `probe_audio`, `prepare_audio` | ffprobe metadata; explicit mono/16 kHz policy; explicit resampling | Metadata/probe policy can inform source validation | `YES` |
| `src/robot_heard/io/manifest.py:12-155` | `validate_item`, `read_manifest`, `write_jsonl` | JSONL manifest validation, relative paths, duplicate IDs, unknown-field preservation | Reference metadata ingestion, subject to timeline limitations | `YES` |
| `src/robot_heard/asr/batch.py:24-480` | `BatchProvenance`, `run_batch` | Sequential batch decode, one backend instance, fsync, failure JSONL, resume | Final-result persistence and provenance only | `YES` |
| `src/robot_heard/asr/scoring.py:148-335` | `compute_rtf`, `score_result_record`, `score_results_jsonl` | Final-result normalization, CER, per-record/global RTF | Final-output scoring boundary | `YES` |
| `src/robot_heard/misp/kaldi_adapter.py:29-590` | `read_kaldi_segments`, `prepare_misp_kaldi_recording` | Reads Stage-2 Kaldi artifacts and materializes selected-channel segment WAVs | Artifact/reference evidence; current extraction is not continuous replay | `YES` |
| `src/robot_heard/misp/baseline_integration.py` | `prepare_baseline_manifest`, `reconcile_ids` | Baseline-facing frontend selection and exact ID reconciliation | Input compatibility evidence | `YES` |
| `src/robot_heard/misp/task2_export.py` | `export_misp2025_task2_submission` | Strict final Task-2 transcript/zip export | Final consumer boundary only | `YES` |
| `src/robot_heard/misp/s6i_remediation.py` | `remediate_s6i_record` | Derived retry/numeric surface fields while preserving primary text | Diagnostic evidence only | `YES` |
| `scripts/decode_manifest.py` | CLI around `run_batch` | Argparse wrapper and config/provenance setup | Operational reference | `YES` |
| `scripts/run_misp_whisper.py` | prepare/decode/score/export orchestration | Maintained MISP sidecar entry point | Artifact preparation reference; not a scheduler | `YES` |
| `configs/*.yaml` | Current YAML configs | Explicit backend/audio/decode/scoring settings | Project config-style reference | `YES` |
| `tests/*.py` | pytest unit/contract tests | Synthetic audio, fake backends/models, resume and reconciliation checks | Test-style reference | `YES` |

The project has no `streaming/`, `replay/`, `scheduler/`, `clock/`,
`consumer/`, or event-runtime package in the actual tree.

## 4. ASRBackend Contract

### Current facts

`CONFIRMED`: `src/robot_heard/asr/base.py:9-28` defines:

```python
@dataclass(frozen=True)
class ASRResult:
    text: str
    language: Optional[str]
    decode_sec: float

class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Union[str, Path]) -> ASRResult:
        ...
```

`CONFIRMED`: the backend instance lifetime is one model lifetime. The base
docstring requires implementations to load model state during initialization
and reuse it across calls. `OpenAIWhisperBackend.__init__` loads the model once
at lines 60-63. `run_batch` creates one backend at line 411 when work remains,
then calls it for each manifest item at lines 421-445. Tests explicitly verify
one factory/model load and multiple calls (`tests/test_batch.py:98-140`,
`tests/test_openai_whisper_backend.py:57-83`).

`CONFIRMED`: `ASRResult` has no partial/final distinction, hypothesis version,
timestamp, token timestamp, confidence, endpoint, model metadata, reset,
finalize, or flush field/method. Model configuration and provenance are kept
outside the result in batch sidecars and result fields.

`CONFIRMED`: `OpenAIWhisperBackend.transcribe` requires an existing path
(`openai_whisper.py:65-68`), probes it (`:70`), rejects non-mono and non-16 kHz
inputs (`:71-80`), and passes a file path to Whisper (`:82-90`). The repository
contract describes that file as one target waveform per segment. The method
does not accept an audio buffer/chunk or a producer callback.

`CURRENT_FACT`: no cross-utterance acoustic/decoder state is defined. The
Whisper model object is reused, but there is no utterance context state exposed
by `ASRBackend` and no method to reset or flush it.

`POSSIBLE_COMPATIBILITY_RISK`: changing `ASRBackend.transcribe` would affect
the frozen batch runner, factory, smoke CLI, fake backends, and existing result
contract. Inheriting from `ASRBackend` alone would not express lifecycle,
partial/revision, endpoint, or flush semantics. A separate streaming contract
would be less invasive to this offline boundary, but the final interface choice
is intentionally not made in this audit.

### Backend compatibility status

| Question | Classification | Evidence |
|---|---|---|
| Minimal call is path-in, `ASRResult`-out | `CONFIRMED` | `asr/base.py:9-28` |
| Model loaded once and reused | `CONFIRMED` | `openai_whisper.py:60-63`; `batch.py:409-435` |
| Input is complete file rather than incremental buffer | `CONFIRMED` | `openai_whisper.py:65-90`; README segment contract |
| Persistent utterance state | `NOT_PRESENT` | no state/lifecycle method in `base.py` |
| Partial/final/revision output | `NOT_PRESENT` | only `text/language/decode_sec` |
| Timestamp/token timestamp/confidence | `NOT_PRESENT` | no result fields or code references |
| `reset/finalize/flush` | `NOT_PRESENT` | no matching method in package |

## 5. Manifest and Timeline Contract

### Generic manifest schema

`CONFIRMED`: `src/robot_heard/io/manifest.py:12-14` makes only
`segment_id` and `audio_path` required. It recognizes optional string fields
`session_id`, `speaker_id`, `frontend`, and `reference`; `start` and `duration`
are optional non-negative/positive numbers (`:55-67`). Unknown fields are
preserved rather than rejected (`write_jsonl` docstring at `:141-155`).

`CONFIRMED`: the generic schema has no required `recording_id`, `end`,
`source_end`, sample-rate unit, channel field, or explicit timeline authority.
Relative audio paths resolve against the manifest directory (`:16-28`,
`:124-128`).

### Real M028 schemas

The following are real JSONL artifacts, not test-only examples:

| Artifact | Rows | Fields observed | Role |
|---|---:|---|---|
| `/home/gc6_challenge/work/whisper_s6i/m028_dev_ch0/manifest.jsonl` | 144 | `segment_id`, `audio_path`, `reference`, `frontend`, `start`, `duration`, `recording_id`, `channel_id`, `source_audio_path`, `source_recording_stem`, `source_end`, `adapter_policy` | raw_ch0 prepared manifest |
| `/home/gc6_challenge/work/whisper_s6i/baseline_integration_b4_m028_rawch0/prepared/manifest.jsonl` | 3 selected from 144 | same fields | maintained adapter smoke artifact |
| `/home/gc6_challenge/work/whisper_s6i/m028_gss_full144_default_v0/gss_full144_manifest.jsonl` | 144 | same fields, `frontend=misp_baseline_gss` | GSS segment artifact manifest |
| `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/segments` | 144 | `segment_id recording_id begin_sec end_sec` | authoritative Stage-2 timeline |

The generic manifest has no `end`; M028 adapter manifests add `source_end`.
`duration` is the segment duration in seconds, and `source_end` is the explicit
absolute end in the prepared recording timeline.

### Timeline authority and ID encoding

```text
TIMELINE_AUTHORITY = MISP Stage-2 `segments` for the prepared recording timeline.
```

`CONFIRMED`: the MISP baseline code at
`s0/gss_main/prepare_gss_data.py:303-374` writes Kaldi-style
`segment_id recording_id start end`. It floors the TextGrid start and rounds it
to 0.01 seconds, rounds the end upward to a 0.01-second value after 0.04-second
alignment, and constructs the ID at lines 357-364 with `int(round(time*100))`.
The same `start_stamp` and `end_stamp` are written to `segments` at lines
372-374.

Therefore the ID suffix is **centiseconds / hundredths of a second**, not
milliseconds or frames. Existing preparation notes call the components
`start-ms` in prose, but the actual code and artifacts prove the divisor is
100 seconds-units.

`CONFIRMED` cross-check over five real M028 segments:

| Segment ID | Stage-2 start/end (s) | Manifest start/duration/end (s) | ID suffix decoded /100 (s) | Result |
|---|---:|---:|---:|---|
| `S197_M028_S197199201241_F8N_014152-015076` | 141.52 / 150.76 | 141.52 / 9.24 / 150.76 | 141.52 / 150.76 | exact |
| `S197_M028_S197199201241_F8N_016820-016944` | 168.20 / 169.44 | 168.20 / 1.24 / 169.44 | 168.20 / 169.44 | exact |
| `S199_M028_S197199201241_F8N_006952-007068` | 69.52 / 70.68 | 69.52 / 1.16 / 70.68 | 69.52 / 70.68 | exact |
| `S201_M028_S197199201241_F8N_002212-002256` | 22.12 / 22.56 | 22.12 / 0.44 / 22.56 | 22.12 / 22.56 | exact |
| `S241_M028_S197199201241_F8N_033808-033964` | 338.08 / 339.64 | 338.08 / 1.56 / 339.64 | 338.08 / 339.64 | exact |

`CONFIRMED`: M028 has 144 Stage-2 segments. The minimum start is 6.88 s, the
maximum end is 798.08 s, and the prepared source duration is 798.299 s. All
starts and ends are exactly sample-aligned at 16,000 Hz under the inspected
floating values.

`INFERRED`: for the prepared M028 source, original recording timeline, segment
start, and segment end can be reconstructed exactly enough for sample-indexed
replay from Stage-2 `segments` plus the prepared source metadata. For a generic
manifest without `recording_id`, `source_end`, or a declared unit, this cannot
be guaranteed by the current schema alone.

`POSSIBLE_COMPATIBILITY_RISK`: a future runtime that treats the generic
manifest's float `start`/`duration` as its only authority could lose the
baseline's sample/centisecond identity or confuse segment-local time with
recording time.

## 6. Continuous Recording Availability

### M028 source classification

#### `CONTINUOUS_SOURCE`

`CONFIRMED`: the prepared recording is available at:

```text
/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename/
  M028_S197199201241_F8N_Far_0.wav ... _7.wav
```

Each of the eight files is a complete continuous mono channel, not a segment
file. Read-only WAV inspection reported for every channel:

```text
channels=1, rate=16000, sample width=4 bytes, frames=12772784,
duration=798.299 sec, compression=NONE
```

The Stage-2 `wav.scp` and `channels.scp` both point to the shared stem
`.../M028_S197199201241_F8N_Far`, and the adapter resolves that stem to
`stem_0.wav` through `stem_7.wav` (`src/robot_heard/misp/kaldi_adapter.py:272-320`).
The two SCP files are byte-identical in the M028 artifact.

The original source is also present:

```text
/home/gc6_challenge/data/dev/audio/CSOBx3/dev-CSOBx3/M028/
  M028-CSOBx3/M028-CSOBx3.pcm
```

Its file size is 501,219,328 bytes. Under the baseline's confirmed format
(signed 32-bit, 8 interleaved channels, 16 kHz), that is 978.944 seconds.
`s0/local/pcm2wav.py:21-39` uses exactly that int32/8-channel/16-kHz
interpretation and does not resample.

#### `SEGMENT_ARTIFACT`

`CONFIRMED`: raw-channel materialization is at
`/home/gc6_challenge/work/whisper_s6i/m028_dev_ch0/audio/`. The adapter calls
ffmpeg with `-ss start`, `-t duration`, `-ar 16000`, `-ac 1`, and `pcm_s32le`
(`src/robot_heard/misp/kaldi_adapter.py:323-356`, called at `:507-514`). It
therefore emits one local segment WAV per authoritative Stage-2 item. A full
144-file check found all 144 outputs mono/16 kHz with exactly
`round(duration*16000)` frames.

#### `FRONTEND_SEGMENT_ARTIFACT`

`CONFIRMED`: persisted GSS artifacts are segment-level, not continuous:

```text
/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/misp_data/M028_wave/
  far/wpe/gss_new/enhanced/M028_S197199201241_F8N_Far/*.flac
```

There are 144 FLAC files. Representative files are 16-bit mono 16 kHz and
their sample counts match the manifest durations. The historical baseline
output directory also contains 144 per-segment WAVs under
`_gss_sandbox/outputs/M028_far_audio_segment/enhanced/.../`; those filenames
use underscore time keys and are not automatically interchangeable with the
authoritative hyphen IDs.

### Timeline mapping result

`CONFIRMED`: `recording_id + segment.start/source_end` maps exactly to the
prepared continuous channel files for M028. `CONFIRMED`: the original raw PCM
has a different zero point. M028 `timestamp.txt` contains:

```text
1:22.463
14:40.762
```

The difference is 880.762 - 82.463 = 798.299 seconds, exactly the prepared
source duration. Thus prepared timeline `t` maps to raw PCM time `t + 82.463`
for this M028 preparation, subject to the baseline timestamp semantics.

`POSSIBLE_COMPATIBILITY_RISK`: the current `adapter.json` records the prepared
source path, hash, duration, sample rate, and channels, but not the raw PCM
path or the 82.463-second crop offset. A replay implementation must not infer
that raw and prepared timelines start at zero merely because their recording
IDs are related.

`CONFIRMED`: M028 full GSS results are driven by the 144 segment FLACs. The
current raw_ch0 adapter output is also per-segment WAVs. Neither current
adapter output is a continuous source stream.

## 7. Timing / RTF Implementation

### Current timing facts

`CONFIRMED`: `src/robot_heard/asr/openai_whisper.py:82-91` uses
`time.perf_counter()` immediately before and after `self._model.transcribe()`.
`decode_sec` is the elapsed Python-side call duration. Probe time and model
load time are outside this interval; model loading occurs in `__init__` at
`:60-63`.

`CONFIRMED`: S5 computes per-record RTF as `decode_sec / audio_sec` at
`src/robot_heard/asr/scoring.py:148-157`; global RTF is
`sum(decode_sec) / sum(audio_sec)` at `:261-280`. `audio_sec` comes from
`probe_audio` in `batch.py:432-445`, not from a replay clock.

`CONFIRMED`: `datetime.now(timezone.utc)` is used for metadata and error
record timestamps (`batch.py:54-55, 340-363` and related scripts). These are
wall-clock provenance timestamps, not audio-timeline timestamps or latency
measurements.

`CONFIRMED`: there is no `time.monotonic()` call, no explicit
`torch.cuda.synchronize()`, no event-log schema, no partial-output timestamp,
no scheduler/sleep/pacing code, and no `asyncio`, queue, thread, or process
runtime in the Whisper package. The baseline's `prepare_misp.py` imports a
`ProcessPoolExecutor` for preparation, but that is outside the current ASR
runtime.

`CONFIRMED`: batch progress is printed to stderr (`scripts/decode_manifest.py:99-100`)
and successful records are flushed/fsynced (`batch.py:297-300, 419-469`). This
is operational progress/durability, not an event log suitable for replay
latency analysis.

### Timing compatibility risks

`POTENTIAL_RISK`: `perf_counter()` is appropriate for elapsed host timing, but
the current code has no explicit GPU completion boundary. The measured value
may depend on when the underlying Whisper call synchronizes internally.

`POTENTIAL_RISK`: shared-GPU contention, first-use CUDA effects, and model/cache
state are not represented in the result schema or timing metadata. Existing
M028 metrics are diagnostic run records, not a controlled real-time latency
protocol.

`CONFIRMED`: no current code mixes model-load time into `decode_sec`; any R0
comparison that includes load/warm-up time would be a new measurement, not the
current metric.

## 8. Config and CLI Conventions

`CONFIRMED`: YAML is loaded with `yaml.safe_load` in the scripts. CLI entry
points use `argparse` (`scripts/*.py`). Paths generally use
`expanduser().resolve(strict=True)` for required inputs and `strict=False` for
outputs; see `scripts/decode_manifest.py:70-79` and
`scripts/run_misp_whisper.py:311-319`.

`CONFIRMED`: there is no project-wide schema library. Validation is distributed:

- `create_asr_backend` requires known core keys and rejects unsupported backend,
  invalid `fp16`, missing model/language/task/device, and invalid task via
  `src/robot_heard/asr/factory.py:13-64` and the backend constructor.
- `AudioInputPolicy` validates the frozen 16 kHz/mono values in
  `src/robot_heard/audio/loader.py:34-82`, but the factory does not reject all
  unknown top-level keys and the backend itself hard-checks 16 kHz/mono.
- `score_results.py` explicitly rejects unknown scoring keys at
  `scripts/score_results.py:29-39`.
- Backend/general YAML loaders accept and preserve extra mappings for
  provenance; unknown backend keys are not globally rejected.

`CONFIRMED`: bool validation is explicit for `fp16` and audio policy; enum-like
validation is explicit for backend name and Whisper task, but `device` and
some string values are only non-empty-string checked. Path provenance and
commit hashes are recorded in batch/integration/scoring/export sidecars, not by
one central config schema.

`INFERRED`: a replay configuration exposed through the existing `configs/`
and `scripts/` pattern would be `CONSISTENT_WITH_CURRENT_PROJECT`; it would not
require a new project-wide configuration pattern based on the current code.
This is a style compatibility finding, not a proposed R0 YAML schema.

## 9. Test Conventions

`CONFIRMED`: pytest is configured in `pyproject.toml:17-19` with `src` on the
Python path and `tests/` as the test root. Tests use `tmp_path`, Python's
`wave`/`struct` to generate small WAV fixtures, and monkeypatching rather than
committed audio fixtures.

`CONFIRMED`: existing fakes include `_FakeBackend` in
`tests/test_batch.py:17-31`, `_FakeModel` in
`tests/test_openai_whisper_backend.py:11-45`, and fake `probe_audio` functions.
These are the closest available style references for a future dummy consumer;
there is no `DummyConsumer` and no `FakeClock`.

`CONFIRMED`: test coverage includes:

- normal/error manifest validation and relative-path resolution;
- mono/16 kHz audio policy and ffmpeg-gated audio tests;
- one backend load with multiple transcriptions;
- sequential batch success, per-segment failure, resume, trailing partial-line
  repair, changed-config rejection, and changed-manifest rejection;
- exact ID reconciliation and deterministic prepared-manifest generation;
- deterministic scoring/edit alignment and global RTF calculation;
- strict Task-2 export and S6I derived-field behavior.

`CONFIRMED`: there are no dedicated timing-clock tests, scheduler tests,
partial-hypothesis tests, revision tests, endpoint tests, or async/thread/queue
runtime tests. Resume is tested at the batch artifact level; there is no
repository test that drives an actual OS interruption process, although the PR
work report records a separate real Ctrl-C/resume server validation.

`CONFIRMED`: no `integration` marker is configured. ffmpeg-dependent tests use
`pytest.mark.skipif` based on executable availability. In the current shell,
`pytest --collect-only -q` could not run because `pytest` is not on PATH; no
package was installed to change that condition.

## 10. Dependencies and Runtime

`CONFIRMED` from `pyproject.toml` and `requirements.txt`:

- Python minimum: `>=3.9`.
- Direct runtime/build declarations: `setuptools>=68`, `wheel`,
  `openai-whisper`, `PyYAML`, and `pytest`.
- `asyncio`, `queue`, and `dataclasses` are Python standard-library facilities;
  the current package does not declare or implement a scheduler runtime.
- `numpy`, `torch`, and other Whisper transitive dependencies are not direct
  entries in this repository's `requirements.txt`.
- `pydantic` and `attrs` are not declared project dependencies.
- Current audio boundary depends on external `ffprobe`/`ffmpeg`; the project
  deliberately does not infer headerless raw PCM metadata.
- No environment YAML, conda file, or pip-freeze artifact was found in the
  repository/work artifact scope.

An informational import probe in the current shell reported
`asyncio=True`, `queue=True`, `dataclasses=True`, `pydantic=True`,
`numpy=True`, `soundfile=True`, `scipy=True`, `torch=True`, `yaml=True`,
while `attrs=False`, `whisper=False`; `pytest` was not available as a command.
This is host-environment evidence only and must not be treated as the project's
reproducible runtime contract.

`INFERRED`: a replay scheduler/event data layer could be implemented with the
Python standard library plus numpy if audio-source access and model adapters
remain outside that claim. No evidence in the current repository establishes a
need for a new dependency for scheduling itself. This audit does not select or
design that runtime.

## 11. Existing MISP Artifacts

### Authoritative metadata

`CONFIRMED`:

- `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/segments`
  — 144 Stage-2 segment IDs, recording IDs, starts, ends.
- `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/text`
  — 144 Dev references matching the segment ID set.
- `.../wav.scp` and `.../channels.scp` — one shared recording stem for the
  eight physical channels; they are byte-identical in M028.
- `.../utt2spk` and `.../spk2utt` — baseline speaker mappings.
- `.../selection.json` — recorded source directory, recording, speaker counts,
  and hashes for selected Stage-2 artifacts.
- Original M028 `timestamp.txt` and CSOBx3 PCM — source-preparation evidence,
  with the raw-to-prepared crop relation described in Section 6.

### Derived waveforms and manifests

`CONFIRMED`:

- `work/whisper_s6i/m028_dev_ch0/manifest.jsonl` and `adapter.json` — full
  144-segment raw_ch0 manifest plus adapter provenance.
- `work/whisper_s6i/m028_dev_ch0/audio/` — 144 raw_ch0 segment WAVs.
- `work/whisper_s6i/m028_gss_full144_default_v0/gss_full144_manifest.jsonl` —
  144-segment GSS manifest pointing to per-segment FLACs.
- `_gss_sandbox/misp_data/M028_wave/.../enhanced/*.flac` — 144 GSS segment
  artifacts.
- `_gss_sandbox/outputs/M028_far_audio_segment/.../*.wav` — 144 historical
  segment WAVs with a baseline-specific filename/key rewrite.

### Diagnostic and experimental results

`CONFIRMED`:

- `m028_dev_ch0/results*.jsonl` and metrics — raw_ch0 full-session diagnostic
  results; the raw result has 144 records and 33 empty `text_raw` values.
- `m028_gss_full144_default_v0/results*.jsonl` and metrics — full GSS
  diagnostic results; 144 records and 16 empty `text_raw` values.
- `m028_gss_empty33_existing/` and `m028_retry_experiments/` — residual-empty
  and retry diagnostics; not primary baseline metadata.
- `m028_digit_normalization_experiment/` and `m028_far_near_compare/` —
  experiments/comparisons, not authoritative timeline sources.

`INFERRED`: the richest ready-to-consume replay metadata is
`m028_dev_ch0/manifest.jsonl` because it carries source path, frontend,
recording, start, duration, and source end. The authoritative source for those
times remains Stage-2 `segments`; the manifest is a derived integration view.

### Input compatibility facts

`CONFIRMED`: baseline Kaldi `segments/text` and the Whisper manifest are both
available and compatible for M028, but they serve different roles. Stage-2
Kaldi artifacts own segmentation/time/reference identity; the Whisper manifest
adds materialized target waveform paths and frontend metadata. The current
maintained entry point reads baseline artifacts and produces the manifest; no
current replay reader supports both as one unified schema.

`CONFIRMED`: a pre-materialized GSS waveform map is accepted only when its keys
exactly match authoritative segment IDs. The README records the actual
hyphen-to-underscore mismatch in common Stage-3 output and the code fails
closed rather than rewriting IDs.

## 12. Future Consumer Compatibility Matrix

The matrix describes what the **current repository interface** can express. It
does not claim a model capability that was not present in this repository.

| Current interface feature | Offline Whisper | Rolling-buffer Whisper wrapper | WeNet U2/U2++ | FastConformer/RNNT |
|---|---|---|---|---|
| `current transcribe()` one-shot final text | `SUPPORTED` | `ADAPTER_REQUIRED` | `ADAPTER_REQUIRED` | `ADAPTER_REQUIRED` |
| Persistent state | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` |
| Partial hypothesis | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` |
| Final hypothesis field | `SUPPORTED` (`ASRResult.text`) | `ADAPTER_REQUIRED` | `ADAPTER_REQUIRED` | `ADAPTER_REQUIRED` |
| Revision/version of hypothesis | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` |
| Endpoint event | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` |
| Segment/token timestamps | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` |

`UNKNOWN`: whether a future U2++ or FastConformer/RNNT implementation in another
runtime can produce partials, revisions, endpoints, or token timestamps. No
such implementation exists in this repository and no external model study was
part of this audit. The only confirmed fact is that the current ASRBackend
cannot carry those values.

## 13. Governance / Git Integration Constraints

`CONFIRMED`:

1. Issue #1 recommends feature-branch development and names
   `feat/whisper-backend-v0` as the current branch. Issue #3 says to keep the
   current feature branch unless a dedicated branch is clearly required.
2. No streaming/realtime/replay issue exists in the inspected current issue
   list. Issue #1 is the only parent execution charter and Issue #3 is the
   current S6I child.
3. PR #2 is still a **DRAFT** and remains the V0 work-report authority. Its
   current scope is S0-S6 plus S6I adapter/baseline integration; Issue #1 and
   Issue #3 explicitly say streaming is not part of that implementation scope.
4. The repository has a gate naming convention (`S0`-`S9`, `S6I`, and Issue #3
   phases `B0`-`B5`), a PR Work Report convention, explicit server-validation
   conventions, and provenance/evidence conventions.
5. Work reports are deliberately kept in GitHub PR description/comments rather
   than added as repository Markdown artifacts; the cleanup commit
   `472c0a1` records that convention.

`UNRESOLVED`: whether the future replay harness should continue under PR #2,
receive a new child issue, or receive an independent Execution Charter. The
existing governance provides precedent for parent issue + child execution task,
but no streaming-specific precedent.

## 14. Risks for a Model-Agnostic Replay Runtime

| Risk check | Classification | Evidence / fact |
|---|---|---|
| Float-second accumulation/error | `POTENTIAL_RISK` | Current manifest/adapter use float seconds; baseline IDs are centiseconds and source samples are 16 kHz. Current per-segment extraction is independently addressed, not an accumulating loop. |
| Segment time-base inconsistency | `CONFIRMED_RISK` | Stage-2 hyphen IDs are authoritative, while historical GSS filenames use underscore keys; README:253-255 and real paths show the mismatch. |
| F8N / CSOBx3 clock/zero-point mismatch | `CONFIRMED_RISK` | Raw M028 PCM is 978.944 s; timestamp crop begins at 82.463 s and prepared/F8N timeline is 798.299 s. |
| Precomputed segmentation future leakage | `POTENTIAL_RISK` | Current evaluation artifacts use TextGrid/Stage-2 precomputed segment boundaries; no online segmentation semantics exist. |
| Current audio adapter automatically cuts segments | `CONFIRMED_RISK` | `kaldi_adapter.py:323-356, 507-514` materializes one WAV per segment. |
| Current Whisper backend assumes complete file | `CONFIRMED_RISK` | `openai_whisper.py:65-90` takes a path and calls file-based Whisper. |
| Model load mixed into `decode_sec` | `NOT_OBSERVED` | Load is in `__init__` (`:60-63`); timing starts at `:82`. |
| CUDA synchronization changes timing | `POTENTIAL_RISK` | No explicit `torch.cuda.synchronize()` is present. |
| `perf_counter` coverage of async GPU completion | `POTENTIAL_RISK` | Host call timing is measured, but completion boundary is implicit in the Whisper call. |
| Shared GPU causes RTF variance | `POTENTIAL_RISK` | No device contention/stream/thermal/provenance field is in current result schema; existing metrics are uncontrolled run evidence. |
| GSS waveform only segment-level | `CONFIRMED` | M028 GSS manifests and directories contain 144 per-segment FLAC/WAV artifacts, no continuous GSS source. |
| Preprocessing lost continuous timeline | `POTENTIAL_RISK` | M028 prepared continuous timeline survives, but raw PCM crop offset is not carried in current adapter metadata; general-session preservation is not established. |
| 8-channel continuous source safely random-accessible | `POTENTIAL_RISK` | Prepared M028 channels are verified complete and frame-aligned; original raw PCM is headerless and needs explicit format/crop metadata. |
| Result schema cannot represent partial revision | `CONFIRMED_RISK` | `ASRResult` and result JSONL have only final `text_raw` plus status; no revision/event fields. |
| Scoring accepts only final hypothesis | `CONFIRMED_RISK` | `score_result_record` requires `status == "success"` and a string `text_raw` (`scoring.py:160-221`). |

## 15. Facts the R0 Architecture Must Respect

This section intentionally records constraints, not a replay-harness design.

### MUST_PRESERVE

- The existing offline `ASRBackend.transcribe(path) -> ASRResult` contract and
  current batch/resume/result behavior remain valid for V0 users.
- MISP Stage-2 `segments` is the authoritative M028 segment identity and
  timeline source; exact IDs, recording IDs, starts, and ends must not be
  silently rewritten.
- M028 segment ID time suffixes encode hundredths of seconds (`time * 100`),
  while prepared audio is sample-addressable at 16 kHz.
- The distinction between a continuous source, a raw selected-channel
  segment artifact, and a GSS/frontend segment artifact.
- Explicit frontend identity such as `raw_ch0` and `misp_baseline_gss`.
- No implicit 8-channel averaging/downmix. Current policy requires an explicit
  mono target waveform.
- `text_raw` as immutable primary evidence, with scoring/export as separate
  post-processing stages.
- Existing provenance patterns: manifest/config/source hashes, code commit,
  frontend/source metadata, and exact ID reconciliation.
- Current scoring and exporter boundaries: final successful records only, strict
  expected-ID equality, and fail-closed Task-2 surface validation.
- M028 prepared source dimensions: eight synchronized mono PCM-s32le WAVs,
  16 kHz, 798.299 s, with Stage-2 ends within that timeline.

### MUST_NOT_ASSUME

- That a generic manifest has an absolute recording timeline. `start` and
  `duration` are optional and their units are not declared by the generic
  validator.
- That a generic manifest contains an explicit `end`; M028 adapter metadata
  adds `source_end`, but this is not a generic field requirement.
- That segment ID suffixes are milliseconds or frames.
- That a raw CSOBx3 PCM and prepared F8N/Stage-2 time zero are identical.
- That headerless raw PCM is self-describing.
- That a current adapter's per-segment WAV is a continuous source.
- That GSS artifacts expose a continuous replay stream.
- That `transcribe()` accepts incremental audio or maintains utterance state.
- That `ASRResult.text` indicates whether a hypothesis is partial, final, or a
  revision.
- That `decode_sec` includes model load, explicit CUDA synchronization, or a
  standardized end-to-end real-time latency boundary.
- That current scoring can score partial/revised hypotheses.
- That existing GitHub PR head, local tracking ref, and local HEAD are the
  same without an explicit synchronization check.

### UNRESOLVED

- Whether R0 replay should use M028's prepared continuous channel files,
  original raw PCM plus timestamp mapping, or both as separately typed sources.
- Whether all future MISP sessions preserve an equivalent continuous prepared
  source and explicit raw-to-prepared offset metadata.
- Whether the future task intends to replay baseline precomputed segmentation
  or evaluate online segmentation/endpointing; current artifacts only prove the
  former.
- The exact future event/clock/result contract for partials, revisions,
  endpoints, finalization, and token timestamps.
- Actual streaming capability and adapter requirements of future U2++ and
  FastConformer/RNNT consumers.
- Whether replay work belongs in PR #2 or a new child issue/charter.
- Official MISP evaluator availability/parity beyond the documented baseline
  scorer evidence; no separate challenge evaluator was found in the server
  inventory.

## 16. Recommended Inputs for ChatGPT R0 Design

Use these paths as the primary evidence set for the next architecture/contract
discussion:

### Repository contracts

- `/home/gc6_challenge/whisper_asr/src/robot_heard/asr/base.py`
- `/home/gc6_challenge/whisper_asr/src/robot_heard/asr/openai_whisper.py`
- `/home/gc6_challenge/whisper_asr/src/robot_heard/asr/batch.py`
- `/home/gc6_challenge/whisper_asr/src/robot_heard/asr/scoring.py`
- `/home/gc6_challenge/whisper_asr/src/robot_heard/io/manifest.py`
- `/home/gc6_challenge/whisper_asr/src/robot_heard/audio/loader.py`
- `/home/gc6_challenge/whisper_asr/src/robot_heard/misp/kaldi_adapter.py`
- `/home/gc6_challenge/whisper_asr/src/robot_heard/misp/baseline_integration.py`
- `/home/gc6_challenge/whisper_asr/scripts/run_misp_whisper.py`
- `/home/gc6_challenge/whisper_asr/configs/whisper_openai_v0.yaml`
- `/home/gc6_challenge/whisper_asr/configs/scoring_v0.yaml`
- `/home/gc6_challenge/whisper_asr/README.md`

### Baseline authority and source mapping

- `/home/gc6_challenge/code/AVSR/MISP2025-AVSR-baseline-main/s0/gss_main/prepare_gss_data.py`
- `/home/gc6_challenge/code/AVSR/MISP2025-AVSR-baseline-main/s0/gss_main/prepare_misp.py`
- `/home/gc6_challenge/code/AVSR/MISP2025-AVSR-baseline-main/s0/local/pcm2wav.py`
- `/home/gc6_challenge/code/AVSR/MISP2025-AVSR-baseline-main/s0/local/wav2multi_channel_wav.py`
- `/home/gc6_challenge/code/AVSR/MISP2025-AVSR-baseline-main/s0/local/prepare_gss_data.sh`
- `/home/gc6_challenge/code/AVSR/MISP2025-AVSR-baseline-main/s0/run_asr_far.sh`
- `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/segments`
- `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/text`
- `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/wav.scp`
- `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/channels.scp`
- `/home/gc6_challenge/data/dev/audio/CSOBx3/dev-CSOBx3/M028/M028-CSOBx3/M028-CSOBx3.pcm`
- `/home/gc6_challenge/data/dev/audio/CSOBx3/dev-CSOBx3/M028/M028-CSOBx3/timestamp.txt`
- `/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename/M028_S197199201241_F8N_Far_0.wav` through `_7.wav`

### Existing reference artifacts

- `/home/gc6_challenge/work/whisper_s6i/m028_dev_ch0/manifest.jsonl`
- `/home/gc6_challenge/work/whisper_s6i/m028_dev_ch0/adapter.json`
- `/home/gc6_challenge/work/whisper_s6i/m028_dev_ch0/audio/`
- `/home/gc6_challenge/work/whisper_s6i/m028_gss_full144_default_v0/gss_full144_manifest.jsonl`
- `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/misp_data/M028_wave/far/wpe/gss_new/enhanced/M028_S197199201241_F8N_Far/`
- `/home/gc6_challenge/work/whisper_s6i/baseline_integration_b5_m028_rawch0/`
- `/home/gc6_challenge/work/whisper_s6i/m028_gss_empty33_existing/`

### Governance inputs

- GitHub Issue #1: `https://github.com/lyxin651/lyx_Robot-HEARD/issues/1`
- GitHub Issue #3: `https://github.com/lyxin651/lyx_Robot-HEARD/issues/3`
- GitHub Draft PR #2: `https://github.com/lyxin651/lyx_Robot-HEARD/pull/2`

## 17. Evidence Appendix

### Repository and contract evidence

| Conclusion | Evidence |
|---|---|
| Clean starting repository, branch, HEAD, remote | `pwd`; `git status --short`; `git branch --show-current`; `git rev-parse HEAD`; `git remote -v` at `/home/gc6_challenge/whisper_asr` |
| Backend shape | `src/robot_heard/asr/base.py:9-28` |
| Model load and decode clock | `src/robot_heard/asr/openai_whisper.py:48-105` |
| One backend per batch and durable result writes | `src/robot_heard/asr/batch.py:366-480` |
| Result fields | `src/robot_heard/asr/batch.py:303-337` |
| Manifest validation limitations | `src/robot_heard/io/manifest.py:12-138` |
| RTF/scoring final-only behavior | `src/robot_heard/asr/scoring.py:148-235, 238-280` |
| Segment WAV extraction | `src/robot_heard/misp/kaldi_adapter.py:323-356, 383-590` |
| Config/CLI behavior | `scripts/decode_manifest.py:21-115`; `scripts/run_misp_whisper.py:77-339`; `scripts/score_results.py:29-39` |
| Test conventions | `tests/test_batch.py:17-320`; `tests/test_openai_whisper_backend.py:11-122`; `tests/test_misp_kaldi_adapter.py:27-220`; `tests/test_baseline_integration.py:26-220` |

### Baseline code evidence

| Conclusion | Evidence |
|---|---|
| Raw PCM interpretation and no resample | `s0/local/pcm2wav.py:21-39` |
| Timestamp crop and per-channel prepared WAVs | `s0/local/wav2multi_channel_wav.py:8-59` |
| Rename/copy behavior | `s0/local/rename_wav_textgrid.py:41-74` |
| Stage-2 ID/time construction | `s0/gss_main/prepare_gss_data.py:303-374` |
| Stage-2 source stem and eight-channel resolution | `s0/gss_main/prepare_misp.py:92-155` |
| Stage boundaries and GSS entry path | `s0/run_asr_far.sh:67-119`; `s0/local/prepare_gss_data.sh:22-58` |

### Command and artifact evidence

```text
M028 Stage-2 segments: 144 rows
M028 prepared source: 8 files, each mono/16000 Hz/PCM 32-bit/798.299 s
M028 segment bounds: min start 6.88 s, max end 798.08 s
M028 raw PCM: 501219328 bytes; s32le/8ch/16k interpretation = 978.944 s
M028 timestamp crop: 1:22.463 -> 14:40.762
M028 raw_ch0 materialized segments: 144/144 exact expected frame counts
M028 GSS enhanced FLAC artifacts: 144 segment files
```

Representative real manifest lines are in:

- `work/whisper_s6i/m028_dev_ch0/manifest.jsonl:1-5`
- `work/whisper_s6i/m028_gss_full144_default_v0/gss_full144_manifest.jsonl:1-5`

The current-shell lightweight checks also established:

```text
pytest --collect-only -q -> /bin/bash: pytest: command not found
git status --short after read-only checks -> empty
```

No test dependency was installed in response to that missing command.

### GitHub evidence

- Issue #1 body: V0 excludes streaming, defines S0-S9/S6I gates, branch/report/
  provenance rules, and requires real MISP evidence.
- Issue #3 body: S6I-B is a child task under Issue #1, PR #2 is the work-report
  authority, and streaming is an explicit non-goal.
- PR #2 page: state `DRAFT`, base `main`, head branch
  `feat/whisper-backend-v0`, current displayed head
  `a0e19cef3a5a09e79d4de8ddcec03098826f4d5d`, and current S6I status.
- PR #2 latest public comment: `S6I inventory — authoritative MISP artifacts
  found`, dated 2026-09-17.

