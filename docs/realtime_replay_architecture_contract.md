# Real-Time Replay Harness — R0 Architecture & Contract

**Status: R0 PASS**
**Scope: architecture and contract documentation only**
**Current Gate: R1 — IN PROGRESS**
**IMPLEMENTED: R0 documentation only**
**SERVER_VALIDATED: no**

This document is the repository-side R0 contract requested by GitHub Issue #4.
It defines the boundary that future R1–R5 work must implement and test. It does
not create a scheduler, runner, clock, consumer, adapter, dataclass, test, or
configuration file.

## 1. Purpose and Scope

The replay harness is a model-agnostic streaming experiment platform for
recorded MISP audio. Its stable surface is the relationship between source
audio availability, packet delivery, consumer state, observed output, trace,
and derived evaluation metrics.

The runtime is not bound to Whisper, U2++, FastConformer/RNNT, or any other
model. Only consumer adapters vary across models. The current offline
Whisper V0 interface remains a compatibility surface and is not redesigned.

R0 freezes the audio and ASR event contract required by R1–R5:

- continuous source packetization;
- logical/audio and wall-clock semantics;
- causality and metadata visibility;
- synchronous or stateful consumer lifecycle;
- partial, revision, endpoint, final, and error semantics;
- append-only trace and provenance;
- model-independent backlog, latency, RTF, and stability metric definitions;
- the MISP source and Stage-2 timeline authority.

R6 is outside this schema. Enhanced audio, diarization, online speaker
association, video, RGB-D, LiDAR, and multimodal events require a separate
schema extension review.

## 2. Evidence and Authority

The authority order is:

1. GitHub Issue #4, **Real-Time Replay Harness — Model-Agnostic Streaming
   Runtime Execution Charter**;
2. this document after R0 review;
3. the pre-implementation audit and the current repository code/artifacts;
4. model- or frontend-specific adapter documentation.

Issue #1, Issue #3, and Draft PR #2 govern Whisper V0 separately. Replay work
must not be reported in PR #2. The current offline `ASRBackend` remains
unchanged.

The following are confirmed facts, not design guesses:

| Confirmed fact | Evidence |
|---|---|
| Current ASR is synchronous, complete-file, and final-result only. | `src/robot_heard/asr/base.py`; `src/robot_heard/asr/openai_whisper.py`; `docs/realtime_replay_preimplementation_audit.md` |
| `ASRResult` contains `text`, `language`, and `decode_sec`; it has no lifecycle/event fields. | `src/robot_heard/asr/base.py` |
| No replay, scheduler, audio-clock, queue runtime, or event-trace runtime exists. | Repository search recorded in the audit |
| M028 Stage-2 `segments` is the authoritative segment identity/timeline source. | `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/segments`; MISP preparation code; audit |
| Segment-ID time suffixes encode centiseconds. | MISP `prepare_gss_data.py`; M028 cross-check in audit |
| M028 has eight synchronized prepared mono WAV channels at 16 kHz, PCM-s32le, 798.299 s. | `/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename/`; audit |
| Original CSOBx3 PCM has a different zero point; prepared time `t` maps to raw time `t + 82.463 s` for M028. | M028 `timestamp.txt`, raw PCM, prepared WAV metadata; audit |
| Existing `raw_ch0` and GSS outputs are per-segment artifacts, not continuous streams. | `src/robot_heard/misp/kaldi_adapter.py`; M028 work artifacts; audit |
| Existing timing uses host `time.perf_counter()` covering the synchronous Whisper call. | `src/robot_heard/asr/openai_whisper.py`; audit |
| Existing scoring accepts final successful records only. | `src/robot_heard/scoring.py`; audit |

These facts constrain R0. They do not imply that a future consumer has any
particular native streaming capability.

## 3. System Architecture

The conceptual flow is:

```text
continuous recorded source
        -> ReplaySource
        -> Clock / Scheduler
        -> AudioChunk delivery
        -> StreamingConsumer adapter
        -> OutputEvent
        -> append-only Trace
        -> derived runtime / latency / accuracy evaluation
```

The runtime core must not branch on model names. It must not contain branches
equivalent to `if whisper`, `if u2`, or `if fastconformer`.

### Ownership

| Owner | Contract responsibility |
|---|---|
| Replay runtime | Source packetization, logical/audio availability, wall/monotonic observation, delivery scheduling, backlog accounting, event ordering, causality guards, append-only raw trace, model-independent metrics |
| Consumer adapter | Model buffering, processing chunk, right context/look-ahead, rolling windows, attention/CNN/RNNT cache, decoder policy, model endpointing, model-specific GPU synchronization |
| Evaluator | Reference visibility, final CER/scoring, latency summaries, stability/flicker summaries, challenge-specific metrics |

The runtime owns the facts of what was available and delivered. The adapter
owns model state. The evaluator owns reference and scoring data. No owner may
silently assume another owner's data.

## 4. Time Authority

Integer sample index is the only audio timeline authority.

For sample rate `Fs` in samples/second:

```text
audio_time_sec(sample) = sample / Fs
```

All audio intervals are half-open:

```text
[start_sample, end_sample)
```

`start_sample` is inclusive and `end_sample` is exclusive. Float seconds are
derived presentation values only. Packet boundaries, availability, source
end, segment mapping, and causality checks must use integer sample positions.

For M028 prepared 16 kHz audio, one centisecond is exactly 160 samples. A
future source whose authoritative decimal boundary does not map exactly to an
integer sample at its declared `Fs` must fail closed; it must not be silently
rounded by accumulated float arithmetic.

## 5. Four Distinct Runtime Positions

Four positions are distinct and must be recorded separately.

### A. `available_through_sample`

The exclusive highest source sample that is fully available to the runtime at
the observation point. For packet `[S,E)`, availability advances to at least
`E` only when the source clock reaches the packet's end.

### B. `delivered_through_sample`

The exclusive highest sample handed to the consumer. A packet can be fully
available while remaining queued and therefore not delivered.

### C. `consumed_through_sample`

The optional, nullable exclusive highest sample that the adapter declares
incorporated into its hypothesis state. This is consumer metadata, not proof
that future audio was not visible. The runtime must enforce visibility
independently.

An adapter may report this position only when it can define and support a
trustworthy relation between the value and its actual model state. The
capability is `reports_consumed_position`. When that capability is false, the
normalized event value is `null` and the runtime must not substitute
`delivered_through_sample`.

### D. Output observed time

The runtime-stamped monotonic wall time at which an output is received. The
consumer cannot backdate this time. The runtime also records the available and
delivered positions observed when the output arrived.

### Required invariant

When a valid consumed position is present:

```text
consumed_through_sample <= delivered_through_sample <= available_through_sample
```

When consumed position is unavailable:

```text
delivered_through_sample <= available_through_sample
```

Consumption-specific metrics are unavailable in the second case unless a
future contract defines another explicit authority.

The relation between source availability, delivery, consumption, and output
observation is causal state, not one timestamp.

## 6. Packet Availability Contract

A source packet is `[S,E)`, with `S < E`. It is fully available only when the
logical source clock reaches `E / Fs`. `S / Fs` is the packet start time, not
the complete-packet release time.

For compute-aware replay with wall origin `W0`, its scheduled wall availability
is:

```text
scheduled_availability_wall_time = W0 + E / Fs
```

The first packet `[0,P)` becomes available at `P/Fs`. There is no zero-length
release event at time zero.

The tail packet is explicit. If source length `N` is not a multiple of the
packet size, the final packet is `[S,N)` and has
`is_final_source_chunk = true`. If `N` is an exact multiple, the final full
packet is still marked final. Source end is visible only when the final packet
has become available at `N/Fs`.

A packet is released exactly once, delivered at most once, and is never
implicitly split, duplicated, dropped, or overwritten under the R3 reference
queue policy.

## 7. Clock Contract

### Logical / audio clock

The logical clock is an integer sample position derived from source packet
boundaries. It determines which samples and time-gated metadata are available.
It is deterministic and independent of machine speed.

### Wall / monotonic clock

The wall clock for elapsed runtime measurements is a monotonic, high-resolution
clock such as `time.perf_counter()`. It timestamps delivery, consumer-call
start/return, output observation, finish, queue delay, and wall-overrun
metrics. `time.time()` and `datetime` are not latency authorities.

### UTC provenance clock

UTC timestamps are optional provenance fields for identifying when a run was
started or recorded. They are not used for elapsed latency, deadline, or RTF
calculations.

The runtime must record the clock kind and run origin in provenance. Mixing UTC
time with monotonic offsets is a contract failure.

## 8. Replay Modes

### `COMPUTE_UNAWARE` / `DETERMINISTIC`

This mode is for deterministic contract tests, input-consumption analysis, and
causality tests. Logical source availability follows the deterministic event
schedule and is not delayed by consumer compute. Consumer call duration may be
recorded as an observation, but it must not be used to claim measured
real-time end-to-end latency.

With the same source, configuration, and deterministic consumer, the logical
trace must be identical. A wall-clock field is not required to be bitwise
identical.

### `COMPUTE_AWARE` / `REALTIME_PACED`

Source availability follows wall time. Wall time continues while a synchronous
`consume()` call blocks. Newly available packets are enqueued during that
interval. The runtime must not pause source time to accommodate slow compute.

The R3 reference policy is a single-stream, lossless, unbounded queue. Bounded
queues, drops, overwrites, and source-capture blocking are separate named
experimental conditions and cannot be compared to the reference condition
without their policy labels.

## 9. Scheduler Same-Time Ordering

At one logical or wall observation time, the runtime must process events in
this order:

1. advance source availability;
2. enqueue all newly available packets whose full-availability time is `<= now`;
3. record completion/return of the in-flight consumer call and all outputs it
   returned;
4. update available, delivered, consumed, and in-flight state;
5. if the consumer is idle, deliver the oldest queued packet;
6. record delivery and start the next consumer call.

This ordering ensures that packets which became available while the consumer
was blocked are visible in state before the returned output is evaluated, but
are not delivered ahead of the completion event. Fake-clock tests and trace
reconstruction must use this ordering.

## 10. AudioChunk Conceptual Schema

This is a conceptual schema, not a Python dataclass. Field definitions are
frozen here before implementation.

| Field | Type | Required | Authority | Semantics and invariant |
|---|---|---:|---|---|
| `stream_id` | opaque string | yes | Runtime | Identifies one replay stream. It is stable for the run and unique in the trace. |
| `sequence_id` | non-negative integer | yes | Runtime | Strictly increases in source delivery order. No reuse. |
| `start_sample` | integer | yes | Source packetizer | Inclusive packet start. `0 <= start_sample < end_sample`. |
| `end_sample` | integer | yes | Source packetizer | Exclusive packet end. Adjacent normal packets have previous `end_sample == next start_sample`. |
| `sample_rate` | positive integer Hz | yes | Source metadata | Applies to every sample index in the chunk. It is constant within a stream. |
| `channel_ids` | ordered non-empty list of opaque IDs | yes | Source metadata | Channel order for `samples`. All channels cover the same `[start_sample,end_sample)`. |
| `samples` | channel-major or explicitly documented array | yes | ReplaySource | Contains exactly `end_sample - start_sample` samples per channel. Runtime does not implicitly average or downmix. |
| `is_final_source_chunk` | boolean | yes | Source packetizer | True only for the final source packet. Exactly one packet is final. |
| `scheduled_availability_logical_sample` | integer | yes | Clock | Must equal `end_sample`; it is the sample position at which the packet becomes fully available. |
| `scheduled_availability_wall_time` | monotonic offset/time or null | compute-aware | Clock | Scheduled wall release for the packet. Null is permitted in compute-unaware mode. |
| `delivery_wall_time` | monotonic offset/time or null | after delivery | Runtime | Actual monotonic observation when packet delivery begins. It is not the scheduled availability time. |

Source packetization must fail if samples, channel count, or channel lengths do
not match the declared range. A shorter tail is valid only when it is the one
explicit final chunk.

## 11. Stream Metadata and Metadata Visibility

Runtime and evaluator code may know source paths, hashes, total source samples,
the Stage-2 segment list, speaker labels, references, and raw/prepared source
mapping. Knowledge by the evaluator does not authorize disclosure to the
consumer.

### Consumer-visible startup metadata

The default startup metadata contains only:

- `stream_id`;
- `sample_rate`;
- channel count and ordered `channel_ids`;
- sample layout/format information needed to consume current packets;
- a named condition identifier if an oracle condition is active.

### Default prohibited early disclosure

The consumer must not receive the following at startup unless a named oracle
condition explicitly grants it:

- full future segment list;
- future segment end;
- reference text;
- speaker identity;
- future transcript;
- future file paths;
- a future end-of-stream time not yet reached by the source clock.

The runtime must emit `ORACLE_METADATA_RELEASE` when a gated segment descriptor
becomes visible. The event records what was released and the logical sample
position at release. References and speaker labels remain evaluator-only in the
R4 default condition.

## 12. Source Packet vs Model Chunk vs Look-ahead

The following are independent quantities:

1. `source_packet_ms`: how often the simulated source makes samples available;
2. `consumer_processing_chunk`: how much arrived audio an adapter accumulates
   before model processing;
3. `model_lookahead`: future context required by the model or decoder.

For example, a source can release 20 ms packets while an adapter processes a
320 ms rolling chunk. A model may also require a model-specific right context.
The runtime controls source delivery only. Processing chunk and look-ahead are
adapter-owned provenance and cannot be inferred from packet size.

## 13. StreamingConsumer Conceptual Protocol

The streaming contract is independent of the existing `ASRBackend`. This
section defines a conceptual protocol only; it creates no production class.

| Operation | Inputs | Output | Lifecycle contract |
|---|---|---|---|
| `start()` | consumer-visible startup metadata | initialization result or lifecycle events | Called once before the first packet. It must not receive prohibited future metadata. |
| `consume()` | one delivered `AudioChunk` | zero or more `OutputEvent` values | May be synchronous and blocking. It may emit snapshots, endpoint, or error events. Runtime stamps observation fields. |
| `finish()` | no new audio | zero or more final/tail events | Called only after source end is visible. Outputs count toward tail latency. |
| `reset()` | no stream payload | no required output | Clears all model and decoder state before a new stream. State must not leak across streams. |
| `capabilities()` | no stream payload | immutable capability declaration | Must be available before metrics interpret consumer events. |

An adapter may have internal processing queues and model state. It must expose
only the contract above to the runtime. An asynchronous implementation is not
required by R0; a synchronous implementation is valid when wall pacing and
backlog semantics remain correct.

## 14. Consumer Capability Contract

Capabilities are declared by the adapter and consumed by the evaluator. An
unsupported capability is a valid value; the runtime must not fabricate events
or metrics for it.

| Capability | Type | Meaning | Required effect if false |
|---|---|---|---|
| `supports_partial` | boolean | Consumer can emit non-final hypothesis snapshots. | No PARTIAL event or partial metric is attributed to the consumer. |
| `supports_revision` | boolean | Consumer can replace a prior snapshot in the same scope. | Revision count is unavailable and multiple snapshots must not be interpreted as revisions. |
| `supports_stable_prefix` | boolean | Consumer can make irrevocable prefix declarations under Section 17. | TTFT-stable and stable-prefix growth are unavailable. |
| `supports_endpoint` | boolean | Consumer can emit an endpoint belief distinct from finalization. | No ENDPOINT capability is claimed. |
| `supports_token_timestamps` | boolean | Consumer emits causally valid token-level timing. | Token timing metrics are unavailable. |
| `supports_word_timestamps` | boolean | Consumer emits causally valid word-level timing. | Word timing metrics are unavailable. |
| `reports_consumed_position` | boolean | Adapter reports a trustworthy, defined exclusive sample position incorporated into model state. | `consumed_through_sample` is nullable and normalized to `null`; consumption-position metrics and processing lag are unavailable. |
| `native_streaming` | boolean | Model/decoder natively consumes incremental audio/state rather than an offline wrapper. | Adapter is not labeled native streaming. |
| `stateful` | boolean | Adapter maintains state across `consume()` calls within one stream. | Each call must be interpreted as stateless unless the adapter contract says otherwise. |

Capabilities must be recorded in provenance and trace metadata. Capability
values do not override runtime causality or visibility checks.

## 15. Hypothesis/Event Semantics

R0 event schema is limited to audio input and ASR/lifecycle outputs. Each
hypothesis event is a complete snapshot, not an append delta.

### Common hypothesis fields

| Field | Required | Authority | Contract |
|---|---:|---|---|
| `scope_id` | yes | Adapter, validated by runtime | Identifies the utterance or output scope. It must be stable for the scope. |
| `event_id` | yes | Runtime | Globally unique within the trace. |
| `revision_index` | yes for PARTIAL/FINAL hypothesis | Adapter, validated by runtime | The first hypothesis snapshot is `0`; every replacement is the immediately previous index plus one. No alternate base is permitted. |
| `text` | yes for PARTIAL/FINAL | Adapter | Complete current hypothesis text, including unchanged prefix. It is not a delta. |
| `replaces_event_id` | optional for first snapshot, required for a replacement | Adapter/runtime | Identifies the current snapshot replaced by this event. It must refer to the immediately active non-final snapshot. |
| `stable_prefix` | optional unless capability is true | Adapter/runtime | Exact prefix of `text`, with the irrevocability rule in Section 17. |
| `consumed_through_sample` | optional / nullable | Adapter declaration, runtime range check | Exclusive sample position incorporated into the hypothesis state. It is `null` when `reports_consumed_position = false`; when present it must not exceed delivered position. |
| runtime observation fields | yes | Runtime | Monotonic observation offset, available position, delivered position, and event index at observation. |

The runtime rejects or marks invalid any event with duplicate event ID, an
unknown replacement, a non-monotonic revision, text treated as a delta, or a
non-null consumed position outside the delivered range. A missing or null
consumed position is valid when `reports_consumed_position = false`; it is
invalid when `reports_consumed_position = true`.

## 16. Revision Semantics

The first PARTIAL or FINAL snapshot in each `scope_id` has
`revision_index = 0`. A new PARTIAL replaces the previous current snapshot for
that scope and has `revision_index = previous + 1`. A FINAL that supersedes a
PARTIAL also has `revision_index = previous + 1`. A replacement may change any
unstable suffix and may also change its length.

The sequence is exactly `0, 1, 2, ...`. A one-based index, a skipped index, a
duplicate index, or a descending index is invalid.

The replacement relationship is explicit. If a consumer emits a new snapshot
without a valid `replaces_event_id`, the runtime records the event as invalid
for revision metrics and must not silently infer a replacement.

After FINAL, no PARTIAL or replacement is valid for that scope. A scope with a
fatal ERROR is closed and cannot later emit FINAL.

## 17. Stable Prefix Contract

`stable_prefix` must be an exact prefix of the same event's `text`. Once a
prefix has been declared stable for a scope, every later PARTIAL and FINAL in
that scope must begin with that exact prefix. The prefix can grow but cannot be
shortened or changed.

If an adapter cannot provide this guarantee, it must declare
`supports_stable_prefix = false` and omit stable-prefix claims.

TTFT-stable is computable only when `supports_stable_prefix = true` and a
non-empty stable prefix is first observed. A consumer's assertion alone does
not make an invalid prefix valid; the runtime verifies exact-prefix relations.

## 18. FINAL / ENDPOINT / ERROR Semantics

### FINAL

FINAL is the complete immutable successful transcript snapshot for one scope.
There may be at most one FINAL per scope. FINAL supersedes earlier PARTIAL
snapshots and closes the scope for hypothesis revision.

### ENDPOINT

ENDPOINT is a consumer belief that an utterance or input scope ended. It is not
equivalent to FINAL. A valid sequence is `ENDPOINT -> rescoring -> FINAL`.
The adapter must document whether it emits FINAL before, with, or after
ENDPOINT.

### ERROR

Every ERROR is classified as fatal or recoverable and identifies its scope.

- A fatal ERROR closes the scope and prohibits later FINAL.
- A recoverable ERROR permits later events only when the adapter explicitly
  declares the recovery transition.

### `finish()`

`finish()` is called only after source end is visible. It may flush delayed
hypotheses or FINAL events. It must not claim audio or metadata was available
before its actual logical release. Outputs from `finish()` belong to tail
latency, not steady-state consume time.

The R4 offline Whisper reference emits FINAL only. It does not claim PARTIAL,
revision, stable-prefix, native-streaming, or model endpointing capability.

## 19. Trace Contract

The raw trace is persisted as append-only JSONL evidence. The canonical raw
trace path is conceptually `trace.jsonl`.

Each JSONL line is exactly one runtime event record. Each record includes a
`schema_version` field so event schema evolution is explicit. The writer
appends new lines only. No post-processing step may overwrite, reorder, delete,
or modify an existing JSONL line.

Run-level provenance is a separate sidecar, conceptually
`trace.run.json`. It is not mixed into the event stream. The sidecar includes
at least:

| Field | Required | Meaning |
|---|---:|---|
| `schema_version` | yes | Version of the run-sidecar contract. |
| `run_id` | yes | Unique identity of the replay run. |
| `replay_mode` | yes | `COMPUTE_UNAWARE` or `COMPUTE_AWARE`. |
| source identity / SHA | yes | Canonical source identity and hashes. |
| source sample rate | yes | Samples/second used for all sample positions. |
| channel IDs | yes | Ordered source channels and layout. |
| packet configuration | yes | Integer packet width and derived presentation width. |
| queue policy | yes | Named queue/overload condition. |
| consumer identity/config | yes | Adapter identity and adapter configuration provenance. |
| consumer capability declaration | yes | The complete capability set, including `reports_consumed_position`. |
| `code_commit` | yes | Exact repository commit used for the run. |
| oracle condition | yes | Metadata visibility and segmentation condition. |
| warm-up/load policy | yes | Load, warm-up, measured interval, and tail policy. |
| device provenance | conditional | Device, synchronization, and shared-load information when relevant. |
| clock policy | yes | Logical, monotonic, and UTC provenance rules. |
| metric window | yes | Measured sample and wall boundaries. |
| trace path | yes | Path of the JSONL evidence file. |
| trace SHA | conditional | Hash of the completed JSONL file when finalized. |

JSONL is the persisted evidence format. Python dataclasses, dictionaries, or
other in-memory objects are implementation details and are not frozen by R0.
Derived metrics are separate artifacts and must never be written back into the
raw JSONL stream.

The trace must cover these event categories:

- `SOURCE_AVAILABLE`;
- `PACKET_DELIVERED`;
- `CONSUMER_CALL_START`;
- `CONSUMER_CALL_RETURN`;
- `HYPOTHESIS`;
- `ENDPOINT`;
- `ERROR`;
- `ORACLE_METADATA_RELEASE`;
- `SOURCE_END`;
- `FINISH_START`;
- `FINISH_RETURN`;
- `RESET`.

Implementations may choose different literal names only when the event
semantics remain one-to-one with this list.

Every trace record contains:

| Field | Authority | Unit/meaning | Invariant |
|---|---|---|---|
| `schema_version` | Runtime | version identifier | Present on every persisted event record; the reader rejects an unsupported version. |
| `event_index` | Runtime | non-negative integer | Strictly increases by one in append order. |
| `stream_id` | Runtime | opaque ID | Same for all records in one stream. |
| logical/sample position | Runtime/source | integer sample position or explicit null for non-audio lifecycle events | Must identify the availability/delivery boundary relevant to the event. |
| monotonic/wall offset | Runtime | monotonic elapsed time | Used for elapsed metrics; never UTC. |
| event-specific fields | Event producer, runtime-validated | schema of the event category | Must satisfy the category invariants above. |

The sidecar records source identity and hashes, sample rate, channels and
layout, packet size, replay mode, queue policy, timeline authority,
adapter/model/config identity, capability declaration, code commit,
load/warm-up policy, device identity when relevant, clock origin, metric
window, and the JSONL trace path/hash.

## 20. MISP Canonical Source Contract

The first canonical continuous source is the prepared M028 eight-channel WAV
set:

```text
/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename/<recording>_0.wav
/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename/<recording>_1.wav
...
/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename/<recording>_7.wav
```

For `M028_S197199201241_F8N_Far`, confirmed properties are:

- 16,000 Hz;
- PCM-s32le;
- one mono WAV per channel;
- eight synchronized channel files;
- 12,772,784 frames per channel;
- 798.299 seconds;
- common prepared/F8N Stage-2 time zero.

The raw interleaved CSOBx3 PCM is not the R0 canonical source. For M028 it
starts at a different zero point and needs the explicit `+82.463 s` mapping to
prepared time. A later raw source adapter must carry raw format, channel
layout, sample rate, crop offset, and hashes explicitly; it may not infer them
from a headerless file.

The runtime must not implicitly downmix the eight channels. Channel selection
or a frontend transformation is an explicit source/frontend condition.

The first R4 reference condition is frozen as:

| Provenance field | Required value |
|---|---|
| `source` | `prepared_continuous_8ch` |
| `frontend_condition` | `raw_ch0` |
| `selected_channel_id` | `0` |
| `materialization` | only arrived continuous samples in `[S,E)` |

The source remains the complete continuous eight-channel recording. The
`raw_ch0` condition is an explicit adapter selection, not a replacement of the
source by an existing segment artifact.

## 21. Stage-2 Timeline Authority

The MISP Stage-2 artifacts are authoritative for segment identity and reference
timeline:

```text
/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/segments
/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/text
/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/utt2spk
/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/spk2utt
```

The authority fields are `segment_id`, `recording_id` as represented by the
authoritative IDs, `start`, and `end`. The Stage-2 segment ID time suffix
encodes centiseconds (`time * 100`). It is not a millisecond or frame index.

Replay scheduling converts authoritative boundaries to integer sample
positions at the source `Fs`. For M028 this conversion is exact because
centisecond boundaries at 16 kHz are multiples of 160 samples. A non-exact
conversion must fail closed rather than round silently.

The current Whisper manifest and materialized raw/GSS WAV/FLAC files are
derived artifacts. They may be used after exact ID reconciliation, but they do
not replace Stage-2 as the segment-time authority.

## 22. Oracle / Segmentation Visibility Matrix

The evaluator condition must be named in the run provenance. The following
matrix is the minimum R0 vocabulary.

| Condition | Segment start visible when | Segment end visible when | Speaker ID visible to consumer? | Reference text visible to consumer? | Full future segment list visible? | Fully online claim? |
|---|---|---|---:|---:|---:|---:|
| Reference-only evaluator | Evaluator may know at run setup | Evaluator may know at run setup | No | No | No to consumer | No; evaluator-only oracle |
| R4 default pseudo-online | At the named R4 gate; complete descriptor is released at `E` | When sample `E` becomes available | No | No | No | No; precomputed boundary oracle |
| Start-known oracle | When source reaches `S` | When source reaches `E` | No | No | No | No; start oracle remains |
| Full-segmentation oracle | At condition setup | At condition setup | Only if separately named | Only if separately named | Yes | No |
| Future online segmentation | Only after an online causal detector emits it | Only after an online causal detector emits it | No by default | No | No | Yes, if no future data or metadata is disclosed |

For R4, the only allowed segment descriptor release is the named
`ORACLE_METADATA_RELEASE` at or after `E`: `segment_id`, `S`, and `E`. The gate
must not release future segment lists, speaker labels, or reference text.

## 23. R4 Offline Whisper Contract

Existing Whisper V0 is not modified.

R4 executes the following conceptual path:

```text
continuous replay
  -> source reaches E
  -> oracle gate releases segment_id, S, E
  -> select channel_id = 0 from arrived continuous samples
  -> materialize [S,E) as explicit mono, 16 kHz Whisper-compatible waveform
  -> existing ASRBackend.transcribe(path)
  -> FINAL-only OutputEvent
```

R4 provenance records `source = prepared_continuous_8ch`,
`frontend_condition = raw_ch0`, `selected_channel_id = 0`, and the exact
materialization range `[S,E)`. The replay runtime still delivers eight-channel
`AudioChunk` values. Only the R4 adapter selects channel 0 after the oracle
gate; the runtime does not downmix or average channels.

R4 is permanently labeled:

**segment-level pseudo-online / oracle-boundary-assisted offline-backend
reference**

It must not be labeled native streaming, fully online segmentation, or
causal endpointing. Existing segment-level `raw_ch0` WAVs and GSS segment
artifacts are not replay sources. Future beamformed, GSS/block-online, FIVE,
or neural frontend conditions must use distinct explicit frontend names and
must not silently replace `raw_ch0`.

The existing `transcribe()` call remains a complete-file call. Model load is
outside its current `decode_sec`; replay compute metrics must not relabel that
field as end-to-end streaming latency.

## 24. Causality Contract

The runtime must physically constrain both future audio and future metadata.
It must control which samples are materialized and which descriptors are
passed to the consumer. A consumer-reported
`consumed_through_sample` is useful for metrics but cannot establish causality
by itself.

At any observation boundary, the consumer may have received only:

- audio samples in the delivered packet history;
- metadata whose explicit release event is at or before that boundary;
- startup metadata permitted by Section 11.

The runtime must reject a packet or metadata release with a logical release
position later than the current causal boundary. A future-information leak is
a contract failure even if the consumer reports an earlier consumed position.

## 25. Future-Perturbation Test

The required causality test is:

1. choose cutoff sample `T`;
2. construct Run A and Run B with identical source prefix `[0,T)`;
3. make the futures after `T` different;
4. use identical deterministic runtime configuration and consumer behavior;
5. inspect the trace before any audio or metadata beyond `T` is delivered;
6. compare outputs observed before that boundary.

The comparison includes:

- `scope_id`;
- hypothesis text;
- revision sequence;
- stable-prefix declarations;
- logical event order.

The comparison does not require bitwise-equal GPU wall times. A logical
difference before `T` is classified as a future-information leak or as named
controlled nondeterminism requiring separate diagnosis. The test must inspect
actual visibility, not only self-reported consumption.

## 26. Metric Measurement Phases

Every run separates four phases:

1. **model load** — adapter/model construction and weight loading;
2. **warm-up** — explicitly configured calls before measured streaming;
3. **measured streaming interval** — the declared source/consume window;
4. **finish/tail** — source-end handling and `finish()` outputs.

The phase boundaries are recorded in the trace/provenance. The default
steady-state RTF excludes model load and warm-up. Tail compute is included in
the total-compute metric and is not silently folded into consume time.

If load or warm-up is skipped, the run must record that fact. If the measured
interval is empty, streaming RTF is unavailable rather than zero.

## 27. RTF Contract

For the measured interval define:

- `D_audio`: source audio duration in seconds represented by the measured
  interval, computed from integer sample boundaries;
- `C_consume`: sum of wall elapsed time inside measured `consume()` calls;
- `C_finish`: wall elapsed time inside `finish()`.

The required metrics are:

```text
RTF_consume       = C_consume / D_audio
RTF_total_compute = (C_consume + C_finish) / D_audio
```

The metric producer is the runtime using monotonic wall offsets; the evaluator
consumes the values and their phase/window provenance. A denominator of zero,
missing phase boundary, or missing GPU completion boundary makes the affected
metric unavailable.

Current Whisper V0 `decode_sec / audio_sec` measures a synchronous file decode
call against that file's audio duration. It excludes model load because timing
starts inside `transcribe()`, and it does not include replay queue delay,
delivery lag, source pacing, or `finish()` tail. It is not interchangeable with
`RTF_consume`, `RTF_total_compute`, or wall-clock E2E latency.

## 28. Backlog and Processing Lag

At a recorded observation point:

```text
queue_backlog_samples   = available_through_sample - delivered_through_sample
queue_backlog_sec        = queue_backlog_samples / Fs
```

Queue backlog measures audio available but not yet delivered. Processing lag
is defined only when `reports_consumed_position = true` and a non-null,
range-valid consumed position is present:

```text
processing_lag_samples  = available_through_sample - consumed_through_sample
processing_lag_sec        = processing_lag_samples / Fs
```

Processing lag also includes audio already delivered but not yet incorporated
into consumer state. The two metrics must be reported separately.

Queue backlog must be non-negative under the delivery invariant. Processing lag
is `UNAVAILABLE` when consumed position is null or the capability is false; it
must not be computed by substituting delivered position. Any negative value
when the required position is present is a runtime state error, not a valid
metric.

## 29. Backlog Statistics

Compute-aware traces report:

- maximum queue backlog;
- time-weighted mean queue backlog;
- P50 queue backlog;
- P95 queue backlog;
- final queue backlog.

The reference percentile sampling policy is packet-availability transitions:
sample the step-function backlog immediately after each transition that makes a
new packet fully available and before delivery caused by that transition.
The policy, sample count, and sample unit are recorded. P50 and P95 are
percentiles of those recorded sample values, not percentiles of arbitrary log
line counts.

Time-weighted mean is:

```text
integral(queue_backlog_samples(t) dt) / measured_wall_interval
```

The integration interval and boundary convention are recorded. If no
compute-aware wall interval exists, backlog percentiles may be unavailable and
must not be synthesized from compute-unaware events.

## 30. Compute Duty

For a single consumer:

```text
compute_duty = total_consumer_busy_wall_time / measured_wall_interval
```

Busy time is the union of measured consumer-call intervals. Overlapping
intervals are counted once. The numerator producer is the runtime trace; the
evaluator consumes the value and phase/window provenance.

Compute duty is unavailable when call start/return boundaries are missing or
the measured wall interval is non-positive. A multi-consumer extension requires
a separate ownership and overlap contract.

## 31. Delivery Lag and Deadline Miss

For each packet in compute-aware mode:

```text
delivery_lag = delivery_wall_time - scheduled_availability_wall_time
```

Delivery lag is measured in monotonic wall seconds and may be zero or positive.
A negative value is a clock/order failure.

Deadline miss is computed only when configuration declares
`deadline_budget` in seconds:

```text
deadline_miss = delivery_lag > deadline_budget
```

Without an explicit budget, delivery lag is reported and deadline miss is
`UNAVAILABLE`. No CHiME or challenge threshold is hard-coded.

## 32. Completion Overrun

Completion overrun is:

```text
completion_overrun = last_final_or_finish_wall_time
                      - source_end_available_wall_time
```

The last useful event is FINAL when a final output exists; otherwise it is the
return of `finish()`. A negative value means useful completion was observed
before source-end availability under the named condition; positive means tail
overrun. The source-end and output event IDs are recorded with the metric.

## 33. CHiME-style vs Real Wall Latency

Two latency families are intentionally separate.

**CHiME-style input-consumption-position latency** reports how far into the
input the consumer had incorporated when it emitted an output. It uses sample
positions and is meaningful for algorithmic streaming analysis only when
`reports_consumed_position = true` and the event has a valid non-null consumed
position. Otherwise it is `UNAVAILABLE`.

**Wall-clock end-to-end delay** reports how long elapsed between the relevant
source availability boundary and runtime observation of the output. It includes
queue delay, consumer compute, and scheduling effects.

Neither metric may be substituted for the other. A report must name the family,
sample/wall authority, start boundary, end boundary, and scope.

## 34. Stability Metrics

The following metrics are available only when the declared capabilities and
event contract support them:

| Metric | Definition | Availability condition |
|---|---|---|
| `stream_TTFT` | First observed PARTIAL or FINAL wall time minus measured stream start wall time. | `supports_partial` or a FINAL-only consumer with a first final. This is stream-relative and does not represent utterance onset. |
| `stream_TTFT-stable` | First observed hypothesis with a non-empty valid irrevocable `stable_prefix` minus measured stream start wall time. | `supports_stable_prefix = true`; exact-prefix checks pass. This is stream-relative. |
| `scope_TTFT` | First observed PARTIAL or FINAL wall time minus a trusted causal scope-start wall time. | Optional future metric; `scope_start_wall` must have an explicit authority. It is `UNAVAILABLE` when no trusted scope start exists. |
| `scope_TTFT-stable` | First valid non-empty stable-prefix observation minus a trusted causal scope-start wall time. | Optional future metric; stable-prefix capability and trusted scope start are both required. |
| TTLT | Final observation wall time minus the declared scope start wall boundary. | One valid FINAL exists. |
| Revision count | Number of valid replacement PARTIAL events in a scope. | `supports_revision = true`; replacement links valid. |
| Flicker | Edit distance or equivalent declared difference between consecutive complete snapshots. | At least two valid snapshots; text unit and algorithm recorded. |
| Stable-prefix growth | Sequence of declared stable-prefix lengths at observation events. | `supports_stable_prefix = true`. |

If the capability or trusted timing authority is false, the evaluator reports
the affected metric as unavailable, not zero. The trace still preserves events
that were actually emitted. A continuous meeting's measured stream start is
not an utterance onset. A Stage-2 oracle start is not a fully-online scope
start and must not be used as one without a named oracle condition and explicit
metric authority.

## 35. GPU Timing Policy

Host `perf_counter()` covering a model call does not automatically prove that
asynchronous GPU work has completed. A GPU-backed adapter must declare the
synchronization strategy and the point at which the measured call represents
completed work. R0 does not implement GPU timing.

The run provenance records device identity, adapter synchronization policy,
CUDA/PyTorch-relevant version identity when available, and whether shared GPU
load was present. Shared load is a provenance condition and a source of RTF
variation; it must not be silently treated as model compute behavior.

If completion synchronization is absent or unknown, GPU-compute-specific
metrics are unavailable. Host wall observation of the returned Python call is
still a valid wall observation, but it must not be labeled synchronized GPU
duration.

## 36. Queue / Overload Policy

The R3 reference policy is:

- one logical input stream;
- one synchronous consumer;
- lossless queue;
- unbounded queue;
- no drop;
- no overwrite;
- source time never pauses while compute-aware inference blocks.

A consumer with sustained `RTF_consume > 1` causes queue backlog to grow
according to source and consumer rates. A bounded/drop/overwrite policy is a
different experiment condition with a distinct policy name, trace provenance,
and metric interpretation.

## 37. Required 20ms / 300ms Worked Timeline

Use `Fs = 16,000` Hz. A 20 ms source packet is 320 samples.

```text
P0 = [0, 320)
```

P0 becomes available at 20 ms and is delivered at 20 ms. The synchronous
consumer blocks for 300 ms and returns at wall time 320 ms.

At return time:

```text
available_through_sample  = 5120
delivered_through_sample  = 320
consumed_through_sample   <= 320 when reported; otherwise null
```

Packets P1 through P15 have become fully available during the blocked call.
Before the scheduler delivers the next packet, the queue contains 15 packets,
which is:

```text
4800 samples = 300 ms
```

If the consumer returns a PARTIAL, the runtime observes and records it at 320
ms. It then updates state and only then delivers P1. P1 was scheduled for
availability at 40 ms, so:

```text
delivery_lag(P1) = 320 ms - 40 ms = 280 ms
```

After P1 is delivered, P2 through P15 remain queued: 14 packets, 4,480
samples, 280 ms. The ordering is therefore:

```text
availability of P1..P15
-> consumer return and PARTIAL observation
-> state update
-> delivery of P1
```

R1/R3 fake-slow-consumer tests must reproduce this ordering and these sample
counts. Any implementation that pauses source availability while `consume()`
blocks violates the contract.

## 38. Planned Config Schema

This is a planned schema only. R0 does not create YAML or JSON configuration.

| Group/field | Type | Required | Owner | Contract |
|---|---|---:|---|---|
| `source.path` | path/string | yes | Runtime source | Explicit continuous source path; must resolve to the selected source artifact. |
| `source.stream_id` | string | yes | Runtime | Stable logical stream identity. |
| `source.sample_rate` | positive integer | yes or source-derived with verification | Runtime source | Must match decoded source and timeline conversion. |
| `source.channel_ids` | ordered list | yes | Runtime source | Must match samples and source files; no implicit downmix. |
| `source.sha256` | hash map | provenance required | Runtime | Hashes canonical source files before replay. |
| `consumer.frontend_condition` | enum/string | yes for R4 | Adapter boundary | R4 reference value is `raw_ch0`; other frontend values are distinct named conditions. |
| `consumer.selected_channel_id` | non-negative integer | required for channel-selecting conditions | Adapter boundary | R4 reference value is `0`; it never authorizes runtime downmix. |
| `packet_samples` | positive integer | exactly one of samples/ms unless consistent | Runtime | Primary integer packet width. |
| `packet_ms` | decimal | exactly one of samples/ms unless consistent | Runtime | Derived/config presentation width; must convert exactly to samples at `Fs`. |
| `replay_mode` | enum | yes | Runtime | `COMPUTE_UNAWARE` or `COMPUTE_AWARE`. |
| `queue_policy` | enum | yes | Runtime | R3 reference is `lossless_unbounded`; alternatives must be named experiments. |
| `consumer.adapter` | string | yes | Adapter boundary | Identifies adapter without model branching in runtime core. |
| `consumer.config` | mapping | adapter-defined | Adapter | Processing chunk, look-ahead, decoder, and model state settings. |
| `warmup.enabled` | boolean | yes | Runtime/adapter | Records whether warm-up is part of the run. |
| `warmup.calls` | non-negative integer | conditional | Adapter | Exact warm-up policy; excluded from measured streaming interval. |
| `metrics.measured_start_sample` | integer | yes | Runtime/evaluator | Start of measured audio interval. |
| `metrics.measured_end_sample` | integer | yes | Runtime/evaluator | Exclusive end of measured audio interval. |
| `deadline_budget` | non-negative seconds | optional | Runtime/evaluator | Required before deadline miss is computed. |
| `oracle_condition` | enum/string | yes when segment metadata is used | Runtime/evaluator | Names visibility condition and disclosure rules. |
| `trace.raw_path` | path | yes | Runtime | Append-only raw trace destination. |
| `trace.run_provenance_path` | path | yes | Runtime | Separate run-level sidecar, conceptually `trace.run.json`. |
| `trace.schema_version` | version identifier | yes | Runtime | Version used by persisted JSONL records and sidecar. |
| `trace.derived_paths` | path map | optional | Evaluator | Derived metrics/results; never overwrite raw trace. |
| `provenance.code_commit` | commit SHA | yes | Runtime | Exact code commit used for the run. |
| `provenance.device` | mapping | conditional | Runtime/adapter | Device and shared-load identity when relevant. |

The runtime owns source, packet, mode, queue, clock, oracle, trace, and metric
window fields. The adapter owns model identity, model processing chunk,
look-ahead, decoder, cache, and synchronization fields. A config loader must
reject a missing required authority or a non-exact packet conversion.

## 39. Planned Package Boundary

The planned conceptual package boundary is:

```text
src/robot_heard/
  replay/
    source
    clock
    scheduler
    events
    runtime
    metrics
  streaming/
    base
    adapters/
      offline_whisper
      rolling_whisper
      u2
      fastconformer
  misp/
    existing modules unchanged
```

This section records ownership only. R0 does not create these modules, classes,
or files. Model logic must remain in adapters and must not be moved into the
replay core.

## 40. R1 Test Plan

R1 is a future synthetic, CPU/model-free gate. This section plans tests only;
no tests are created in R0.

The minimum R1/R3/R4 test inventory is:

- exact integer-sample packetization;
- exact tail packet and final-source event;
- no loss;
- no duplication;
- multichannel aligned ranges;
- no implicit downmix;
- fake clock behavior;
- deterministic trace reproducibility;
- fixed same-time ordering;
- reset state isolation;
- PARTIAL snapshot and revision rules;
- first hypothesis `revision_index = 0`, exact +1 replacement sequence, and
  first-FINAL-at-zero;
- stable-prefix exactness and irrevocability;
- fatal and recoverable error rules;
- FINAL and `finish()` rules;
- future perturbation causality;
- metadata leakage prevention;
- slow synchronous consumer behavior;
- sustained `RTF > 1` backlog growth;
- queue backlog versus processing lag;
- delivery lag and explicit deadline budget;
- source-end/tail completion overrun;
- oracle segment release at E only for R4;
- existing offline Whisper interface unchanged;
- exact MISP sample mapping and source hashes in R2.

Every test must state its source of time, sample unit, expected trace ordering,
and failure condition. Tests that require Whisper, CUDA, GSS, diarization, or
MISP replay are not R1 tests.

## 41. R0 Acceptance Checklist

The following checklist quotes the 22 R0 invariants from Issue #4. `DEFINED`
means this document gives a testable contract. It does not mean that R1–R5
implementation or server validation has occurred. R0 status is PASS; R1
implementation and server validation remain under review.

| # | Issue #4 invariant | Status | Contract location / failure condition |
|---:|---|---|---|
| 1 | Same input/config + deterministic consumer => same logical trace. | DEFINED | Sections 8, 19, 25. Logical event mismatch is a deterministic-trace failure. |
| 2 | Every normal source sample is released exactly once. | DEFINED | Sections 6, 10. Missing, duplicate, or overlapping normal packet range fails. |
| 3 | Packet availability, delivery, consumption, and output observation are distinct. | DEFINED | Sections 5, 15, and 19. Consumption is optional/nullable; absence is recorded, not fabricated. |
| 4 | `consumed <= delivered <= available`. | DEFINED | Section 5. When consumed is present, `consumed <= delivered <= available`; when absent, `delivered <= available`. |
| 5 | Same-time event order is fixed. | DEFINED | Section 9 and Section 37. Different ordering fails deterministic scheduler tests. |
| 6 | Prepared MISP time uses integer sample addressing. | DEFINED | Sections 4, 20, 21. Float accumulation or non-exact conversion fails closed. |
| 7 | 8ch input is never implicitly downmixed. | DEFINED | Sections 10, 20, and 23. R4 explicitly selects `raw_ch0`/channel 0; missing channel policy fails source validation. |
| 8 | Consumer startup metadata does not silently reveal future annotations/endpoints. | DEFINED | Sections 11, 22, 24. Undeclared disclosure is a causality failure. |
| 9 | Partial is a full snapshot, not an unspecified delta. | DEFINED | Section 15. Delta-only text fails event validation. |
| 10 | Revision replacement and stable-prefix rules are explicit. | DEFINED | Sections 16 and 17. Invalid replacement or prefix withdrawal fails. |
| 11 | FINAL/ENDPOINT/ERROR/`finish()` semantics are explicit. | DEFINED | Section 18. Final-after-fatal, duplicate final, or early finish fails. |
| 12 | Compute-aware source time does not pause while inference is slow. | DEFINED | Sections 8, 36, 37. Missing packets during a blocked call fails pacing tests. |
| 13 | Queue backlog and processing lag are different metrics. | DEFINED | Section 28. A report using one as the other is invalid. |
| 14 | RTF numerator/denominator/window are explicit. | DEFINED | Sections 26 and 27. Missing phase/window makes RTF unavailable. |
| 15 | Model load/warm-up/tail are separated. | DEFINED | Section 26. Unlabeled load or tail time fails provenance validation. |
| 16 | CHiME-style input-position latency and wall E2E delay have different names. | DEFINED | Section 33. Position latency requires `reports_consumed_position = true`; substitution without label fails metric validation. |
| 17 | Causality test constrains actual visible audio/metadata, not just self-reported consumption. | DEFINED | Sections 11, 24, 25. Consumer self-report alone cannot pass. |
| 18 | R4 precomputed segmentation remains explicitly oracle/pseudo-online. | DEFINED | Sections 22 and 23. Native/fully-online label fails R4 validation. |
| 19 | Raw trace is append-only; derived metrics do not mutate it. | DEFINED | Section 19. Any derived rewrite fails evidence validation. |
| 20 | Current offline Whisper V0 contract remains unchanged. | DEFINED | Sections 2 and 23. Replay implementation requiring ASRBackend redesign fails scope review. |
| 21 | A new ASR adapter does not require replay clock/scheduler changes. | DEFINED | Sections 3, 13, 23, and 39. R4 channel selection is adapter policy; a model-name branch in runtime fails boundary review. |
| 22 | R1–R5 schema scope is audio+ASR; R6 extensions require review. | DEFINED | Sections 1, 19, 39, 42. Unreviewed AVDR schema expansion fails R0 scope. |

The remaining questions are future implementation or governance questions, not
unfrozen R0 core definitions:

- which exact GPU synchronization primitive each future adapter will declare;
- whether a later raw-PCM source adapter is needed in addition to the canonical
  prepared source;
- whether PR base should be retargeted after PR #2 merges.

## 42. Explicit Non-goals

R0 does not implement or decide:

- scheduler;
- replay runner;
- Python `AudioChunk` dataclass;
- `FakeClock`;
- `DummyConsumer`;
- production `StreamingConsumer` class;
- Whisper adapter implementation;
- U2/U2++ adapter;
- FastConformer/RNNT adapter;
- pyannote;
- GSS, WPE, SpatialNet, MossFormer2, or FIVE;
- online diarization or speaker tracking;
- online segmentation implementation;
- cpCER or official challenge latency compliance;
- multimodal fusion;
- video, RGB-D, or LiDAR;
- model training, fine-tuning, or GPU kernel optimization;
- changes to `src/`, `scripts/`, `configs/`, `tests/`, README, dependencies,
  Whisper V0, or the MISP baseline.

The current permitted gate is **R1 review and validation only**. R1 must not be
treated as externally passed until implementation and server validation are
explicitly reviewed.
