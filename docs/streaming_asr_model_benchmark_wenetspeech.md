# WenetSpeech U2++ Streaming ASR Benchmark & Closure Report

## 实验概览（Quick Summary）

本报告整理 WeNet streaming ASR 线的最终 closure evidence，回答两个问题：第一，WenetSpeech U2++ 是否能在 MISP far-field raw input 上作为稳定的 native-streaming Chinese ASR baseline；第二，服务器上是否存在可以直接接入当前 replay harness、满足严格 causal multichannel contract 的现成 frontend。本次最终执行的数据是 MISP 2025 `dev_far` 中的 `M028_S197199201241_F8N_Far`，即 1 个 meeting、1 个连续录音、798.299 秒音频、144 个 Stage-2 scopes 和 2395 个 reference characters。连续 8-channel source 位于 `/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename`，Stage-2 authority 位于 `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far`。

输入条件是连续 8-channel、16 kHz、PCM32LE 远场 WAV，但 ASR 条件明确选择 `raw_ch0`，也就是第 0 个物理通道；除冻结的 PCM32LE 到 PCM16LE 转换外，没有 GSS、beamforming、speech enhancement 或 separation。Stage-2 START/END scope boundary 是 oracle ASR boundary，因此本结果评估的是 native streaming ASR，不是 fully-online segmentation + ASR；reference text、未来 RTTM 和 oracle speaker identity 都没有进入 decoder。模型是官方 WenetSpeech U2++ Conformer LibTorch，模型目录为 `/home/gc6_challenge/models/native_streaming/wenet/wenetspeech/20220506_u2pp_conformer_libtorch`，WeNet source revision 为 `d17059667d6afe0680d19b3a4948ab825ef25105`，使用官方 `wenet_init` / `wenet_decode` / `wenet_reset` C API 和 CPU LibTorch 2.2.0 runtime。

本次 raw baseline 的 CER 为 `0.33903966597077245`，S/D/I/N 为 `411/352/49/2395`，empty/non-empty 为 `5/139`；closure run 的 `RTF_consume` 为 `0.1787101070`，`RTF_total_compute` 为 `0.1792607488`，TTFT P50/P95 为 `0.819962/2.098514` 秒，segment-end-to-FINAL P50/P95 为 `0.183999/0.345144` 秒。主脚本是 `/home/gc6_challenge/code/wenet-streaming-frontend-ablation/scripts/run_wenet_frontend_ablation.py`，主 artifact 根目录是 `/home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation`。

服务器 frontend audit 找到了 Standard GSS、OnlineFIVE、batch FIVE/IVE、NBSS OnlineSpatialNet 和 AV-SkiM target-speaker extraction，但没有任何一个同时满足严格的持久 state、有限 lookahead、sample-aligned output、无未来 metadata/oracle、可驱动 8-channel MISP stream 等要求。因此没有执行 Condition B，也没有自行实现新 frontend；按最新任务约束，Condition C offline GSS 也没有执行。本次实际执行数据仍是 M028-only，尚未完成原计划中的 multi-meeting expanded evaluation；这不是因为六条 recording 缺失，而是因为最终范围被明确收敛到单个 M028 meeting。当前结果可作为后续 Whisper、FastConformer 或其他 streaming ASR 的统一 raw-input / oracle-scope baseline，但不能被描述为 full `dev_far` expanded result。

## 关键结果速览

| 项目 | 结果 |
|---|---|
| Dataset | MISP 2025 |
| Split | `dev_far` |
| Meetings | `M028` / `M028_S197199201241_F8N_Far`，1 个 |
| Input | continuous 8ch source，ASR selects `raw_ch0`；无 GSS/beamforming/enhancement/separation |
| Audio duration | 798.299 s（约 0.222 h） |
| Scopes | 144 Stage-2 scopes |
| Reference chars | 2395 |
| Model | WenetSpeech U2++ Conformer LibTorch |
| MISP fine-tune | NO evidence；按当前 provenance 记为 NO |
| Native streaming | YES；stateful official WeNet C API |
| CER | 0.33903966597077245 |
| S/D/I/N | 411 / 352 / 49 / 2395 |
| RTF_total | 0.1792607488290538（closure run） |
| TTFT | mean 1.103375 s；P50 0.819962 s；P95 2.098514 s；max 2.742141 s |
| End-to-FINAL | mean 0.206600 s；P50 0.183999 s；P95 0.345144 s；max 0.438135 s |
| Streaming frontend | 无 `TRUE_STREAMING_REUSABLE` candidate |
| Current status | `WF_NO_REUSABLE_TRUE_STREAMING_FRONTEND`；报告状态 `WENET_CLOSURE_REPORT_READY_FOR_EXTERNAL_REVIEW` |

## 1. Experiment Purpose

本阶段冻结一套可复用于不同 streaming ASR 模型的 benchmark protocol，而不是继续开发模型或 frontend。研究问题是：

1. WenetSpeech U2++ 能否作为稳定的 native-streaming Chinese ASR baseline；
2. 在 MISP far-field raw input 上，accuracy、partial/final 行为和 runtime 如何；
3. 当前服务器是否存在可以直接复用的真正 causal 8-channel frontend；
4. 为什么 Standard GSS、OnlineFIVE 等候选不能直接作为严格 online generic-ASR frontend baseline；
5. 如何冻结数据、输入、scope policy、PCM policy、scorer、runtime 和 latency 指标，便于后续比较 Whisper、FastConformer 或其他 streaming ASR。

本报告不重新训练模型、不修改 ReplaySource/PacedReplayRuntime/Clock/Trace semantics、不修改已通过的 WeNet adapter semantics、不进入 Nemotron 或 R6。

## 2. Dataset & Input Provenance

### 2.1 Authority and coverage

| 字段 | 事实 |
|---|---|
| Dataset | MISP 2025 |
| Split | `dev_far` |
| Full available inventory | 6 recordings，2455 scopes，11661.225 s，44813 reference characters，25 speakers |
| Final evaluation | `M028_S197199201241_F8N_Far` |
| Final meeting count | 1 |
| Final recording count | 1 |
| Final scope count | 144 |
| Final recording duration | 798.299 s |
| Final reference characters | 2395 |
| Final speakers | 4 |
| Sample rate | 16000 Hz |
| Source format | 8 个同步 mono WAV，每个为 PCM 32-bit little-endian；每个物理文件本身为 single-channel WAV |
| Timeline authority | Stage-2 `segments` 的整数 sample timeline；selected source 的 `trace.run.json` 记录 `integer_sample_index` |
| Reference authority | Stage-2 `text`，经 frozen `v0_nfkc_casefold_strip_space_punct` normalization 评分 |
| Selection rule | 最新用户指令要求最终流式实验使用完整 M028；selection 在 decode 前冻结，未按识别结果筛 meeting |

本次实际执行数据是 `M028-ONLY BENCHMARK`。此前 inventory 已确认当前服务器上的 6 条 recording 均为 `AVAILABLE`，连续 8-channel source 和 Stage-2 reference 均完整；因此最终没有使用全部 6 条，不是因为 data missing 或 reference incomplete，而是因为任务范围按最新指令收敛到一个 meeting。不能将本结果称为 `full dev_far evaluation` 或 `expanded multi-meeting experiment`。

### 2.2 Absolute paths

- Continuous far-field source directory: `/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename`
- Selected source stem: `/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename/M028_S197199201241_F8N_Far`
- Selected physical channels: `/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename/M028_S197199201241_F8N_Far_0.wav` through `_7.wav`
- Stage-2 Kaldi data directory: `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far`
- Stage-2 `wav.scp`: `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far/wav.scp`
- Stage-2 `segments`: `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far/segments`
- Stage-2 `text`: `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far/text`
- Stage-2 `utt2spk`: `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far/utt2spk`
- Stage-2 `spk2utt`: `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far/spk2utt`
- Historical M028 parity `segments` authority: `/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/M028_far/segments`

### 2.3 Input condition

`raw_ch0` means channel 0 of the prepared continuous MISP far-field multichannel recording. It is not an enhanced recording. No GSS, beamforming, speech enhancement, separation, normalization, or channel downmix is used. The only input conversion is the frozen signed PCM32LE arithmetic right shift by 16 with saturation to signed PCM16LE.

The Stage-2 START/END boundary is an oracle ASR scope boundary. This boundary is used to open/reset native decoder sessions and to release FINAL, but future scope text, reference transcript, RTTM, or speaker identity is not supplied to the decoder. Consequently, the result is a native-streaming ASR qualification under oracle segmentation, not a fully-online segmentation + ASR result.

## 3. Model Provenance

| 字段 | 事实 |
|---|---|
| Model family | Official WenetSpeech U2++ Conformer LibTorch |
| Model directory | `/home/gc6_challenge/models/native_streaming/wenet/wenetspeech/20220506_u2pp_conformer_libtorch` |
| Checkpoint path | `/home/gc6_challenge/models/native_streaming/wenet/wenetspeech/20220506_u2pp_conformer_libtorch/final.zip` |
| Checkpoint size | 521,167,532 bytes |
| Extracted `final.zip` SHA256 | `b51b67daf66a5a2d9ca8e49c4bab2aec32ec900302eaf13bfbd9ab6765e70e9f` |
| Original model archive | `/home/gc6_challenge/models/native_streaming/wenet/wenetspeech/wenetspeech_u2pp_conformer_libtorch.tar.gz` |
| Original archive SHA256 | `cd82435483dfbcf8dc7385e10c72051be5488893f010ce7866b94600a1ccbca9` |
| Original archive size | 481,976,187 bytes |
| Provenance distinction | `cd824...` is the original archive SHA recorded by qualification evidence; `b51b...` is the current extracted `final.zip` SHA. They are not interchangeable. |
| WeNet source path | `/home/gc6_challenge/code/native_backend_qualification/wenet` |
| WeNet revision | `d17059667d6afe0680d19b3a4948ab825ef25105` |
| Native library | `/home/gc6_challenge/code/native_backend_qualification/wenet/runtime/libtorch/build2/api/libwenet_api.so` |
| Native library SHA256 | `69fb822d236f5b9ac8b2268c5211dfee1187513a45aa6375c7bff10ff2718926` |
| LibTorch path | `/home/gc6_challenge/models/native_streaming/wenet/libtorch` |
| LibTorch runtime | official CPU archive `libtorch-shared-with-deps-2.2.0+cpu.zip` |
| LibTorch archive SHA256 | `e1f6bc48403022ff4680c7299cc8b160df146892c414b8a6b6f7d5aff65bcbce` |
| Language | `chs` |
| MISP in-domain fine-tuning | NO evidence; treated as NO |
| Inference API | `wenet_init`, `wenet_decode`, `wenet_reset` through official C API |

The `train.yaml` in the model directory declares a Conformer encoder with `causal: true`, `use_dynamic_chunk: true`, and 16 kHz resampling configuration. No MISP in-domain fine-tuning artifact or training run is part of this benchmark.

### 3.1 Decoder and chunk policy

- Source packet: 320 samples = 20 ms at 16 kHz.
- Model input chunk: 320 samples per adapter feed.
- Official WeNet `chunk_size`: 16 post-subsampling frames.
- Decoder state is persistent across packets within a scope; it is not rebuilt per packet and the complete prefix is not replayed.
- Decoder reset occurs after scope FINAL; the session pool is reused for later non-overlapping scopes.
- Beam/rescoring parameters are not separately exposed in the closure artifact; they remain those encoded by the official model/runtime.
- Numeric independent right-context/lookahead is not exposed by the native artifact. The model is causal/dynamic-chunk, but this report does not infer a zero-sample algorithmic lookahead from that fact.

## 4. Streaming Semantics

| Capability | Result |
|---|---|
| `native_streaming` | YES |
| `stateful` | YES |
| `supports_partial` | YES |
| `supports_revision` | YES; 724 visible changed-text/revision emissions in this run |
| `supports_stable_prefix` | NO |
| `supports_endpoint` | NO |
| `supports_token_timestamps` | NO |
| `supports_word_timestamps` | NO |
| `reports_consumed_position` | NO |
| FINAL authority | Stage-2 oracle scope-end release; the model does not claim native endpointing |

Persistent native state includes the WeNet `FeaturePipeline`, attention cache, CNN cache, and decoder state. The source packet size, model chunk size, and frontend lookahead are separate concepts; in this run source packet and model chunk are both 320 samples, while `chunk_size=16` is expressed in post-subsampling frames.

`supports_endpoint=false` is not a recognition failure. It means the current adapter does not infer a model endpoint from partial output; FINAL is emitted when the accepted Stage-2 scope-end gate releases the scope. This is why this benchmark must not be advertised as fully-online endpointing.

## 5. PCM / Input Conversion and Parity

The frozen conversion policy is:

```text
signed_pcm32le_arithmetic_right_shift_16_to_pcm16le_with_saturation
```

The existing isolation evidence in `/home/gc6_challenge/work/realtime_replay/r5b_wq/r5b_wq_isolation.json` records:

- continuous-source PCM32 payload and independently sliced segment PCM32 payload hash equality;
- converted PCM16 hash equality;
- exact equality to the `soundfile/libsndfile` `dtype=int16` reference;
- parity across all 144 M028 scopes;
- no resampling, normalization, downmix, denoising, or enhancement.

The same adapter lifecycle and scoring protocol were used for the AISHELL quality-isolation history and the WenetSpeech closure. This evidence makes input slicing, PCM conversion, and adapter lifecycle unlikely explanations for the AISHELL CER anomaly.

## 6. AISHELL Quality-Isolation History

The AISHELL diagnostic history is retained because it separates model/domain quality from streaming geometry:

| Condition | CER |
|---|---:|
| AISHELL U2++ native streaming | 0.8363256785 |
| AISHELL official full-context | 0.8229645094 |
| Absolute full-context improvement | 0.0133611691 |
| WenetSpeech U2++ native streaming closure | 0.33903966597 |

Full-context only improves the AISHELL result by about 0.013 absolute. The evidence therefore supports the interpretation that the large AISHELL-vs-MISP discrepancy was primarily checkpoint/domain suitability, not streaming geometry or a replay harness bug. With the same adapter, PCM policy, and scoring protocol, WenetSpeech gives a materially lower MISP CER.

This is an interpretation, not a claim that full-context and native streaming are identical. The WenetSpeech full-context quality diagnostic was `0.2784968685`, while the formal native-streaming closure is `0.33903966597`; the difference is retained as model/streaming evidence, not hidden.

## 7. Frontend Server Audit

The complete inventory is stored at `/home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation/frontend_inventory.md` and `/home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation/dataset_inventory.json`. The audit searched repository/file names first and then inspected relevant source/configuration without reading large dataset or model binaries.

| Name | Server path | Git/source | Input | Stateful? | Bounded lookahead? | Future metadata / oracle | Output | Classification | Directly reusable? | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| Standard GSS (WPE + CACGMM + Souden MVDR/BAN) | `/home/gc6_challenge/code/gss` | local MISP offline GSS implementation | 8ch, separate mono WAV, 16 kHz | session-level, not causal incremental | NO | full-session activity/segments/RTTM guidance | mono enhanced WAV/FLAC | `OFFLINE_ONLY` | NO | global clustering/filter estimation and no incremental state API |
| OnlineFIVE | `/home/gc6_challenge/code/gss/gss/online_five` | local GSS/FIVE implementation | 8ch STFT stream | YES | block-online only; FFT 1024, hop 256, 32 update frames, about 512 ms plus implementation buffering (~560 ms inventory estimate) | no explicit future RTTM, but target/source ambiguity remains | separated unit stream | `BLOCK_ONLINE_OR_PSEUDO_STREAMING` | NO | block adaptation and one-unit blind-source ambiguity; no generic target selection contract |
| FIVE/IVE 2020 batch | `/home/gc6_challenge/code/five_2020ICASSP` | local batch reference | finite multichannel tensor | NO incremental API | NO | no explicit future metadata, but full finite input | batch separated output | `OFFLINE_ONLY` | NO | no persistent chunk state |
| NBSS OnlineSpatialNet | `/home/gc6_challenge/gc6_model/NBSS` | local NBSS implementation | intended multichannel spatial features | causal internals exist | not exposed at top-level | local MISP config/runner references GSS-derived features | model-dependent enhanced stream | `BLOCK_ONLINE_OR_PSEUDO_STREAMING` | NO | no directly callable persistent cross-call state contract, checkpoint/feature/target adapter missing |
| AV-SkiM target-speaker extraction | `/home/gc6_challenge/gc6_model/ClearerVoice-Studio/train/target_speaker_extraction_online` | local target-speaker extraction code | target-speaker/visual-conditioned input | causal state API exists | bounded state API exists | visual target identity and target-speaker oracle dependency | target speaker stream | `ORACLE_DIAGNOSTIC_ONLY` | NO | not generic MISP 8ch frontend; unsupported checkpoint/target contract |

### 7.1 Audit decision

`PRIMARY_STREAMING_FRONTEND = NO_DIRECTLY_REUSABLE_TRUE_STREAMING_FRONTEND`.

No candidate simultaneously demonstrated all of the following on this server: continuous/chunk input, persistent state across calls, explicitly bounded lookahead, no future Stage-2 segments/RTTM/reference/speaker identity, sample-aligned output timeline, generic 8-channel MISP compatibility, and a replay-harness-callable API. No new frontend was implemented to force a Condition B result.

### 7.2 Why Standard GSS is not streaming

The existing Standard GSS path is a full-session/segment enhancement pipeline. Its activity masks and speaker/segment guidance come from Stage-2 or related diarization artifacts; CACGMM clustering and spatial statistics use global recording/session context; WPE and beamforming filters are estimated over a finite utterance/session rather than maintained through a documented causal incremental state. The local pipeline exposes offline enhancement artifacts and no official/local API that accepts a chunk, updates all required state, and returns a strictly sample-aligned causal output. Cutting its input into blocks would therefore be pseudo-streaming, not evidence of a true streaming frontend.

### 7.3 Why OnlineFIVE is not the primary baseline

OnlineFIVE is the closest candidate because it has state and accepts multichannel STFT updates. However, the inspected implementation uses an adaptation block of 32 frames with FFT 1024 and hop 256; this is approximately 512 ms of frame updates before accounting for window/buffering behavior. It also has a one-unit blind-source ambiguity and does not provide a generic multi-speaker target-selection contract for current MISP recordings. The target may require steering/selection semantics that are not equivalent to a generic ASR frontend. It is therefore `BLOCK_ONLINE_OR_PSEUDO_STREAMING`, not `TRUE_STREAMING_REUSABLE`; it was not inserted into the main ASR comparison.

## 8. Experimental Conditions

The frozen protocol defines three reusable conditions:

| Condition | Definition | This report |
|---|---|---|
| A RAW | continuous far-field 8ch → select `raw_ch0` → WeNet U2++ native streaming ASR | **EXECUTED** |
| B TRUE STREAMING FRONTEND | continuous 8ch → qualified causal frontend → sample-aligned mono stream → same ASR adapter | **NOT RUN**; no qualified candidate |
| C OFFLINE FRONTEND UPPER BOUND | offline GSS → ASR | **NOT RUN BY USER INSTRUCTION** |

All executed A settings share the accepted baseline checkpoint, LibTorch runtime, adapter, scorer, Stage-2 scope policy, source timeline, model chunk policy, and PARTIAL emission policy. The frontend condition is the only intended condition variable, but B was unavailable and C was explicitly skipped; no CER gain/loss between A and a frontend is claimed.

## 9. Accuracy Results

### 9.1 Global A result

| Metric | A: raw_ch0 + WenetSpeech U2++ |
|---|---:|
| Scopes | 144 |
| Reference characters N | 2395 |
| Substitutions S | 411 |
| Deletions D | 352 |
| Insertions I | 49 |
| Total errors | 812 |
| CER | 0.33903966597077245 |
| Empty outputs | 5 |
| Non-empty outputs | 139 |
| Mean per-scope CER | 0.44821758227881897 |

The per-scope CER distribution from `scored.jsonl` is min `0.0`, median `0.3693957115`, p75 `0.6428571429`, p90 `1.0`, p95 `1.0`, max `1.0`. These are scope-level diagnostics, not a meeting-level bootstrap claim.

### 9.2 Meeting-level result

The complete single meeting row is in `/home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation/meeting_level_results.csv`:

| meeting_id | scopes | audio duration (s) | reference chars | CER | S/D/I | empty | status |
|---|---:|---:|---:|---:|---|---:|---|
| M028 | 144 | 798.299 | 2395 | 0.33903966597 | 411/352/49 | 5 | PASS |

Because there is only one meeting, a multi-meeting macro CER and paired meeting-level improvement count are not reported. Condition B has no result, so no meeting improved/worsened comparison exists. Condition C was not run.

## 10. Runtime and Latency Metrics

### 10.1 Closure run

| Metric | Value |
|---|---:|
| `RTF_consume` / ASR RTF | 0.1787101070 |
| `frontend_RTF` | 0.0 |
| `RTF_total_compute` | 0.1792607488 |
| `C_consume` | 142.6640997356 s |
| `C_finish` | 0.4395767939 s |
| Completion overrun | -0.0656324590 s |
| TTFT mean / P50 / P95 / max | 1.103375 / 0.819962 / 2.098514 / 2.742141 s |
| Segment-end-to-FINAL mean / P50 / P95 / max | 0.206600 / 0.183999 / 0.345144 / 0.438135 s |
| Backlog P50 / P95 / max | 0.020 / 0.080 / 0.640 s |
| PARTIAL returns | 30410 native returns; 724 visible changed-text emissions |
| REVISION count | 724 |
| FINAL count | 144 |
| Flicker | mean 2.671 chars; P50 3; P95 5; max 6 |
| Native decoder calls | 30554 |
| Native decoder sessions | 2; max active sessions 2 |
| Model load | 23.9919621170 s |
| Peak RSS | 1,377,673,216 bytes |
| Device | CPU |

In this closure run, total compute RTF is below 1 and completion finishes 65.6 ms before source end. This supports sustained realtime feasibility for this run on this host configuration only; it is not a hardware-independent guarantee.

### 10.2 RTF reconciliation with accepted prior run

The accepted prior M028 WenetSpeech run at `/home/gc6_challenge/work/realtime_replay/r5b_wenet_full_wenetspeech_m028_v2/` reported:

| Metric | Prior accepted run | Closure raw rerun |
|---|---:|---:|
| `RTF_consume` | 0.6049516408 | 0.1787101070 |
| `RTF_total_compute` | 0.6053801243 | 0.1792607488 |
| `C_consume` | 482.9322898597 s | 142.6640997356 s |
| `C_finish` | 0.3420579879 s | 0.4395767939 s |
| Model load | 4.9331670529 s | 23.9919621170 s |
| P95 backlog | 12.800 s | 0.080 s |
| Max backlog | 53.760 s | 0.640 s |
| Completion overrun | +0.5185162129 s | -0.0656324590 s |
| Source duration | 798.299 s | 798.299 s |
| Native decoder calls | 30554 | 30554 |

Accuracy, source duration, native call structure, and adapter semantics are consistent, but the artifacts do not establish identical CPU thread settings, server load, runtime launch, affinity, or warm-up behavior. The prior report records model archive SHA `ace400d74a0eb073b0ee284e0c21be18796a7ab095a8b013d0e703e0180b1968`, while current closure provenance records the WenetSpeech original archive SHA `cd82435483dfbcf8dc7385e10c72051be5488893f010ce7866b94600a1ccbca9` and current extracted `final.zip` SHA `b51b67...`; this is another provenance difference that must not be silently ignored.

Therefore the reconciliation conclusion is:

```text
RTF_CROSS_RUN_NOT_DIRECTLY_COMPARABLE
```

This report uses `0.1792607488` as the closure benchmark measurement and retains `0.6053801243` as accepted prior evidence. Neither number replaces the other, and the difference is not presented as a measured speedup.

## 11. Hardware / Runtime Environment

The following provenance was read from the current server and closure artifacts:

| Field | Value |
|---|---|
| Hostname | `ubuntu` |
| OS/kernel | Ubuntu 22.04 environment; Linux `6.2.0-26-generic` x86_64 |
| CPU | Intel Xeon Platinum 8476C |
| CPU topology | 2 sockets × 48 cores/socket × 2 threads/core; 192 logical CPUs |
| RAM | 503 GiB installed according to current host audit |
| GPU | Not used; current benchmark device is CPU |
| Python in audit shell | 3.13.9 |
| Python Torch | unavailable in the audit shell; inference uses native C++ LibTorch, not Python Torch |
| LibTorch | official `2.2.0+cpu` runtime archive |
| `OMP_NUM_THREADS` | unset in closure audit shell |
| `MKL_NUM_THREADS` | unset in closure audit shell |
| `OPENBLAS_NUM_THREADS` | unset in closure audit shell |
| CPU affinity | `0-191` in closure audit shell |
| Observed host load | approximately 31 during provenance audit; exact load at every decode interval was not persisted |
| Warm-up | 16000 samples; output ignored and decoder reset before replay; model load excluded from streaming metric window |

Because thread and concurrent-load provenance was incomplete for the two historical runs, RTF comparison is explicitly not direct.

## 12. Causality & Leakage

| Check | A raw result | B | C |
|---|---|---|---|
| Future audio leakage | NO beyond paced causal source delivery | N/A | N/A |
| Future metadata leakage into decoder | NO | N/A | N/A |
| Reference text leakage | NO | N/A | N/A |
| Speaker label leakage | NO oracle speaker identity | N/A | N/A |
| Oracle segment boundary | YES; Stage-2 START/END | N/A | would be offline guidance; not run |
| Oracle speaker identity | NO | N/A | N/A |
| Future RTTM dependency | NO in A decoder | N/A | not run |
| Future perturbation test | N/A; no B frontend | N/A | N/A |

The no-leakage statement for A is specifically about decoder input and frontend processing. It does not remove the declared oracle scope-boundary limitation.

## 13. Tests & Regression

Existing validation evidence for the closure branch is:

| Check | Result |
|---|---|
| Focused tests | 29 passed |
| Full pytest | 204 passed, 1 failed |
| Failing test | `tests/test_baseline_integration.py::test_maintained_cli_dry_run_does_not_load_backend` |
| Failure cause | `ffprobe` subprocess exited `-2` after OpenBLAS pthread creation failure under the inherited host/thread environment |
| Failure classification | `KNOWN_TEST_ENVIRONMENT_FAILURE` |
| Introduced by benchmark/report files? | Evidence indicates no; it is an environment-type CLI subprocess failure and does not execute the benchmark path |
| `compileall` | PASS |
| New closure script `ruff` | PASS |
| New closure script `pyflakes` | PASS |
| Full-repository `ruff`/`pyflakes` | Existing unrelated unused-import violations remain |
| `git diff --check` | PASS |

The report does not claim “all tests passed”; the single environment failure remains visible for external review.

## 14. Reproducibility

### 14.1 Repository state

- Repository: `lyxin651/lyx_Robot-HEARD`
- Worktree: `/home/gc6_challenge/code/wenet-streaming-frontend-ablation`
- Branch: `feat/wenet-streaming-frontend-ablation`
- Starting closure base: `0e9676da6635232eaf37e46ec543d2ff71281b9c`
- Current pre-report HEAD: `1f3103779124f5b3c09131f977030207a0f37fa2`
- Accepted PR #8 worktree: `/home/gc6_challenge/code/wenet-u2pp-streaming-adapter`
- Accepted PR #8 HEAD: `0e9676da6635232eaf37e46ec543d2ff71281b9c`
- Accepted PR #8 implementation was not modified.
- Closure worktree was clean before this report was added.

The online latest contents of GitHub Issue #4, Issue #7, and Draft PR #8 could not be retrieved in this environment because the `gh` CLI is unavailable and GitHub API/network attempts timed out. The local accepted PR #8 ref and repository evidence are recorded above; this limitation must not be mistaken for a verified live GitHub read.

### 14.2 Main entrypoint and config

- Main script: `/home/gc6_challenge/code/wenet-streaming-frontend-ablation/scripts/run_wenet_frontend_ablation.py`
- Frozen contract: `/home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation/experiment_contract.json`
- Reproducible command form (the exact shell transcript was not persisted; defaults correspond to the frozen contract):

```bash
PYTHONPATH=/home/gc6_challenge/code/wenet-streaming-frontend-ablation/src \
python3 /home/gc6_challenge/code/wenet-streaming-frontend-ablation/scripts/run_wenet_frontend_ablation.py \
  --data-dir /home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far \
  --source-dir /home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename \
  --recording-id M028_S197199201241_F8N_Far \
  --output-root /home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation
```

- Raw report: `/home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation/raw_baseline_report.json`
- Raw paced metrics: `/home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation/raw_baseline/M028_S197199201241_F8N_Far/paced_metrics.json`
- Raw native metrics: `/home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation/raw_baseline/M028_S197199201241_F8N_Far/r5b_metrics.json`
- Raw trace provenance: `/home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation/raw_baseline/M028_S197199201241_F8N_Far/trace.run.json`
- Meeting results: `/home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation/meeting_level_results.csv`
- Frontend inventory: `/home/gc6_challenge/work/realtime_replay/wenet_frontend_ablation/frontend_inventory.md`

## 15. Limitations

1. The executed benchmark is one meeting only: M028, 144 scopes, 798.299 seconds, 2395 reference characters.
2. The planned multi-meeting expanded evaluation was not completed; the full six-recording inventory is background evidence only.
3. Stage-2 START/END boundaries are oracle; fully-online segmentation and endpointing were not evaluated.
4. The raw condition uses only `raw_ch0`; no multichannel enhancement benefit is measured.
5. There is no MISP in-domain fine-tuning evidence.
6. No qualified reusable true causal streaming frontend was found, so no raw-vs-streaming-frontend ablation exists.
7. Offline GSS was not executed by explicit user instruction; even if existing artifacts are available, they cannot enter online latency/RTF conclusions.
8. RTF is sensitive to CPU model, thread variables, affinity, concurrent server load, runtime launch and warm-up; closure and prior RTF are not directly comparable.
9. Online diarization was not evaluated.
10. This is not a complete RoboHEARD online or multimodal integration result.
11. GitHub live Issue/PR content and mutation status were not verifiable from this environment; use `GITHUB_MUTATION_PENDING_CHATGPT` for external GitHub mutation follow-up.

## 16. Fact / Interpretation / Limitation / Next Use

### FACT

- M028 raw_ch0 + WenetSpeech U2++ produced CER `0.33903966597077245` with S/D/I/N `411/352/49/2395`.
- Closure `RTF_total_compute` is `0.1792607488290538`; latency and backlog values are recorded in Section 10.
- The executed coverage is one meeting, 144 scopes, 798.299 seconds and 2395 reference characters.
- The server audit found no `TRUE_STREAMING_REUSABLE` frontend.
- Condition B was not run and Condition C was not run.

### INTERPRETATION

- WenetSpeech is a materially better MISP-suitable checkpoint than the AISHELL U2++ checkpoint under the same adapter/scoring history.
- The closure run is computationally realtime on the observed CPU configuration, but this is not a cross-run speedup claim.
- Standard GSS and OnlineFIVE cannot be promoted as strict generic causal frontend baselines under the current contract.

### LIMITATION

- Oracle segmentation, raw single-channel input, M028-only coverage, no MISP fine-tuning, no qualified frontend, no offline upper-bound run, and incomplete cross-run timing provenance remain explicit limitations.

### NEXT USE

- Use this report and `model_benchmark_summary.json` as the frozen raw-input baseline schema for later Whisper, FastConformer or other streaming ASR comparisons.
- Reuse the same M028 source, Stage-2 scope policy, scorer, PCM conversion, model-chunk policy, and latency definitions unless a future experiment explicitly registers a changed condition.
- Do not rank a future model against this result unless dataset coverage, oracle boundary policy, frontend condition and metric definitions are comparable.

## 17. Cross-Model Comparison Table

The table is intentionally conservative. Rows with incomplete evidence are marked `NOT DIRECTLY COMPARABLE` rather than used to create a ranking.

| Model | Checkpoint | Dataset / meetings / duration / scopes / refs | Input / frontend / fine-tune | Native / stateful / oracle boundary / oracle speaker | CER | S/D/I | RTF consume / total | TTFT / FINAL / backlog | Lookahead | CPU/GPU / memory | Qualification status |
|---|---|---|---|---|---:|---|---|---|---|---|---|
| WenetSpeech U2++ | `/home/gc6_challenge/models/native_streaming/wenet/wenetspeech/20220506_u2pp_conformer_libtorch/final.zip` | MISP 2025 `dev_far`; M028; 798.299 s; 144; 2395 | raw_ch0; NONE; NO MISP fine-tune | YES / YES / YES Stage-2 boundary / NO | 0.33903966597 | 411/352/49 | 0.178710 / 0.179261 | TTFT P50 0.819962 s; FINAL P50 0.183999 s; backlog P95 0.080 s | not independently exposed | Xeon 8476C CPU; peak RSS 1.378 GB | `PASS` for native streaming raw baseline; M028-only |
| Offline Whisper | evidence under `/home/gc6_challenge/work/realtime_replay/r4_m028_rawch0_offline_whisper_v2/r4_report.json` | M028; 144 scopes; same broad source history | raw_ch0; offline/segment-end gated; fine-tune not confirmed | NO native streaming claim / N/A / oracle segment gate / NO | 0.39749478079 | S/D/I not persisted in report | N/A | N/A | offline | runtime/memory not directly comparable | `NOT DIRECTLY COMPARABLE` |
| Nemotron ASR | not run in this task | — | — | — | — | — | — | — | — | — | `NOT DIRECTLY COMPARABLE` |
| FastConformer or future model | fill only after same protocol | M028 or a newly registered common split | register input/frontend/fine-tune | register all oracle/state fields | — | — | — | — | — | — | pending |

No model ranking is claimed from this table. In particular, the Whisper row is historical context, not an online latency comparison, and Nemotron is intentionally outside this task.

## 给竞赛成员的简短结论

现在 WeNet 已经完成了一个可复用的 native-streaming Chinese ASR baseline：在 MISP 2025 `dev_far` 的 M028 连续远场录音上，用 raw_ch0、Stage-2 oracle scope boundary 和官方 WenetSpeech U2++ LibTorch C API，得到 CER `0.33904`，closure total compute RTF `0.17926`，并保留了 TTFT、FINAL、backlog、partial/revision 和 memory evidence。当前数据实际只有 1 个 meeting、144 个 scopes、798.299 秒和 2395 个 reference characters；这不是 full dev_far，也不是 multi-meeting expanded evaluation。

服务器上没有找到可以直接接入当前 harness 的严格 causal 8ch streaming frontend，所以没有 raw-vs-streaming frontend 消融；GSS 是 offline，OnlineFIVE 是 block-online/pseudo-streaming，其他候选分别有 state/target/oracle/API 问题。下一位成员比较 Whisper、FastConformer 或其他 streaming ASR 时，应直接复用 `/home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename`、`/home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far`、本报告定义的 raw_ch0/PCM/scorer/Stage-2 protocol，以及 `/home/gc6_challenge/code/wenet-streaming-frontend-ablation/scripts/run_wenet_frontend_ablation.py`；必须先保证数据覆盖、oracle boundary 和 timing definition 可比。

## Delivery Status

`WENET_CLOSURE_REPORT_READY_FOR_EXTERNAL_REVIEW`

GitHub mutation/push was not confirmed in this environment. If external GitHub creation or update is required, return:

```text
GITHUB_MUTATION_PENDING_CHATGPT
```
