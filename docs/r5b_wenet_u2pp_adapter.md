# R5B-W WeNet U2++ native streaming adapter

This adapter is the ASR-only R5B-W condition. It consumes the frozen replay
runtime's causal Stage-2 boundary releases; it does not own the Stage-2
timeline, references, speakers, future segments, or replay scheduling.

## Qualified native runtime

- WeNet revision: d17059667d6afe0680d19b3a4948ab825ef25105
- Model: official AISHELL U2++ Conformer LibTorch
- Model archive SHA256:
  ace400d74a0eb073b0ee284e0c21be18796a7ab095a8b013d0e703e0180b1968
- Official LibTorch CPU runtime SHA256:
  e1f6bc48403022ff4680c7299cc8b160df146892c414b8a6b6f7d5aff65bcbce
- Native API: wenet_init, wenet_decode, wenet_reset through the qualified
  libwenet_api.so; no file-level transcription is used.

## Input and state

The replay source remains 8-channel, 16 kHz, PCM32LE with 320-sample
20-ms packets. The adapter explicitly selects raw_ch0 and converts each
signed PCM32 sample to PCM16LE with arithmetic right shift 16 and saturation.
It feeds fixed 320-sample model chunks incrementally; the official WeNet
chunk_size=16 is the model's post-subsampling frame setting, not the replay
packet size.

One native decoder session preserves the FeaturePipeline, attention cache, CNN
cache, and decoder state for each active scope. It is reset after FINAL and
pooled for later non-overlapping scopes. A second session is created only
when the audited Stage-2 timeline has a true concurrent overlap; model state is
never rebuilt per source packet or by replaying a complete prefix.

The official API does not expose a strict incorporated-sample position, so
reports_consumed_position=false and consumed_through_sample=null.
Endpoint capability is also false: the adapter does not infer an endpoint from
partial results; FINAL is emitted only after the runtime releases the causal
scope end.

## Emission policy

Every complete model chunk is fed to the native decoder. Raw decoder-call,
partial-return, final-return, empty-partial, changed-text, and repeated-text
counts are retained as audit evidence. Visible PARTIAL events suppress initial
empty results and repeated visible text, emitting only the first non-empty or
changed text. FINAL is emitted exactly once and supersedes the preceding
hypothesis. Revision indices start at zero and form an immediate replacement
chain.

The full M028 result is written by
scripts/run_r5b_m028_wenet.py; references are opened only after replay
completion. The official sample and 30-second M028 native smoke evidence is
produced by scripts/smoke_wenet_native.py.
