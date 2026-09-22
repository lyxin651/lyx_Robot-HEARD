# S6I M028 Dev Validation Status and Far/Near Diagnostic

Date: 2026-09-22  
Authority: Issue #1 `Whisper Backend V0 — Execution Charter`  
Branch: `feat/whisper-backend-v0`  
Gate at start of this update: **S6I MISP Integration Compatibility**  
Last fully closed Gate: **S6 SERVER_VALIDATED**

## 1. Gate decision

**S6I remains OPEN. It must not be marked PASS or MISP-ready yet.**

The real M028 Dev integration has established the baseline input contract, full-session adapter execution, full-session Whisper S4 execution, and S5 reconciliation. However, the Charter requires a real Dev session to complete through S6 with official-format/export completeness validation and scorer parity when the baseline scorer is available. Those requirements are not yet satisfied.

No core backend, decoder, scoring, adapter, exporter, or V0 config is changed by this documentation update.

## 2. Real M028 Dev evidence established

Session: `M028_S197199201241_F8N_Far`

Authoritative Dev Stage-2 artifacts:
- `segments=144`
- `text=144`
- `utt2spk=144`
- `spk2utt=4`
- `wav.scp=1`
- `channels.scp=1`
- exact `segments/text` ID-set reconciliation: PASS
- 8/8 physical far-field channels exist, mono 16 kHz, 32-bit PCM

Existing S6I raw-channel integration:
- adapter: 144/144 emitted with explicit `raw_ch0`
- Whisper S4: 144/144 execution success, 0 runtime failures
- S5: exact ID reconciliation PASS
- local diagnostic global CER: `0.3974947807933194`
- persistent rerun reproduced the same 33 empty hypotheses
- S6: FAIL-CLOSED on real Dev output because the exporter rejects empty submission text

The S5 CER above is a local diagnostic metric and is not labeled official MISP CER.

## 3. Empty-hypothesis diagnosis

Observed on the complete M028 raw_ch0 run:
- total: 144
- empty `text_raw`: 33/144 (22.92%)
- non-empty: 111/144
- empty median duration: 1.16 s
- non-empty median duration: 4.68 s
- `<1.5 s`: 20/31 empty (64.52%)
- `>=5 s`: 0/52 empty

Representative direct reruns of five empty samples reproduced `segments=[]` and `text=''` under the frozen OpenAI Whisper V0 config. This confirms the immediate output path for those samples is Whisper's no-speech skip behavior. The returned API result does not preserve the skipped window's `no_speech_prob` / `avg_logprob`, so the numerical margin for all 33 empties remains unavailable without changing the diagnostic method.

Short duration is therefore a strong statistical correlate, but subsequent acoustic checks show it is not sufficient as a root-cause explanation.

## 4. Manual far-field inspection

Manual listening of representative empty samples showed mixed failure modes:

- some official speaker-specific segments contain only a very short target utterance while another speaker overlaps or continues nearby;
- some far-field clips are strongly noise/overlap dominated and the target content is difficult to understand;
- some clips are human-intelligible and still return an empty Whisper hypothesis;
- sampled ch0--ch7 clips sounded very similar, so simple raw-channel selection does not explain these sampled failures.

The MISP Stage-2 `segments/text` are speaker-specific reference artifacts, while a single raw far-field microphone remains a meeting mixture. Therefore a target speaker's reference can be correct even when the raw single-channel segment is dominated by overlap, reverberation, or another speaker.

This does not establish that the Dev set is corrupt, nor does it establish a global time-alignment failure.

## 5. Far vs near-field diagnostic

The server contains speaker-wise M028 F8N recordings:

- `M028-F8N-197.wav`
- `M028-F8N-199.wav`
- `M028-F8N-201.wav`
- `M028-F8N-241.wav`

Seven representative M028 segments were selected from the raw_ch0 empty set. For each segment, the same frozen Whisper V0 backend/config was run on:
1. the existing far `raw_ch0` segment actually used by the S6I run;
2. the corresponding speaker F8N near-field segment using the same reference interval.

Observed result:
- far raw_ch0: **7/7 empty**
- F8N near-field: **0/7 empty**

Examples include:
- reference `好`: far empty; near-field Whisper output `好` (local CER 0)
- reference `创业的话`: far empty; near-field Whisper output `创业的话` (local CER 0)
- other near-field samples were non-empty but still contained substitutions/deletions

Interpretation: on these selected failure samples, Whisper can emit text even for very short segments when the target speaker is acoustically prominent. The evidence therefore supports a substantial far-field mixture / overlap / target-speaker-acoustics contribution to the empty-hypothesis failure.

Limitations:
- these 7 samples were deliberately selected from raw-far empty cases and must not be used to estimate full-Dev improvement;
- F8N is a diagnostic/oracle condition here, not evidence that the far-field Task-2 integration is complete;
- the result does not yet quantify how much GSS or another frontend will recover on all 144 segments.

## 6. Why S6I cannot close yet

Against the Charter's S6I requirements:

1. **Real contract test** — PASS.
2. **One real Dev session through baseline segmentation -> target waveform -> Whisper -> S5 -> S6** — NOT PASS. The real M028 run reaches S5, but S6 fails on empty/non-Han outputs.
3. **Artifact reconciliation** — PASS through S5; final export reconciliation is incomplete because no complete 144-row S6 artifact exists.
4. **Official submission contract validation** — NOT PASS for the real full session.
5. **Scorer parity when available** — NOT PASS. The MISP AVSR baseline `compute-wer.py --char=1` is available and parity has not yet been executed on the same complete M028 hypothesis set.
6. **AVDR Task 3 speaker assignment** — out of scope for the current Task-2 S6I path; the ASR backend continues not to guess speakers.

Therefore the allowed state remains **backend-ready**, not **MISP-ready**, and S7 must not start yet.

## 7. Current exporter boundary

The current S6 exporter intentionally:
- preserves exact segment IDs;
- requires every exported transcript to remain non-empty after Task-2 normalization;
- rejects unsupported non-Han semantic content instead of silently deleting it;
- validates exact expected-ID completeness.

The full M028 raw result contains:
- 33 empty Whisper hypotheses;
- at least one non-Han/digit case previously observed in the real Dev run.

These are real integration boundaries. Do not manually fill missing text, drop segment IDs, silently delete digits, disable fail-closed behavior, or relax the exporter only to make the Gate pass.

## 8. Immediate S6I work, before S7

The next work remains inside S6I:

1. Run the available MISP baseline scorer on the complete M028 reference/hypothesis ID set and record exact ID reconciliation plus baseline `N/S/D/I/CER`. Empty hypotheses must be represented in the scorer input in the baseline-supported form without changing the ASR result.
2. Verify the authoritative Task-2 submission rule for empty hypotheses and non-Han/digit output. The baseline scorer's ability to score an empty hypothesis is not by itself proof that the challenge submission parser accepts an ID-only row.
3. If the official submission contract permits a representation that the exporter does not yet support, change the exporter only with explicit tests and server validation. If the official contract requires non-empty Chinese text, keep the exporter strict and resolve the upstream target-waveform/decode integration without weakening the submission contract.
4. Re-run one complete M028 path through S6 and exact 144-ID reconciliation. Only after S6 succeeds and scorer parity/limitations are recorded may S6I be marked PASS.

A one-session use of an existing baseline frontend may be considered only as an S6I target-waveform integration remedy if required to satisfy the real submission contract; this must not be mislabeled as the S7 raw/GSS/neural benchmark. S7 remains blocked until S6I closes.

## 9. State boundary

Current declaration:

- S0--S6: **SERVER_VALIDATED**
- S6I real-input contract / adapter / M028 S4-S5: **SERVER_VALIDATED**
- S6I full-session export: **BLOCKED**
- S6I baseline scorer parity: **PENDING**
- S6I overall: **OPEN**
- MISP-ready: **NO**
- S7: **NOT STARTED**
- OpenAI Whisper V0 Frozen: **NO**
