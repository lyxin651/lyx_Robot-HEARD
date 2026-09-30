"""R5A oracle-segmented rolling-prefix Whisper wrapper.

This is a model-agnostic ``StreamingConsumer`` adapter around the existing
file-based ``ASRBackend``.  It is intentionally an oracle-segmented,
segment-level pseudo-online reference: scope boundaries are released by the
runtime, audio is received only through delivered chunks, and each snapshot
is a fresh complete-file decode of the arrived raw_ch0 prefix.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from robot_heard.asr.base import ASRBackend, ASRResult
from robot_heard.replay.events import (
    AudioChunk,
    ConsumerCapabilities,
    ContractValidationError,
    HypothesisEvent,
    HypothesisKind,
    OutputEvent,
    StreamMetadata,
    TraceEvent,
)
from robot_heard.replay.oracle import ScopeEndRelease, ScopeStartRelease
from robot_heard.streaming.base import StreamingConsumer


class RollingWhisperError(ContractValidationError):
    """Raised when the R5A adapter violates a causal or lifecycle contract."""


_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class _ScopeState:
    start: ScopeStartRelease
    end: Optional[ScopeEndRelease] = None
    last_decode_end_sample: int = 0
    revision_index: int = -1
    last_event_id: Optional[str] = None
    closed: bool = False


class RollingWhisperConsumer(StreamingConsumer):
    """Wrap a frozen file-based Whisper backend with rolling prefix snapshots.

    The constructor accepts no oracle gate and no source path.  The runtime
    forwards only individual causal ``ScopeStartRelease`` and
    ``ScopeEndRelease`` records.  Therefore the adapter cannot enumerate a
    future Stage-2 timeline or read future source samples directly.
    """

    def __init__(
        self,
        backend: ASRBackend,
        *,
        materialized_dir: Path,
        spool_path: Path,
        snapshot_evidence_path: Path,
        update_interval_samples: int = 16000,
        selected_channel_id: str = "ch0",
        frontend_condition: str = "raw_ch0",
        gpu_sync_policy: str = "none",
        wall_clock: Optional[Any] = None,
    ) -> None:
        if not isinstance(backend, ASRBackend):
            raise RollingWhisperError("backend must implement ASRBackend")
        if isinstance(update_interval_samples, bool) or not isinstance(update_interval_samples, int):
            raise RollingWhisperError("update_interval_samples must be a positive integer")
        if update_interval_samples <= 0:
            raise RollingWhisperError("update_interval_samples must be a positive integer")
        if not isinstance(selected_channel_id, str) or not selected_channel_id.strip():
            raise RollingWhisperError("selected_channel_id must be non-empty")
        if gpu_sync_policy not in (
            "none",
            "cuda_synchronize_before_and_after_backend_transcribe",
        ):
            raise RollingWhisperError("unsupported gpu_sync_policy")
        self.backend = backend
        self.materialized_dir = Path(materialized_dir).expanduser().resolve(strict=False)
        self.spool_path = Path(spool_path).expanduser().resolve(strict=False)
        self.snapshot_evidence_path = Path(snapshot_evidence_path).expanduser().resolve(strict=False)
        if self.spool_path.exists() or self.snapshot_evidence_path.exists():
            raise FileExistsError("R5A adapter artifacts must be fresh; overwrite is refused")
        self.update_interval_samples = update_interval_samples
        self.selected_channel_id = selected_channel_id
        self.frontend_condition = frontend_condition
        self.gpu_sync_policy = gpu_sync_policy
        self.wall_clock = wall_clock
        self._metadata: Optional[StreamMetadata] = None
        self._stream_id: Optional[str] = None
        self._origin: Optional[float] = None
        self._spool = None
        self._received_through_sample = 0
        self._scopes: Dict[str, _ScopeState] = {}
        self._snapshots: List[Dict[str, Any]] = []
        self._finished = False

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities(
            supports_partial=True,
            supports_revision=True,
            supports_stable_prefix=False,
            supports_endpoint=False,
            supports_token_timestamps=False,
            supports_word_timestamps=False,
            reports_consumed_position=True,
            native_streaming=False,
            stateful=True,
        )

    @property
    def startup_metadata(self) -> Optional[Dict[str, Any]]:
        return None if self._metadata is None else dict(self._metadata.consumer_visible_dict())

    @property
    def received_through_sample(self) -> int:
        return self._received_through_sample

    @property
    def snapshots(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(dict(row) for row in self._snapshots)

    @property
    def active_scope_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(scope_id for scope_id, state in self._scopes.items() if not state.closed))

    def start(self, metadata: StreamMetadata) -> None:
        if self._metadata is not None:
            raise RollingWhisperError("adapter cannot start twice")
        if metadata.sample_rate != 16000:
            raise RollingWhisperError("R5A Whisper input must be 16 kHz")
        if metadata.sample_format != "pcm_s32le":
            raise RollingWhisperError("R5A Whisper input must be PCM32 little-endian")
        if self.selected_channel_id not in metadata.channel_ids:
            raise RollingWhisperError(
                f"selected channel {self.selected_channel_id!r} is not in source metadata"
            )
        self._metadata = metadata
        self._stream_id = metadata.stream_id
        self.materialized_dir.mkdir(parents=True, exist_ok=False)
        self.spool_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self._spool = self.spool_path.open("xb+")

    def on_run_wall_origin(self, origin: float) -> None:
        if self._metadata is None:
            raise RollingWhisperError("wall origin received before start")
        self._origin = origin

    def on_scope_start_release(self, release: ScopeStartRelease) -> None:
        if self._metadata is None:
            raise RollingWhisperError("scope start received before start")
        if release.segment_id in self._scopes:
            raise RollingWhisperError(f"duplicate scope start: {release.segment_id}")
        self._scopes[release.segment_id] = _ScopeState(
            start=release,
            last_decode_end_sample=release.start_sample,
        )

    def on_scope_end_release(self, release: ScopeEndRelease) -> None:
        state = self._scopes.get(release.segment_id)
        if state is None:
            raise RollingWhisperError(
                f"scope end arrived before causal scope start: {release.segment_id}"
            )
        if state.end is not None:
            raise RollingWhisperError(f"duplicate scope end: {release.segment_id}")
        if release.end_sample <= state.start.start_sample:
            raise RollingWhisperError(f"scope end precedes scope start: {release.segment_id}")
        state.end = release

    def on_oracle_release(self, release: object) -> None:
        """Accept exactly one already-released boundary from the runtime."""

        if isinstance(release, ScopeStartRelease):
            self.on_scope_start_release(release)
        elif isinstance(release, ScopeEndRelease):
            self.on_scope_end_release(release)
        else:
            raise RollingWhisperError(
                f"rolling adapter received unsupported oracle release: {type(release).__name__}"
            )

    def _wall_offset(self, absolute_time: float) -> float:
        if self._origin is None:
            raise RollingWhisperError("runtime wall origin is not established")
        return max(0.0, absolute_time - self._origin)

    @staticmethod
    def _pack_pcm32(channel: Sequence[Any]) -> bytes:
        values = []
        for index, value in enumerate(channel):
            if isinstance(value, bool) or not isinstance(value, int):
                raise RollingWhisperError(f"channel sample {index} is not an integer")
            if value < -2147483648 or value > 2147483647:
                raise RollingWhisperError(f"channel sample {index} exceeds PCM32 range")
            values.append(value)
        return struct.pack(f"<{len(values)}i", *values)

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        if self._metadata is None or self._spool is None or self._stream_id is None:
            raise RollingWhisperError("consume called before start")
        if self._finished:
            raise RollingWhisperError("consume called after finish")
        if chunk.stream_id != self._stream_id:
            raise RollingWhisperError("chunk stream_id does not match adapter stream")
        if chunk.sample_rate != self._metadata.sample_rate:
            raise RollingWhisperError("chunk sample rate does not match metadata")
        if chunk.channel_ids != self._metadata.channel_ids:
            raise RollingWhisperError("chunk channel IDs do not match metadata")
        if chunk.start_sample != self._received_through_sample:
            raise RollingWhisperError(
                "delivered chunks must be contiguous: "
                f"start={chunk.start_sample}, received={self._received_through_sample}"
            )
        channel_index = chunk.channel_ids.index(self.selected_channel_id)
        self._spool.seek(chunk.start_sample * 4)
        self._spool.write(self._pack_pcm32(chunk.samples[channel_index]))
        self._received_through_sample = chunk.end_sample

        outputs: List[OutputEvent] = []
        for scope_id in sorted(
            self._scopes,
            key=lambda item: (
                self._scopes[item].start.start_sample,
                self._scopes[item].end.end_sample if self._scopes[item].end else 2**63 - 1,
                item,
            ),
        ):
            state = self._scopes[scope_id]
            if state.closed:
                continue
            if state.end is not None and self._received_through_sample >= state.end.end_sample:
                outputs.append(self._decode(state, state.end.end_sample, HypothesisKind.FINAL))
                state.closed = True
                continue
            current_end = self._received_through_sample
            if current_end - state.last_decode_end_sample >= self.update_interval_samples:
                outputs.append(self._decode(state, current_end, HypothesisKind.PARTIAL))
        return tuple(outputs)

    def _sync_gpu(self) -> None:
        if self.gpu_sync_policy == "none":
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - GPU-only path
            raise RollingWhisperError("CUDA sync policy requires torch") from exc
        if not torch.cuda.is_available():
            raise RollingWhisperError("CUDA sync policy requested but CUDA is unavailable")
        torch.cuda.synchronize()

    def _materialize(
        self,
        state: _ScopeState,
        decode_end_sample: int,
        revision_index: int,
        kind: HypothesisKind,
    ) -> Tuple[Path, str, float, float]:
        if decode_end_sample > self._received_through_sample:
            raise RollingWhisperError("decode range is not fully delivered")
        if decode_end_sample <= state.start.start_sample:
            raise RollingWhisperError("decode range must be non-empty")
        if self._spool is None or self._metadata is None:
            raise RollingWhisperError("spool is not open")
        safe = _SAFE_ID.sub("_", state.start.segment_id).strip("._") or "segment"
        suffix = hashlib.sha256(
            f"{state.start.segment_id}:{revision_index}:{kind.value}".encode("utf-8")
        ).hexdigest()[:12]
        path = self.materialized_dir / f"{safe}.{revision_index}.{kind.value.lower()}.{suffix}.wav"
        if path.exists():
            raise FileExistsError(f"materialized prefix already exists: {path}")
        start_wall = self.wall_clock.now() if self.wall_clock is not None else 0.0
        self._spool.flush()
        self._spool.seek(state.start.start_sample * 4)
        raw = self._spool.read((decode_end_sample - state.start.start_sample) * 4)
        expected = (decode_end_sample - state.start.start_sample) * 4
        if len(raw) != expected:
            raise RollingWhisperError("spool did not contain the exact prefix range")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(4)
            handle.setframerate(self._metadata.sample_rate)
            handle.writeframes(raw)
        end_wall = self.wall_clock.now() if self.wall_clock is not None else start_wall
        return path, hashlib.sha256(path.read_bytes()).hexdigest(), start_wall, end_wall

    def _decode(
        self,
        state: _ScopeState,
        decode_end_sample: int,
        kind: HypothesisKind,
    ) -> HypothesisEvent:
        revision = state.revision_index + 1
        path, materialized_sha, materialization_start, materialization_end = self._materialize(
            state, decode_end_sample, revision, kind
        )
        self._sync_gpu()
        invocation_start = self.wall_clock.now() if self.wall_clock is not None else 0.0
        result = self.backend.transcribe(path)
        if not isinstance(result, ASRResult) or not isinstance(result.text, str):
            raise RollingWhisperError("ASR backend must return ASRResult with string text")
        self._sync_gpu()
        invocation_return = self.wall_clock.now() if self.wall_clock is not None else invocation_start
        event_id = f"r5a:{state.start.segment_id}:{revision}"
        previous_event_id = state.last_event_id
        event = HypothesisEvent(
            stream_id=self._stream_id,
            scope_id=state.start.segment_id,
            event_id=event_id,
            revision_index=revision,
            text=result.text,
            kind=kind,
            replaces_event_id=previous_event_id,
            consumed_through_sample=decode_end_sample,
        )
        state.last_decode_end_sample = decode_end_sample
        state.revision_index = revision
        state.last_event_id = event_id
        self._snapshots.append(
            {
                "schema_version": "r5a.snapshot.v1",
                "segment_id": state.start.segment_id,
                "scope_id": state.start.segment_id,
                "start_sample": state.start.start_sample,
                "end_sample": None if state.end is None else state.end.end_sample,
                "decode_end_sample": decode_end_sample,
                "frame_count": decode_end_sample - state.start.start_sample,
                "kind": kind.value,
                "revision_index": revision,
                "event_id": event_id,
                "replaces_event_id": previous_event_id,
                "frontend": self.frontend_condition,
                "selected_channel_id": self.selected_channel_id,
                "materialized_path": str(path),
                "materialized_sha256": materialized_sha,
                "sample_rate": self._metadata.sample_rate,
                "channels": 1,
                "sample_width": 4,
                "scope_start_available_wall_offset_sec": state.start.scope_start_available_wall_offset_sec,
                "scope_start_release_wall_offset_sec": state.start.oracle_release_wall_offset_sec,
                "scope_end_available_wall_offset_sec": (
                    None if state.end is None else state.end.scope_end_available_wall_offset_sec
                ),
                "scope_end_release_wall_offset_sec": (
                    None if state.end is None else state.end.oracle_release_wall_offset_sec
                ),
                "materialization_start_wall_offset_sec": self._wall_offset(materialization_start),
                "materialization_end_wall_offset_sec": self._wall_offset(materialization_end),
                "asr_invocation_start_wall_offset_sec": self._wall_offset(invocation_start),
                "asr_invocation_return_wall_offset_sec": self._wall_offset(invocation_return),
                "backend_decode_sec": result.decode_sec,
                "text_raw": result.text,
                "text_empty": result.text == "",
                "observed_event_index": None,
                "observed_wall_offset_sec": None,
            }
        )
        return event

    def on_output_observed(self, event: OutputEvent, trace_event: TraceEvent) -> None:
        if not isinstance(event, HypothesisEvent):
            return
        row = next((item for item in self._snapshots if item["event_id"] == event.event_id), None)
        if row is None:
            raise RollingWhisperError(f"missing snapshot evidence for {event.event_id}")
        if trace_event.wall_offset_sec is None:
            raise RollingWhisperError("R5A output observation requires wall offset")
        row["observed_event_index"] = trace_event.event_index
        row["observed_wall_offset_sec"] = trace_event.wall_offset_sec

    def finish(self) -> Sequence[OutputEvent]:
        if self._metadata is None or self._spool is None:
            raise RollingWhisperError("finish called before start")
        pending = [scope_id for scope_id, state in self._scopes.items() if not state.closed]
        if pending:
            raise RollingWhisperError("source ended with unfinalized scopes: " + ",".join(sorted(pending)))
        self._spool.flush()
        self._spool.close()
        self._spool = None
        if self.snapshot_evidence_path.exists():
            raise FileExistsError("snapshot evidence already exists")
        temporary = self.snapshot_evidence_path.with_name(self.snapshot_evidence_path.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            for record in self._snapshots:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.snapshot_evidence_path)
        self._finished = True
        return ()

    def reset(self) -> None:
        if self._spool is not None:
            self._spool.close()
        self._metadata = None
        self._stream_id = None
        self._origin = None
        self._spool = None
        self._received_through_sample = 0
        self._scopes.clear()
        self._snapshots.clear()
        self._finished = False
