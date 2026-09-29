"""R4 oracle-gated offline Whisper reference adapter.

This adapter is deliberately a segment-level pseudo-online reference.  It
receives only delivered chunks and released descriptors, materializes exact
sample ranges, and invokes the existing file-based ASRBackend unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import wave
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
from robot_heard.replay.oracle import OracleRelease, OracleSegmentDescriptor
from robot_heard.streaming.base import StreamingConsumer


class OfflineWhisperReferenceError(ContractValidationError):
    """Raised when the R4 adapter violates its causal materialization contract."""


_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


class OfflineWhisperReferenceConsumer(StreamingConsumer):
    """Wrap an existing ASRBackend behind the R4 raw_ch0 policy.

    ``oracle_gate`` is an interface to runtime-owned release state; this class
    never enumerates or stores the full Stage-2 timeline.  The only metadata
    retained from startup is the StreamMetadata whitelist.
    """

    def __init__(
        self,
        backend: ASRBackend,
        *,
        materialized_dir: Path,
        spool_path: Path,
        segment_evidence_path: Path,
        selected_channel_id: str = "ch0",
        frontend_condition: str = "raw_ch0",
        gpu_sync_policy: str = "none",
        wall_clock: Optional[Any] = None,
    ) -> None:
        if not isinstance(backend, ASRBackend):
            raise OfflineWhisperReferenceError("backend must implement ASRBackend")
        if not isinstance(materialized_dir, Path):
            materialized_dir = Path(materialized_dir)
        if not isinstance(spool_path, Path):
            spool_path = Path(spool_path)
        if not isinstance(segment_evidence_path, Path):
            segment_evidence_path = Path(segment_evidence_path)
        if spool_path.exists() or segment_evidence_path.exists():
            raise FileExistsError("R4 adapter artifacts must be fresh; overwrite is refused")
        if gpu_sync_policy not in (
            "none",
            "cuda_synchronize_before_and_after_backend_transcribe",
        ):
            raise OfflineWhisperReferenceError("unsupported gpu_sync_policy")
        if not isinstance(selected_channel_id, str) or not selected_channel_id.strip():
            raise OfflineWhisperReferenceError("selected_channel_id must be non-empty")
        self.backend = backend
        self.materialized_dir = materialized_dir.resolve()
        self.spool_path = spool_path.resolve()
        self.segment_evidence_path = segment_evidence_path.resolve()
        self.selected_channel_id = selected_channel_id
        self.frontend_condition = frontend_condition
        self.gpu_sync_policy = gpu_sync_policy
        self.wall_clock = wall_clock
        self._metadata: Optional[StreamMetadata] = None
        self._stream_id: Optional[str] = None
        self._origin: Optional[float] = None
        self._spool = None
        self._received_through_sample = 0
        self._pending: List[OracleSegmentDescriptor] = []
        self._release_evidence: Dict[str, OracleRelease] = {}
        self._evidence: Dict[str, Dict[str, Any]] = {}
        self._processed = set()
        self._finished = False

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities(
            supports_partial=False,
            supports_revision=False,
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
    def segment_evidence(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(self._evidence[key] for key in sorted(self._evidence))

    def start(self, metadata: StreamMetadata) -> None:
        if self._metadata is not None:
            raise OfflineWhisperReferenceError("adapter cannot start twice")
        if metadata.sample_rate != 16000:
            raise OfflineWhisperReferenceError("R4 Whisper input must be 16 kHz")
        if metadata.sample_format != "pcm_s32le":
            raise OfflineWhisperReferenceError("R4 Whisper input must be PCM32 little-endian")
        if self.selected_channel_id not in metadata.channel_ids:
            raise OfflineWhisperReferenceError(
                f"selected channel {self.selected_channel_id!r} is not in source metadata"
            )
        self._metadata = metadata
        self._stream_id = metadata.stream_id
        self.materialized_dir.mkdir(parents=True, exist_ok=False)
        self.spool_path.parent.mkdir(parents=True, exist_ok=True)
        self.segment_evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self._spool = self.spool_path.open("xb+")

    def on_run_wall_origin(self, origin: float) -> None:
        if self._metadata is None:
            raise OfflineWhisperReferenceError("wall origin received before start")
        self._origin = origin

    def on_oracle_release(self, release: OracleRelease) -> None:
        """Receive runtime evidence without receiving evaluator metadata."""

        segment_id = release.descriptor.segment_id
        if segment_id in self._release_evidence:
            raise OfflineWhisperReferenceError(f"duplicate oracle release: {segment_id}")
        self._release_evidence[segment_id] = release
        self._pending.append(release.descriptor)
        self._pending.sort(key=lambda descriptor: (descriptor.end_sample, descriptor.segment_id))

    def _wall_offset(self, absolute_time: float) -> float:
        if self._origin is None:
            raise OfflineWhisperReferenceError("runtime wall origin is not established")
        return max(0.0, absolute_time - self._origin)

    def _sync_gpu(self) -> None:
        if self.gpu_sync_policy == "none":
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised on GPU server
            raise OfflineWhisperReferenceError("CUDA sync policy requires torch") from exc
        if not torch.cuda.is_available():
            raise OfflineWhisperReferenceError("CUDA sync policy requested but CUDA is unavailable")
        torch.cuda.synchronize()

    @staticmethod
    def _pack_pcm32(channel: Sequence[Any]) -> bytes:
        values = []
        for index, value in enumerate(channel):
            if isinstance(value, bool) or not isinstance(value, int):
                raise OfflineWhisperReferenceError(f"channel sample {index} is not an integer")
            if value < -2147483648 or value > 2147483647:
                raise OfflineWhisperReferenceError(f"channel sample {index} exceeds PCM32 range")
            values.append(value)
        return struct.pack("<{}i".format(len(values)), *values)

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        if self._metadata is None or self._spool is None or self._stream_id is None:
            raise OfflineWhisperReferenceError("consume called before start")
        if self._finished:
            raise OfflineWhisperReferenceError("consume called after finish")
        if chunk.stream_id != self._stream_id:
            raise OfflineWhisperReferenceError("chunk stream_id does not match adapter stream")
        if chunk.sample_rate != self._metadata.sample_rate:
            raise OfflineWhisperReferenceError("chunk sample rate does not match metadata")
        if chunk.channel_ids != self._metadata.channel_ids:
            raise OfflineWhisperReferenceError("chunk channel IDs do not match metadata")
        if chunk.start_sample != self._received_through_sample:
            raise OfflineWhisperReferenceError(
                "delivered chunks must be contiguous: "
                f"start={chunk.start_sample}, received={self._received_through_sample}"
            )
        channel_index = chunk.channel_ids.index(self.selected_channel_id)
        self._spool.seek(chunk.start_sample * 4)
        self._spool.write(self._pack_pcm32(chunk.samples[channel_index]))
        self._received_through_sample = chunk.end_sample
        self._pending.sort(key=lambda descriptor: (descriptor.end_sample, descriptor.segment_id))
        outputs: List[OutputEvent] = []
        still_pending = []
        for descriptor in self._pending:
            if descriptor.end_sample > self._received_through_sample:
                still_pending.append(descriptor)
            else:
                outputs.append(self._decode_descriptor(descriptor))
        self._pending = still_pending
        return tuple(outputs)

    def _materialize(self, descriptor: OracleSegmentDescriptor) -> Tuple[Path, str, float, float]:
        if descriptor.end_sample > self._received_through_sample:
            raise OfflineWhisperReferenceError("descriptor end is not yet in arrived audio")
        if self._spool is None or self._metadata is None:
            raise OfflineWhisperReferenceError("spool is not open")
        safe = _SAFE_ID.sub("_", descriptor.segment_id).strip("._") or "segment"
        suffix = hashlib.sha256(descriptor.segment_id.encode("utf-8")).hexdigest()[:12]
        path = self.materialized_dir / f"{safe}.{suffix}.wav"
        if path.exists():
            raise FileExistsError(f"materialized segment already exists: {path}")
        start_wall = self.wall_clock.now() if self.wall_clock is not None else 0.0
        self._spool.flush()
        self._spool.seek(descriptor.start_sample * 4)
        raw = self._spool.read((descriptor.end_sample - descriptor.start_sample) * 4)
        expected = (descriptor.end_sample - descriptor.start_sample) * 4
        if len(raw) != expected:
            raise OfflineWhisperReferenceError("spool did not contain the exact segment range")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(4)
            handle.setframerate(self._metadata.sample_rate)
            handle.writeframes(raw)
        end_wall = self.wall_clock.now() if self.wall_clock is not None else start_wall
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return path, digest, start_wall, end_wall

    def _decode_descriptor(self, descriptor: OracleSegmentDescriptor) -> HypothesisEvent:
        if descriptor.segment_id in self._processed:
            raise OfflineWhisperReferenceError(f"duplicate descriptor: {descriptor.segment_id}")
        release = self._release_evidence.get(descriptor.segment_id)
        if release is None:
            raise OfflineWhisperReferenceError(
                f"oracle release evidence missing for {descriptor.segment_id!r}"
            )
        path, materialized_sha, materialization_start, materialization_end = self._materialize(
            descriptor
        )
        self._sync_gpu()
        invocation_start = self.wall_clock.now() if self.wall_clock is not None else 0.0
        result = self.backend.transcribe(path)
        if not isinstance(result, ASRResult):
            raise OfflineWhisperReferenceError("ASR backend must return ASRResult")
        self._sync_gpu()
        invocation_return = self.wall_clock.now() if self.wall_clock is not None else invocation_start
        if not isinstance(result.text, str):
            raise OfflineWhisperReferenceError("ASRResult.text must be a string")
        event_id = f"r4-final:{descriptor.segment_id}"
        self._evidence[descriptor.segment_id] = {
            "schema_version": "r4.segment.v1",
            "segment_id": descriptor.segment_id,
            "start_sample": descriptor.start_sample,
            "end_sample": descriptor.end_sample,
            "frame_count": descriptor.end_sample - descriptor.start_sample,
            "frontend": self.frontend_condition,
            "selected_channel_id": self.selected_channel_id,
            "materialized_path": str(path),
            "materialized_sha256": materialized_sha,
            "sample_rate": self._metadata.sample_rate,
            "channels": 1,
            "sample_width": 4,
            "segment_end_available_wall_offset_sec": release.segment_end_available_wall_offset_sec,
            "oracle_release_wall_offset_sec": release.oracle_release_wall_offset_sec,
            "oracle_release_quantization_sec": release.oracle_release_quantization_sec,
            "oracle_condition": release.condition,
            "materialization_start_wall_offset_sec": self._wall_offset(materialization_start),
            "materialization_end_wall_offset_sec": self._wall_offset(materialization_end),
            "asr_invocation_start_wall_offset_sec": self._wall_offset(invocation_start),
            "asr_invocation_return_wall_offset_sec": self._wall_offset(invocation_return),
            "backend_decode_sec": result.decode_sec,
            "text_raw": result.text,
            "text_empty": result.text == "",
            "event_id": event_id,
            "final_event_index": None,
            "final_observed_wall_offset_sec": None,
            "segment_end_to_final_wall_delay_sec": None,
        }
        self._processed.add(descriptor.segment_id)
        return HypothesisEvent(
            stream_id=self._stream_id,
            scope_id=descriptor.segment_id,
            event_id=event_id,
            revision_index=0,
            text=result.text,
            kind=HypothesisKind.FINAL,
            consumed_through_sample=descriptor.end_sample,
        )

    def on_output_observed(self, event: OutputEvent, trace_event: TraceEvent) -> None:
        if not isinstance(event, HypothesisEvent) or event.kind is not HypothesisKind.FINAL:
            return
        evidence = next(
            (value for value in self._evidence.values() if value["event_id"] == event.event_id),
            None,
        )
        if evidence is None:
            raise OfflineWhisperReferenceError(f"missing evidence for observed event {event.event_id}")
        observed = trace_event.wall_offset_sec
        if observed is None:
            raise OfflineWhisperReferenceError("R4 final observation requires wall offset")
        evidence["final_event_index"] = trace_event.event_index
        evidence["final_observed_wall_offset_sec"] = observed
        evidence["segment_end_to_final_wall_delay_sec"] = observed - (
            evidence["end_sample"] / evidence["sample_rate"]
        )

    def finish(self) -> Sequence[OutputEvent]:
        if self._metadata is None or self._spool is None:
            raise OfflineWhisperReferenceError("finish called before start")
        if self._pending:
            raise OfflineWhisperReferenceError("source ended with unprocessed released segments")
        if len(self._processed) != len(self._evidence):
            raise OfflineWhisperReferenceError("segment processing evidence is incomplete")
        self._spool.flush()
        self._spool.close()
        self._spool = None
        if self.segment_evidence_path.exists():
            raise FileExistsError("segment evidence already exists")
        temporary = self.segment_evidence_path.with_name(self.segment_evidence_path.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            for record in self.segment_evidence:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.segment_evidence_path)
        self._finished = True
        return ()

    def reset(self) -> None:
        if self._spool is not None:
            self._spool.close()
        self._metadata = None
        self._stream_id = None
        self._origin = None
        self._received_through_sample = 0
        self._pending.clear()
        self._release_evidence.clear()
        self._evidence.clear()
        self._processed.clear()
        self._finished = False
