"""Native WeNet U2++ streaming adapter for the replay harness.

The adapter deliberately keeps the WeNet boundary narrow.  The replay runtime
owns the Stage-2 timeline and forwards only already-released scope boundaries;
this module owns PCM conversion, model chunking, native decoder state, and
native-result-to-event policy.

The production bridge uses the official WeNet LibTorch C API
(``wenet_init``/``wenet_decode``/``wenet_reset``).  Tests inject a small fake
decoder factory at the same boundary, so lifecycle and causality tests never
need a model or a native library.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

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


PCM_CONVERSION_POLICY = (
    "signed_pcm32le_arithmetic_right_shift_16_to_pcm16le_with_saturation"
)
DEFAULT_MODEL_CHUNK_SAMPLES = 320
DEFAULT_WENET_CHUNK_SIZE = 16


class WeNetU2PPError(ContractValidationError):
    """Raised when the native adapter cannot satisfy its lifecycle contract."""


def pcm32_to_pcm16le(samples: Sequence[int]) -> bytes:
    """Convert signed PCM32 samples to little-endian PCM16 explicitly.

    MISP's canonical replay source is PCM32LE.  WeNet's qualified LibTorch
    API accepts signed PCM16LE.  Arithmetic right shift preserves the high
    order signal bits; saturation is retained explicitly for defensive input
    validation and conversion provenance.
    """

    if isinstance(samples, (str, bytes, bytearray)):
        raise WeNetU2PPError("PCM32 samples must be an integer sequence")
    converted = bytearray()
    for index, value in enumerate(samples):
        if isinstance(value, bool) or not isinstance(value, int):
            raise WeNetU2PPError(f"PCM32 sample {index} must be an integer")
        if value < -2**31 or value > 2**31 - 1:
            raise WeNetU2PPError(f"PCM32 sample {index} is outside signed PCM32 range")
        pcm16 = value >> 16
        pcm16 = max(-32768, min(32767, pcm16))
        converted.extend(struct.pack("<h", pcm16))
    return bytes(converted)


@dataclass(frozen=True)
class NativeDecodeResult:
    """Normalized result returned by one native ``wenet_decode`` call."""

    result_type: str
    text: str
    raw_json: str

    def __post_init__(self) -> None:
        if self.result_type not in {"partial_result", "final_result"}:
            raise WeNetU2PPError(f"unsupported WeNet result type: {self.result_type!r}")
        if not isinstance(self.text, str):
            raise WeNetU2PPError("WeNet result sentence must be a string")


def parse_native_result(payload: Any) -> NativeDecodeResult:
    """Parse one official WeNet JSON result without inventing endpoint state."""

    if isinstance(payload, NativeDecodeResult):
        return payload
    if isinstance(payload, bytes):
        try:
            raw_json = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WeNetU2PPError("WeNet result is not UTF-8") from exc
    elif isinstance(payload, str):
        raw_json = payload
    else:
        raise WeNetU2PPError("native decoder must return JSON text or NativeDecodeResult")
    try:
        obj = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise WeNetU2PPError("WeNet native result is invalid JSON") from exc
    if not isinstance(obj, Mapping):
        raise WeNetU2PPError("WeNet native result must be a JSON object")
    result_type = obj.get("type")
    if result_type not in {"partial_result", "final_result"}:
        raise WeNetU2PPError("WeNet native result has no supported type")
    nbest = obj.get("nbest", [])
    if not isinstance(nbest, list):
        raise WeNetU2PPError("WeNet native nbest must be a list")
    text = ""
    if nbest:
        first = nbest[0]
        if not isinstance(first, Mapping) or not isinstance(first.get("sentence", ""), str):
            raise WeNetU2PPError("WeNet native nbest[0].sentence must be a string")
        text = first.get("sentence", "")
    return NativeDecodeResult(result_type=result_type, text=text, raw_json=raw_json)


class NativeStreamingDecoder(Protocol):
    """Minimal official-runtime boundary used by the adapter."""

    def decode(self, pcm16le: bytes, *, last: bool) -> Any:
        """Consume one finite PCM16LE chunk and return native JSON evidence."""

    def reset(self) -> None:
        """Reset feature/decoder/cache state for a new scope."""

    def close(self) -> None:
        """Release native resources."""


class _WeNetApi:
    """ctypes declarations for the qualified official LibTorch C API."""

    def __init__(self, library_path: Path) -> None:
        self.library_path = Path(library_path).expanduser().resolve(strict=True)
        try:
            self.library = ctypes.CDLL(str(self.library_path))
        except OSError as exc:
            raise WeNetU2PPError(f"cannot load WeNet native library: {self.library_path}") from exc

        self.wenet_init = self.library.wenet_init
        self.wenet_init.argtypes = [ctypes.c_char_p]
        self.wenet_init.restype = ctypes.c_void_p
        self.wenet_free = self.library.wenet_free
        self.wenet_free.argtypes = [ctypes.c_void_p]
        self.wenet_free.restype = None
        self.wenet_reset = self.library.wenet_reset
        self.wenet_reset.argtypes = [ctypes.c_void_p]
        self.wenet_reset.restype = None
        self.wenet_decode = self.library.wenet_decode
        self.wenet_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.wenet_decode.restype = ctypes.c_char_p
        self.wenet_set_language = self.library.wenet_set_language
        self.wenet_set_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.wenet_set_language.restype = None
        self.wenet_set_chunk_size = self.library.wenet_set_chunk_size
        self.wenet_set_chunk_size.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.wenet_set_chunk_size.restype = None
        self.wenet_set_nbest = self.library.wenet_set_nbest
        self.wenet_set_nbest.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.wenet_set_nbest.restype = None
        self.wenet_set_timestamp = self.library.wenet_set_timestamp
        self.wenet_set_timestamp.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.wenet_set_timestamp.restype = None
        self.wenet_set_continuous_decoding = self.library.wenet_set_continuous_decoding
        self.wenet_set_continuous_decoding.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.wenet_set_continuous_decoding.restype = None


class _CtypesWeNetDecoder:
    def __init__(self, api: _WeNetApi, handle: int) -> None:
        self._api = api
        self._handle = handle

    def decode(self, pcm16le: bytes, *, last: bool) -> NativeDecodeResult:
        if not isinstance(pcm16le, bytes) or len(pcm16le) % 2:
            raise WeNetU2PPError("native WeNet input must be an even-length PCM16LE byte string")
        # create_string_buffer also gives the native call a valid pointer for
        # an empty final flush, although the adapter normally finalizes with a
        # non-empty retained model chunk.
        buffer = ctypes.create_string_buffer(pcm16le or b"\0", len(pcm16le) or 1)
        result = self._api.wenet_decode(
            self._handle,
            ctypes.cast(buffer, ctypes.c_void_p),
            len(pcm16le),
            1 if last else 0,
        )
        if result is None:
            raise WeNetU2PPError("wenet_decode returned a null result")
        return parse_native_result(result)

    def reset(self) -> None:
        self._api.wenet_reset(self._handle)

    def close(self) -> None:
        if self._handle is not None:
            self._api.wenet_free(self._handle)
            self._handle = None


class CtypesWeNetDecoderFactory:
    """Create official WeNet LibTorch sessions from a shared loaded library."""

    def __init__(
        self,
        library_path: Path | str,
        model_dir: Path | str,
        *,
        wenet_chunk_size: int = DEFAULT_WENET_CHUNK_SIZE,
        language: str = "chs",
    ) -> None:
        if isinstance(wenet_chunk_size, bool) or not isinstance(wenet_chunk_size, int):
            raise WeNetU2PPError(
                "wenet_chunk_size must be a positive integer or -1 for "
                "official non-streaming mode"
            )
        if wenet_chunk_size == 0 or wenet_chunk_size < -1:
            raise WeNetU2PPError(
                "wenet_chunk_size must be a positive integer or -1 for "
                "official non-streaming mode"
            )
        if not isinstance(language, str) or not language.strip():
            raise WeNetU2PPError("language must be non-empty")
        self.api = _WeNetApi(Path(library_path))
        self.model_dir = Path(model_dir).expanduser().resolve(strict=True)
        self.wenet_chunk_size = wenet_chunk_size
        self.language = language
        self.created_count = 0

    def __call__(self) -> NativeStreamingDecoder:
        handle = self.api.wenet_init(os.fsencode(str(self.model_dir)))
        if not handle:
            raise WeNetU2PPError(f"wenet_init failed for model directory: {self.model_dir}")
        try:
            self.api.wenet_set_language(handle, os.fsencode(self.language))
            self.api.wenet_set_chunk_size(handle, self.wenet_chunk_size)
            self.api.wenet_set_nbest(handle, 1)
            self.api.wenet_set_timestamp(handle, 0)
            # Endpoint/final semantics are scope-owned by this adapter.  The
            # official API's continuous endpoint path is therefore disabled.
            self.api.wenet_set_continuous_decoding(handle, 0)
        except Exception:
            self.api.wenet_free(handle)
            raise
        self.created_count += 1
        return _CtypesWeNetDecoder(self.api, handle)


@dataclass
class _ScopeState:
    start: ScopeStartRelease
    end: Optional[ScopeEndRelease] = None
    decoder: Optional[NativeStreamingDecoder] = None
    pending_pcm32: List[int] = field(default_factory=list)
    next_input_sample: int = 0
    revision_index: int = -1
    last_event_id: Optional[str] = None
    last_emitted_text: Optional[str] = None
    emitted_any: bool = False
    last_native_partial_text: Optional[str] = None
    native_call_count: int = 0
    native_partial_return_count: int = 0
    native_final_return_count: int = 0
    native_non_empty_partial_count: int = 0
    native_changed_text_count: int = 0
    native_repeated_identical_count: int = 0
    native_previous_partial_seen: bool = False
    reset_count: int = 0
    closed: bool = False


class WeNetU2PPStreamingConsumer(StreamingConsumer):
    """Oracle-segmented native WeNet U2++ ASR consumer.

    A native decoder session is stateful for one scope.  Sessions are pooled:
    the first session is created before the measured replay interval and an
    additional session is created only when overlapping released scopes require
    it.  A session is reset after FINAL and reused for later non-overlapping
    scopes, so model state is never rebuilt per 20 ms source packet.
    """

    def __init__(
        self,
        *,
        decoder_factory: Callable[[], NativeStreamingDecoder],
        model_chunk_samples: int = DEFAULT_MODEL_CHUNK_SAMPLES,
        selected_channel_id: str = "ch0",
        frontend_condition: str = "raw_ch0",
        warmup_pcm16le: Optional[bytes] = None,
        snapshot_evidence_path: Optional[Path | str] = None,
        wall_clock: Optional[Any] = None,
    ) -> None:
        if not callable(decoder_factory):
            raise WeNetU2PPError("decoder_factory must be callable")
        if isinstance(model_chunk_samples, bool) or not isinstance(model_chunk_samples, int):
            raise WeNetU2PPError("model_chunk_samples must be a positive integer")
        if model_chunk_samples <= 0:
            raise WeNetU2PPError("model_chunk_samples must be a positive integer")
        if not isinstance(selected_channel_id, str) or not selected_channel_id.strip():
            raise WeNetU2PPError("selected_channel_id must be non-empty")
        if not isinstance(frontend_condition, str) or not frontend_condition.strip():
            raise WeNetU2PPError("frontend_condition must be non-empty")
        if warmup_pcm16le is not None and (
            not isinstance(warmup_pcm16le, bytes) or len(warmup_pcm16le) % 2
        ):
            raise WeNetU2PPError("warmup_pcm16le must be an even-length bytes value or null")
        self.decoder_factory = decoder_factory
        self.model_chunk_samples = model_chunk_samples
        self.selected_channel_id = selected_channel_id
        self.frontend_condition = frontend_condition
        self.warmup_pcm16le = warmup_pcm16le
        self.snapshot_evidence_path = (
            None
            if snapshot_evidence_path is None
            else Path(snapshot_evidence_path).expanduser().resolve(strict=False)
        )
        self.wall_clock = wall_clock
        self._metadata: Optional[StreamMetadata] = None
        self._stream_id: Optional[str] = None
        self._run_wall_origin: Optional[float] = None
        self._scopes: Dict[str, _ScopeState] = {}
        self._idle_decoders: List[NativeStreamingDecoder] = []
        self._snapshots: List[Dict[str, Any]] = []
        self._finished = False
        self._model_load_sec = 0.0
        self._warmup_sec = 0.0
        self._decoder_session_count = 0
        self._decoder_reset_count = 0
        self._max_active_decoder_sessions = 0

    def capabilities(self) -> ConsumerCapabilities:
        return ConsumerCapabilities(
            supports_partial=True,
            supports_revision=True,
            supports_stable_prefix=False,
            supports_endpoint=False,
            supports_token_timestamps=False,
            supports_word_timestamps=False,
            reports_consumed_position=False,
            native_streaming=True,
            stateful=True,
        )

    @property
    def startup_metadata(self) -> Optional[Dict[str, Any]]:
        return None if self._metadata is None else dict(self._metadata.consumer_visible_dict())

    @property
    def snapshots(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(dict(row) for row in self._snapshots)

    @property
    def active_scope_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(scope_id for scope_id, state in self._scopes.items() if not state.closed))

    @property
    def model_load_sec(self) -> float:
        return self._model_load_sec

    @property
    def warmup_sec(self) -> float:
        return self._warmup_sec

    @property
    def decoder_session_count(self) -> int:
        return self._decoder_session_count

    @property
    def decoder_reset_count(self) -> int:
        return self._decoder_reset_count

    @property
    def max_active_decoder_sessions(self) -> int:
        return self._max_active_decoder_sessions

    @property
    def native_audit(self) -> Mapping[str, Any]:
        return {
            "decoder_call_count": sum(state.native_call_count for state in self._scopes.values()),
            "partial_return_count": sum(
                state.native_partial_return_count for state in self._scopes.values()
            ),
            "final_return_count": sum(
                state.native_final_return_count for state in self._scopes.values()
            ),
            "non_empty_partial_count": sum(
                state.native_non_empty_partial_count for state in self._scopes.values()
            ),
            "changed_text_count": sum(
                state.native_changed_text_count for state in self._scopes.values()
            ),
            "repeated_identical_count": sum(
                state.native_repeated_identical_count for state in self._scopes.values()
            ),
            "model_load_sec": self.model_load_sec,
            "decoder_session_count": self.decoder_session_count,
            "decoder_reset_count": self.decoder_reset_count,
            "max_active_decoder_sessions": self.max_active_decoder_sessions,
            "model_chunk_samples": self.model_chunk_samples,
            "pcm_conversion_policy": PCM_CONVERSION_POLICY,
        }

    def start(self, metadata: StreamMetadata) -> None:
        if self._metadata is not None:
            raise WeNetU2PPError("adapter cannot start twice")
        if metadata.sample_rate != 16000:
            raise WeNetU2PPError("WeNet U2++ input must be 16 kHz")
        if metadata.sample_format != "pcm_s32le":
            raise WeNetU2PPError("WeNet U2++ input must be PCM32 little-endian")
        if self.selected_channel_id not in metadata.channel_ids:
            raise WeNetU2PPError(
                f"selected channel {self.selected_channel_id!r} is not in source metadata"
            )
        self._metadata = metadata
        self._stream_id = metadata.stream_id
        self._finished = False
        # Load one model-backed native session before the measured replay clock
        # begins.  It is reused after each scope reset.
        decoder = self._create_decoder()
        self._idle_decoders.append(decoder)
        if self.warmup_pcm16le is not None:
            warmup_start = time.perf_counter()
            parse_native_result(decoder.decode(self.warmup_pcm16le, last=True))
            decoder.reset()
            self._decoder_reset_count += 1
            self._warmup_sec = time.perf_counter() - warmup_start

    def on_run_wall_origin(self, origin: float) -> None:
        if self._metadata is None:
            raise WeNetU2PPError("wall origin received before start")
        self._run_wall_origin = origin

    def _now(self) -> float:
        return float(self.wall_clock.now()) if self.wall_clock is not None else time.perf_counter()

    def _wall_offset(self, absolute: float) -> Optional[float]:
        if self._run_wall_origin is None:
            return None
        return max(0.0, absolute - self._run_wall_origin)

    def _create_decoder(self) -> NativeStreamingDecoder:
        start = time.perf_counter()
        decoder = self.decoder_factory()
        self._model_load_sec += time.perf_counter() - start
        self._decoder_session_count += 1
        self._max_active_decoder_sessions = max(
            self._max_active_decoder_sessions,
            self._decoder_session_count - len(self._idle_decoders),
        )
        return decoder

    def _acquire_decoder(self) -> NativeStreamingDecoder:
        if self._idle_decoders:
            return self._idle_decoders.pop()
        decoder = self._create_decoder()
        return decoder

    def _release_decoder(self, state: _ScopeState) -> None:
        if state.decoder is None:
            raise WeNetU2PPError(f"scope {state.start.segment_id!r} has no native decoder")
        state.decoder.reset()
        state.reset_count += 1
        self._decoder_reset_count += 1
        self._idle_decoders.append(state.decoder)
        state.decoder = None

    def on_oracle_release(self, release: object) -> None:
        if self._metadata is None:
            raise WeNetU2PPError("oracle release received before start")
        if isinstance(release, ScopeStartRelease):
            if release.segment_id in self._scopes:
                raise WeNetU2PPError(f"duplicate scope start: {release.segment_id}")
            state = _ScopeState(
                start=release,
                next_input_sample=release.start_sample,
            )
            self._scopes[release.segment_id] = state
            return
        if isinstance(release, ScopeEndRelease):
            state = self._scopes.get(release.segment_id)
            if state is None:
                raise WeNetU2PPError(
                    f"scope end arrived before causal scope start: {release.segment_id}"
                )
            if state.end is not None:
                raise WeNetU2PPError(f"duplicate scope end: {release.segment_id}")
            if release.end_sample <= state.start.start_sample:
                raise WeNetU2PPError(f"scope end precedes scope start: {release.segment_id}")
            state.end = release
            return
        raise WeNetU2PPError(
            f"adapter received unsupported oracle release: {type(release).__name__}"
        )

    def on_scope_start_release(self, release: ScopeStartRelease) -> None:
        self.on_oracle_release(release)

    def on_scope_end_release(self, release: ScopeEndRelease) -> None:
        self.on_oracle_release(release)

    def _validate_chunk(self, chunk: AudioChunk) -> int:
        if self._metadata is None or self._stream_id is None:
            raise WeNetU2PPError("consume called before start")
        if self._finished:
            raise WeNetU2PPError("consume called after finish")
        if chunk.stream_id != self._stream_id:
            raise WeNetU2PPError("chunk stream_id does not match adapter stream")
        if chunk.sample_rate != self._metadata.sample_rate:
            raise WeNetU2PPError("chunk sample rate does not match adapter metadata")
        if chunk.channel_ids != self._metadata.channel_ids:
            raise WeNetU2PPError("chunk channel IDs do not match adapter metadata")
        try:
            return chunk.channel_ids.index(self.selected_channel_id)
        except ValueError as exc:  # pragma: no cover - start() checks this first.
            raise WeNetU2PPError("selected channel is absent from delivered chunk") from exc

    def consume(self, chunk: AudioChunk) -> Sequence[OutputEvent]:
        channel_index = self._validate_chunk(chunk)
        outputs: List[OutputEvent] = []
        for scope_id in sorted(
            self._scopes,
            key=lambda item: (
                self._scopes[item].start.start_sample,
                self._scopes[item].end.end_sample
                if self._scopes[item].end is not None
                else 2**63 - 1,
                item,
            ),
        ):
            state = self._scopes[scope_id]
            if state.closed:
                continue
            if chunk.end_sample <= state.next_input_sample:
                continue
            if chunk.start_sample > state.next_input_sample:
                raise WeNetU2PPError(
                    f"delivered audio gap for scope {scope_id!r}: "
                    f"chunk starts {chunk.start_sample}, expected {state.next_input_sample}"
                )
            start_sample = max(chunk.start_sample, state.next_input_sample, state.start.start_sample)
            end_limit = state.end.end_sample if state.end is not None else chunk.end_sample
            end_sample = min(chunk.end_sample, end_limit)
            if end_sample > start_sample:
                offset_start = start_sample - chunk.start_sample
                offset_end = end_sample - chunk.start_sample
                state.pending_pcm32.extend(chunk.samples[channel_index][offset_start:offset_end])
                state.next_input_sample = end_sample
            if state.end is not None and state.next_input_sample >= state.end.end_sample:
                outputs.extend(self._feed_state(state, force_final=True))
                if not state.closed:
                    raise WeNetU2PPError(
                        f"scope {scope_id!r} reached E without a FINAL native result"
                    )
            else:
                outputs.extend(self._feed_state(state, force_final=False))
        return tuple(outputs)

    def _feed_state(self, state: _ScopeState, *, force_final: bool) -> List[OutputEvent]:
        outputs: List[OutputEvent] = []
        if not force_final and len(state.pending_pcm32) <= self.model_chunk_samples:
            return outputs
        if state.decoder is None:
            # Decoder allocation is intentionally delayed until the scope has
            # a model chunk to process.  This lets a scope ending at the same
            # sample as a new scope release its pooled decoder before the new
            # scope acquires one, without changing the runtime's frozen
            # START-before-END release ordering.
            state.decoder = self._acquire_decoder()
            self._max_active_decoder_sessions = max(
                self._max_active_decoder_sessions,
                sum(
                    item.decoder is not None and not item.closed
                    for item in self._scopes.values()
                ),
            )
        if force_final:
            while len(state.pending_pcm32) > self.model_chunk_samples:
                outputs.extend(self._call_native(state, self.model_chunk_samples, last=False))
            if not state.pending_pcm32:
                raise WeNetU2PPError(
                    f"scope {state.start.segment_id!r} has no retained PCM for FINAL"
                )
            outputs.extend(self._call_native(state, len(state.pending_pcm32), last=True))
            return outputs
        # Retain one complete model chunk until E is known.  This guarantees
        # that the final native call carries last=1 with real samples and that
        # no empty or guessed final flush is needed.
        while len(state.pending_pcm32) > self.model_chunk_samples:
            outputs.extend(self._call_native(state, self.model_chunk_samples, last=False))
        return outputs

    def _call_native(
        self, state: _ScopeState, sample_count: int, *, last: bool
    ) -> List[OutputEvent]:
        if state.decoder is None:
            raise WeNetU2PPError("native decoder is unavailable")
        samples = state.pending_pcm32[:sample_count]
        del state.pending_pcm32[:sample_count]
        pcm16le = pcm32_to_pcm16le(samples)
        call_start = self._now()
        native = parse_native_result(state.decoder.decode(pcm16le, last=last))
        call_return = self._now()
        state.native_call_count += 1
        if native.result_type == "partial_result":
            state.native_partial_return_count += 1
            if native.text:
                state.native_non_empty_partial_count += 1
            if state.native_previous_partial_seen:
                if native.text == state.last_native_partial_text:
                    state.native_repeated_identical_count += 1
                else:
                    state.native_changed_text_count += 1
            state.last_native_partial_text = native.text
            state.native_previous_partial_seen = True
            if last:
                raise WeNetU2PPError("native returned PARTIAL for a final input call")
            # Initial empty results are raw decoder evidence, not visible
            # hypotheses.  Once text is visible, a changed empty text remains a
            # valid replacement snapshot and is emitted normally.
            if not state.emitted_any and native.text == "":
                return []
            if state.emitted_any and native.text == state.last_emitted_text:
                return []
            return [self._make_event(state, native, HypothesisKind.PARTIAL, call_start, call_return)]

        state.native_final_return_count += 1
        if not last:
            raise WeNetU2PPError("native returned FINAL before oracle scope end")
        event = self._make_event(state, native, HypothesisKind.FINAL, call_start, call_return)
        state.closed = True
        self._release_decoder(state)
        return [event]

    def _make_event(
        self,
        state: _ScopeState,
        native: NativeDecodeResult,
        kind: HypothesisKind,
        call_start: float,
        call_return: float,
    ) -> HypothesisEvent:
        state.revision_index += 1
        event_id = f"r5b:{state.start.segment_id}:{state.revision_index}"
        event = HypothesisEvent(
            stream_id=self._stream_id or "",
            scope_id=state.start.segment_id,
            event_id=event_id,
            revision_index=state.revision_index,
            text=native.text,
            kind=kind,
            replaces_event_id=state.last_event_id,
            consumed_through_sample=None,
        )
        end_sample = None if state.end is None else state.end.end_sample
        row = {
            "schema_version": "r5b.wenet.snapshot.v1",
            "scope_id": state.start.segment_id,
            "segment_id": state.start.segment_id,
            "start_sample": state.start.start_sample,
            "end_sample": end_sample,
            "kind": kind.value,
            "event_id": event_id,
            "revision_index": state.revision_index,
            "replaces_event_id": state.last_event_id,
            "text_raw": native.text,
            "text_empty": native.text == "",
            "native_result_type": native.result_type,
            "native_call_count": state.native_call_count,
            "native_partial_return_count": state.native_partial_return_count,
            "native_final_return_count": state.native_final_return_count,
            "native_non_empty_partial_count": state.native_non_empty_partial_count,
            "native_changed_text_count": state.native_changed_text_count,
            "native_repeated_identical_count": state.native_repeated_identical_count,
            "native_call_start_wall_offset_sec": self._wall_offset(call_start),
            "native_call_return_wall_offset_sec": self._wall_offset(call_return),
            "scope_start_available_wall_offset_sec": state.start.scope_start_available_wall_offset_sec,
            "scope_start_release_wall_offset_sec": state.start.oracle_release_wall_offset_sec,
            "scope_end_available_wall_offset_sec": (
                None if state.end is None else state.end.scope_end_available_wall_offset_sec
            ),
            "scope_end_release_wall_offset_sec": (
                None if state.end is None else state.end.oracle_release_wall_offset_sec
            ),
            "frontend_condition": self.frontend_condition,
            "selected_channel_id": self.selected_channel_id,
            "pcm_conversion_policy": PCM_CONVERSION_POLICY,
            "model_chunk_samples": self.model_chunk_samples,
            "reports_consumed_position": False,
            "observed_event_index": None,
            "observed_wall_offset_sec": None,
        }
        self._snapshots.append(row)
        state.last_event_id = event_id
        state.last_emitted_text = native.text
        state.emitted_any = True
        return event

    def on_output_observed(self, event: OutputEvent, trace_event: TraceEvent) -> None:
        if not isinstance(event, HypothesisEvent):
            return
        for row in reversed(self._snapshots):
            if row["event_id"] == event.event_id:
                row["observed_event_index"] = trace_event.event_index
                row["observed_wall_offset_sec"] = trace_event.wall_offset_sec
                return
        raise WeNetU2PPError(f"missing snapshot evidence for {event.event_id}")

    def finish(self) -> Sequence[OutputEvent]:
        if self._metadata is None:
            raise WeNetU2PPError("finish called before start")
        pending = [scope_id for scope_id, state in self._scopes.items() if not state.closed]
        if pending:
            raise WeNetU2PPError(
                "source ended with unfinalized scopes: " + ",".join(sorted(pending))
            )
        self._write_snapshot_evidence()
        self._close_idle_decoders()
        self._finished = True
        return ()

    def _write_snapshot_evidence(self) -> None:
        if self.snapshot_evidence_path is None:
            return
        destination = self.snapshot_evidence_path
        if destination.exists():
            raise FileExistsError(f"WeNet snapshot evidence already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for row in self._snapshots:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def _close_idle_decoders(self) -> None:
        while self._idle_decoders:
            self._idle_decoders.pop().close()

    def reset(self) -> None:
        for state in self._scopes.values():
            if state.decoder is not None:
                state.decoder.close()
        self._close_idle_decoders()
        self._metadata = None
        self._stream_id = None
        self._run_wall_origin = None
        self._scopes.clear()
        self._snapshots.clear()
        self._finished = False
        self._model_load_sec = 0.0
        self._warmup_sec = 0.0
        self._decoder_session_count = 0
        self._decoder_reset_count = 0
        self._max_active_decoder_sessions = 0


__all__ = [
    "CtypesWeNetDecoderFactory",
    "DEFAULT_MODEL_CHUNK_SAMPLES",
    "DEFAULT_WENET_CHUNK_SIZE",
    "NativeDecodeResult",
    "PCM_CONVERSION_POLICY",
    "WeNetU2PPError",
    "WeNetU2PPStreamingConsumer",
    "parse_native_result",
    "pcm32_to_pcm16le",
]
