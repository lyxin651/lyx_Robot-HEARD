from __future__ import annotations

import inspect
import json
import struct
from pathlib import Path
from typing import Iterable, Sequence

import pytest

from robot_heard.replay.clock import FakeWallClock
from robot_heard.replay.events import (
    AudioChunk,
    ConsumerCapabilities,
    StreamMetadata,
)
from robot_heard.replay.oracle import (
    ScopeEndRelease,
    ScopeStartRelease,
    Stage2ScopeBoundaryReleaseGate,
)
from robot_heard.replay.paced_runtime import PacedReplayRuntime
from robot_heard.replay.source import SyntheticSource
from robot_heard.replay.trace import TraceWriter
from robot_heard.streaming.adapters.wenet_u2pp import (
    NativeDecodeResult,
    WeNetU2PPError,
    WeNetU2PPStreamingConsumer,
    parse_native_result,
    pcm32_to_pcm16le,
)


class _Row:
    def __init__(self, segment_id: str, start_sample: int, end_sample: int):
        self.segment_id = segment_id
        self.start_sample = start_sample
        self.end_sample = end_sample


class _FakeNativeDecoder:
    def __init__(self, texts: Sequence[str] | None = None):
        self.texts = list(texts or ())
        self.calls: list[tuple[tuple[int, ...], bool]] = []
        self.reset_calls = 0
        self.closed = False

    def decode(self, pcm16le: bytes, *, last: bool):
        samples = struct.unpack(f"<{len(pcm16le) // 2}h", pcm16le)
        self.calls.append((samples, last))
        if self.texts:
            text = self.texts.pop(0)
        else:
            text = f"call-{len(self.calls)}"
        return json.dumps(
            {
                "type": "final_result" if last else "partial_result",
                "nbest": [{"sentence": text}],
            }
        )

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True


class _FakeFactory:
    def __init__(self, scripts: Iterable[Sequence[str]] = ()):
        self.scripts = [list(script) for script in scripts]
        self.instances: list[_FakeNativeDecoder] = []

    def __call__(self):
        script = self.scripts[len(self.instances)] if len(self.instances) < len(self.scripts) else ()
        decoder = _FakeNativeDecoder(script)
        self.instances.append(decoder)
        return decoder


def _source(sample_count: int, *, packet_samples: int = 2):
    # The high-order 16 bits preserve the visible integer sample number after
    # the explicit PCM32 -> PCM16 conversion.
    values = tuple(index << 16 for index in range(sample_count))
    other = tuple((1000 + index) << 16 for index in range(sample_count))
    return SyntheticSource(
        (values, other),
        sample_rate=16000,
        packet_samples=packet_samples,
        stream_id="wenet-test",
        channel_ids=("ch0", "ch1"),
        sample_format="pcm_s32le",
        channel_layout="separate_mono_files",
    )


def _writer(tmp_path: Path, source, consumer):
    trace_path = tmp_path / "trace.jsonl"
    run_path = tmp_path / "trace.run.json"
    provenance = {
        "schema_version": "r5b.test.v1",
        "run_id": "r5b-test",
        "replay_mode": "COMPUTE_AWARE",
        "source_identity": source.identity(),
        "source_sha256": source.content_sha256,
        "source_sample_rate": source.metadata.sample_rate,
        "channel_ids": list(source.metadata.channel_ids),
        "packet_samples": source.packet_samples,
        "queue_policy": "lossless_unbounded_fifo",
        "timeline_authority": "integer_sample_index",
        "consumer_identity": type(consumer).__name__,
        "consumer_config": {"model_chunk_samples": consumer.model_chunk_samples},
        "consumer_capabilities": dict(consumer.capabilities().to_dict()),
        "code_commit": "test",
        "oracle_condition": "r5b_test",
        "warmup_load_policy": {"model_load": "fake", "warmup": "not_applicable"},
        "clock_policy": "fake_wall_clock",
        "clock_origin": {"kind": "fake_wall", "run_wall_origin": None},
        "metric_window": {"start_sample": 0, "end_sample": source.total_samples},
        "trace_path": str(trace_path),
        "device_provenance": {"status": "not_applicable"},
        "trace_durability_policy": "buffered_close_fsync",
    }
    return TraceWriter(trace_path, run_path, provenance), trace_path


def _run(tmp_path: Path, factory, rows, *, sample_count, packet_samples=2, model_chunk=2):
    source = _source(sample_count, packet_samples=packet_samples)
    clock = FakeWallClock()
    consumer = WeNetU2PPStreamingConsumer(
        decoder_factory=factory,
        model_chunk_samples=model_chunk,
        wall_clock=clock,
    )
    writer, trace_path = _writer(tmp_path, source, consumer)
    gate = Stage2ScopeBoundaryReleaseGate(
        rows,
        sample_rate=16000,
        packet_duration_sec=packet_samples / 16000,
    )
    result = PacedReplayRuntime(
        source,
        consumer,
        writer,
        wall_clock=clock,
        oracle_gate=gate,
    ).run()
    return source, consumer, result, trace_path


def test_pcm32_to_pcm16_exact_edges_and_saturation():
    raw = pcm32_to_pcm16le(
        (-2**31, -2147418113, -65536, -1, 0, 65535, 65536, 2147483647)
    )
    assert struct.unpack("<8h", raw) == (-32768, -32768, -1, -1, 0, 0, 1, 32767)


def test_pcm_conversion_rejects_implicit_or_invalid_inputs():
    with pytest.raises(WeNetU2PPError):
        pcm32_to_pcm16le([True])
    with pytest.raises(WeNetU2PPError):
        pcm32_to_pcm16le([2**31])
    with pytest.raises(WeNetU2PPError):
        pcm32_to_pcm16le(b"\0\0\0\0")


def test_native_result_parser_and_capabilities_are_explicit():
    parsed = parse_native_result('{"type":"partial_result","nbest":[] }')
    assert parsed == NativeDecodeResult("partial_result", "", '{"type":"partial_result","nbest":[] }')
    factory = _FakeFactory()
    consumer = WeNetU2PPStreamingConsumer(decoder_factory=factory)
    assert consumer.capabilities() == ConsumerCapabilities(
        supports_partial=True,
        supports_revision=True,
        native_streaming=True,
        stateful=True,
    )
    assert consumer.capabilities().reports_consumed_position is False
    assert consumer.capabilities().supports_endpoint is False


def test_scope_start_is_causal_and_end_is_not_needed_until_released(tmp_path):
    factory = _FakeFactory([("visible", "final")])
    source, consumer, result, trace_path = _run(
        tmp_path,
        factory,
        [_Row("s", 2, 6)],
        sample_count=8,
    )
    assert result.source_end_sample == 8
    assert len(factory.instances) == 1
    assert factory.instances[0].calls == [((2, 3), False), ((4, 5), True)]
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    hypotheses = [row["payload"] for row in records if row["event_type"] == "HYPOTHESIS"]
    assert [row["kind"] for row in hypotheses] == ["PARTIAL", "FINAL"]
    assert all(row["consumed_through_sample"] is None for row in hypotheses)
    assert consumer.active_scope_ids == ()


def test_partial_policy_suppresses_empty_and_repeated_native_snapshots(tmp_path):
    factory = _FakeFactory([("", "", "甲", "甲", "乙", "乙")])
    _, consumer, _, trace_path = _run(
        tmp_path,
        factory,
        [_Row("policy", 0, 12)],
        sample_count=12,
    )
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    hypotheses = [row["payload"] for row in records if row["event_type"] == "HYPOTHESIS"]
    assert [(row["kind"], row["text"], row["revision_index"]) for row in hypotheses] == [
        ("PARTIAL", "甲", 0),
        ("PARTIAL", "乙", 1),
        ("FINAL", "乙", 2),
    ]
    assert hypotheses[1]["replaces_event_id"] == hypotheses[0]["event_id"]
    assert hypotheses[2]["replaces_event_id"] == hypotheses[1]["event_id"]
    assert consumer.native_audit["partial_return_count"] == 5
    assert consumer.native_audit["changed_text_count"] == 2
    assert consumer.native_audit["repeated_identical_count"] == 2


def test_final_resets_and_adjacent_scopes_reuse_native_session(tmp_path):
    factory = _FakeFactory([("a", "a-final", "b", "b-final")])
    _, consumer, _, trace_path = _run(
        tmp_path,
        factory,
        [_Row("a", 0, 4), _Row("b", 4, 8)],
        sample_count=8,
    )
    assert len(factory.instances) == 1
    assert factory.instances[0].reset_calls == 2
    assert consumer.decoder_reset_count == 2
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    finals = [
        row["payload"]
        for row in records
        if row["event_type"] == "HYPOTHESIS" and row["payload"]["kind"] == "FINAL"
    ]
    assert {row["scope_id"] for row in finals} == {"a", "b"}
    assert all(row["revision_index"] == 1 for row in finals)


def test_overlapping_scopes_get_separate_state_without_gate_or_reference_metadata(tmp_path):
    factory = _FakeFactory([("a", "a-final"), ("b", "b-final")])
    _, consumer, _, trace_path = _run(
        tmp_path,
        factory,
        [_Row("a", 0, 10), _Row("b", 2, 8)],
        sample_count=10,
    )
    assert len(factory.instances) == 2
    assert all(instance.reset_calls == 1 for instance in factory.instances)
    assert not hasattr(consumer, "_oracle_gate")
    assert not hasattr(consumer, "_segments")
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    releases = [row["payload"] for row in records if row["event_type"] == "ORACLE_METADATA_RELEASE"]
    assert all(not ({"reference", "speaker_id", "future_segments"} & set(row)) for row in releases)
    finals = [
        row["payload"]["scope_id"]
        for row in records
        if row["event_type"] == "HYPOTHESIS" and row["payload"]["kind"] == "FINAL"
    ]
    assert set(finals) == {"a", "b"}


def test_no_event_after_final_and_final_is_exactly_once(tmp_path):
    factory = _FakeFactory([("partial", "final")])
    _, consumer, _, trace_path = _run(
        tmp_path,
        factory,
        [_Row("s", 0, 4)],
        sample_count=8,
    )
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    finals = [
        row for row in records if row["event_type"] == "HYPOTHESIS" and row["payload"]["kind"] == "FINAL"
    ]
    assert len(finals) == 1
    assert consumer.active_scope_ids == ()


def test_invalid_lifecycle_and_metadata_fail_closed():
    consumer = WeNetU2PPStreamingConsumer(decoder_factory=_FakeFactory())
    with pytest.raises(WeNetU2PPError):
        consumer.on_scope_end_release(
            ScopeEndRelease(
                "missing", 2, 0.1, 0.1, 0.0, "test"
            )
        )
    with pytest.raises(WeNetU2PPError):
        consumer.start(
            StreamMetadata(
                stream_id="bad",
                sample_rate=8000,
                channel_ids=("ch0",),
                sample_format="pcm_s32le",
            )
        )


def test_future_perturbation_before_delivery_has_identical_visible_state():
    factory_a = _FakeFactory([("same", "final")])
    factory_b = _FakeFactory([("same", "final")])
    metadata = StreamMetadata(
        stream_id="causal",
        sample_rate=16000,
        channel_ids=("ch0",),
        sample_format="pcm_s32le",
        channel_layout="separate_mono_files",
    )
    a = WeNetU2PPStreamingConsumer(
        decoder_factory=factory_a, model_chunk_samples=2, wall_clock=FakeWallClock()
    )
    b = WeNetU2PPStreamingConsumer(
        decoder_factory=factory_b, model_chunk_samples=2, wall_clock=FakeWallClock()
    )
    a.start(metadata)
    b.start(metadata)
    start = ScopeStartRelease("s", 0, 0.0, 0.0, 0.0, "test")
    a.on_scope_start_release(start)
    b.on_scope_start_release(start)
    prefix_a = AudioChunk(
        stream_id="causal", sequence_id=0, start_sample=0, end_sample=4,
        sample_rate=16000, channel_ids=("ch0",), samples=((0, 1 << 16, 2 << 16, 3 << 16),),
        is_final_source_chunk=False, scheduled_availability_logical_sample=4,
    )
    prefix_b = AudioChunk(
        stream_id="causal", sequence_id=0, start_sample=0, end_sample=4,
        sample_rate=16000, channel_ids=("ch0",), samples=((0, 1 << 16, 2 << 16, 3 << 16),),
        is_final_source_chunk=False, scheduled_availability_logical_sample=4,
    )
    assert a.consume(prefix_a) == b.consume(prefix_b)
    assert a.snapshots == b.snapshots
    assert factory_a.instances[0].calls == factory_b.instances[0].calls
    assert factory_a.instances[0].calls

    end = ScopeEndRelease("s", 6, 6 / 16000, 6 / 16000, 0.0, "test")
    a.on_scope_end_release(end)
    b.on_scope_end_release(end)
    future_a = AudioChunk(
        stream_id="causal", sequence_id=1, start_sample=4, end_sample=6,
        sample_rate=16000, channel_ids=("ch0",), samples=((4 << 16, 5 << 16),),
        is_final_source_chunk=False, scheduled_availability_logical_sample=6,
    )
    future_b = AudioChunk(
        stream_id="causal", sequence_id=1, start_sample=4, end_sample=6,
        sample_rate=16000, channel_ids=("ch0",), samples=((400 << 16, 500 << 16),),
        is_final_source_chunk=False, scheduled_availability_logical_sample=6,
    )
    assert a.consume(future_a) == b.consume(future_b)
    assert a.active_scope_ids == b.active_scope_ids == ()
    assert a.snapshots == b.snapshots
    assert factory_a.instances[0].calls[0] == factory_b.instances[0].calls[0]
    assert factory_a.instances[0].calls[-1] != factory_b.instances[0].calls[-1]


def test_adapter_has_no_oracle_gate_constructor_parameter():
    assert "oracle_gate" not in inspect.signature(WeNetU2PPStreamingConsumer).parameters
