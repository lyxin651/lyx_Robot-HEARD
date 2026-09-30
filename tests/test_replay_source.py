from __future__ import annotations

import pytest

from robot_heard.replay.clock import FakeClock
from robot_heard.replay.events import ContractValidationError
from robot_heard.replay.source import SyntheticSource


def test_synthetic_source_exact_packetization_tail_and_alignment():
    source = SyntheticSource(
        (tuple(range(105)), tuple(range(1000, 1105))),
        sample_rate=1000,
        packet_samples=20,
        stream_id="synthetic-105",
        channel_ids=("left", "right"),
    )

    chunks = list(source)

    assert [(chunk.start_sample, chunk.end_sample) for chunk in chunks] == [
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 80),
        (80, 100),
        (100, 105),
    ]
    assert [chunk.sequence_id for chunk in chunks] == list(range(6))
    assert [chunk.is_final_source_chunk for chunk in chunks] == [False] * 5 + [True]
    assert all(chunk.scheduled_availability_logical_sample == chunk.end_sample for chunk in chunks)
    assert all(chunk.num_channels == 2 for chunk in chunks)
    assert chunks[-1].samples == (tuple(range(100, 105)), tuple(range(1100, 1105)))

    flattened_left = tuple(value for chunk in chunks for value in chunk.samples[0])
    flattened_right = tuple(value for chunk in chunks for value in chunk.samples[1])
    assert flattened_left == tuple(range(105))
    assert flattened_right == tuple(range(1000, 1105))


def test_synthetic_source_is_repeatable_without_mutating_input():
    samples = ([1, 2, 3], [4, 5, 6])
    source = SyntheticSource(samples, sample_rate=1000, packet_samples=2)
    first = list(source)
    second = list(source)

    assert first == second
    assert samples == ([1, 2, 3], [4, 5, 6])


def test_synthetic_source_hash_is_independent_of_packetization_and_stream_label():
    source_a = SyntheticSource(
        ((1, 2, 3), (4, 5, 6)),
        sample_rate=1000,
        packet_samples=1,
        stream_id="stream-a",
        channel_ids=("left", "right"),
    )
    source_b = SyntheticSource(
        ((1, 2, 3), (4, 5, 6)),
        sample_rate=1000,
        packet_samples=2,
        stream_id="stream-a",
        channel_ids=("left", "right"),
    )
    source_c = SyntheticSource(
        ((1, 2, 3), (4, 5, 6)),
        sample_rate=1000,
        packet_samples=1,
        stream_id="stream-b",
        channel_ids=("left", "right"),
    )

    assert source_a.content_sha256 == source_b.content_sha256 == source_c.content_sha256
    assert source_a.identity() == source_b.identity()
    assert source_a.identity() != source_c.identity()


def test_synthetic_source_hash_is_sensitive_to_content_and_source_semantics():
    source = SyntheticSource(
        ((1, 2, 3), (4, 5, 6)),
        sample_rate=1000,
        packet_samples=1,
        stream_id="stream-a",
        channel_ids=("left", "right"),
    )
    different_content = SyntheticSource(
        ((1, 2, 4), (4, 5, 6)),
        sample_rate=1000,
        packet_samples=1,
        stream_id="stream-a",
        channel_ids=("left", "right"),
    )
    different_rate = SyntheticSource(
        ((1, 2, 3), (4, 5, 6)),
        sample_rate=2000,
        packet_samples=1,
        stream_id="stream-a",
        channel_ids=("left", "right"),
    )
    different_channel_order = SyntheticSource(
        ((1, 2, 3), (4, 5, 6)),
        sample_rate=1000,
        packet_samples=1,
        stream_id="stream-a",
        channel_ids=("right", "left"),
    )
    repeated_construction = SyntheticSource(
        ((1, 2, 3), (4, 5, 6)),
        sample_rate=1000,
        packet_samples=1,
        stream_id="stream-a",
        channel_ids=("left", "right"),
    )

    assert source.content_sha256 != different_content.content_sha256
    assert source.content_sha256 != different_rate.content_sha256
    assert source.content_sha256 != different_channel_order.content_sha256
    assert source.content_sha256 == repeated_construction.content_sha256


def test_synthetic_source_rejects_unhashable_sample_type():
    with pytest.raises(ContractValidationError, match="unsupported type"):
        SyntheticSource(
            ((object(),),), sample_rate=1000, packet_samples=1, stream_id="hash"
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sample_rate": 0, "packet_samples": 1}, "sample_rate"),
        ({"sample_rate": 1000, "packet_samples": 0}, "packet_samples"),
        ({"sample_rate": 1000, "packet_samples": 1, "samples": ([1], [2, 3])}, "frames"),
        ({"sample_rate": 1000, "packet_samples": 1, "samples": ()}, "at least one channel"),
        ({"sample_rate": 1000, "packet_samples": 1, "samples": ([],)}, "at least one sample"),
    ],
)
def test_synthetic_source_rejects_invalid_input(kwargs, message):
    values = {
        "samples": ([1, 2],),
        "sample_rate": 1000,
        "packet_samples": 1,
    }
    values.update(kwargs)
    with pytest.raises(ContractValidationError, match=message):
        SyntheticSource(**values)


def test_fake_clock_is_deterministic_and_monotonic():
    clock = FakeClock()
    clock.advance_to(20)
    clock.advance_to(20)
    assert clock.current_sample == 20
    with pytest.raises(ContractValidationError, match="backwards"):
        clock.advance_to(19)
