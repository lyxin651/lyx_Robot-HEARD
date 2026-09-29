from dataclasses import dataclass

import pytest

from robot_heard.replay.events import ContractValidationError
from robot_heard.replay.oracle import Stage2OracleReleaseGate


@dataclass(frozen=True)
class _Row:
    segment_id: str
    start_sample: int
    end_sample: int


def test_stage2_gate_releases_only_at_first_available_packet_and_is_descriptor_only():
    gate = Stage2OracleReleaseGate(
        [_Row("seg-b", 100, 500), _Row("seg-a", 500, 640)],
        sample_rate=16000,
        packet_duration_sec=0.02,
    )
    assert gate.on_source_available(320, 0.02) == ()
    releases = gate.on_source_available(640, 0.04)
    assert [release.descriptor.segment_id for release in releases] == ["seg-b", "seg-a"]
    assert releases[0].segment_end_available_wall_offset_sec == pytest.approx(0.03125)
    assert releases[0].oracle_release_quantization_sec == pytest.approx(0.00875)
    descriptors = gate.poll_released()
    assert descriptors[0].end_sample == 500
    assert not hasattr(descriptors[0], "reference")
    assert not hasattr(descriptors[0], "speaker_id")
    assert gate.release_count == 2
    assert gate.remaining_count == 0
    assert gate.poll_released() == ()


def test_stage2_gate_rejects_duplicate_ids_and_large_quantization():
    with pytest.raises(ContractValidationError, match="duplicate"):
        Stage2OracleReleaseGate(
            [_Row("same", 0, 10), _Row("same", 10, 20)], sample_rate=1000
        )
    gate = Stage2OracleReleaseGate(
        [_Row("seg", 0, 10)], sample_rate=1000, packet_duration_sec=0.01
    )
    with pytest.raises(ContractValidationError, match="quantization"):
        gate.on_source_available(10, 0.03)


def test_gate_release_order_is_end_then_segment_id():
    gate = Stage2OracleReleaseGate(
        [_Row("z", 0, 20), _Row("a", 10, 20), _Row("m", 0, 10)],
        sample_rate=1000,
    )
    releases = gate.on_source_available(20, 0.02)
    assert [item.descriptor.segment_id for item in releases] == ["m", "a", "z"]


def test_future_timeline_perturbation_cannot_change_prefix_releases():
    prefix_a = Stage2OracleReleaseGate(
        [_Row("prefix", 0, 10), _Row("future-a", 20, 30)], sample_rate=1000
    )
    prefix_b = Stage2OracleReleaseGate(
        [_Row("prefix", 0, 10), _Row("future-b", 20, 40)], sample_rate=1000
    )
    first_a = prefix_a.on_source_available(10, 0.01)
    first_b = prefix_b.on_source_available(10, 0.01)
    assert [item.descriptor for item in first_a] == [item.descriptor for item in first_b]
    assert prefix_a.poll_released() == prefix_b.poll_released()
