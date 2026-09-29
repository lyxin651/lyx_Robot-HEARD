from dataclasses import dataclass

import pytest

from robot_heard.replay.events import ContractValidationError
from robot_heard.replay.oracle import (
    ScopeEndRelease,
    ScopeStartRelease,
    Stage2ScopeBoundaryReleaseGate,
)


@dataclass(frozen=True)
class _Row:
    segment_id: str
    start_sample: int
    end_sample: int


def test_start_release_has_no_end_and_end_release_is_causal():
    gate = Stage2ScopeBoundaryReleaseGate(
        [_Row("seg", 100, 500)], sample_rate=1000, packet_duration_sec=0.1
    )
    assert gate.on_source_available(99, 0.099) == ()
    start = gate.on_source_available(100, 0.1)
    assert len(start) == 1
    assert isinstance(start[0], ScopeStartRelease)
    assert start[0].segment_id == "seg"
    assert start[0].start_sample == 100
    assert not hasattr(start[0], "end_sample")
    assert gate.remaining_count == 1
    assert gate.on_source_available(499, 0.499) == ()
    end = gate.on_source_available(500, 0.5)
    assert len(end) == 1
    assert isinstance(end[0], ScopeEndRelease)
    assert end[0].end_sample == 500
    assert not hasattr(end[0], "start_sample")
    assert gate.remaining_count == 0


def test_overlapping_boundaries_have_deterministic_order():
    gate = Stage2ScopeBoundaryReleaseGate(
        [_Row("b", 100, 300), _Row("a", 100, 200), _Row("c", 200, 400)],
        sample_rate=1000,
    )
    releases = gate.on_source_available(400, 0.4)
    assert [(type(item).__name__, item.segment_id) for item in releases] == [
        ("ScopeStartRelease", "a"),
        ("ScopeStartRelease", "b"),
        ("ScopeStartRelease", "c"),
        ("ScopeEndRelease", "a"),
        ("ScopeEndRelease", "b"),
        ("ScopeEndRelease", "c"),
    ]


def test_future_schedule_perturbation_does_not_change_start_prefix():
    gate_a = Stage2ScopeBoundaryReleaseGate(
        [_Row("prefix", 100, 200), _Row("future-a", 400, 500)], sample_rate=1000
    )
    gate_b = Stage2ScopeBoundaryReleaseGate(
        [_Row("prefix", 100, 200), _Row("future-b", 800, 900)], sample_rate=1000
    )
    first_a = gate_a.on_source_available(100, 0.1)
    first_b = gate_b.on_source_available(100, 0.1)
    assert first_a == first_b
    assert first_a[0].start_sample == 100
    assert not hasattr(first_a[0], "end_sample")


def test_scope_gate_rejects_duplicate_and_invalid_rows():
    with pytest.raises(ContractValidationError, match="duplicate"):
        Stage2ScopeBoundaryReleaseGate(
            [_Row("same", 0, 10), _Row("same", 10, 20)], sample_rate=1000
        )
    with pytest.raises(ContractValidationError, match="greater"):
        Stage2ScopeBoundaryReleaseGate([_Row("bad", 10, 10)], sample_rate=1000)
