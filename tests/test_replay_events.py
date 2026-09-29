from __future__ import annotations

import pytest

from robot_heard.replay.events import (
    AudioChunk,
    ConsumerCapabilities,
    ContractValidationError,
    HypothesisEvent,
    HypothesisKind,
)


def _valid_chunk(**overrides):
    values = {
        "stream_id": "s",
        "sequence_id": 0,
        "start_sample": 0,
        "end_sample": 2,
        "sample_rate": 1000,
        "channel_ids": ("ch0", "ch1"),
        "samples": ((1, 2), (3, 4)),
        "is_final_source_chunk": True,
        "scheduled_availability_logical_sample": 2,
    }
    values.update(overrides)
    return AudioChunk(**values)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"start_sample": -1}, "start_sample"),
        ({"end_sample": 0}, "end_sample"),
        ({"sample_rate": 0}, "sample_rate"),
        ({"channel_ids": ()}, "channel_ids"),
        ({"samples": ((1, 2),)}, "channel count"),
        ({"samples": ((1, 2), (3,))}, "frames"),
        ({"samples": ((1,), (2,))}, "frame count"),
        ({"sequence_id": -1}, "sequence_id"),
    ],
)
def test_audio_chunk_fails_closed(overrides, message):
    with pytest.raises(ContractValidationError, match=message):
        _valid_chunk(**overrides)


def test_audio_chunk_preserves_multi_channel_shape_without_downmix():
    chunk = _valid_chunk()

    assert chunk.num_channels == 2
    assert chunk.frame_count == 2
    assert chunk.channel_ids == ("ch0", "ch1")
    assert chunk.samples == ((1, 2), (3, 4))


def test_capabilities_have_explicit_false_defaults():
    capabilities = ConsumerCapabilities()

    assert capabilities.to_dict() == {
        "supports_partial": False,
        "supports_revision": False,
        "supports_stable_prefix": False,
        "supports_endpoint": False,
        "supports_token_timestamps": False,
        "supports_word_timestamps": False,
        "reports_consumed_position": False,
        "native_streaming": False,
        "stateful": False,
    }


def test_hypothesis_snapshot_is_not_a_delta():
    event = HypothesisEvent(
        stream_id="s",
        scope_id="scope",
        event_id="e0",
        revision_index=0,
        text="complete snapshot",
        kind=HypothesisKind.PARTIAL,
    )

    assert event.text == "complete snapshot"
    assert event.revision_index == 0
