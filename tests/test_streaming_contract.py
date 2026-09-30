from __future__ import annotations

from robot_heard.asr.base import ASRBackend
from robot_heard.streaming.base import StreamingConsumer

from replay_helpers import RecordingConsumer


def test_streaming_consumer_is_parallel_to_offline_asr_backend():
    assert not issubclass(StreamingConsumer, ASRBackend)
    assert isinstance(RecordingConsumer(), StreamingConsumer)


def test_reset_is_explicit_and_isolates_consumer_state():
    consumer = RecordingConsumer()
    consumer.reset()

    assert consumer.reset_count == 1
    assert consumer.chunks == []
    assert consumer.metadata is None
