from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from robot_heard.replay.events import ContractValidationError
from robot_heard.replay.runtime import DeterministicReplayRuntime
from robot_heard.replay.wav_source import MultiChannelPcmWavSource, WavSourceError

from replay_helpers import RecordingConsumer
from test_replay_runtime import _writer


def _write_mono_wav(
    path: Path,
    values,
    *,
    sample_rate: int = 1000,
    sample_width: int = 4,
) -> Path:
    if sample_width == 1:
        payload = bytes((int(value) + 128) % 256 for value in values)
    elif sample_width == 2:
        payload = struct.pack("<" + "h" * len(values), *values)
    elif sample_width == 4:
        payload = struct.pack("<" + "i" * len(values), *values)
    else:
        raise ValueError("test helper supports PCM8/16/32")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)
    return path


def _pair(tmp_path: Path, *, sample_rate=1000, sample_width=4, right_values=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    left = _write_mono_wav(
        tmp_path / "left.wav",
        (1, 2, 3, 4, 5),
        sample_rate=sample_rate,
        sample_width=sample_width,
    )
    right = _write_mono_wav(
        tmp_path / "right.wav",
        right_values if right_values is not None else (101, 102, 103, 104, 105),
        sample_rate=sample_rate,
        sample_width=sample_width,
    )
    return left, right


def _source(paths, *, packet_samples=2, stream_id="stream"):
    return MultiChannelPcmWavSource(
        paths,
        channel_ids=("left", "right"),
        stream_id=stream_id,
        packet_samples=packet_samples,
    )


def test_pcm32_packetization_tail_alignment_and_no_downmix(tmp_path):
    paths = _pair(tmp_path)
    source = _source(paths, packet_samples=2)

    chunks = list(source)

    assert [(chunk.start_sample, chunk.end_sample) for chunk in chunks] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert chunks[-1].is_final_source_chunk is True
    assert chunks[-1].frame_count == 1
    assert [chunk.samples for chunk in chunks] == [
        ((1, 2), (101, 102)),
        ((3, 4), (103, 104)),
        ((5,), (105,)),
    ]
    assert all(chunk.num_channels == 2 for chunk in chunks)


def test_wav_source_consumes_generic_replay_runtime(tmp_path):
    paths = _pair(tmp_path)
    source = _source(paths, packet_samples=2)
    consumer = RecordingConsumer()
    writer, trace_path, _ = _writer(tmp_path / "run", source, consumer)

    result = DeterministicReplayRuntime(source, consumer, writer).run()

    assert result.source_end_sample == 5
    assert [(chunk.start_sample, chunk.end_sample) for chunk in consumer.chunks] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert trace_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"right_sample_rate": 2000}, "sample rate"),
        ({"right_values": (101, 102)}, "frame count"),
        ({"right_sample_width": 2}, "sample width"),
    ],
)
def test_wav_source_rejects_mismatched_channel_headers(tmp_path, kwargs, message):
    left = _write_mono_wav(tmp_path / "left.wav", (1, 2, 3, 4, 5))
    right = _write_mono_wav(
        tmp_path / "right.wav",
        kwargs.get("right_values", (101, 102, 103, 104, 105)),
        sample_rate=kwargs.get("right_sample_rate", 1000),
        sample_width=kwargs.get("right_sample_width", 4),
    )

    with pytest.raises(WavSourceError, match=message):
        _source((left, right))


def test_wav_source_rejects_path_and_channel_id_mismatch(tmp_path):
    paths = _pair(tmp_path)

    with pytest.raises(ContractValidationError, match="count"):
        MultiChannelPcmWavSource(
            paths[:1],
            channel_ids=("left", "right"),
            stream_id="stream",
            packet_samples=2,
        )
    with pytest.raises(ContractValidationError, match="unique"):
        MultiChannelPcmWavSource(
            paths,
            channel_ids=("same", "same"),
            stream_id="stream",
            packet_samples=2,
        )
    with pytest.raises(FileNotFoundError, match="regular file"):
        MultiChannelPcmWavSource(
            (paths[0], tmp_path / "missing.wav"),
            channel_ids=("left", "right"),
            stream_id="stream",
            packet_samples=2,
        )


def test_wav_source_hash_is_independent_of_packet_size_and_stream_id(tmp_path):
    paths = _pair(tmp_path)
    source_a = _source(paths, packet_samples=1, stream_id="stream-a")
    source_b = _source(paths, packet_samples=3, stream_id="stream-b")

    assert source_a.content_sha256 == source_b.content_sha256
    assert source_a.identity() != source_b.identity()


def test_wav_source_hash_changes_for_content_and_channel_order(tmp_path):
    paths = _pair(tmp_path)
    changed_paths = _pair(tmp_path / "changed", right_values=(201, 202, 203, 204, 205))
    source = _source(paths)
    changed = _source(changed_paths)
    reordered = _source((paths[1], paths[0]))

    assert source.content_sha256 != changed.content_sha256
    assert source.content_sha256 != reordered.content_sha256


def test_wav_source_repeated_construction_and_iteration_are_deterministic(tmp_path):
    paths = _pair(tmp_path)
    source_a = _source(paths, packet_samples=2)
    source_b = _source(paths, packet_samples=2)

    assert source_a.content_sha256 == source_b.content_sha256
    assert list(source_a) == list(source_b)
    assert list(source_a) == list(source_a)


def test_wav_source_closes_handles_after_decode_error(tmp_path):
    paths = _pair(tmp_path)
    source = _source(paths, packet_samples=2)
    original = paths[1].read_bytes()
    paths[1].write_bytes(original[:-1])

    with pytest.raises(WavSourceError, match="short PCM packet"):
        list(source)

    # The failed generator has exited its ExitStack; the file remains usable.
    with wave.open(str(paths[0]), "rb") as handle:
        assert handle.getnframes() == 5
