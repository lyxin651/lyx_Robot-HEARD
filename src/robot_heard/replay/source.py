"""Synthetic integer-sample replay source for R1."""

from __future__ import annotations

from typing import Any, Iterator, Optional, Sequence, Tuple

from robot_heard.replay.events import (
    AudioChunk,
    ContractValidationError,
    StreamMetadata,
)


class SyntheticSource:
    """Produce exact, deterministic packets from channel-major sample data.

    The source stores an immutable tuple copy of the supplied synthetic data.
    It never pads, truncates, resamples, or combines channels.
    """

    def __init__(
        self,
        samples: Sequence[Sequence[Any]],
        *,
        sample_rate: int,
        packet_samples: int,
        stream_id: str = "synthetic",
        channel_ids: Optional[Sequence[str]] = None,
        sample_format: str = "synthetic",
        channel_layout: Optional[str] = None,
    ) -> None:
        if isinstance(samples, (str, bytes)):
            raise ContractValidationError("samples must be a channel-major sequence")
        try:
            normalized = tuple(tuple(channel) for channel in samples)
        except TypeError as exc:
            raise ContractValidationError("samples must be a channel-major sequence") from exc
        if not normalized:
            raise ContractValidationError("synthetic source requires at least one channel")
        frame_count = len(normalized[0])
        if frame_count <= 0:
            raise ContractValidationError("synthetic source requires at least one sample")
        for index, channel in enumerate(normalized):
            if len(channel) != frame_count:
                raise ContractValidationError(
                    f"source channel {index} has {len(channel)} frames; expected {frame_count}"
                )
        if isinstance(packet_samples, bool) or not isinstance(packet_samples, int):
            raise ContractValidationError("packet_samples must be a positive integer")
        if packet_samples <= 0:
            raise ContractValidationError("packet_samples must be a positive integer")

        if channel_ids is None:
            channel_ids = tuple(f"ch{index}" for index in range(len(normalized)))
        channel_ids_tuple = tuple(channel_ids)
        self._metadata = StreamMetadata(
            stream_id=stream_id,
            sample_rate=sample_rate,
            channel_ids=channel_ids_tuple,
            sample_format=sample_format,
            channel_layout=channel_layout,
        )
        if len(channel_ids_tuple) != len(normalized):
            raise ContractValidationError(
                "channel count must equal channel_ids count"
            )
        self._samples: Tuple[Tuple[Any, ...], ...] = normalized
        self.packet_samples = packet_samples

    @property
    def metadata(self) -> StreamMetadata:
        return self._metadata

    @property
    def total_samples(self) -> int:
        return len(self._samples[0])

    @property
    def num_channels(self) -> int:
        return len(self._samples)

    def identity(self) -> str:
        """Return a stable synthetic source identity without filesystem access."""

        return f"synthetic:{self._metadata.stream_id}:{self.total_samples}"

    def chunks(self) -> Iterator[AudioChunk]:
        """Yield adjacent half-open packets using only integer boundaries."""

        sequence_id = 0
        for start_sample in range(0, self.total_samples, self.packet_samples):
            end_sample = min(start_sample + self.packet_samples, self.total_samples)
            packet = tuple(
                channel[start_sample:end_sample] for channel in self._samples
            )
            yield AudioChunk(
                stream_id=self._metadata.stream_id,
                sequence_id=sequence_id,
                start_sample=start_sample,
                end_sample=end_sample,
                sample_rate=self._metadata.sample_rate,
                channel_ids=self._metadata.channel_ids,
                samples=packet,
                is_final_source_chunk=end_sample == self.total_samples,
                scheduled_availability_logical_sample=end_sample,
            )
            sequence_id += 1

    def __iter__(self) -> Iterator[AudioChunk]:
        return self.chunks()
