"""Streaming multi-channel PCM WAV replay source for R2."""

from __future__ import annotations

import hashlib
import struct
import wave
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

from robot_heard.replay.events import AudioChunk, ContractValidationError, StreamMetadata


class WavSourceError(ContractValidationError):
    """Raised when a replay WAV source violates the R2 source contract."""


@dataclass(frozen=True)
class _WavMetadata:
    channels: int
    sample_rate: int
    sample_width: int
    frame_count: int
    compression_type: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_field(digest: "hashlib._Hash", label: str, value: object) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(label.encode("ascii"))
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _decode_pcm(raw: bytes, sample_width: int, frame_count: int) -> Tuple[int, ...]:
    expected_bytes = frame_count * sample_width
    if len(raw) != expected_bytes:
        raise WavSourceError(
            f"short PCM packet: received {len(raw)} bytes, expected {expected_bytes}"
        )
    if sample_width == 1:
        # PCM8 WAV samples are unsigned; expose a signed integer sample shape.
        return tuple(value - 128 for value in raw)
    if sample_width in (2, 4):
        format_code = "h" if sample_width == 2 else "i"
        return struct.unpack(f"<{format_code * frame_count}", raw)
    if sample_width == 3:
        values = []
        for offset in range(0, len(raw), 3):
            value = raw[offset] | (raw[offset + 1] << 8) | (raw[offset + 2] << 16)
            if value & 0x800000:
                value -= 0x1000000
            values.append(value)
        return tuple(values)
    raise WavSourceError(f"unsupported PCM sample width: {sample_width} bytes")


class MultiChannelPcmWavSource:
    """Stream synchronized separated-mono PCM WAV files as AudioChunks.

    The source validates all headers at construction but opens audio handles
    only while iterating.  Packetization uses integer frame boundaries and
    never resamples, pads, truncates, averages, or downmixes channels.
    """

    def __init__(
        self,
        channel_paths: Sequence[Path | str],
        *,
        channel_ids: Sequence[str],
        stream_id: str,
        packet_samples: int,
    ) -> None:
        if isinstance(channel_paths, (str, bytes)):
            raise WavSourceError("channel_paths must be an ordered sequence")
        if isinstance(channel_ids, (str, bytes)):
            raise WavSourceError("channel_ids must be an ordered sequence")
        try:
            normalized_paths = tuple(
                Path(path).expanduser().resolve(strict=False) for path in channel_paths
            )
            normalized_ids = tuple(channel_ids)
        except (TypeError, ValueError) as exc:
            raise WavSourceError("channel_paths and channel_ids must be sequences") from exc
        if not normalized_paths:
            raise WavSourceError("at least one channel path is required")
        if len(normalized_paths) != len(normalized_ids):
            raise WavSourceError("channel_paths count must equal channel_ids count")
        if isinstance(packet_samples, bool) or not isinstance(packet_samples, int):
            raise WavSourceError("packet_samples must be a positive integer")
        if packet_samples <= 0:
            raise WavSourceError("packet_samples must be a positive integer")

        self._metadata_by_path = tuple(
            self._probe(path, index) for index, path in enumerate(normalized_paths)
        )
        reference = self._metadata_by_path[0]
        for index, metadata in enumerate(self._metadata_by_path[1:], start=1):
            if metadata.sample_rate != reference.sample_rate:
                raise WavSourceError(
                    f"channel {index} sample rate {metadata.sample_rate} does not match "
                    f"channel 0 sample rate {reference.sample_rate}"
                )
            if metadata.sample_width != reference.sample_width:
                raise WavSourceError(
                    f"channel {index} sample width {metadata.sample_width} does not match "
                    f"channel 0 sample width {reference.sample_width}"
                )
            if metadata.frame_count != reference.frame_count:
                raise WavSourceError(
                    f"channel {index} frame count {metadata.frame_count} does not match "
                    f"channel 0 frame count {reference.frame_count}"
                )
            if metadata.compression_type != reference.compression_type:
                raise WavSourceError(
                    f"channel {index} compression {metadata.compression_type!r} does not "
                    f"match channel 0 compression {reference.compression_type!r}"
                )
        if reference.frame_count <= 0:
            raise WavSourceError("WAV source must contain at least one frame")
        if reference.sample_width not in (1, 2, 3, 4):
            raise WavSourceError(
                f"unsupported PCM sample width: {reference.sample_width} bytes"
            )

        self._channel_paths = normalized_paths
        self._channel_ids = normalized_ids
        self._packet_samples = packet_samples
        sample_format = (
            "pcm_u8" if reference.sample_width == 1 else f"pcm_s{reference.sample_width * 8}le"
        )
        self._metadata = StreamMetadata(
            stream_id=stream_id,
            sample_rate=reference.sample_rate,
            channel_ids=normalized_ids,
            sample_format=sample_format,
            channel_layout="separate_mono_files",
        )
        self._sample_width = reference.sample_width
        self._compression_type = reference.compression_type
        self._total_samples = reference.frame_count
        self._channel_hashes = tuple(_sha256_file(path) for path in self._channel_paths)
        self._content_sha256 = self._calculate_content_sha256()

    @staticmethod
    def _probe(path: Path, index: int) -> _WavMetadata:
        if not path.is_file():
            raise FileNotFoundError(f"channel {index} is not a regular file: {path}")
        try:
            with wave.open(str(path), "rb") as handle:
                metadata = _WavMetadata(
                    channels=handle.getnchannels(),
                    sample_rate=handle.getframerate(),
                    sample_width=handle.getsampwidth(),
                    frame_count=handle.getnframes(),
                    compression_type=handle.getcomptype(),
                )
        except (EOFError, OSError, wave.Error) as exc:
            raise WavSourceError(f"invalid WAV channel {index}: {path}: {exc}") from exc
        if metadata.compression_type != "NONE":
            raise WavSourceError(
                f"channel {index} must be uncompressed PCM; "
                f"got {metadata.compression_type!r}"
            )
        if metadata.sample_rate <= 0 or metadata.sample_width <= 0:
            raise WavSourceError(f"channel {index} has invalid WAV format: {path}")
        if metadata.channels != 1:
            raise WavSourceError(
                f"channel {index} must be mono; got {metadata.channels} channels"
            )
        return metadata

    def _calculate_content_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"robot-heard.multi-channel-pcm-wav.v1\n")
        reference = self._metadata_by_path[0]
        _hash_field(digest, "sample_rate", reference.sample_rate)
        _hash_field(digest, "sample_width", reference.sample_width)
        _hash_field(digest, "compression_type", reference.compression_type)
        _hash_field(digest, "channel_layout", "separate_mono_files")
        _hash_field(digest, "channel_count", len(self._channel_ids))
        for index, (channel_id, file_hash) in enumerate(
            zip(self._channel_ids, self._channel_hashes)
        ):
            _hash_field(digest, f"channel_id[{index}]", channel_id)
            _hash_field(digest, f"channel_file_sha256[{index}]", file_hash)
        return digest.hexdigest()

    @property
    def metadata(self) -> StreamMetadata:
        return self._metadata

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def packet_samples(self) -> int:
        return self._packet_samples

    @property
    def content_sha256(self) -> str:
        return self._content_sha256

    @property
    def sample_width(self) -> int:
        return self._sample_width

    @property
    def compression_type(self) -> str:
        return self._compression_type

    @property
    def channel_paths(self) -> Tuple[Path, ...]:
        return self._channel_paths

    @property
    def channel_ids(self) -> Tuple[str, ...]:
        return self._channel_ids

    @property
    def channel_hashes(self) -> Tuple[str, ...]:
        return self._channel_hashes

    def identity(self) -> str:
        """Return a readable identity without packet/run configuration."""

        return (
            f"wav:{self._metadata.stream_id}:{self.total_samples}:"
            f"sha256:{self.content_sha256}"
        )

    def chunks(self) -> Iterator[AudioChunk]:
        """Yield synchronized packets while keeping only one packet in memory."""

        with ExitStack() as stack:
            readers = [
                stack.enter_context(wave.open(str(path), "rb"))
                for path in self._channel_paths
            ]
            start_sample = 0
            sequence_id = 0
            while start_sample < self.total_samples:
                end_sample = min(start_sample + self.packet_samples, self.total_samples)
                frame_count = end_sample - start_sample
                samples = []
                for index, reader in enumerate(readers):
                    raw = reader.readframes(frame_count)
                    try:
                        samples.append(_decode_pcm(raw, self.sample_width, frame_count))
                    except (struct.error, ValueError) as exc:
                        raise WavSourceError(
                            f"channel {index} failed decoding packet "
                            f"[{start_sample},{end_sample}): {exc}"
                        ) from exc
                yield AudioChunk(
                    stream_id=self.metadata.stream_id,
                    sequence_id=sequence_id,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    sample_rate=self.metadata.sample_rate,
                    channel_ids=self.metadata.channel_ids,
                    samples=tuple(samples),
                    is_final_source_chunk=end_sample == self.total_samples,
                    scheduled_availability_logical_sample=end_sample,
                )
                start_sample = end_sample
                sequence_id += 1

    def __iter__(self) -> Iterator[AudioChunk]:
        return self.chunks()
