from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class ASRResult:
    """Minimal backend-neutral transcription result for V0."""

    text: str
    language: Optional[str]
    decode_sec: float


class ASRBackend(ABC):
    """Backend-neutral ASR interface.

    Implementations must load model state during initialization and reuse that
    state across calls to :meth:`transcribe`.
    """

    @abstractmethod
    def transcribe(self, audio_path: Union[str, Path]) -> ASRResult:
        """Transcribe one target waveform."""
        raise NotImplementedError
