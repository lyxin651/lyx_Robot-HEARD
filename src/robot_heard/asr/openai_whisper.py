from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

from .base import ASRBackend, ASRResult


class OpenAIWhisperBackend(ASRBackend):
    """Reference backend using the official OpenAI Whisper package.

    V0 intentionally keeps decoding options small and explicit. The model is
    loaded exactly once during construction and reused for all transcriptions.
    """

    def __init__(
        self,
        *,
        model_name: str,
        language: str,
        task: str,
        device: str,
        fp16: bool,
        temperature: float,
        download_root: Optional[str] = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be a non-empty string")
        if not language:
            raise ValueError("language must be a non-empty string")
        if task not in {"transcribe", "translate"}:
            raise ValueError("task must be either 'transcribe' or 'translate'")
        if device == "cpu" and fp16:
            raise ValueError("fp16=true is incompatible with device='cpu'; set fp16=false explicitly")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise TypeError("temperature must be a numeric value")

        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Whisper is not installed. Run `python -m pip install -r requirements.txt`."
            ) from exc

        self.model_name = model_name
        self.language = language
        self.task = task
        self.device = device
        self.fp16 = fp16
        self.temperature = float(temperature)
        self.download_root = download_root

        load_kwargs = {"device": device}
        if download_root:
            load_kwargs["download_root"] = download_root

        self._model = whisper.load_model(model_name, **load_kwargs)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def transcribe(self, audio_path: Union[str, Path]) -> ASRResult:
        path = Path(audio_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"audio file does not exist: {path}")

        start = time.perf_counter()
        result = self._model.transcribe(
            str(path),
            language=self.language,
            task=self.task,
            temperature=self.temperature,
            fp16=self.fp16,
            verbose=False,
        )
        decode_sec = time.perf_counter() - start

        text = result.get("text")
        if not isinstance(text, str):
            raise RuntimeError("Whisper result did not contain a string 'text' field")

        detected_language = result.get("language")
        if detected_language is not None and not isinstance(detected_language, str):
            detected_language = str(detected_language)

        return ASRResult(
            text=text,
            language=detected_language or self.language,
            decode_sec=decode_sec,
        )
