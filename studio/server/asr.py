"""Local speech recognition for verification: Whisper base.en via transformers.

The weights are already in the Hugging Face cache (`openai/whisper-base.en`), and the model
is loaded with `HF_HUB_OFFLINE=1`, so nothing is downloaded. It returns word timestamps,
which the talking-head sync uses. It is a checker, not a source of truth.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

MODEL_ID = "openai/whisper-base.en"


class ASRError(Exception):
    pass


class LocalASR:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._pipe = None
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        from huggingface_hub import try_to_load_from_cache

        try:
            hit = try_to_load_from_cache(self.model_id, "model.safetensors")
        except Exception:
            hit = None
        return (True, "cached") if isinstance(hit, str) else (False, f"{self.model_id} is not in the local cache")

    def _load(self):
        if self._pipe is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import pipeline

            try:
                self._pipe = pipeline("automatic-speech-recognition", model=self.model_id, device="cpu")
            except Exception as exc:
                raise ASRError(f"The local speech recogniser could not load: {exc.__class__.__name__}: {exc}") from exc
        return self._pipe

    def transcribe(self, audio: Path) -> dict:
        import librosa

        try:
            wav, _ = librosa.load(audio, sr=16000, mono=True)
        except Exception as exc:
            raise ASRError(f"Could not read {audio.name} for transcription: {exc}") from exc
        with self._lock:
            pipe = self._load()
            out = pipe({"raw": wav, "sampling_rate": 16000}, return_timestamps="word")
        chunks = [{"word": c["text"].strip(), "start": float(c["timestamp"][0] or 0.0),
                   "end": float(c["timestamp"][1] if c["timestamp"][1] is not None else c["timestamp"][0] or 0.0)}
                  for c in out.get("chunks") or [] if c["text"].strip()]
        return {"model": self.model_id, "text": out.get("text", "").strip(), "words": chunks,
                "duration": round(len(wav) / 16000, 3)}
