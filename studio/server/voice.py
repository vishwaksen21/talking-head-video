"""The existing Chatterbox installation, used for text-to-speech in a reference voice.

This is the same `chatterbox.tts.ChatterboxTTS` the repo's Gradio app uses, loaded from
the same Hugging Face cache, conditioned on a reference clip through
`generate(..., audio_prompt_path=...)` - Chatterbox's own zero-shot cloning mechanism.
The speaker in the generated video is never used as the target: its audio only supplies
the words (via transcription) and the timing.

The model is loaded once per process and guarded by a lock, because one MPS model must
not run two generations at once.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("studio.voice")

MAX_CHUNK_CHARS = 280  # Chatterbox degrades on very long inputs; split at sentence/clause edges
INTRA_GAP_S = 0.12     # pause inserted between chunks of one transcript segment
BUILTIN_VOICE = "builtin"


class VoiceError(Exception):
    pass


def pick_device(preference: str = "auto") -> str:
    import torch

    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Chunks of at most `limit` chars, as few as possible, broken at sentence edges first,
    then clause edges, then word edges. Short text stays one chunk (one Chatterbox call,
    so prosody runs across sentences). Words are never cut and order is preserved.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        if len(sentence) <= limit:
            pieces.append(sentence)
            continue
        for part in re.split(r"(?<=[,;:—])\s+", sentence):
            if len(part) <= limit:
                pieces.append(part)
                continue
            buf = ""
            for w in part.split(" "):
                if buf and len(buf) + 1 + len(w) > limit:
                    pieces.append(buf)
                    buf = w
                else:
                    buf = f"{buf} {w}".strip()
            if buf:
                pieces.append(buf)
    chunks: list[str] = []
    for piece in pieces:  # re-pack greedily up to the limit
        if chunks and len(chunks[-1]) + 1 + len(piece) <= limit:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    return chunks


def trim_silence(wav: np.ndarray, sr: int, *, margin_s: float = 0.04) -> np.ndarray:
    """Drop leading/trailing near-silence so placement uses the real speech edges."""
    if wav.size == 0:
        return wav
    frame = max(1, int(sr * 0.01))
    n = wav.size // frame
    if n == 0:
        return wav
    energy = np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    threshold = max(float(energy.max()) * 0.03, 1e-4)
    voiced = np.nonzero(energy > threshold)[0]
    if voiced.size == 0:
        return wav
    margin = int(sr * margin_s)
    start = max(0, voiced[0] * frame - margin)
    end = min(wav.size, (voiced[-1] + 1) * frame + margin)
    return wav[start:end]


class ChatterboxVoice:
    def __init__(self, device: str = "auto"):
        self._device_pref = device
        self._model = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()
        self.load_error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device or pick_device(self._device_pref)

    @property
    def sample_rate(self) -> int:
        return int(getattr(self._model, "sr", 24000))

    @staticmethod
    def installed() -> tuple[bool, str]:
        try:
            import chatterbox  # noqa: F401
            from chatterbox.tts import ChatterboxTTS  # noqa: F401
        except Exception as exc:  # report the real reason (e.g. missing pkg_resources)
            return False, f"{exc.__class__.__name__}: {exc}"
        try:
            import perth

            if perth.PerthImplicitWatermarker is None:
                return False, "resemble-perth watermarker failed to import (needs setuptools<81)."
        except Exception as exc:
            return False, f"resemble-perth: {exc}"
        return True, "ok"

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        ok, why = self.installed()
        if not ok:
            raise VoiceError(f"Chatterbox is not available: {why}")
        from chatterbox.tts import ChatterboxTTS

        device = pick_device(self._device_pref)
        log.info("loading ChatterboxTTS on %s", device)
        try:
            self._model = ChatterboxTTS.from_pretrained(device)
        except Exception as exc:
            self.load_error = f"{exc.__class__.__name__}: {exc}"
            raise VoiceError(f"Chatterbox model failed to load: {self.load_error}") from exc
        self._device = device
        self.load_error = None
        return self._model

    def synthesize(
        self,
        text: str,
        reference: Optional[Path],
        *,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> np.ndarray:
        """Speak `text` in the reference voice; returns float32 mono at `sample_rate`."""
        if reference is not None and not reference.exists():
            raise VoiceError(f"Reference voice file is missing: {reference.name}")
        chunks = split_text(text)
        if not chunks:
            raise VoiceError("There is no text to speak.")
        with self._lock:
            model = self._ensure_loaded()
            sr = self.sample_rate
            pieces: list[np.ndarray] = []
            for chunk in chunks:
                if should_stop():
                    raise VoiceError("Cancelled.")
                try:
                    wav = model.generate(
                        chunk,
                        audio_prompt_path=str(reference) if reference else None,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        self._free_memory()
                        raise VoiceError(
                            "Ran out of memory while synthesising speech. Close other apps and retry."
                        ) from exc
                    raise VoiceError(f"Chatterbox failed on a sentence: {exc}") from exc
                audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
                if not np.isfinite(audio).all() or audio.size == 0:
                    raise VoiceError("Chatterbox produced invalid audio for a sentence.")
                pieces.append(trim_silence(audio, sr))
            gap = np.zeros(int(sr * INTRA_GAP_S), dtype=np.float32)
            out = pieces[0]
            for p in pieces[1:]:
                out = np.concatenate([out, gap, p])
            return out

    def convert(self, source: Path, reference: Path, *, should_stop: Callable[[], bool] = lambda: False) -> np.ndarray:
        """Re-speak a whole recording in the reference voice (Chatterbox voice conversion).

        The complete `source` file is converted as one piece: nothing is trimmed, split or
        re-timed. Chatterbox reads it at 16 kHz mono (its tokenizer's input) and returns 24 kHz
        mono. The output timeline follows the source frame for frame: its speech tokens are
        40 ms frames, and the decoder rebuilds the audio frame by frame.
        """
        if not reference or not reference.exists():
            raise VoiceError("The reference voice file is missing.")
        if not source.exists():
            raise VoiceError("The audio to convert is missing.")
        ok, why = self.installed()
        if not ok:
            raise VoiceError(f"Chatterbox is not available: {why}")
        with self._lock:
            if should_stop():
                raise VoiceError("Cancelled.")
            if getattr(self, "_vc", None) is None:
                from chatterbox.vc import ChatterboxVC

                try:
                    self._vc = ChatterboxVC.from_pretrained(pick_device(self._device_pref))
                except Exception as exc:
                    raise VoiceError(f"Chatterbox voice conversion failed to load: {exc}") from exc
            try:
                wav = self._vc.generate(str(source), target_voice_path=str(reference))
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    self._free_memory()
                    raise VoiceError("Ran out of memory while converting the voice. Close other apps and retry.") from exc
                raise VoiceError(f"Chatterbox voice conversion failed: {exc}") from exc
        out = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        if out.size == 0 or not np.isfinite(out).all():
            raise VoiceError("Chatterbox produced invalid audio.")
        return out

    def _free_memory(self) -> None:
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
