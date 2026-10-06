"""Speech transcription through the gateway (`/llm/generate`, Gemini with inline audio).

Timing comes from the audio, not from the model. Gemini's own timestamps were measured
1.7 s early on a real Veo clip, which would put every cloned line in the wrong place. So:

1. `media.speech_regions` finds where sound actually is (adaptive threshold).
2. Each region is transcribed on its own, so its words inherit the region's measured
   start and end. A region with no words (music, an effect) is dropped.
3. Regions are not split further: a region may hold several sentences, and the voice
   module chunks long text at sentence edges anyway.

Without any detected region (or with an unreasonable number of them) the whole clip is
transcribed once and the model's timestamps are used, with a warning.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import soundfile as sf

from .utils_api import UtilsAPIError, UtilsClient

LOW_CONFIDENCE = 0.80
REGION_PAD_S = 0.15
MAX_REGIONS = 24

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "has_speech": {"type": "BOOLEAN"},
        "language": {"type": "STRING"},
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start": {"type": "NUMBER"},
                    "end": {"type": "NUMBER"},
                    "text": {"type": "STRING"},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["start", "end", "text", "confidence"],
            },
        },
    },
    "required": ["has_speech", "segments"],
}

_INSTRUCTION = (
    "Transcribe the spoken words in this audio exactly as spoken. "
    "Return one segment per sentence, in order, with start and end times in seconds. "
    "Do not add, correct, translate or summarise words. Do not transcribe music, "
    "sound effects or background noise. If nothing is spoken, set has_speech to false "
    "and return no segments. confidence is your 0-1 certainty that the words are right."
)


class TranscriptionError(Exception):
    pass


def _ask(client: UtilsClient, wav_bytes: bytes, *, model: str, location: str) -> dict:
    body = {
        "model": model,
        "location": location,
        "contents": [{"role": "user", "parts": [
            {"text": _INSTRUCTION},
            {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(wav_bytes).decode("ascii")}},
        ]}],
        "config": {"temperature": 0, "response_mime_type": "application/json", "response_schema": _SCHEMA},
    }
    try:
        response = client.generate(body)
    except UtilsAPIError as exc:
        raise TranscriptionError(f"Transcription request failed: {exc}") from exc
    text = response.get("text")
    if not text:
        reason = response.get("finish_reason") or "no text returned"
        raise TranscriptionError(f"The transcription model returned nothing ({reason}).")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TranscriptionError("The transcription model returned malformed JSON.") from exc
    segments = []
    for seg in parsed.get("segments") or []:
        words = str(seg.get("text") or "").strip()
        if words:
            conf = seg.get("confidence")
            segments.append({
                "start": max(0.0, float(seg.get("start") or 0)),
                "end": max(0.0, float(seg.get("end") or 0)),
                "text": words,
                "confidence": round(float(conf if conf is not None else 0), 3),
            })
    return {"has_speech": bool(parsed.get("has_speech")) and bool(segments), "language": parsed.get("language"),
            "segments": segments, "usage": response.get("usage") or {}}


def _wav_bytes(samples, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def transcribe(client: UtilsClient, audio_16k: Path, regions: list[tuple[float, float]], duration: float, *,
               model: str, location: str) -> dict:
    wav, sr = sf.read(audio_16k, dtype="float32")
    tokens = 0
    language = None
    timing = "measured"
    segments: list[dict] = []

    if regions and len(regions) <= MAX_REGIONS:
        for start, end in regions:
            a = max(0, int((start - REGION_PAD_S) * sr))
            b = min(wav.size, int((end + REGION_PAD_S) * sr))
            result = _ask(client, _wav_bytes(wav[a:b], sr), model=model, location=location)
            tokens += int(result["usage"].get("total_tokens") or 0)
            if not result["has_speech"]:
                continue
            language = language or result["language"]
            segments.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "text": " ".join(s["text"] for s in result["segments"]),
                "confidence": min(s["confidence"] for s in result["segments"]),
            })
    else:
        timing = "model_estimate"
        result = _ask(client, _wav_bytes(wav, sr), model=model, location=location)
        tokens += int(result["usage"].get("total_tokens") or 0)
        language = result["language"]
        prev = 0.0
        for s in sorted(result["segments"], key=lambda s: s["start"]):
            start = min(max(s["start"], prev), duration)
            end = min(max(s["end"], start + 0.2), duration)
            segments.append({**s, "start": round(start, 3), "end": round(end, 3)})
            prev = end

    return {"model": model, "has_speech": bool(segments), "language": language, "segments": segments,
            "timing": timing, "total_tokens": tokens, "requests": max(1, len(regions or [])) if timing == "measured" else 1}


def low_confidence(segments: list[dict]) -> list[int]:
    return [i for i, s in enumerate(segments) if s.get("confidence", 0) < LOW_CONFIDENCE]
