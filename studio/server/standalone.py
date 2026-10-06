"""Standalone Chatterbox: text-to-speech and voice conversion without any video.

Separate from the video pipeline:
- It uses the same installed Chatterbox (through the app's shared `ChatterboxVoice`, so the
  model is loaded once) and the same voice library.
- It never builds a provider client, so it works with the provider gate off.

Each result is kept in `data/chatterbox/<id>/` (`output.wav`, `meta.json`, and the uploaded
source for conversions) and listed in its history.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import dialogue
from .media import MediaError, probe, run_ffmpeg
from .voice import BUILTIN_VOICE, VoiceError

MAX_TEXT = 2000
MAX_UPLOAD = 100 * 1024 * 1024
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".mp4", ".mov"}


class TTSIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    voice: str = BUILTIN_VOICE
    exaggeration: float = Field(0.5, ge=0, le=1)
    cfg_weight: float = Field(0.5, ge=0, le=1)
    temperature: float = Field(0.8, ge=0.1, le=1.5)
    check_words: bool = True


def router(data_dir: Path, voice, pipeline, asr_factory) -> APIRouter:
    r = APIRouter(prefix="/api/chatterbox")
    root = data_dir / "chatterbox"
    root.mkdir(parents=True, exist_ok=True)

    def new_item(kind: str) -> tuple[str, Path]:
        item_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        folder = root / item_id
        folder.mkdir()
        return item_id, folder

    def resolve(voice_id: str, allow_builtin: bool) -> Optional[Path]:
        if voice_id == BUILTIN_VOICE:
            if not allow_builtin:
                raise HTTPException(400, "Choose a voice to convert into; the built-in voice cannot be a conversion target.")
            return None
        try:
            return pipeline.resolve_voice(voice_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    def finish(item_id: str, folder: Path, wav: np.ndarray, meta: dict) -> dict:
        sr = voice.sample_rate
        sf.write(folder / "output.wav", wav, sr)
        meta.update({"id": item_id, "created_at": time.time(), "sample_rate": sr,
                     "duration_s": round(wav.size / sr, 3), "samples": int(wav.size)})
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        return present(meta)

    def present(meta: dict) -> dict:
        return {**meta, "audio_url": f"/api/chatterbox/items/{meta['id']}/output.wav",
                "download_url": f"/api/chatterbox/items/{meta['id']}/output.wav?download=true"}

    @r.get("/status")
    def status():
        ok, why = voice.installed()
        return {"installed": ok, "detail": why, "loaded": voice.loaded, "device": voice.device,
                "voices": pipeline.list_voices(), "provider_calls_needed": False}

    @r.post("/tts")
    def tts(body: TTSIn):
        text = dialogue.normalize_script(body.text)
        if not text:
            raise HTTPException(400, "Type something to speak.")
        reference = resolve(body.voice, allow_builtin=True)
        item_id, folder = new_item("tts")
        t0 = time.time()
        try:
            wav = voice.synthesize(text, reference, exaggeration=body.exaggeration, cfg_weight=body.cfg_weight,
                                   temperature=body.temperature)
        except VoiceError as exc:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(500, str(exc))
        meta = {"kind": "tts", "text": text, "voice": body.voice,
                "settings": {"exaggeration": body.exaggeration, "cfg_weight": body.cfg_weight,
                             "temperature": body.temperature}, "seconds_to_make": round(time.time() - t0, 1)}
        result = finish(item_id, folder, wav, meta)
        if body.check_words:
            result["check"] = meta["check"] = word_check(folder / "output.wav", text)
            (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        return result

    @r.post("/convert")
    async def convert(file: UploadFile = File(...), voice_id: str = Form(...)):
        name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "audio").name)
        if Path(name).suffix.lower() not in AUDIO_EXT:
            raise HTTPException(400, "Upload an audio file (WAV, MP3, M4A, FLAC, AAC, OGG) or a video with sound.")
        data = await file.read()
        if not data:
            raise HTTPException(400, "The uploaded file is empty.")
        if len(data) > MAX_UPLOAD:
            raise HTTPException(400, "Files must be under 100 MB.")
        reference = resolve(voice_id, allow_builtin=False)
        item_id, folder = new_item("convert")
        upload = folder / f"source_{name}"
        upload.write_bytes(data)
        try:
            info = probe(upload)
            if not info["has_audio"] or info["duration"] <= 0:
                raise MediaError("no audio stream")
            source = folder / "source.wav"   # the complete audio, decoded losslessly to PCM
            run_ffmpeg(["-i", str(upload), "-vn", "-map", "0:a:0", "-c:a", "pcm_s16le", str(source)],
                       what="reading the uploaded audio")
        except MediaError as exc:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(400, f"That file has no usable audio: {exc}")
        t0 = time.time()
        try:
            from starlette.concurrency import run_in_threadpool

            wav = await run_in_threadpool(voice.convert, source, reference)
        except VoiceError as exc:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(500, str(exc))
        src = sf.info(source)
        n = int(round(src.frames / src.samplerate * voice.sample_rate))
        raw_len = int(wav.size)
        wav = wav[:n] if wav.size >= n else np.concatenate([wav, np.zeros(n - wav.size, wav.dtype)])
        meta = {"kind": "convert", "source_file": name, "voice": voice_id,
                "source": {"seconds": round(src.frames / src.samplerate, 3), "sample_rate": src.samplerate,
                           "channels": src.channels},
                "length_matched_from_ms": round((raw_len - n) / voice.sample_rate * 1000, 3),
                "seconds_to_make": round(time.time() - t0, 1)}
        return finish(item_id, folder, wav, meta)

    @r.get("/items")
    def items():
        out = []
        for meta_path in sorted(root.glob("*/meta.json"), reverse=True):
            try:
                out.append(present(json.loads(meta_path.read_text())))
            except (ValueError, KeyError):
                continue
        return out

    @r.get("/items/{item_id}/output.wav")
    def item_audio(item_id: str, download: bool = False):
        if not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", item_id) or not (root / item_id / "output.wav").exists():
            raise HTTPException(404, "No such result.")
        return FileResponse(root / item_id / "output.wav", media_type="audio/wav",
                            filename=f"chatterbox-{item_id}.wav" if download else None,
                            content_disposition_type="attachment" if download else "inline")

    def word_check(path: Path, text: str) -> dict:
        try:
            heard = asr_factory().transcribe(path)
        except Exception as exc:  # the check is optional; never lose the audio over it
            return {"available": False, "reason": str(exc)}
        m = dialogue.compare(text, heard["text"])
        return {"available": True, "heard": heard["text"], "word_accuracy": m["word_accuracy"], "passed": m["passed"],
                "missing": m["missing"], "extra": m["extra"]}

    return r
