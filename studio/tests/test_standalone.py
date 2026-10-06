"""Standalone Chatterbox page: text-to-speech and voice conversion with no video.
It never builds a provider client and never creates a video project."""

from __future__ import annotations

import httpx
import soundfile as sf
from fastapi.testclient import TestClient

from .conftest import FakeVoice, tone_bursts
from .test_media_and_flow import make_app


class FakeASR:
    def transcribe(self, path):
        return {"text": "Hello there, this is a standalone test.", "words": [], "model": "fake", "duration": 1.0}


def app_with_counter(tmp_path):
    app = make_app(tmp_path, lambda req: httpx.Response(500), FakeVoice())
    pipe = app.state.pipeline
    pipe.strict.asr = FakeASR()
    calls = []
    original = pipe._client_factory
    pipe._client_factory = lambda s: (calls.append(1), original(s))[1]
    return app, calls


def test_text_to_speech_makes_and_keeps_audio(tmp_path):
    app, calls = app_with_counter(tmp_path)
    with TestClient(app) as c:
        st = c.get("/api/chatterbox/status").json()
        assert st["installed"] and st["provider_calls_needed"] is False
        r = c.post("/api/chatterbox/tts", json={"text": "Hello there, this is a standalone test.", "voice": "reference.wav"})
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["kind"] == "tts" and res["duration_s"] > 0 and res["check"]["passed"]
        wav = c.get(res["audio_url"])
        assert wav.status_code == 200 and wav.headers["content-type"] == "audio/wav"
        dl = c.get(res["download_url"])
        assert "attachment" in dl.headers["content-disposition"]
        builtin = c.post("/api/chatterbox/tts", json={"text": "Built in voice.", "voice": "builtin", "check_words": False})
        assert builtin.status_code == 200 and "check" not in builtin.json()
        assert [i["kind"] for i in c.get("/api/chatterbox/items").json()] == ["tts", "tts"]
        assert c.get("/api/projects").json() == []                    # no video project created
    assert calls == []                                                 # no provider client built


def test_voice_conversion_keeps_the_source_length(tmp_path, speech_like_video):
    app, calls = app_with_counter(tmp_path)
    src = tmp_path / "speech.wav"
    tone_bursts(src, [(0.3, 1.4), (2.0, 3.1)], 3.5, sr=44100)
    with TestClient(app) as c:
        r = c.post("/api/chatterbox/convert", data={"voice_id": "reference.wav"},
                   files={"file": ("speech.wav", src.read_bytes(), "audio/wav")})
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["kind"] == "convert" and res["samples"] == int(round(3.5 * 24000))
        out = tmp_path / "data" / "chatterbox" / res["id"] / "output.wav"
        assert sf.info(out).frames == res["samples"]
        vid = c.post("/api/chatterbox/convert", data={"voice_id": "reference.wav"},
                     files={"file": ("clip.mp4", speech_like_video.read_bytes(), "video/mp4")})
        assert vid.status_code == 200 and vid.json()["source"]["seconds"] == 4.0   # audio taken from a video
    assert calls == []


def test_bad_inputs_are_refused_clearly(tmp_path, silent_video):
    app, _ = app_with_counter(tmp_path)
    with TestClient(app) as c:
        assert c.post("/api/chatterbox/tts", json={"text": "", "voice": "builtin"}).status_code == 422
        r = c.post("/api/chatterbox/tts", json={"text": "hi", "voice": "missing.wav"})
        assert r.status_code == 400 and "missing" in r.json()["detail"]
        r = c.post("/api/chatterbox/convert", data={"voice_id": "builtin"}, files={"file": ("a.wav", b"RIFF", "audio/wav")})
        assert r.status_code == 400 and "built-in voice" in r.json()["detail"]
        r = c.post("/api/chatterbox/convert", data={"voice_id": "reference.wav"}, files={"file": ("notes.txt", b"hello", "text/plain")})
        assert r.status_code == 400 and "audio file" in r.json()["detail"]
        r = c.post("/api/chatterbox/convert", data={"voice_id": "reference.wav"},
                   files={"file": ("silent.mp4", silent_video.read_bytes(), "video/mp4")})
        assert r.status_code == 400 and "no usable audio" in r.json()["detail"]
        assert c.get("/api/chatterbox/items/../../studio.db/output.wav").status_code == 404
        assert c.get("/api/chatterbox/items").json() == []
