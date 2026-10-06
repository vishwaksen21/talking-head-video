"""FFmpeg stages on synthetic media, and the whole job flow through the HTTP API.

The gateway here is an httpx.MockTransport - isolated test only, never evidence that the
real API works. The real-API checks live in test_real_integration.py.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from studio.server.compose import compose
from studio.server.main import create_app
from studio.server.media import extract_audio, mean_volume_db, probe, speech_regions
from studio.server.sync import plan, render
from studio.server.utils_api import UtilsClient

from .conftest import FakeVoice, tone_bursts

# ------------------------------------------------------------------ media


def test_probe_and_extract(speech_like_video, tmp_path):
    info = probe(speech_like_video)
    assert info["has_video"] and info["has_audio"] and info["duration"] == pytest.approx(4.0, abs=0.1)
    files = extract_audio(speech_like_video, tmp_path)
    a = probe(files["original"])
    s = probe(files["speech"])
    assert a["audio_codec"] == "pcm_s16le" and a["sample_rate"] == info["sample_rate"]
    assert s["sample_rate"] == 16000 and s["channels"] == 1
    assert probe(speech_like_video)["duration"] == info["duration"]  # original untouched


def test_speech_regions_match_the_bursts(speech_like_video, tmp_path):
    files = extract_audio(speech_like_video, tmp_path)
    regions = speech_regions(files["speech"])
    assert len(regions) == 2
    (a0, a1), (b0, b1) = regions
    assert a0 == pytest.approx(0.5, abs=0.08) and a1 == pytest.approx(1.5, abs=0.08)
    assert b0 == pytest.approx(2.2, abs=0.08) and b1 == pytest.approx(3.4, abs=0.08)


def test_video_without_audio_is_detected(silent_video):
    assert probe(silent_video)["has_audio"] is False


def _voice_wav(tmp_path, seconds, sr=24000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def test_compose_replaces_audio_and_keeps_duration(speech_like_video, tmp_path):
    clips = [_voice_wav(tmp_path, 0.8), _voice_wav(tmp_path, 1.0)]
    placement = plan([{"start": 0.5}, {"start": 2.2}], [0.8, 1.0], 4.0)
    render(clips, 24000, placement, tmp_path / "voice.wav")
    info = compose(speech_like_video, tmp_path / "voice.wav", tmp_path / "final.mp4",
                   final_duration=placement["final_duration"], freeze_seconds=0)
    assert info["duration"] == pytest.approx(4.0, abs=0.1)
    assert info["has_audio"] and info["audio_codec"] == "aac" and info["video_codec"] == "h264"
    # The final soundtrack is the cloned track: silent where the voice track is silent.
    extract = extract_audio(tmp_path / "final.mp4", tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path)
    wav, sr = sf.read(extract["speech"])
    assert np.abs(wav[int(1.6 * sr): int(2.1 * sr)]).max() < 0.01  # gap between lines
    assert np.abs(wav[int(0.7 * sr): int(1.1 * sr)]).max() > 0.05  # cloned line 1


def test_compose_holds_last_frame_when_speech_is_longer(speech_like_video, tmp_path):
    clips = [_voice_wav(tmp_path, 1.0), _voice_wav(tmp_path, 3.0)]
    placement = plan([{"start": 0.5}, {"start": 2.2}], [1.0, 3.0], 4.0)
    assert placement["freeze_seconds"] > 0
    render(clips, 24000, placement, tmp_path / "voice.wav")
    info = compose(speech_like_video, tmp_path / "voice.wav", tmp_path / "final.mp4",
                   final_duration=placement["final_duration"], freeze_seconds=placement["freeze_seconds"])
    assert info["duration"] == pytest.approx(placement["final_duration"], abs=0.1)
    assert mean_volume_db(tmp_path / "final.mp4") > -60


# ------------------------------------------------------------------ fake gateway + full flow


class FakeGateway:
    def __init__(self, video: Path, *, rai_block: bool = False, transcripts=None):
        self.video_b64 = base64.urlsafe_b64encode(video.read_bytes()).decode()
        self.polls = 0
        self.started = 0
        self.rai_block = rai_block
        self.transcripts = list(transcripts or ["Hello there.", "This is the second line."])
        self.generate_calls = 0
        self.requests: list[dict] = []
        self.fail_next = False

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.headers.get("x-api-key") != "test-key":
            return httpx.Response(401, json={"detail": "unauthorized"})
        path = req.url.path
        if path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/llm/models":
            return httpx.Response(200, json={"location": "us-central1", "count": 2, "models": [
                {"id": "veo-3.1-fast-generate-001"}, {"id": "gemini-2.5-flash"}]})
        if path == "/llm/videos" and req.method == "POST":
            self.started += 1
            body = json.loads(req.content)
            self.requests.append(body)
            self.polls = 0
            return httpx.Response(200, json={"operation": f"ops/{self.started}", "done": False, "status": "running"})
        if path == "/llm/videos" and req.method == "GET":
            self.polls += 1
            if self.polls < 2:
                return httpx.Response(200, json={"operation": "ops/1", "done": False, "status": "running"})
            if self.fail_next:
                self.fail_next = False
                return httpx.Response(200, json={"operation": "ops/x", "done": True, "status": "failed", "error": "internal"})
            if self.rai_block:
                return httpx.Response(200, json={"operation": "ops/1", "done": True, "status": "succeeded", "videos": [],
                                                 "rai_filtered_count": 1, "rai_filtered_reasons": ["violence"]})
            return httpx.Response(200, json={"operation": "ops/1", "done": True, "status": "succeeded",
                                             "videos": [{"data": self.video_b64, "mime_type": "video/mp4"}]})
        if path == "/llm/generate":
            text = self.transcripts[self.generate_calls % len(self.transcripts)]
            self.generate_calls += 1
            payload = {"has_speech": True, "language": "en",
                       "segments": [{"start": 0.1, "end": 0.9, "text": text, "confidence": 0.95 if self.generate_calls == 1 else 0.6}]}
            return httpx.Response(200, json={"text": json.dumps(payload), "model": "gemini-2.5-flash", "usage": {"total_tokens": 10}})
        return httpx.Response(404, json={"detail": "Not Found"})


def make_app(tmp_path, gateway, voice=None):
    data = tmp_path / "data"
    (data / "voices").mkdir(parents=True)
    tone_bursts(data / "voices" / "reference.wav", [(0.2, 5.0)], 6.0)
    (data / "settings.json").write_text(json.dumps({"base_url": "https://gw.test", "api_key": "test-key"}))
    factory = lambda s: UtilsClient(s.base_url, s.api_key, transport=httpx.MockTransport(gateway), sleep=lambda x: None)
    app = create_app(data, client_factory=factory, voice=voice or FakeVoice(), recover=False)
    app.state.pipeline.sleep = lambda s: None
    return app


def wait_for(client, pid, states, timeout=60):
    seen = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = client.get(f"/api/projects/{pid}").json()
        if not seen or seen[-1] != p["state"]:
            seen.append(p["state"])
        if p["state"] in states:
            return p, seen
        time.sleep(0.05)
    raise AssertionError(f"timed out; states seen: {seen}")


def test_full_flow_through_the_api(speech_like_video, tmp_path):
    gateway = FakeGateway(speech_like_video)
    voice = FakeVoice()
    app = make_app(tmp_path, gateway, voice)
    with TestClient(app) as c:
        settings = c.get("/api/settings").json()
        assert settings["api_key_set"] and "test-key" not in json.dumps(settings)
        assert c.post("/api/settings/test").json()["video_models"] == 1

        models = c.get("/api/models").json()["models"]
        assert [m["id"] for m in models] == ["veo-3.1-fast-generate-001"]

        r = c.post("/api/projects", json={"prompt": 'A man says: "Hello there."', "model": "veo-3.1-fast-generate-001",
                                          "duration": 4, "aspect_ratio": "16:9", "resolution": "720p", "generate_audio": True})
        assert r.status_code == 200, r.text
        pid = r.json()["id"]
        p, seen = wait_for(c, pid, {"AWAITING_TRANSCRIPT_REVIEW", "FAILED"})
        assert p["state"] == "AWAITING_TRANSCRIPT_REVIEW", p["error"]
        assert "operation" not in p  # internal id not exposed

        t = p["transcript"]
        assert t["timing"] == "measured" and [s["text"] for s in t["original"]] == ["Hello there.", "This is the second line."]
        assert t["original"][0]["start"] == pytest.approx(0.5, abs=0.08)
        assert t["original"][1]["start"] == pytest.approx(2.2, abs=0.08)
        assert t["low_confidence"] == [1]
        assert any("unsure" in w for w in p["warnings"])
        assert set(p["files"]) >= {"generated.mp4", "original_audio.wav"}
        assert gateway.started == 1
        assert gateway.requests[0]["config"]["generate_audio"] is True  # creative mode asks Veo for sound

        # The run does not continue until the transcript is submitted.
        time.sleep(0.3)
        assert c.get(f"/api/projects/{pid}").json()["state"] == "AWAITING_TRANSCRIPT_REVIEW"

        edited = [{"start": s["start"], "text": s["text"]} for s in t["original"]]
        edited[1]["text"] = "This is the corrected second line."
        r = c.post(f"/api/projects/{pid}/transcript", json={"segments": edited, "voice": "reference.wav"})
        assert r.status_code == 200, r.text
        p, seen2 = wait_for(c, pid, {"COMPLETED", "FAILED"})
        assert p["state"] == "COMPLETED", p["error"]
        assert seen2[-1] == "COMPLETED"
        assert [call[0] for call in voice.calls] == ["Hello there.", "This is the corrected second line."]
        assert all(call[1].name == "reference.wav" for call in voice.calls)  # target voice, not the video's speaker
        assert p["transcript"]["original"][1]["text"] == "This is the second line."  # original kept

        final = p["media"]["final"]
        assert final["has_audio"] and final["duration"] == pytest.approx(p["sync"]["final_duration"], abs=0.15)
        r = c.get(p["files"]["final.mp4"] + "?download=true")
        assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
        assert "attachment" in r.headers["content-disposition"]
        r = c.get(p["files"]["final.mp4"], headers={"Range": "bytes=0-99"})
        assert r.status_code == 206 and len(r.content) == 100  # seekable in the browser

        # History lists it and it can be reopened without regenerating.
        history = c.get("/api/projects").json()
        assert history[0]["id"] == pid and history[0]["has_final"]
        assert gateway.started == 1


def test_no_audio_video_waits_for_typed_text(silent_video, tmp_path):
    app = make_app(tmp_path, FakeGateway(silent_video))
    with TestClient(app) as c:
        pid = c.post("/api/projects", json={"prompt": "A quiet beach", "model": "veo-3.1-fast-generate-001",
                                            "duration": 4, "generate_audio": True}).json()["id"]
        p, _ = wait_for(c, pid, {"AWAITING_TRANSCRIPT_REVIEW", "FAILED"})
        assert p["state"] == "AWAITING_TRANSCRIPT_REVIEW"
        assert p["transcript"]["has_speech"] is False
        assert any("no audio track" in w for w in p["warnings"])
        r = c.post(f"/api/projects/{pid}/transcript", json={"segments": [{"start": 0.3, "text": "Welcome to the beach."}]})
        assert r.status_code == 200
        p, _ = wait_for(c, pid, {"COMPLETED", "FAILED"})
        assert p["state"] == "COMPLETED", p["error"]


def test_safety_filter_failure_is_explained(speech_like_video, tmp_path):
    app = make_app(tmp_path, FakeGateway(speech_like_video, rai_block=True))
    with TestClient(app) as c:
        pid = c.post("/api/projects", json={"prompt": "x", "model": "veo-3.1-fast-generate-001", "duration": 4}).json()["id"]
        p, _ = wait_for(c, pid, {"FAILED", "AWAITING_TRANSCRIPT_REVIEW"})
        assert p["state"] == "FAILED" and p["failed_stage"] == "GENERATING_VIDEO"
        assert "safety filter" in p["error"] and "violence" in p["error"]


def test_cancel_during_generation(speech_like_video, tmp_path):
    gateway = FakeGateway(speech_like_video)
    app = make_app(tmp_path, gateway)
    with TestClient(app) as c:
        slow = []
        app.state.pipeline.sleep = lambda s: (slow.append(1), time.sleep(0.2))
        gateway.polls = -1000  # never finishes
        pid = c.post("/api/projects", json={"prompt": "x", "model": "veo-3.1-fast-generate-001", "duration": 4}).json()["id"]
        wait_for(c, pid, {"GENERATING_VIDEO"})
        assert c.post(f"/api/projects/{pid}/cancel").status_code == 200
        p, _ = wait_for(c, pid, {"CANCELLED", "FAILED"})
        assert p["state"] == "CANCELLED"
        assert any("may still finish" in w for w in p["warnings"])
        # Retry resumes polling the same operation instead of paying for a new one.
        gateway.polls = 0
        app.state.pipeline.sleep = lambda s: None
        assert c.post(f"/api/projects/{pid}/retry").status_code == 200
        p, _ = wait_for(c, pid, {"AWAITING_TRANSCRIPT_REVIEW", "FAILED"})
        assert p["state"] == "AWAITING_TRANSCRIPT_REVIEW" and gateway.started == 1


def test_api_validation_and_errors(speech_like_video, tmp_path):
    app = make_app(tmp_path, FakeGateway(speech_like_video))
    with TestClient(app) as c:
        r = c.post("/api/projects", json={"prompt": "x", "model": "veo-9-imaginary"})
        assert r.status_code == 400 and "not a video model" in r.json()["detail"]
        r = c.post("/api/projects", json={"prompt": "x", "model": "veo-3.1-fast-generate-001", "duration": 5})
        assert r.status_code == 400 and "durations" in r.json()["detail"]
        r = c.post("/api/projects", json={"prompt": "", "model": "veo-3.1-fast-generate-001"})
        assert r.status_code == 400 and "Write a prompt" in r.json()["detail"]
        assert c.get("/api/projects/nope").status_code == 404
        pid = c.post("/api/projects", json={"prompt": "x", "model": "veo-3.1-fast-generate-001", "duration": 4}).json()["id"]
        wait_for(c, pid, {"AWAITING_TRANSCRIPT_REVIEW"})
        assert c.get(f"/api/projects/{pid}/files/..%2Fstudio.db").status_code == 404
        assert c.get(f"/api/projects/{pid}/files/settings.json").status_code == 404
        r = c.post(f"/api/projects/{pid}/transcript", json={"segments": [{"start": 99, "text": "late"}]})
        assert r.status_code == 400 and "outside" in r.json()["detail"]
        r = c.post(f"/api/projects/{pid}/transcript", json={"segments": [{"start": 0, "text": "  "}]})
        assert r.status_code == 400
        r = c.post(f"/api/projects/{pid}/transcript", json={"segments": [{"start": 0, "text": "hi"}], "voice": "../../x.wav"})
        assert r.status_code == 400 and "missing" in r.json()["detail"]

        # Wrong key -> clear 401 message surfaced to the UI.
        c.put("/api/settings", json={"api_key": "wrong"})
        r = c.post("/api/settings/test")
        assert r.status_code == 400 and "rejected" in r.json()["detail"]
        assert "wrong" not in r.text


def test_frontend_is_served(speech_like_video, tmp_path):
    app = make_app(tmp_path, FakeGateway(speech_like_video))
    with TestClient(app) as c:
        html = c.get("/").text
        assert "Voice Studio" in html and "logo.png" in html
        assert c.get("/app.js").status_code == 200
        assert "test-key" not in c.get("/app.js").text


# ------------------------------------------------------------------ multi-scene + references + script


def _png(tmp_path) -> Path:
    from PIL import Image

    path = tmp_path / "person.png"
    Image.new("RGB", (400, 500), (200, 180, 160)).save(path)
    return path


SCRIPT = "Every moment with family is precious.\nWhen health matters, peace of mind matters too.\nWith Prudential Health India, focus on living well and making every moment count."


def test_two_scenes_with_reference_and_approved_script(speech_like_video, tmp_path):
    gateway = FakeGateway(speech_like_video, transcripts=["Something else entirely."])
    voice = FakeVoice(seconds_per_char=0.02)
    app = make_app(tmp_path, gateway, voice)
    with TestClient(app) as c:
        up = c.post("/api/uploads/image", files={"file": ("person.png", _png(tmp_path).read_bytes(), "image/png")})
        assert up.status_code == 200, up.text
        ref = up.json()
        assert ref["mime_type"] == "image/png" and ref["width"] == 400

        bad = c.post("/api/projects", json={"scenes": [{"prompt": "a"}], "model": "veo-3.1-fast-generate-001",
                                            "reference_ids": [ref["id"]], "duration": 4})
        assert bad.status_code == 400 and "8-second" in bad.json()["detail"]

        r = c.post("/api/projects", json={
            "name": "prudential-test-001", "model": "veo-3.1-fast-generate-001",
            "scenes": [{"prompt": "Scene one"}, {"prompt": "Scene two", "use_references": False}], "reference_ids": [ref["id"]],
            "aspect_ratio": "16:9", "resolution": "720p", "generate_audio": True, "script": SCRIPT,
            "end_card": True, "export_name": "final_prudential_video"})
        assert r.status_code == 200, r.text
        assert r.json()["id"] == "prudential-test-001"
        dup = c.post("/api/projects", json={"name": "prudential-test-001", "prompt": "x", "model": "veo-3.1-fast-generate-001"})
        assert dup.status_code == 400 and "already exists" in dup.json()["detail"]

        p, _ = wait_for(c, "prudential-test-001", {"AWAITING_TRANSCRIPT_REVIEW", "FAILED"})
        assert p["state"] == "AWAITING_TRANSCRIPT_REVIEW", p["error"]
        assert gateway.started == 2 and p["params"]["generations_started"] == 2
        assert [b["prompt"] for b in gateway.requests] == ["Scene one", "Scene two"]  # sent verbatim
        refs = gateway.requests[0]["config"]["reference_images"]
        assert len(refs) == 1 and refs[0]["reference_type"] == "asset"
        assert base64.b64decode(refs[0]["image"]["image_bytes"]) == _png(tmp_path).read_bytes()
        assert "reference_images" not in gateway.requests[1]["config"]  # scene 2 opted out
        assert all(b["config"]["duration_seconds"] == 8 for b in gateway.requests)  # a reference pins Veo to 8 s
        assert {"scene_01.mp4", "scene_02.mp4", "generated.mp4"} <= set(p["files"])
        gen = p["media"]["generated"]
        assert [s["start"] for s in gen["scenes"]] == [0.0, pytest.approx(3.5, abs=0.05)]
        assert gen["duration"] == pytest.approx(7.5, abs=0.1)  # 4 + 4 - 0.5 crossfade

        t = p["transcript"]
        assert t["approved_script"] == SCRIPT.split("\n")
        assert [x["text"] for x in t["proposed"]] == SCRIPT.split("\n")
        assert [x["scene"] for x in t["proposed"]] == [1, 1, 2]
        assert t["proposed"][2]["start"] == pytest.approx(3.9, abs=0.05)
        assert t["comparison"]["status"] == "differs"
        assert any("differs from the approved script" in w for w in p["warnings"])

        r = c.post("/api/projects/prudential-test-001/transcript",
                   json={"segments": [{"start": x["start"], "text": x["text"]} for x in t["proposed"]], "voice": "reference.wav"})
        assert r.status_code == 200, r.text
        p, _ = wait_for(c, "prudential-test-001", {"COMPLETED", "FAILED"})
        assert p["state"] == "COMPLETED", p["error"]
        assert [call[0] for call in voice.calls] == SCRIPT.split("\n")
        folder = tmp_path / "data" / "projects" / "prudential-test-001"
        for rel in ["reference/person.png", "reference/reference.wav", "video/scene_01.mp4", "video/scene_02.mp4",
                    "audio/original_veo_audio.wav", "audio/approved_script.txt", "audio/cloned_voice.wav",
                    "output/final_prudential_video.mp4"]:
            assert (folder / rel).exists(), rel
        assert (folder / "audio" / "approved_script.txt").read_text().strip() == SCRIPT.replace("\n", "\n\n")
        # End card: the last frame is the white card with the logo, not the scene.
        last = folder / "last.png"
        dur = p["media"]["final"]["duration"]
        from .conftest import ffmpeg as _ff
        _ff("-sseof", "-0.1", "-i", str(folder / "final.mp4"), "-frames:v", "1", str(last))
        from PIL import Image
        px = Image.open(last).convert("RGB").getpixel((5, 5))
        assert min(px) > 240, px
        assert gateway.started == 2  # nothing regenerated


def test_quota_guard_and_explicit_regeneration(speech_like_video, tmp_path):
    gateway = FakeGateway(speech_like_video)
    app = make_app(tmp_path, gateway)
    with TestClient(app) as c:
        gateway.fail_next = True
        pid = c.post("/api/projects", json={"scenes": [{"prompt": "a"}, {"prompt": "b"}],
                                            "model": "veo-3.1-fast-generate-001", "duration": 4}).json()["id"]
        p, _ = wait_for(c, pid, {"FAILED", "AWAITING_TRANSCRIPT_REVIEW"})
        assert p["state"] == "FAILED" and "scene 1 of 2" in p["error"]
        assert gateway.started == 1
        # Plain retry stays within the approved 2 generations (scene 1 again + scene 2 = 3 > 2): refused.
        c.post(f"/api/projects/{pid}/retry")
        p, _ = wait_for(c, pid, {"FAILED", "AWAITING_TRANSCRIPT_REVIEW"})
        assert gateway.started == 2  # scene 1 regenerated within budget (the failed one did not produce video)
        assert p["state"] == "FAILED" and "Generate again" in p["error"]
        r = c.post(f"/api/projects/{pid}/retry?approve_new_generation=true")
        assert r.status_code == 200
        p, _ = wait_for(c, pid, {"FAILED", "AWAITING_TRANSCRIPT_REVIEW"})
        assert p["state"] == "AWAITING_TRANSCRIPT_REVIEW", p["error"]
        assert gateway.started == 3


def test_voice_upload(speech_like_video, tmp_path):
    app = make_app(tmp_path, FakeGateway(speech_like_video))
    wav = tmp_path / "my voice.wav"
    tone_bursts(wav, [(0.1, 4.0)], 5.0)
    with TestClient(app) as c:
        r = c.post("/api/voices", files={"file": ("my voice.wav", wav.read_bytes(), "audio/wav")})
        assert r.status_code == 200 and r.json()["id"] == "my_voice.wav"
        assert c.post("/api/voices", files={"file": ("my voice.wav", wav.read_bytes(), "audio/wav")}).status_code == 200
        other = tmp_path / "o.wav"
        tone_bursts(other, [(0.1, 3.5)], 5.0)
        r = c.post("/api/voices", files={"file": ("my voice.wav", other.read_bytes(), "audio/wav")})
        assert r.status_code == 400 and "already exists" in r.json()["detail"]
        assert c.post("/api/voices", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 400
        r = c.post("/api/uploads/image", files={"file": ("x.png", b"notapng", "image/png")})
        assert r.status_code == 400


def test_mono_voice_reaches_both_channels_at_full_level(speech_like_video, tmp_path):
    clips = [_voice_wav(tmp_path, 1.0)]
    placement = plan([{"start": 0.5}], [1.0], 4.0)
    render(clips, 24000, placement, tmp_path / "voice.wav")
    compose(speech_like_video, tmp_path / "voice.wav", tmp_path / "final.mp4", final_duration=4.0, freeze_seconds=0)
    import subprocess
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(tmp_path / "final.mp4"), "-vn", "-f", "f32le",
                          "-ac", "2", "-ar", "24000", "-"], capture_output=True).stdout
    st = np.frombuffer(raw, np.float32).reshape(-1, 2)
    v, _ = sf.read(tmp_path / "voice.wav", dtype="float32")
    n = min(len(st), len(v))
    gain_db = 20 * np.log10(np.sqrt((st[:n, 0] ** 2).mean()) / np.sqrt((v[:n] ** 2).mean()))
    assert abs(gain_db) < 0.5                                   # no -3 dB pan-law loss
    assert np.abs(st[:, 0] - st[:, 1]).max() < 0.01            # same signal in both channels


def test_render_keeps_gaps_silent_after_each_clip(tmp_path):
    sr = 24000
    clips = [np.full(int(1.03 * sr), 0.2, np.float32)]          # 30 ms longer than planned
    placement = {"final_duration": 3.0, "placements": [{"start": 0.5, "end": 1.5}]}
    render(clips, sr, placement, tmp_path / "v.wav")
    v, _ = sf.read(tmp_path / "v.wav")
    assert np.abs(v[int(1.5 * sr) + 1:]).max() == 0.0
