"""Strict Talking Head mode: the critical failure modes and the full orchestration.

Mocked providers only:
- the video gateway is an httpx.MockTransport
- Chatterbox is FakeVoice
- Whisper is FakeASR
- Apple Vision is a fake `facekit.analyse`
FFmpeg runs for real on synthetic media. No paid quota is used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from studio.server import dialogue, facekit, lipsync, strict
from studio.server.media import MediaError, probe

from .conftest import FakeVoice
from .test_media_and_flow import FakeGateway, _png, make_app, wait_for

SCRIPT = "Prudential Health India helps people make informed choices for their health and well-being."
WORDS = SCRIPT.split()

# ------------------------------------------------------------------ dialogue matching (unit)


def test_exact_and_formatting_only_differences_pass():
    r = dialogue.compare(SCRIPT, "prudential health india helps people make informed choices for their health and wellbeing")
    assert r["passed"] and r["word_accuracy"] == 1.0 and r["char_similarity"] == 1.0


def test_incorrect_transcript_is_rejected():
    r = dialogue.compare(SCRIPT, "Welcome everyone. Hello, this is a test of the emergency broadcast system.")
    assert not r["passed"] and r["word_accuracy"] < 0.2


def test_extra_words_are_detected():
    r = dialogue.compare(SCRIPT, "Hello and welcome. " + SCRIPT)
    assert not r["passed"] and {"hello", "welcome"} <= set(r["extra"])


def test_missing_words_are_detected():
    r = dialogue.compare(SCRIPT, SCRIPT.replace("informed ", ""))
    assert not r["passed"] and r["missing"] == ["informed"]


def test_reordered_words_are_detected():
    r = dialogue.compare(SCRIPT, SCRIPT.replace("make informed choices", "choices make informed"))
    assert not r["passed"] and "choices" in r["reordered"]


def test_duplicated_words_are_detected():
    r = dialogue.compare(SCRIPT, SCRIPT.replace("helps people", "helps people people people"))
    assert not r["passed"] and r["duplicated"] == ["people"]


def test_script_normalisation_is_formatting_only():
    assert dialogue.validate_script("  Prudential ’s   plan\n") == "Prudential 's plan"
    with pytest.raises(dialogue.ScriptError):
        dialogue.validate_script("   ")
    with pytest.raises(dialogue.ScriptError):
        dialogue.validate_script('He said "hi"')


# ------------------------------------------------------------------ fakes for the flow

FACE = {"confidence": 0.95, "quality": 0.7, "yaw": 0.0, "roll": 0.0, "pitch": 0.0,
        "left_eye": [[90, 100], [110, 100]], "right_eye": [[190, 100], [210, 100]],
        "nose": [[140, 110], [160, 110], [150, 160], [140, 150], [160, 150]],
        "outer_lips": [[120, 190], [150, 185], [180, 190], [150, 200]],
        "inner_lips": [[125, 192], [150, 190], [175, 192], [150, 195]],
        "face_contour": [[60, 120], [80, 220], [150, 260], [220, 220], [240, 120]]}


def make_face(scale_mouth: float = 1.0) -> dict:
    f = json.loads(json.dumps(FACE))
    f["bbox"] = [50, 50, 200, 230]
    if scale_mouth != 1.0:
        f["outer_lips"] = [[150 + (x - 150) * scale_mouth, y] for x, y in f["outer_lips"]]
        f["face_contour"] = [[150 + (x - 150) * scale_mouth, y * 1.2] for x, y in f["face_contour"]]
    return f


class FakeFaceKit:
    def __init__(self, faces_in_reference: int = 1, video_face_scale: float = 1.0):
        self.faces_in_reference = faces_in_reference
        self.video_face_scale = video_face_scale
        self.calls: list[str] = []

    def __call__(self, images, batch=64):
        out = []
        for p in images:
            self.calls.append(Path(p).name)
            if "reference" in str(p) or Path(p).name.endswith("person.png"):
                faces = [make_face() for _ in range(self.faces_in_reference)]
            else:
                faces = [make_face(self.video_face_scale)]
            out.append({"file": str(p), "faces": faces})
        return out


class FakeASR:
    """Word timestamps: the cloned voice says the script; the generated video says `veo_text`
    inside its two sound bursts (0.5-1.5 s and 2.2-3.4 s)."""

    def __init__(self, tts_text: str = SCRIPT, veo_text: str = SCRIPT):
        self.tts_text, self.veo_text = tts_text, veo_text

    def available(self):
        return True, "fake"

    def transcribe(self, audio: Path) -> dict:
        name = Path(audio).name
        dur = sf.info(audio).duration
        if name.startswith("speech_16k"):
            words = self.veo_text.split()
            half = len(words) // 2
            times = np.r_[np.linspace(0.5, 1.4, half, endpoint=False), np.linspace(2.2, 3.3, len(words) - half, endpoint=False)]
            ws = [{"word": w, "start": float(t), "end": float(t) + 0.08} for w, t in zip(words, times)]
            return {"model": "fake", "text": self.veo_text, "words": ws, "duration": dur}
        text = self.tts_text if name.startswith("tts_take") else SCRIPT
        words = text.split()
        half = len(words) // 2
        a = np.linspace(0.0, dur * 0.45, half, endpoint=False)
        b = np.linspace(dur * 0.55, dur * 0.98, len(words) - half, endpoint=False)
        ws = [{"word": w, "start": float(t), "end": float(t) + 0.05} for w, t in zip(words, np.r_[a, b])]
        return {"model": "fake", "text": text, "words": ws, "duration": dur}


def fake_frame_series(video: Path, work: Path) -> dict:
    """The mouth moves with the generated video's own soundtrack, i.e. in sync by construction."""
    info = probe(video)
    n = int(round(info["duration"] * 24))
    native = video.parent / "original_audio.wav"
    if native.exists():
        env = lipsync.envelope(native, 24, n)
    else:  # silent clip: the mouth "talks" at 0.5-1.5 s and 2.2-3.4 s
        t = np.arange(n) / 24
        env = (((t >= 0.5) & (t < 1.5)) | ((t >= 2.2) & (t < 3.4))).astype(float) * 0.3
    return {"fps": 24.0, "frames": n, "activity": list(env + 1e-3), "face_counts": [1] * n, "yaw_deg": [0.0] * n}


def strict_app(tmp_path, monkeypatch, *, gateway=None, asr=None, facekit_fake=None, voice=None, video=None):
    gateway = gateway or FakeGateway(video, transcripts=["unused"])
    app = make_app(tmp_path, gateway, voice or FakeVoice(seconds_per_char=0.03))
    pipe = app.state.pipeline
    pipe.strict.asr = asr or FakeASR()
    monkeypatch.setattr(facekit, "analyse", facekit_fake or FakeFaceKit())
    monkeypatch.setattr(facekit, "segment_person", lambda src, out: (__import__("shutil").copy(src, out), 0.6)[1])
    monkeypatch.setattr(lipsync, "frame_series", fake_frame_series)
    return app, gateway


def upload(c, tmp_path) -> str:
    r = c.post("/api/uploads/image", files={"file": ("person.png", _png(tmp_path).read_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def create(c, ref_id, **over):
    body = {"voice_method": "tts", "veo_audio": "reference", "reference_id": ref_id, "script": SCRIPT, "voice": "reference.wav",
            "consent": {"attested": True, "by": "Test Operator"}, "name": None}
    body.update(over)
    return c.post("/api/strict/projects", json=body)


# ------------------------------------------------------------------ input validation (6, 7)


def test_reference_image_is_mandatory(speech_like_video, tmp_path, monkeypatch):
    app, _ = strict_app(tmp_path, monkeypatch, video=speech_like_video)
    with TestClient(app) as c:
        r = create(c, None)
        assert r.status_code == 400 and "Upload a photo" in r.json()["detail"]
        r = create(c, "0123456789abcdef")
        assert r.status_code == 400 and "not found" in r.json()["detail"]


def test_script_voice_and_consent_are_mandatory(speech_like_video, tmp_path, monkeypatch):
    app, _ = strict_app(tmp_path, monkeypatch, video=speech_like_video)
    with TestClient(app) as c:
        ref = upload(c, tmp_path)
        assert "empty" in create(c, ref, script="  ").json()["detail"]
        assert "Choose the authorised voice" in create(c, ref, voice=None).json()["detail"]
        assert "built-in voice" in create(c, ref, voice="builtin").json()["detail"]
        assert "authorised this" in create(c, ref, consent={"attested": False, "by": "x"}).json()["detail"]
        assert "authorised this" in create(c, ref, consent={"attested": True, "by": " "}).json()["detail"]


# ------------------------------------------------------------------ face handling (8)


@pytest.mark.parametrize("faces,message", [(0, "No face was detected"), (2, "exactly one person")])
def test_unusable_reference_face_fails_before_any_spending(speech_like_video, tmp_path, monkeypatch, faces, message):
    app, gw = strict_app(tmp_path, monkeypatch, video=speech_like_video, facekit_fake=FakeFaceKit(faces_in_reference=faces))
    with TestClient(app) as c:
        pid = create(c, upload(c, tmp_path)).json()["id"]
        p, _ = wait_for(c, pid, {"FAILED", "AWAITING_AUDIO_APPROVAL"})
        assert p["state"] == "FAILED" and p["failed_stage"] == "PREPARING_REFERENCE" and message in p["error"]
        assert gw.started == 0


# ------------------------------------------------------------------ the full orchestration


def run_to_review(c, tmp_path, **over):
    pid = create(c, upload(c, tmp_path), **over).json()["id"]
    p, seen = wait_for(c, pid, {"AWAITING_AUDIO_APPROVAL", "FAILED"})
    return pid, p, seen


def test_full_flow_reference_mode_veo_audio_only_times_the_mouth(speech_like_video, tmp_path, monkeypatch):
    voice = FakeVoice(seconds_per_char=0.03)
    app, gw = strict_app(tmp_path, monkeypatch, video=speech_like_video, voice=voice)
    with TestClient(app) as c:
        pid, p, seen = run_to_review(c, tmp_path, name="strict-test-001")
        assert p["state"] == "AWAITING_AUDIO_APPROVAL", p["error"]
        assert {"VALIDATING_INPUT", "PREPARING_REFERENCE", "GENERATING_SPEECH", "VALIDATING_DIALOGUE"} & set(seen)
        st = p["params"]["strict"]
        assert voice.calls[0][0] == SCRIPT                              # (1) exact script, unchanged
        assert voice.calls[0][1].name == "reference.wav"                 # the chosen voice, not a default
        assert st["dialogue"]["passed"] and st["dialogue"]["word_accuracy"] == 1.0
        assert st["speech_seconds"] == pytest.approx(len(SCRIPT) * 0.03, abs=0.06)  # (9) measured
        assert gw.started == 0                                           # nothing paid before approval
        folder = tmp_path / "data" / "projects" / pid
        assert (folder / "reference" / "person.png").read_bytes() == _png(tmp_path).read_bytes()  # immutable original
        assert p["params"]["reference_files"][0]["path"] == "reference/person.png"   # the full photo, unchanged
        assert (folder / "audio" / "approved_script.txt").read_text().strip() == SCRIPT

        assert c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": ""}).status_code == 400
        r = c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "Test Operator"})
        assert r.status_code == 200, r.text
        p, seen = wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        assert p["state"] == "NEEDS_REVIEW", p["error"]
        assert gw.started == 1
        body = gw.requests[0]
        assert f'"{SCRIPT}"' in body["prompt"]                           # the script reaches the video model
        assert body["config"]["reference_images"][0]["reference_type"] == "asset"
        assert body["config"]["duration_seconds"] == 8 and body["config"]["generate_audio"] is True

        v = p["params"]["strict"]["validation"]
        assert v["checks"].pop("exact_photo_animated") is False                           # stated, never claimed
        assert all(v["checks"].values()), v["checks"]
        assert v["checks"]["veo_audio_absent"] and v["checks"]["audio_is_cloned_voice"]   # (13)
        final = folder / "output" / "final_talking_head.mp4"
        info = probe(final)
        assert info["duration"] == pytest.approx(probe(folder / "generated.mp4")["duration"], abs=0.1)  # (10)
        for name in ("dialogue_comparison", "veo_speech_check", "sync_plan", "lipsync_validation",
                     "identity_validation", "final_validation"):
            assert (folder / "reports" / f"{name}.json").exists(), name
            assert c.get(f"/api/projects/{pid}/reports/{name}").status_code == 200
        assert v["run_id"] == p["params"]["strict"]["run_id"]                              # (15)

        r = c.post(f"/api/projects/{pid}/strict/confirm-identity", json={"same_person": True, "reviewer": "QA"})
        # A likeness is never reported as the exact person: the provider cannot animate the photo.
        assert r.status_code == 200 and r.json()["state"] == "LIKENESS_ACCEPTED"
        assert c.get(r.json()["files"]["final_talking_head.mp4"] + "?download=true").status_code == 200


def test_dialogue_mismatch_blocks_until_explicit_override(speech_like_video, tmp_path, monkeypatch):
    app, gw = strict_app(tmp_path, monkeypatch, video=speech_like_video,
                         asr=FakeASR(tts_text="Welcome everyone, this is a test."))
    with TestClient(app) as c:
        pid, p, _ = run_to_review(c, tmp_path)
        d = p["params"]["strict"]["dialogue"]
        assert not d["passed"] and d["reasons"]
        r = c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        assert r.status_code == 400 and "did not match" in r.json()["detail"]
        assert gw.started == 0
        r = c.post(f"/api/projects/{pid}/strict/regenerate-speech")
        assert r.status_code == 200
        p, _ = wait_for(c, pid, {"AWAITING_AUDIO_APPROVAL", "FAILED"})
        assert p["params"]["strict"]["take"] == 2


def test_veo_inventing_dialogue_is_never_success(speech_like_video, tmp_path, monkeypatch):
    app, gw = strict_app(tmp_path, monkeypatch, video=speech_like_video,
                         asr=FakeASR(veo_text="Welcome everyone. Hello, this is a test of the emergency broadcast system."))
    with TestClient(app) as c:
        pid, p, _ = run_to_review(c, tmp_path)
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        p, _ = wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        v = p["params"]["strict"]["validation"]
        assert p["state"] == "NEEDS_REVIEW" and not v["checks"]["generated_person_says_the_script"]
        assert v["checks"]["final_speech_matches_script"] and v["checks"]["veo_audio_absent"]  # still our audio only
        r = c.post(f"/api/projects/{pid}/strict/confirm-identity", json={"same_person": True, "reviewer": "QA"})
        assert r.status_code == 400 and "checks failed" in r.json()["detail"]
        assert c.get(f"/api/projects/{pid}").json()["state"] == "NEEDS_REVIEW"


def test_failed_identity_validation_prevents_success(speech_like_video, tmp_path, monkeypatch):    # (12)
    app, _ = strict_app(tmp_path, monkeypatch, video=speech_like_video, facekit_fake=FakeFaceKit(video_face_scale=1.6))
    with TestClient(app) as c:
        pid, p, _ = run_to_review(c, tmp_path)
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        p, _ = wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        assert not p["params"]["strict"]["validation"]["checks"]["face_proportions_similar_to_photo"]
        r = c.post(f"/api/projects/{pid}/strict/confirm-identity", json={"same_person": True, "reviewer": "QA"})
        assert r.status_code == 400 and c.get(f"/api/projects/{pid}").json()["state"] == "NEEDS_REVIEW"
        r = c.post(f"/api/projects/{pid}/strict/confirm-identity", json={"same_person": False, "reviewer": "QA"})
        assert r.json()["state"] == "FAILED"


def test_cached_or_altered_output_is_not_accepted(speech_like_video, tmp_path, monkeypatch):    # (14)
    app, _ = strict_app(tmp_path, monkeypatch, video=speech_like_video)
    with TestClient(app) as c:
        pid, p, _ = run_to_review(c, tmp_path)
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        wait_for(c, pid, {"NEEDS_REVIEW"}, timeout=120)
        final = tmp_path / "data" / "projects" / pid / "output" / "final_talking_head.mp4"
        final.write_bytes(final.read_bytes() + b"\0")  # a different file than the one validated
        r = c.post(f"/api/projects/{pid}/strict/confirm-identity", json={"same_person": True, "reviewer": "QA"})
        assert r.status_code == 400 and "do not belong" in r.json()["detail"]


def test_ffmpeg_failure_is_reported(speech_like_video, tmp_path, monkeypatch):    # (11)
    app, _ = strict_app(tmp_path, monkeypatch, video=speech_like_video)

    def broken(*a, **k):
        raise MediaError("FFmpeg failed while composing the final video: simulated")

    monkeypatch.setattr(strict, "compose", broken)
    with TestClient(app) as c:
        pid, p, _ = run_to_review(c, tmp_path)
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        p, _ = wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        assert p["state"] == "FAILED" and p["failed_stage"] == "COMPOSING" and "simulated" in p["error"]


def test_script_too_long_for_one_clip_stops_before_spending(speech_like_video, tmp_path, monkeypatch):
    app, gw = strict_app(tmp_path, monkeypatch, video=speech_like_video, voice=FakeVoice(seconds_per_char=0.09))
    with TestClient(app) as c:
        pid, p, _ = run_to_review(c, tmp_path)
        assert p["state"] == "FAILED" and "Shorten the script" in p["error"] and gw.started == 0


def test_artifacts_belong_to_their_own_project(speech_like_video, tmp_path, monkeypatch):    # (15)
    app, _ = strict_app(tmp_path, monkeypatch, video=speech_like_video)
    with TestClient(app) as c:
        a, _, _ = run_to_review(c, tmp_path, name="job-aaa")
        b, _, _ = run_to_review(c, tmp_path, name="job-bbb", script="Choose care. Choose peace of mind.")
        root = tmp_path / "data" / "projects"
        assert (root / "job-aaa" / "audio" / "approved_script.txt").read_text().strip() == SCRIPT
        assert (root / "job-bbb" / "audio" / "approved_script.txt").read_text().strip() == "Choose care. Choose peace of mind."
        assert c.get("/api/projects/job-aaa").json()["files"]["tts.wav"].startswith("/api/projects/job-aaa/")


def test_lipsync_validation_rejects_offset_audio(tmp_path):
    from .conftest import tone_bursts

    native, shifted = tmp_path / "n.wav", tmp_path / "s.wav"
    tone_bursts(native, [(0.5, 1.2), (2.0, 2.6), (3.4, 4.4)], 6.0)
    tone_bursts(shifted, [(0.9, 1.6), (2.4, 3.0), (3.8, 4.8)], 6.0)  # 400 ms late
    n = 144
    series = {"fps": 24.0, "frames": n, "activity": list(lipsync.envelope(native, 24, n) + 1e-3),
              "face_counts": [1] * n, "yaw_deg": [0.0] * n}
    good = lipsync.validate(Path("v.mp4"), native, native, series=series)
    bad = lipsync.validate(Path("v.mp4"), shifted, native, series=series)
    assert good["passed"] and good["lag_delta_frames"] == 0
    assert not bad["passed"] and bad["lag_delta_frames"] < -2 and "behind the mouth" in bad["reasons"][0]


def test_reference_original_is_never_modified_and_backdrop_is_replaced_in_the_copy(tmp_path, monkeypatch):
    from studio.server import reference

    monkeypatch.setattr(facekit, "analyse", FakeFaceKit())
    seen = {}

    def fake_segment(src, out):
        seen["src"] = Path(src).name
        __import__("shutil").copy(src, out)
        return 0.55

    monkeypatch.setattr(facekit, "segment_person", fake_segment)
    original = _png(tmp_path)
    before = original.read_bytes()
    meta = reference.prepare(original, tmp_path / "ref", mode="face_crop")
    assert original.read_bytes() == before
    assert seen["src"] != original.name  # segmentation runs on the derived crop, not the upload
    assert meta["prepared"]["background"].startswith("plain grey") and meta["original"]["sha256"] == reference.sha256(original)


def test_align_by_regions_follows_measured_pauses_and_cuts_between_words():
    # Cloned speech: 14 words over 4.46 s. On screen: three stretches of sound, as measured on a real Veo clip.
    t = [(0.0, 0.38), (0.38, 0.72), (0.72, 0.92), (0.92, 1.34), (1.34, 1.64), (1.64, 1.96), (1.96, 2.24),
         (2.24, 2.92), (2.92, 3.26), (3.26, 3.4), (3.4, 3.64), (3.64, 3.84), (3.84, 4.02), (4.02, 4.46)]
    words = [{"word": f"w{i}", "start": a, "end": b} for i, (a, b) in enumerate(t)]
    regions = [(0.9, 2.1), (2.5, 5.66), (6.1, 6.78)]
    pieces = lipsync.align_by_regions(words, 4.46, regions)
    assert [p["veo_start"] for p in pieces] == [0.9, 2.5, 6.1]
    boundaries = {round((a[1] + b[0]) / 2, 3) for a, b in zip(t, t[1:])}
    assert all(p["tts_end"] in boundaries or p["tts_end"] == 4.46 for p in pieces)  # never inside a word
    total = sum(b - a for a, b in regions)
    for p, (a, b) in zip(pieces, regions):  # each part's share matches its region's share (within a word)
        assert abs((p["tts_end"] - p["tts_start"]) / 4.46 - (b - a) / total) < 0.08


def test_resync_reuses_the_clip_and_never_generates_again(speech_like_video, tmp_path, monkeypatch):
    app, gw = strict_app(tmp_path, monkeypatch, video=speech_like_video)
    with TestClient(app) as c:
        pid, p, _ = run_to_review(c, tmp_path)
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        first, _ = wait_for(c, pid, {"NEEDS_REVIEW"}, timeout=120)
        r = c.post(f"/api/projects/{pid}/strict/resync")
        assert r.status_code == 200
        second, _ = wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        assert second["state"] == "NEEDS_REVIEW" and gw.started == 1
        assert second["params"]["strict"]["validation"]["run_id"] != first["params"]["strict"]["validation"]["run_id"]


def test_tts_method_default_turns_veo_audio_off_and_times_from_the_mouth(silent_video, tmp_path, monkeypatch):
    voice = FakeVoice(seconds_per_char=0.02)
    app, gw = strict_app(tmp_path, monkeypatch, video=silent_video, voice=voice)
    with TestClient(app) as c:
        body = {"voice_method": "tts", "reference_id": upload(c, tmp_path), "script": SCRIPT, "voice": "reference.wav",
                "consent": {"attested": True, "by": "Test Operator"}}           # veo_audio omitted: default (off)
        pid = c.post("/api/strict/projects", json=body).json()["id"]
        wait_for(c, pid, {"AWAITING_AUDIO_APPROVAL"})
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        p, _ = wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        assert p["state"] == "NEEDS_REVIEW", p["error"]
        assert gw.requests[0]["config"]["generate_audio"] is False               # Veo audio disabled
        assert f'"{SCRIPT}"' in gw.requests[0]["prompt"]                         # the mouth is told the words
        v = p["params"]["strict"]["validation"]
        assert v["checks"]["generated_person_says_the_script"] is None           # not measurable, not "passed"
        assert v["checks"]["lip_sync_timing"] is None                            # no reference audio: not claimed
        assert v["checks"]["exact_photo_animated"] is False
        assert v["checks"]["veo_audio_absent"] and v["checks"]["audio_is_cloned_voice"]
        assert "visible mouth movement" in p["sync"]["alignment_method"]
        starts = [x["start"] for x in p["sync"]["placements"]]
        assert starts[0] == pytest.approx(0.5, abs=0.25)                          # voice starts when the mouth does
        rows = {x["key"]: x["status"] for x in v["results"]}
        assert rows["visual_lip_sync"] == "UNCERTAIN" and rows["speech_onsets"] == "NOT MEASURABLE"
        assert "Uncertain: Visual lip-sync" in p["message"] and "passed" not in p["message"].lower()
        r = c.post(f"/api/projects/{pid}/strict/confirm-identity", json={"same_person": True, "reviewer": "QA"})
        assert r.json()["state"] == "LIKENESS_ACCEPTED"


def test_default_sends_the_full_uploaded_photo_byte_for_byte(speech_like_video, tmp_path, monkeypatch):
    app, gw = strict_app(tmp_path, monkeypatch, video=speech_like_video)
    with TestClient(app) as c:
        pid, p, _ = run_to_review(c, tmp_path)
        assert p["params"]["strict"]["reference"]["mode"] == "full"
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        body = gw.requests[0]
        refs = body["config"]["reference_images"]
        assert len(refs) == 1 and refs[0]["image"]["mime_type"] == "image/png"
        assert __import__("base64").b64decode(refs[0]["image"]["image_bytes"]) == _png(tmp_path).read_bytes()
        assert "Keep the background and setting exactly as in the reference image." in body["prompt"]
        assert "Keep the framing of the reference image." in body["prompt"]
        assert "light-grey background" not in body["prompt"]
        assert "logos" not in body["config"]["negative_prompt"]          # the photo's own background may contain them
        assert not (tmp_path / "data" / "projects" / pid / "reference" / "reference_prepared.png").exists()


def test_face_crop_mode_sends_the_derived_crop(speech_like_video, tmp_path, monkeypatch):
    app, gw = strict_app(tmp_path, monkeypatch, video=speech_like_video)
    with TestClient(app) as c:
        pid, p, _ = run_to_review(c, tmp_path, reference_mode="face_crop")
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        prepared = (tmp_path / "data" / "projects" / pid / "reference" / "reference_prepared.png").read_bytes()
        sent = __import__("base64").b64decode(gw.requests[0]["config"]["reference_images"][0]["image"]["image_bytes"])
        assert sent == prepared and "light-grey background" in gw.requests[0]["prompt"]


def test_visual_regions_find_the_talking_mouth():
    n = 192
    t = np.arange(n) / 24
    act = np.where(((t > 1.0) & (t < 2.5)) | ((t > 4.0) & (t < 6.0)), 10.0, 2.0) + np.random.default_rng(0).normal(0, 0.3, n)
    regions = lipsync.visual_regions({"fps": 24.0, "activity": list(act)})
    assert len(regions) == 2
    assert regions[0][0] == pytest.approx(1.0, abs=0.2) and regions[1][1] == pytest.approx(6.0, abs=0.2)


def test_face_pixels_must_survive_background_replacement(tmp_path, monkeypatch):
    from PIL import Image

    from studio.server import reference

    monkeypatch.setattr(facekit, "analyse", FakeFaceKit())

    def ruin_face(src, out):  # a segmentation that also paints over the face
        Image.new("RGB", Image.open(src).size, (220, 222, 222)).save(out)
        return 0.6

    monkeypatch.setattr(facekit, "segment_person", ruin_face)
    meta = reference.prepare(_png(tmp_path), tmp_path / "ref", mode="face_crop")
    assert meta["prepared"]["background"].startswith("original (segmentation altered")
    assert meta["prepared"]["face_pixels_identical"] == 1.0 and meta["prepared"]["resized"] is False


def test_cuts_snap_to_silence_not_inside_a_word():
    sr = 16000
    t = np.arange(int(sr * 2.0)) / sr
    wav = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    wav[int(1.05 * sr):int(1.09 * sr)] = 0.0          # the real gap between two words is at 1.05-1.09 s
    assert lipsync.quietest_point(wav, sr, 1.0) == pytest.approx(1.07, abs=0.02)


# ------------------------------------------------------------------ voice conversion of the generated audio (default)


def convert_project(c, tmp_path, **over):
    body = {"reference_id": upload(c, tmp_path), "script": SCRIPT, "voice": "reference.wav",
            "consent": {"attested": True, "by": "Test Operator"}}
    body.update(over)
    r = c.post("/api/strict/projects", json=body)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    p, _ = wait_for(c, pid, {"AWAITING_AUDIO_APPROVAL", "FAILED"})
    return pid, p


def test_default_converts_the_complete_generated_audio_with_identical_timeline(speech_like_video, tmp_path, monkeypatch):
    voice = FakeVoice()
    app, gw = strict_app(tmp_path, monkeypatch, video=speech_like_video, voice=voice)
    with TestClient(app) as c:
        pid, p = convert_project(c, tmp_path)
        assert p["state"] == "AWAITING_AUDIO_APPROVAL" and p["params"]["strict"]["voice_method"] == "convert"
        assert voice.calls == []                                             # no text-to-speech in this method
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        p, _ = wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        assert p["state"] == "NEEDS_REVIEW", p["error"]
        assert gw.requests[0]["config"]["generate_audio"] is True            # the speech comes from the video
        assert f'"{SCRIPT}"' in gw.requests[0]["prompt"]
        folder = tmp_path / "data" / "projects" / pid
        src, ref = voice.converted[0]
        assert Path(src) == folder / "audio" / "original_veo_audio.wav" and Path(ref).name == "reference.wav"
        src_info, video_audio = sf.info(src), probe(folder / "generated.mp4")
        assert abs(src_info.duration - video_audio["duration"]) < 0.03       # the complete track, not trimmed
        rep = json.loads((folder / "reports" / "voice_conversion_timing.json").read_text())
        assert rep["length_identical"] and rep["source"]["trimmed"] is False
        assert abs(rep["global_offset_after_ms"]) <= 1.0                       # whole track aligned on speech
        rows = {x["key"]: x["status"] for x in p["params"]["strict"]["validation"]["results"]}
        assert rows["complete_source_to_vc"] == "PASS" and rows["duration_exact"] == "PASS"
        assert rows["phoneme_timing"] == "NOT GUARANTEED" and rows["photo_animated"] == "NOT GUARANTEED"
        v = p["params"]["strict"]["validation"]["checks"]
        assert v["audio_is_cloned_voice"] and v["generated_person_says_the_script"]
        assert v["veo_audio_absent"]  # the converted voice, not Veo's waveform, is in the final file


def test_conversion_offset_and_length_are_corrected_to_the_sample(speech_like_video, tmp_path, monkeypatch):
    voice = FakeVoice(convert_delay=120, convert_extra=500)                 # 5 ms late, 26 ms too long
    app, _ = strict_app(tmp_path, monkeypatch, video=speech_like_video, voice=voice)
    with TestClient(app) as c:
        pid, _ = convert_project(c, tmp_path)
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        p, _ = wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        rep = json.loads((tmp_path / "data" / "projects" / pid / "reports" / "voice_conversion_timing.json").read_text())
        assert rep["chatterbox_output_length_diff_ms"] > 25 and rep["global_offset_before_ms"] == pytest.approx(5.0, abs=0.3)
        assert rep["length_identical"] and abs(rep["global_offset_after_ms"]) <= 1.0
        rows = {x["key"]: x["status"] for x in p["params"]["strict"]["validation"]["results"]}
        assert rows["global_alignment"] == "PASS" and rows["duration_exact"] == "PASS"


def test_conversion_never_accepts_a_video_that_said_other_words(speech_like_video, tmp_path, monkeypatch):
    app, _ = strict_app(tmp_path, monkeypatch, video=speech_like_video,
                        asr=FakeASR(veo_text="Welcome everyone. Hello, this is a test of the emergency broadcast system."))
    with TestClient(app) as c:
        pid, _ = convert_project(c, tmp_path)
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        p, _ = wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        assert p["params"]["strict"]["validation"]["checks"]["generated_person_says_the_script"] is False
        r = c.post(f"/api/projects/{pid}/strict/confirm-identity", json={"same_person": True, "reviewer": "QA"})
        assert r.status_code == 400


def test_conversion_refuses_a_script_too_long_for_one_clip_before_spending(speech_like_video, tmp_path, monkeypatch):
    app, gw = strict_app(tmp_path, monkeypatch, video=speech_like_video)
    with TestClient(app) as c:
        long_script = " ".join(["Prudential Health India helps people make informed choices."] * 3)
        pid, p = convert_project(c, tmp_path, script=long_script)
        assert p["state"] == "FAILED" and "Shorten the script" in p["error"] and gw.started == 0


def test_timing_helpers_are_sample_exact():
    from studio.server import timing

    sr = 24000
    t = np.arange(sr * 3) / sr
    a = (np.sin(2 * np.pi * 150 * t) * ((t > 0.5) & (t < 1.2) | (t > 1.8) & (t < 2.5))).astype(np.float32)
    late = timing.shift(a, -72)                                                # 3 ms late
    lag, _ = timing.global_lag_samples(a, late, sr=sr)
    assert lag == 72 and np.array_equal(timing.shift(late, lag)[:-72], a[:-72])
    assert timing.match_length(a, 100).size == 100 and timing.match_length(a[:10], 100).size == 100
    comp = timing.onset_comparison(a, late, sr)
    assert [p["diff_ms"] for p in comp["pairs"]] == [3.0, 3.0] and comp["status"] == "PARTIAL"
