"""Long talking head: script splitting, per-take variants, and the clip-by-clip flow + hard-cut stitching.

Mocked providers only (the same fakes as the strict tests); FFmpeg runs for real. No quota.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from studio.server import direction, longform
from studio.server.prompt_compiler import compile_prompt, negative_prompt

from studio.server.strict import pick_take

from .conftest import FakeVoice
from .test_strict import FakeASR, strict_app, upload

THREE = ("Good morning everyone, and thank you so much for joining us here today. "
         "At Prudential Health India, we work hard to make your health choices simpler. "
         "Please take good care of yourself and of the people you love.")


# ------------------------------------------------------------------ splitting


def test_split_keeps_sentences_whole_and_every_clip_fits():
    segs = longform.split_script(THREE)
    assert segs == ["Good morning everyone, and thank you so much for joining us here today.",
                    "At Prudential Health India, we work hard to make your health choices simpler.",
                    "Please take good care of yourself and of the people you love."]
    assert longform.split_script("Hello there. Thank you for coming.") == ["Hello there. Thank you for coming."]
    long = ("We help families compare plans, understand their cover, choose the right hospital, "
            "plan for the future and stay healthy every single day of the year.")
    parts = longform.split_script(long)
    assert len(parts) == 2 and all(len(p.split()) <= longform.MAX_SEGMENT_WORDS for p in parts)
    assert " ".join(parts) == long                                     # words never change, only where clips cut


def test_split_uses_the_fewest_clips():
    s = " ".join(["Stay well."] * 20)                                  # 40 words: 3 clips of at most 17
    segs = longform.split_script(s)
    assert len(segs) == 3 and sum(len(x.split()) for x in segs) == 40


def test_split_rejects_bad_scripts():
    with pytest.raises(longform.LongError, match="double quotes"):
        longform.split_script('He said "hello" to everyone.')
    with pytest.raises(longform.LongError, match="limit is"):
        longform.split_script(" ".join(["This sentence has exactly ten words in it, really."] * 30))
    with pytest.raises(longform.LongError):
        longform.split_script("   ")


# ------------------------------------------------------------------ variants and the compiler


def test_default_direction_prompt_is_unchanged():
    d = direction.for_strict(script="Hello everyone! Your health matters.", reference_mode="full",
                             voice_method="convert", speech_seconds=3.0)
    p = compile_prompt(d)
    assert "Timed actions" not in p and "stressing" not in p
    assert p.endswith('starting right away, taking about 3.0 seconds, then stays silent with a relaxed closed mouth: '
                      '"Hello everyone! Your health matters." Photorealistic, with natural skin texture: no beauty '
                      'filter, no waxy or over-smoothed skin and no artificial facial symmetry. No cuts, no scene '
                      'changes, no other people, no B-roll and no added text or subtitles. No facial distortion. '
                      "Audio: only the person's voice, no music and no sound effects.")
    assert "No unnecessary hand gestures." in p and "Head movement is minimal" in p


def test_variants_differ_between_neighbours_and_always_validate():
    script = " ".join(f"Sentence number {i} keeps this example script moving along nicely." for i in range(15))
    segs = longform.plan(script)
    assert len(segs) == 15
    prompts = [s["prompt"] for s in segs]
    assert len(set(prompts)) == 15
    for a, b in zip(segs, segs[1:]):
        assert a["summary"]["framing"] != b["summary"]["framing"]
        assert a["summary"]["actions"] != b["summary"]["actions"]
    for s in segs:
        assert "Timed actions: [00:00-" in s["prompt"]
        hands = any(x in s["summary"]["actions"] for x in ("open-palm gesture", "two-hand gesture", "hand to chest"))
        assert not hands or s["summary"]["framing"] == "photo framing"   # hands only when they are in frame
    assert "finishes with a small nod" in prompts[-1] and "soft, closed-mouth smile" in prompts[0]
    voice = ("The voice is a warm, mid-pitched, natural speaking voice with the same calm, even energy from start to "
             "finish: no rising excitement, no dramatic pitch changes. The pace is steady and unhurried, about 2.6 words "
             "per second, never speeding up.")
    assert all(voice in p for p in prompts)                             # one voice and speed in every clip
    assert longform.plan(script) == segs                                # deterministic


def test_face_crop_mode_never_asks_for_hands():
    for s in longform.plan(THREE, reference_mode="face_crop"):
        assert "Head-and-shoulders framing." in s["prompt"] and "hand" not in " ".join(s["summary"]["actions"])


def test_direction_rejects_inconsistent_beats():
    base = {"performance": {"hand_gestures": "occasional_natural",
                            "beats": [{"start_s": 0, "end_s": 2, "action": "open_palm"}]}}
    with pytest.raises(direction.DirectionError, match="hands in frame"):
        direction.for_strict(script="Hello there friends.", overrides={**base, "scene": {"composition": "medium_close_up"}})
    with pytest.raises(direction.DirectionError, match="overlap"):
        direction.for_strict(script="Hello there friends.", overrides={"performance": {"beats": [
            {"start_s": 0, "end_s": 3, "action": "small_nod"}, {"start_s": 2, "end_s": 4, "action": "warm_smile"}]}})
    with pytest.raises(direction.DirectionError, match="not in the script"):
        direction.for_strict(script="Hello there friends.", overrides={"speech": {"emphasis": "insurance"}})
    with pytest.raises(direction.DirectionError):
        direction.for_strict(script="Hello there friends.", overrides={"performance": {"beats": [
            {"start_s": 0, "end_s": 2, "action": "juggle"}]}})
    d = direction.for_strict(script="Hello there friends.", overrides=base)
    assert "one hand rises briefly" in compile_prompt(d) and "exaggerated head movement" in negative_prompt(d)


# ------------------------------------------------------------------ the flow


class LongASR(FakeASR):
    """Each child's generated video says that child's own words; each cut piece of the final
    video says its clip's words."""

    def __init__(self, pieces=None):
        super().__init__()
        self.pieces = pieces or []

    def transcribe(self, audio: Path) -> dict:
        audio = Path(audio)
        if audio.name.startswith("piece_"):
            text = self.pieces[int(audio.stem.split("_")[1])]
            return {"model": "fake", "text": text, "words": [], "duration": 1.0}
        script = next((p / "audio" / "approved_script.txt" for p in audio.parents
                       if (p / "audio" / "approved_script.txt").exists()), None)
        text = script.read_text().strip() if script else ""
        if audio.name.startswith("speech_16k"):
            self.veo_text = text
            return super().transcribe(audio)
        words = text.split()
        times = np.linspace(0.2, 3.3, len(words), endpoint=False)
        return {"model": "fake", "text": text, "duration": 4.0,
                "words": [{"word": w, "start": float(t), "end": float(t) + 0.05} for w, t in zip(words, times)]}


def long_app(tmp_path, monkeypatch, video, voice=None):
    asr = LongASR(longform.split_script(THREE))
    app, gw = strict_app(tmp_path, monkeypatch, video=video, asr=asr, voice=voice)
    app.state.longform.poll_s = 0.05
    return app, gw


def create_long(c, tmp_path, script=THREE, **over):
    body = {"reference_id": upload(c, tmp_path), "script": script, "voice": "reference.wav",
            "consent": {"attested": True, "by": "Test Operator"}}
    body.update(over)
    return c.post("/api/long/projects", json=body)


def wait_long(c, pid, cond, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = c.get(f"/api/long/{pid}").json()
        if cond(v):
            return v
        time.sleep(0.1)
    raise AssertionError(f"timed out: {v['state']} {v['message']} {v['error']}")


def test_plan_preview_needs_no_project_and_no_network(speech_like_video, tmp_path, monkeypatch):
    app, gw = long_app(tmp_path, monkeypatch, speech_like_video)
    with TestClient(app) as c:
        r = c.post("/api/long/plan", json={"script": THREE})
        assert r.status_code == 200, r.text
        plan = r.json()
        assert plan["clips"] == plan["generations"] == 3 and len({s["prompt"] for s in plan["segments"]}) == 3
        assert c.get("/api/projects").json() == [] and gw.started == 0
        assert c.post("/api/long/plan", json={"script": 'Say "hi".'}).status_code == 400


def test_long_video_end_to_end_with_hard_cuts(speech_like_video, tmp_path, monkeypatch):
    app, gw = long_app(tmp_path, monkeypatch, speech_like_video)
    with TestClient(app) as c:
        r = create_long(c, tmp_path, name="long-test")
        assert r.status_code == 200, r.text
        pid = r.json()["id"]
        v = wait_long(c, pid, lambda v: v["clips_to_generate"] == 3 and all(
            s["state"] == "AWAITING_AUDIO_APPROVAL" for s in v["segments"]))
        assert [s["project_id"] for s in v["segments"]] == ["long-test-s01", "long-test-s02", "long-test-s03"]
        assert gw.started == 0                                               # nothing paid by planning or preparing
        listed = [p["id"] for p in c.get("/api/projects").json()]
        assert listed == ["long-test"]                                       # clips live inside their video

        bad = c.post(f"/api/long/{pid}/generate", json={"reviewer": "QA", "generations": 2})
        assert bad.status_code == 400 and "uses 3 generations" in bad.json()["detail"] and gw.started == 0
        assert c.post(f"/api/long/{pid}/generate", json={"reviewer": "", "generations": 3}).status_code == 400
        r = c.post(f"/api/long/{pid}/generate", json={"reviewer": "QA", "generations": 3})
        assert r.status_code == 200, r.text
        v = wait_long(c, pid, lambda v: v["state"] == "LONG_READY" and not v["running"])
        assert v["clips_ready"] == 3, (v["error"], [s["state"] for s in v["segments"]])
        assert gw.started == 3
        prompts = [b["prompt"] for b in gw.requests]
        assert len(set(prompts)) == 3 and all("Timed actions" in p for p in prompts)
        for seg, p in zip(longform.split_script(THREE), prompts):
            assert f'"{seg}"' in p                                           # each clip says only its own words

        assert v["clips_with_failed_checks"] == [], [s["failed_checks"] for s in v["segments"]]
        r = c.post(f"/api/long/{pid}/stitch")
        assert r.status_code == 200, r.text
        v = wait_long(c, pid, lambda v: v["state"] in ("NEEDS_REVIEW", "LONG_READY") and not v["running"])
        assert v["state"] == "NEEDS_REVIEW", v["error"]
        rows = {x["key"]: x["status"] for x in v["stitch"]["results"]}
        assert rows == {"frames": "PASS", "audio_length": "PASS", "clipping": "PASS", "script": "PASS", "clips": "PASS"}
        cuts = v["stitch"]["cuts"]
        assert all(0 <= c_["start_frame"] < c_["end_frame"] <= c_["frames"] for c_ in cuts)
        assert sum(c_["end_frame"] - c_["start_frame"] for c_ in cuts) < sum(c_["frames"] for c_ in cuts)  # idle tails cut
        assert all(abs(c_["gain_db"]) <= longform.MAX_GAIN_DB for c_ in cuts)
        assert "xfade" not in json.dumps(v["stitch"]["join"]["filters"])     # hard cuts, never crossfades
        assert (tmp_path / "data" / "projects" / pid / "reports" / "long_form.json").exists()
        assert c.get(v["final_url"] + "?download=true").status_code == 200

        assert c.post(f"/api/long/{pid}/accept", json={"reviewer": ""}).status_code == 400
        r = c.post(f"/api/long/{pid}/accept", json={"reviewer": "QA"})
        assert r.status_code == 200 and r.json()["state"] == "LIKENESS_ACCEPTED"

        # Redo one clip: a new take with a different variant, costing exactly one generation.
        r = c.post(f"/api/long/{pid}/segments/1/redo")
        assert r.status_code == 200, r.text
        v = wait_long(c, pid, lambda v: v["segments"][1]["state"] == "AWAITING_AUDIO_APPROVAL")
        assert v["segments"][1]["project_id"] == "long-test-s02-t2" and v["segments"][1]["takes"] == [
            "long-test-s02", "long-test-s02-t2"]
        assert v["clips_to_generate"] == 1 and v["review"] is None
        r = c.post(f"/api/long/{pid}/generate", json={"reviewer": "QA", "generations": 1})
        assert r.status_code == 200, r.text
        wait_long(c, pid, lambda v: v["state"] == "LONG_READY" and not v["running"])
        assert gw.started == 4 and gw.requests[3]["prompt"] not in prompts


def test_generation_is_refused_while_the_provider_gate_is_off(speech_like_video, tmp_path, monkeypatch):
    from studio.server.utils_api import gated_client

    app, gw = long_app(tmp_path, monkeypatch, speech_like_video)
    with TestClient(app) as c:
        pid = create_long(c, tmp_path).json()["id"]
        wait_long(c, pid, lambda v: v["clips_to_generate"] == 3)
        app.state.pipeline._client_factory = gated_client                   # the real factory; the gate is off
        r = c.post(f"/api/long/{pid}/generate", json={"reviewer": "QA", "generations": 3})
        assert r.status_code == 400 and "switched off" in r.json()["detail"] and gw.started == 0


def test_stitching_waits_for_every_clip(speech_like_video, tmp_path, monkeypatch):
    app, _ = long_app(tmp_path, monkeypatch, speech_like_video)
    with TestClient(app) as c:
        pid = create_long(c, tmp_path).json()["id"]
        wait_long(c, pid, lambda v: v["clips_to_generate"] == 3)
        r = c.post(f"/api/long/{pid}/stitch")
        assert r.status_code == 400 and "not ready" in r.json()["detail"]
        assert c.post(f"/api/long/{pid}/accept", json={"reviewer": "QA"}).status_code == 400
        assert c.get("/api/long/nope").status_code == 404


# ------------------------------------------------------------------ one voice across clips


def _take(n, pitch, status="PARTIAL", lag=0, max_ms=5.0, measured=2):
    return {"take": n, "pitch_hz": pitch, "lag_after": lag,
            "comp": {"status": status, "measured": measured, "max_abs_ms": max_ms}}


def test_take_choice_is_by_timing_without_an_anchor_and_by_pitch_with_one():
    takes = [_take(1, 150, max_ms=2.0), _take(2, 121), _take(3, 98, max_ms=9.0)]
    assert pick_take(takes, None)[0]["take"] == 1                       # timing only
    best, why = pick_take(takes, 120.0)
    assert best["take"] == 2 and "closest to clip 1 (120 Hz)" in why
    # timing still guards the choice: a take that is misaligned or failed is never picked for its pitch
    takes = [_take(1, 150), _take(2, 120, lag=30), _take(3, 119, status="FAIL")]
    assert pick_take(takes, 120.0)[0]["take"] == 1


def test_clips_after_the_first_keep_the_take_closest_to_clip_one(speech_like_video, tmp_path, monkeypatch):
    voice = FakeVoice(tones=(120, 180, 125))         # takes 1-3 of every clip: 120, 180 and 125 Hz
    app, gw = long_app(tmp_path, monkeypatch, speech_like_video, voice=voice)
    with TestClient(app) as c:
        pid = create_long(c, tmp_path, name="voice-test").json()["id"]
        wait_long(c, pid, lambda v: v["clips_to_generate"] == 3)
        assert c.post(f"/api/long/{pid}/voice-target", json={"source": "clip1"}).json()["anchor_source"] == "clip1"
        c.post(f"/api/long/{pid}/generate", json={"reviewer": "QA", "generations": 3})
        v = wait_long(c, pid, lambda v: v["state"] == "LONG_READY" and not v["running"])
        assert v["clips_ready"] == 3 and gw.started == 3
        reps = [json.loads((tmp_path / "data" / "projects" / f"voice-test-s0{i}" / "reports" /
                            "voice_conversion_timing.json").read_text()) for i in (1, 2, 3)]
        assert reps[0]["chosen_by"] == "timing" and reps[0]["pitch_anchor"] is None
        anchor = v["anchor"]["pitch_hz"]
        assert abs(anchor - [120, 180, 125][reps[0]["chosen_take"] - 1]) < 4
        for r in reps[1:]:
            assert r["pitch_anchor"]["pitch_hz"] == anchor and "closest to clip 1" in r["chosen_by"]
            pitches = {t["take"]: t["pitch_hz"] for t in r["takes"]}
            assert r["chosen_take"] == min(pitches, key=lambda k: abs(pitches[k] - anchor))

        # Re-matching uses the clips on disk: no generation.
        assert v["can_match"]
        r = c.post(f"/api/long/{pid}/match-voices")
        assert r.status_code == 200, r.text
        v = wait_long(c, pid, lambda v: v["state"] == "LONG_READY" and not v["running"])
        assert v["error"] is None and v["clips_ready"] == 3 and gw.started == 3


def test_by_default_every_clip_matches_the_voice_recording(speech_like_video, tmp_path, monkeypatch):
    voice = FakeVoice(tones=(120, 180, 125))         # the test voice recording is a 220 Hz tone
    app, gw = long_app(tmp_path, monkeypatch, speech_like_video, voice=voice)
    with TestClient(app) as c:
        plan = c.post("/api/long/plan", json={"script": THREE, "voice": "reference.wav"}).json()
        assert all("light, higher-pitched" in s_["prompt"] for s_ in plan["segments"])   # pitch word from the recording
        pid = create_long(c, tmp_path, name="rec-test").json()["id"]
        v = wait_long(c, pid, lambda v: v["clips_to_generate"] == 3)
        assert v["anchor_source"] == "voice" and abs(v["anchor"]["pitch_hz"] - 220) < 5
        c.post(f"/api/long/{pid}/generate", json={"reviewer": "QA", "generations": 3})
        v = wait_long(c, pid, lambda v: v["state"] == "LONG_READY" and not v["running"])
        assert v["clips_ready"] == 3
        assert all("light, higher-pitched" in b["prompt"] for b in gw.requests)
        for i in (1, 2, 3):                                                 # clip 1 included
            r = json.loads((tmp_path / "data" / "projects" / f"rec-test-s0{i}" / "reports" /
                            "voice_conversion_timing.json").read_text())
            assert "closest to your voice recording" in r["chosen_by"] and r["chosen_take"] == 2   # 180 Hz is nearest
        r = c.post(f"/api/long/{pid}/match-voices")
        assert r.status_code == 200, r.text
        v = wait_long(c, pid, lambda v: v["state"] == "LONG_READY" and not v["running"])
        assert v["error"] is None and gw.started == 3
        assert c.post(f"/api/long/{pid}/voice-target", json={"source": "nope"}).status_code == 422
