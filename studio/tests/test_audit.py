"""Audit-grade checks: duration, alignment, onset certainty, stream mapping, audio source,
status vocabulary, failure handling and old-project compatibility.

Local only:
- synthetic media built with FFmpeg and numpy
- the real assessment and timing code
- no cloud calls
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from studio.server import results, timing
from studio.server.compose import compose
from studio.server.media import extract_audio

from .conftest import FakeVoice, ffmpeg, tone_bursts

SR = 24000
STATUSES = {"PASS", "FAIL", "PARTIAL", "NOT TESTED", "NOT GUARANTEED", "UNCERTAIN", "NOT MEASURABLE"}


def speechlike(bursts, dur, sr=SR, noise_db=-70.0, level=0.3, seed=0, carrier=180.0):
    """Amplitude-modulated harmonic bursts (vowel-like) with optional background noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    for a, b in bursts:
        m = (t >= a) & (t < b)
        ramp = np.clip(np.minimum(t[m] - a, b - t[m]) / 0.01, 0, 1)
        x[m] = level * ramp * (np.sin(2 * np.pi * carrier * t[m]) + 0.5 * np.sin(2 * np.pi * 2 * carrier * t[m]))
    x += 10 ** (noise_db / 20) * rng.standard_normal(t.size)
    return x.astype(np.float32)


# ------------------------------------------------------------------ timing primitives


def test_global_offset_sign_convention_both_directions():
    a = speechlike([(0.5, 1.2), (1.8, 2.6)], 3.0)
    late, early = timing.shift(a, -120), timing.shift(a, 96)        # 5 ms late, 4 ms early
    assert timing.global_lag_samples(a, late)[0] == 120
    assert timing.global_lag_samples(a, early)[0] == -96
    assert np.array_equal(timing.shift(late, 120)[:-120], a[:-120])


def test_alignment_ignores_noise_floor_and_loudness_differences():
    src = speechlike([(0.5, 1.2), (1.8, 2.6), (3.2, 3.6)], 4.0, noise_db=-75)
    conv = timing.shift(speechlike([(0.5, 1.2), (1.8, 2.6), (3.2, 3.6)], 4.0, noise_db=-48, level=0.1, seed=3,
                                   carrier=137.0), -72)                     # 3 ms late, -10 dB, louder noise
    lag, r = timing.global_lag_samples(src, conv)
    assert lag == pytest.approx(72, abs=6) and r > 0.8


def test_silence_is_not_measurable_not_zero():
    silent = np.zeros(SR * 3, np.float32)
    comp = timing.onset_comparison(silent, silent)
    assert comp["status"] == "NOT MEASURABLE" and comp["mean_abs_ms"] is None and comp["measured"] == 0


def test_breaths_in_pauses_are_not_counted_as_speech_onsets():
    src = speechlike([(0.5, 1.2), (2.0, 2.8)], 3.5)
    conv = src.copy()
    t = np.arange(conv.size) / SR
    breath = (t > 1.6) & (t < 1.615)                                        # 15 ms burst, loud
    conv[breath] += 0.25 * np.sin(2 * np.pi * 900 * t[breath])
    assert len(timing.onsets(conv)) == len(timing.onsets(src)) == 2


def test_different_noise_floor_does_not_create_or_move_onsets():
    src = speechlike([(0.5, 1.2), (2.0, 2.8)], 3.5, noise_db=-75)
    conv = speechlike([(0.5, 1.2), (2.0, 2.8)], 3.5, noise_db=-50, seed=9, carrier=150.0)
    comp = timing.onset_comparison(src, conv)
    assert comp["measured"] == 2 and comp["max_abs_ms"] <= 2.0


def test_inconsistent_onsets_are_uncertain_not_numbers():
    src = speechlike([(0.5, 1.2), (2.0, 2.8)], 3.5)
    conv = speechlike([(0.5, 1.2), (2.0, 2.8)], 3.5)
    t = np.arange(conv.size) / SR
    soft = (t > 1.95) & (t < 2.0)                                           # a quiet consonant only in the conversion
    conv[soft] += 0.02 * np.sin(2 * np.pi * 300 * t[soft])
    comp = timing.onset_comparison(src, conv)
    second = comp["pairs"][1]
    assert second["status"] == "UNCERTAIN" or abs(second["diff_ms"]) > 0      # never a fabricated 0
    assert comp["status"] in ("UNCERTAIN", "PARTIAL", "FAIL")


def test_unmatched_onset_is_uncertain():
    src = speechlike([(0.5, 1.2), (2.0, 2.8)], 3.5)
    conv = speechlike([(0.5, 1.2)], 3.5)
    comp = timing.onset_comparison(src, conv)
    assert comp["status"] == "UNCERTAIN" and any(p["status"] == "UNCERTAIN" for p in comp["pairs"])


# ------------------------------------------------------------------ assessment of a composed project


@pytest.fixture
def project(tmp_path, speech_like_video):
    """A strict 'convert' project on disk: generated clip, extracted Veo audio, converted voice,
    final MP4 composed by the real compose(), and the reports the pipeline writes."""
    folder = tmp_path / "proj"
    for sub in ("audio", "reports", "output", "video"):
        (folder / sub).mkdir(parents=True)
    shutil.copy(speech_like_video, folder / "generated.mp4")
    files = extract_audio(folder / "generated.mp4", folder)
    shutil.copy(files["original"], folder / "audio" / "original_veo_audio.wav")
    src, src_sr = sf.read(files["original"], dtype="float32", always_2d=True)
    import librosa

    src24 = librosa.resample(src.mean(1), orig_sr=src_sr, target_sr=SR)
    env = np.sqrt(np.convolve(src24.astype(np.float64) ** 2, np.ones(48) / 48, mode="same"))
    voice = (env * 1.4 * np.sin(2 * np.pi * 137 * np.arange(src24.size) / SR)).astype(np.float32)
    voice = timing.match_length(voice, int(round(4.0 * SR)))
    sf.write(folder / "cloned_voice.wav", voice, SR)
    shutil.copy(folder / "cloned_voice.wav", folder / "audio" / "converted_voice.wav")
    (folder / "reports" / "voice_conversion_timing.json").write_text(json.dumps(
        {"source": {"samples": int(src.shape[0]), "trimmed": False}, "final_length_samples_24k": voice.size,
         "chatterbox_output_length_diff_ms": 0.0, "global_offset_before_ms": 0.0, "constant_gain_db": 0.0}))
    compose(folder / "generated.mp4", folder / "cloned_voice.wav", folder / "output" / "final_talking_head.mp4",
            final_duration=4.0, freeze_seconds=0, log_path=folder / "reports" / "ffmpeg_compose.log")
    (folder / "reports" / "veo_speech_check.json").write_text(json.dumps({"passed": True, "transcript": "hi", "word_accuracy": 1.0}))
    (folder / "reports" / "final_validation.json").write_text(json.dumps(
        {"details": {"final_transcript": {"passed": True, "transcript": "hi", "word_accuracy": 1.0}}}))
    return folder


STRICT = {"voice_method": "convert", "script": "hi"}


def status(res, key):
    return next(r["status"] for r in res["rows"] if r["key"] == key)


def test_assessment_of_a_correct_project(project):
    before = {p.name: p.read_bytes() for p in [project / "output" / "final_talking_head.mp4", project / "cloned_voice.wav"]}
    res = results.assess(project, STRICT)
    for key in ("complete_source_to_vc", "selected_audio_in_final", "veo_audio_excluded", "duration_exact",
                "stream_mapping", "video_frames_preserved", "av_duration", "global_alignment"):
        assert status(res, key) == "PASS", (key, next(r for r in res["rows"] if r["key"] == key))
    assert status(res, "phoneme_timing") == "NOT GUARANTEED" and status(res, "photo_animated") == "NOT GUARANTEED"
    assert {r["status"] for r in res["rows"]} <= STATUSES
    assert res["facts"]["durations"]["final_audio_samples_48k"] == 192000          # sample count, not container time
    after = {p.name: p.read_bytes() for p in [project / "output" / "final_talking_head.mp4", project / "cloned_voice.wav"]}
    assert before == after                                                          # read-only


def test_wrong_voice_sample_count_fails_duration(project):
    v, _ = sf.read(project / "cloned_voice.wav", dtype="float32")
    sf.write(project / "cloned_voice.wav", v[:-240], SR)                            # 10 ms short
    assert status(results.assess(project, STRICT), "duration_exact") == "FAIL"


def test_final_without_audio_stream_fails(project):
    final = project / "output" / "final_talking_head.mp4"
    tmp = project / "noaudio.mp4"
    ffmpeg("-i", str(final), "-an", "-c", "copy", str(tmp))
    res = results.assess(project, STRICT, final=tmp)
    assert status(res, "stream_mapping") == "FAIL" and status(res, "selected_audio_in_final") == "FAIL"
    assert status(res, "duration_exact") == "FAIL"


@pytest.mark.parametrize("rate,channels", [(44100, 2), (48000, 1)])
def test_wrong_audio_sample_rate_or_channels_fails_mapping(project, rate, channels):
    final = project / "output" / "final_talking_head.mp4"
    bad = project / f"bad_{rate}_{channels}.mp4"
    ffmpeg("-i", str(final), "-c:v", "copy", "-c:a", "aac", "-ar", str(rate), "-ac", str(channels), str(bad))
    assert status(results.assess(project, STRICT, final=bad), "stream_mapping") == "FAIL"


def test_final_carrying_veo_audio_is_detected(project):
    final = project / "output" / "final_talking_head.mp4"
    leaked = project / "leaked.mp4"
    ffmpeg("-i", str(project / "generated.mp4"), "-c", "copy", str(leaked))      # Veo's own soundtrack
    res = results.assess(project, STRICT, final=leaked)
    assert status(res, "selected_audio_in_final") == "FAIL" and status(res, "veo_audio_excluded") == "FAIL"


def test_selected_track_must_be_the_recorded_derivative(project):
    other = np.zeros(int(4.0 * SR), np.float32)
    sf.write(project / "audio" / "converted_voice.wav", other, SR)                 # provenance mismatch
    assert status(results.assess(project, STRICT), "selected_audio_in_final") == "FAIL"


def test_truncated_source_fails_completeness(project):
    rep = json.loads((project / "reports" / "voice_conversion_timing.json").read_text())
    rep["source"]["samples"] -= 4800
    (project / "reports" / "voice_conversion_timing.json").write_text(json.dumps(rep))
    assert status(results.assess(project, STRICT), "complete_source_to_vc") == "FAIL"


def test_any_fail_blocks_and_not_guaranteed_does_not(project):
    res = results.assess(project, STRICT)
    assert "identity" not in results.blocking_failures(res)
    res["rows"].append({"key": "x", "label": "x", "status": "FAIL", "evidence": ""})
    assert results.blocking_failures(res) == ["x"]


# ------------------------------------------------------------------ pipeline behaviour


def _strict_app(tmp_path, monkeypatch, video, voice):
    from .test_strict import strict_app

    return strict_app(tmp_path, monkeypatch, video=video, voice=voice)


def test_failed_conversion_never_falls_back_to_veo_audio(speech_like_video, tmp_path, monkeypatch):
    from studio.server.voice import VoiceError

    from .test_strict import convert_project, wait_for

    class Broken(FakeVoice):
        def convert(self, *a, **k):
            raise VoiceError("Chatterbox voice conversion failed: simulated")

    app, _ = _strict_app(tmp_path, monkeypatch, speech_like_video, Broken())
    with TestClient(app) as c:
        pid, _ = convert_project(c, tmp_path)
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        p, _ = wait_for(c, pid, {"FAILED", "NEEDS_REVIEW"}, timeout=120)
        folder = tmp_path / "data" / "projects" / pid
        assert p["state"] == "FAILED" and "simulated" in p["error"]
        assert not (folder / "output" / "final_talking_head.mp4").exists() and not (folder / "final.mp4").exists()
        assert not (folder / "cloned_voice.wav").exists()


def test_stale_audio_from_a_previous_run_is_never_converted(speech_like_video, silent_video, tmp_path, monkeypatch):
    from .test_strict import convert_project, wait_for

    app, _ = _strict_app(tmp_path, monkeypatch, speech_like_video, FakeVoice())
    with TestClient(app) as c:
        pid, _ = convert_project(c, tmp_path)
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        wait_for(c, pid, {"NEEDS_REVIEW"}, timeout=120)
        folder = tmp_path / "data" / "projects" / pid
        assert (folder / "audio" / "original_veo_audio.wav").exists()
        shutil.copy(silent_video, folder / "video" / "scene_01.mp4")                 # the clip now has no audio
        c.post(f"/api/projects/{pid}/strict/resync")
        p, _ = wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        assert p["state"] == "FAILED" and "no audio to convert" in p["error"]
        assert not (folder / "audio" / "original_veo_audio.wav").exists()


def test_status_messages_never_claim_an_overall_pass(speech_like_video, tmp_path, monkeypatch):
    from .test_strict import convert_project, wait_for

    app, _ = _strict_app(tmp_path, monkeypatch, speech_like_video, FakeVoice())
    with TestClient(app) as c:
        pid, _ = convert_project(c, tmp_path)
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        p, _ = wait_for(c, pid, {"NEEDS_REVIEW"}, timeout=120)
        rows = p["params"]["strict"]["validation"]["results"]
        assert {r["status"] for r in rows} <= STATUSES and all(r["evidence"] for r in rows)
        assert "passed" not in p["message"].lower() and "Not guaranteed" in p["message"]
        for key in ("identity", "photo_animated", "phoneme_timing"):
            assert next(r["status"] for r in rows if r["key"] == key) != "PASS"


def test_projects_validated_before_the_status_table_still_work(speech_like_video, tmp_path, monkeypatch):
    from .test_strict import convert_project, wait_for

    app, _ = _strict_app(tmp_path, monkeypatch, speech_like_video, FakeVoice())
    with TestClient(app) as c:
        pid, _ = convert_project(c, tmp_path)
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        wait_for(c, pid, {"NEEDS_REVIEW"}, timeout=120)
        store = app.state.pipeline.store
        params = store.get(pid)["params"]
        params["strict"]["validation"].pop("results")                               # an older record
        params["strict"]["validation"].pop("facts")
        params["strict"].pop("voice_method")
        store.update(pid, params=params)
        assert c.get(f"/api/projects/{pid}").status_code == 200
        r = c.post(f"/api/projects/{pid}/strict/confirm-identity", json={"same_person": True, "reviewer": "QA"})
        assert r.status_code == 200 and r.json()["state"] == "LIKENESS_ACCEPTED"
