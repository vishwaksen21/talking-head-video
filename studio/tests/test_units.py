"""Unit tests: API client behaviour, model catalog, sync planning, text chunking, config."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from studio.server import catalog
from studio.server.config import SettingsStore, normalize_base_url
from studio.server.sync import plan
from studio.server.utils_api import UtilsAPIError, UtilsClient, decode_base64
from studio.server.voice import split_text, trim_silence

# ------------------------------------------------------------------ utils client


def client_with(handler, **kw) -> UtilsClient:
    return UtilsClient("https://gw.test", "secret-key", transport=httpx.MockTransport(handler), sleep=lambda s: None, **kw)


def test_sends_key_in_x_api_key_header():
    seen = {}

    def handler(req):
        seen["key"] = req.headers.get("x-api-key")
        return httpx.Response(200, json={"location": "us-central1", "count": 0, "models": []})

    client_with(handler).list_models("us-central1")
    assert seen["key"] == "secret-key"


def test_401_is_a_clear_message_without_the_key():
    c = client_with(lambda req: httpx.Response(401, json={"detail": "unauthorized"}))
    with pytest.raises(UtilsAPIError) as err:
        c.list_models("us-central1")
    assert err.value.status == 401
    assert "rejected" in str(err.value)
    assert "secret-key" not in str(err.value)


def test_429_is_retried_with_retry_after():
    calls, sleeps = [], []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "7"}, json={"detail": "slow down"})
        return httpx.Response(200, json={"models": [{"id": "veo-3.1-generate-001"}]})

    c = UtilsClient("https://gw.test", "k", transport=httpx.MockTransport(handler), sleep=sleeps.append)
    assert c.list_models("us-central1")[0]["id"] == "veo-3.1-generate-001"
    assert len(calls) == 3 and sleeps == [7.0, 7.0]


def test_rate_limit_gives_up_with_message():
    c = client_with(lambda req: httpx.Response(429, json={"detail": "x"}), max_retries=2)
    with pytest.raises(UtilsAPIError, match="Rate limit"):
        c.list_models("us-central1")


def test_starting_a_video_is_not_retried_on_5xx():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(503, json={"detail": "busy"})

    with pytest.raises(UtilsAPIError):
        client_with(handler).start_video(model="veo", prompt="p", config={}, location="us-central1")
    assert len(calls) == 1  # a retry could start (and bill) a second generation


def test_polling_is_retried_on_5xx():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(502, text="bad gateway")
        assert req.url.params["operation"] == "projects/p/operations/abc"
        return httpx.Response(200, json={"operation": "projects/p/operations/abc", "done": False, "status": "running"})

    op = client_with(handler).poll_video("projects/p/operations/abc")
    assert op.status == "running" and len(calls) == 2


def test_unreachable_gateway():
    def handler(req):
        raise httpx.ConnectError("nope", request=req)

    with pytest.raises(UtilsAPIError, match="Could not reach"):
        client_with(handler, max_retries=1).list_models("global")


def test_validation_error_detail_is_surfaced():
    c = client_with(lambda req: httpx.Response(400, json={"detail": "duration_seconds must be 4, 6 or 8"}))
    with pytest.raises(UtilsAPIError, match="duration_seconds must be"):
        c.start_video(model="veo", prompt="p", config={"duration_seconds": 5}, location="us-central1")


def test_html_error_pages_are_not_dumped_into_messages():
    c = client_with(lambda req: httpx.Response(404, headers={"content-type": "text/html"}, text="<!DOCTYPE html><title>x</title>"))
    with pytest.raises(UtilsAPIError) as err:
        c.list_models("global")
    assert "<" not in str(err.value) and "Check the base URL" in str(err.value)


def test_missing_configuration():
    with pytest.raises(UtilsAPIError, match="base URL"):
        UtilsClient("", "k")
    with pytest.raises(UtilsAPIError, match="API key"):
        UtilsClient("https://x", "")


def test_decode_base64_accepts_both_alphabets():
    raw = bytes(range(256)) * 3
    assert decode_base64(base64.b64encode(raw).decode()) == raw
    assert decode_base64(base64.urlsafe_b64encode(raw).decode().rstrip("=")) == raw


# ------------------------------------------------------------------ catalog

LISTED = [{"id": i} for i in ["gemini-2.5-flash", "veo-2.0-generate-001", "veo-3.1-fast-generate-001",
                              "veo-3.1-generate-001", "veo-3.1-lite-generate-001", "chirp-3"]]


def test_discovery_keeps_only_video_models():
    ids = [m["id"] for m in catalog.discover_video_models(LISTED)]
    assert ids == ["veo-3.1-lite-generate-001", "veo-3.1-generate-001", "veo-3.1-fast-generate-001", "veo-2.0-generate-001"]


def test_capabilities_and_flags():
    caps = {m["id"]: m for m in catalog.discover_video_models(LISTED)}
    assert caps["veo-3.1-fast-generate-001"]["supports_audio"] is True
    assert caps["veo-3.1-fast-generate-001"]["granted_note"] is None
    assert caps["veo-2.0-generate-001"]["supports_audio"] is False
    assert caps["veo-2.0-generate-001"]["granted_note"]
    assert caps["veo-3.1-lite-generate-001"]["capabilities_known"] is False
    assert caps["veo-3.1-generate-001"]["label"] == "Veo 3.1"


def test_build_config_validates_against_model():
    caps = catalog.capabilities_for("veo-3.1-fast-generate-001")
    cfg = catalog.build_video_config(caps, duration=4, aspect_ratio="9:16", resolution="720p",
                                     generate_audio=True, negative_prompt=" music ", seed=7)
    assert cfg == {"number_of_videos": 1, "duration_seconds": 4, "aspect_ratio": "9:16", "resolution": "720p",
                   "generate_audio": True, "negative_prompt": "music", "seed": 7}
    with pytest.raises(ValueError, match="durations"):
        catalog.build_video_config(caps, duration=5, aspect_ratio=None, resolution=None, generate_audio=True,
                                   negative_prompt=None, seed=None)
    with pytest.raises(ValueError, match="cannot generate audio"):
        catalog.build_video_config(catalog.capabilities_for("veo-2.0-generate-001"), duration=None,
                                   aspect_ratio=None, resolution=None, generate_audio=True, negative_prompt=None, seed=None)


# ------------------------------------------------------------------ sync planning

SEGS = [{"start": 0.5, "text": "a"}, {"start": 3.0, "text": "b"}]


def test_lines_that_fit_keep_original_timing():
    p = plan(SEGS, [2.0, 2.0], 8.0)
    assert [x["start"] for x in p["placements"]] == [0.5, 3.0]
    assert all(x["tempo"] == 1.0 for x in p["placements"])
    assert p["status"] == "in_sync" and p["final_duration"] == 8.0 and p["freeze_seconds"] == 0


def test_overrun_is_sped_up_within_the_cap_then_pushed():
    p = plan(SEGS, [3.5, 1.0], 8.0, max_speedup=1.15)
    first, second = p["placements"]
    assert first["tempo"] == 1.15                      # capped, not 1.4
    assert second["start"] >= first["end"] + 0.1 - 1e-6  # pushed, never overlapping
    assert any("sped up" in w for w in p["warnings"])
    assert any("later than in the original" in w for w in p["warnings"])


def test_speech_longer_than_video_freezes_last_frame_and_keeps_last_words():
    p = plan(SEGS, [2.0, 6.0], 6.0)
    assert p["speech_end"] > 6.0
    assert p["final_duration"] >= p["speech_end"]      # last sentence never cut
    assert p["freeze_seconds"] == pytest.approx(p["final_duration"] - 6.0)
    assert p["status"] == "extended"


def test_stretch_strategy_tries_speeding_up_first():
    # Line 2 has 3.0 s before the end; 3.8 s of speech needs +27%: over the 15% cap, under 30%.
    freeze = plan(SEGS, [2.0, 3.8], 6.0, overflow="freeze")
    stretch = plan(SEGS, [2.0, 3.8], 6.0, overflow="stretch")
    assert freeze["freeze_seconds"] > 0
    assert stretch["freeze_seconds"] == 0
    assert stretch["placements"][1]["tempo"] <= 1.30


def test_plan_rejects_mismatched_input():
    with pytest.raises(ValueError):
        plan(SEGS, [1.0], 8.0)


# ------------------------------------------------------------------ voice helpers


def test_split_text_never_cuts_words_and_keeps_order():
    text = ("First sentence here. " + "word " * 120 + "end. Third, with a clause; and another clause.")
    chunks = split_text(text, limit=80)
    assert all(len(c) <= 80 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_split_text_short_text_is_one_chunk():
    assert split_text("Hello there. How are you?") == ["Hello there. How are you?"]
    assert split_text("Aaa aaa. Bbb bbb.", limit=10) == ["Aaa aaa.", "Bbb bbb."]
    assert split_text("   ") == []


def test_trim_silence_removes_padding():
    import numpy as np

    sr = 24000
    wav = np.concatenate([np.zeros(sr), 0.3 * np.ones(sr // 2, dtype=np.float32), np.zeros(sr)]).astype(np.float32)
    out = trim_silence(wav, sr)
    assert 0.5 * sr <= out.size <= 0.62 * sr


# ------------------------------------------------------------------ config


def test_normalize_base_url_strips_docs_page():
    assert normalize_base_url("https://gw.example.run.app/docs#/") == "https://gw.example.run.app"
    assert normalize_base_url("https://gw.example.run.app/") == "https://gw.example.run.app"
    with pytest.raises(ValueError):
        normalize_base_url("gw.example.run.app")


def test_settings_store_keeps_key_private(tmp_path):
    store = SettingsStore(tmp_path, env_file=tmp_path / "missing.env")
    s = store.save(base_url="https://gw.test/docs", api_key="abc123")
    assert s.api_key == "abc123" and s.base_url == "https://gw.test"
    view = s.public_view()
    assert view["api_key_set"] is True and "abc123" not in json.dumps(view)
    assert (tmp_path / "settings.json").stat().st_mode & 0o777 == 0o600
    # A blank key keeps the saved one.
    assert store.save(base_url="https://gw.test", api_key="").api_key == "abc123"


def test_target_length_fits_line_to_the_speaker_within_15_percent():
    segs = [{"start": 1.0, "text": "a", "target_length": 2.0}, {"start": 4.0, "text": "b", "target_length": 2.0}]
    p = plan(segs, [2.2, 1.5], 8.0)
    a, b = p["placements"]
    assert a["tempo"] == pytest.approx(1.1) and a["end"] - a["start"] == pytest.approx(2.0, abs=0.01)
    assert b["tempo"] == pytest.approx(1 / 1.15, abs=1e-3)  # slowed, but only 15%
    assert any("Line 2" in w and "on screen" in w for w in p["warnings"])


def test_align_script_to_speech_regions():
    from studio.server.pipeline import align_script_to_speech

    script = ["Your health matters. Your family's well-being matters even more.", "Choose care. Choose peace of mind."]
    heard = [{"start": 0.5, "end": 3.9, "text": "Your health matters. Your family's wellbeing matters even more."},
             {"start": 4.6, "end": 7.2, "text": "Choose care, choose peace of mind."}]
    lines = align_script_to_speech(script, heard)
    assert [l["text"] for l in lines] == ["Your health matters. Your family's well-being matters even more.",
                                          "Choose care. Choose peace of mind."]
    assert [l["start"] for l in lines] == [0.5, 4.6] and [l["target_length"] for l in lines] == [3.4, 2.6]
    # Region boundaries may fall inside a script line: the exact words are split there.
    split = align_script_to_speech(["Your health matters. Choose care."],
                                   [{"start": 0.2, "end": 1.4, "text": "Your health matters."},
                                    {"start": 2.0, "end": 2.9, "text": "Choose care."}])
    assert [l["text"] for l in split] == ["Your health matters.", "Choose care."]
    assert align_script_to_speech(["Completely different words here."], [{"start": 0, "end": 1, "text": "Hello there"}]) is None
