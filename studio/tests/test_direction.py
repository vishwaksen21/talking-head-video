"""Creative direction: schema, presets, compiler, provider adapters, preview and the gate.

All offline:
- no provider is invoked
- the network is blocked inside the tests that would expose a leak
"""

from __future__ import annotations

import base64
import json
import socket

import httpx
import pytest
from fastapi.testclient import TestClient

from studio.server import adapters, direction, prompt_compiler
from studio.server.direction import DirectionError

SCRIPT = "Prudential Health India helps people make informed choices for their health and well-being."
REF = adapters.ReferenceImage(b"\x89PNG" + b"x" * 300, "image/png")


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")

    from studio.server import utils_api

    monkeypatch.setattr(socket.socket, "connect", refuse)          # any real socket
    monkeypatch.setattr(utils_api.UtilsClient, "_request", refuse)  # any provider request


# 1-5 schema
def test_defaults_are_the_natural_human_preset():
    d = direction.build(speech={"script": SCRIPT})
    assert d.version == "1.0" and d.preset == "natural_human"
    assert d.performance.head_motion == "minimal" and d.performance.facial_expression == "subtle"
    assert d.camera.movement == "locked" and d.visual_constraints.realism_priority == "high"
    assert d.scene.background == "preserve_reference" and d.scene.composition == "preserve_reference"


def test_presets_and_custom_overrides():
    pro = direction.build("professional_presenter", speech={"script": SCRIPT})
    assert pro.performance.head_motion == "low" and pro.performance.facial_expression == "warm"
    custom = direction.build("custom", {"performance": {"head_motion": "low", "facial_expression": "warm"}},
                             speech={"script": SCRIPT})
    assert custom.performance.head_motion == "low" and custom.preset == "custom"


@pytest.mark.parametrize("doc,field", [
    ({"performance": {"head_motion": "wild"}}, "performance.head_motion"),
    ({"camera": {"movement": "orbit"}}, "camera.movement"),
    ({"output": {"aspect_ratio": "4:3"}}, "output.aspect_ratio"),
    ({"performance": {"unknown_knob": 1}}, "performance.unknown_knob"),      # never silently dropped
    ({"version": "2.0"}, "version"),
])
def test_invalid_values_name_the_field(doc, field):
    with pytest.raises(DirectionError) as err:
        direction.build("custom", doc, speech={"script": SCRIPT})
    assert any(e["field"] == field for e in err.value.errors)


def test_unknown_preset_and_missing_script():
    with pytest.raises(DirectionError, match="preset"):
        direction.build("cinematic")
    with pytest.raises(DirectionError) as err:
        prompt_compiler.compile_prompt(direction.build())
    assert err.value.errors[0]["field"] == "speech.script"


def test_contradictions_are_rejected():
    with pytest.raises(DirectionError, match="neutral studio"):
        direction.build("custom", {"scene": {"environment": "neutral_studio"}}, speech={"script": SCRIPT})
    with pytest.raises(DirectionError, match="head motion"):
        direction.build("custom", {"performance": {"head_motion": "moderate"}}, speech={"script": SCRIPT})


def test_json_round_trip_and_schema():
    d = direction.build("expressive", speech={"script": SCRIPT})
    assert direction.from_json(direction.to_json(d)) == d
    assert direction.json_schema()["title"] == "CreativeDirection"


# 6-7 compiler
@pytest.mark.parametrize("script", [SCRIPT, "Hi — it's 9:30, “quoted” words; ok?", "  spaced   words  "])
def test_exact_script_is_quoted_verbatim(script):
    d = direction.build(speech={"script": script})
    assert f'"{script}"' in prompt_compiler.compile_prompt(d)


def test_compiler_is_deterministic_and_has_no_duplicates():
    d = direction.build("professional_presenter", speech={"script": SCRIPT, "speech_seconds": 4.5})
    a, b = prompt_compiler.compile_prompt(d), prompt_compiler.compile_prompt(direction.from_json(direction.to_json(d)))
    assert a == b
    sentences = [x for sec in prompt_compiler.compile_sections(d).values() for x in sec]
    assert len(sentences) == len(set(sentences))
    assert list(prompt_compiler.compile_sections(d)) == list(prompt_compiler.SECTIONS)


def test_natural_preset_wording():
    p = prompt_compiler.compile_prompt(direction.build(speech={"script": SCRIPT}))
    for phrase in ("no repeated nodding", "no rhythmic swaying", "irregular blinking", "no beauty filter",
                   "locked off", "alive, not frozen", "relaxed shoulders"):
        assert phrase in p
    assert "cinematic" not in p.lower()


# 8 Veo mapping
def test_veo_request_mapping_exact():
    d = direction.for_strict(script=SCRIPT, speech_seconds=4.5)
    req = adapters.VeoAdapter("veo-3.1-fast-generate-001").build(d, [REF])
    assert req.endpoint == "POST /llm/videos" and set(req.body) == {"model", "prompt", "config", "location"}
    cfg = req.body["config"]
    assert set(cfg) == {"number_of_videos", "duration_seconds", "aspect_ratio", "resolution", "generate_audio",
                        "negative_prompt", "reference_images"}
    assert cfg["duration_seconds"] == 8 and cfg["aspect_ratio"] == "16:9" and cfg["resolution"] == "720p"
    assert cfg["generate_audio"] is True and cfg["number_of_videos"] == 1
    assert cfg["reference_images"] == [{"image": {"image_bytes": base64.b64encode(REF.data).decode(), "mime_type": "image/png"},
                                        "reference_type": "asset"}]
    assert req.body["prompt"] == prompt_compiler.compile_prompt(d) and req.verified and req.generation_allowed
    assert "<base64 image" in json.dumps(req.preview())                          # previews never carry the bytes


# 9 Omni mapping
def test_omni_request_mapping_and_unverified_status():
    d = direction.for_strict(script=SCRIPT)
    req = adapters.OmniAdapter().build(d, [REF])
    assert req.endpoint == "POST /llm/generate" and set(req.body) == {"model", "contents", "location"}
    parts = req.body["contents"][0]["parts"]
    assert parts[0] == {"text": prompt_compiler.compile_prompt(d)}
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert not req.verified and not req.generation_allowed and req.negative_prompt is None
    assert {u["field"] for u in req.unsupported} >= {"output.duration_seconds", "output.resolution", "video output modality"}
    with pytest.raises(DirectionError):
        adapters.OmniAdapter("gemini-omni-ultra")


# 10 unsupported capability handling
def test_unsupported_capabilities_are_errors_not_substitutions():
    d = direction.for_strict(script=SCRIPT, resolution="1080p")
    with pytest.raises(DirectionError) as err:
        adapters.VeoAdapter("veo-3.1-lite-generate-001").build(d, [])
    assert err.value.errors[0]["field"] == "output.resolution"
    with pytest.raises(DirectionError, match="8-second"):
        adapters.VeoAdapter().build(direction.for_strict(script=SCRIPT, duration_seconds=4), [REF])
    req = adapters.VeoAdapter().build(direction.build("custom", {"camera": {"movement": "slow_push_in"},
                                                                 "scene": {"composition": "medium_close_up"}},
                                                      speech={"script": SCRIPT}), [])
    assert any("prompt instruction only" in n for n in req.notes)
    with pytest.raises(DirectionError, match="voice conversion needs"):
        direction.for_strict(script=SCRIPT, overrides={"audio": {"generated_audio": False}})


# 12-14 API: options, preview (no provider, no network), creation, legacy compatibility, gate
def _app(tmp_path, factory_calls):
    from .test_media_and_flow import make_app
    from .conftest import FakeVoice

    app = make_app(tmp_path, lambda req: httpx.Response(500), FakeVoice())
    pipe = app.state.pipeline
    original = pipe._client_factory

    def counting(s):
        factory_calls.append(1)
        return original(s)

    pipe._client_factory = counting
    return app


def test_preview_never_invokes_a_provider_or_the_network(tmp_path, monkeypatch, no_network):
    calls = []
    app = _app(tmp_path, calls)
    with TestClient(app) as c:
        opts = c.get("/api/direction/options").json()
        assert set(opts["presets"]) == {"natural_human", "professional_presenter", "expressive", "custom"}
        assert opts["providers"]["omni"]["verified"] is False
        body = {"provider": "veo", "script": SCRIPT, "speech_seconds": 4.5,
                "direction": {"preset": "custom", "overrides": {"performance": {"head_motion": "low", "facial_expression": "warm"}}}}
        r = c.post("/api/direction/preview", json=body)
        assert r.status_code == 200, r.text
        p = r.json()
        assert p["mode"].startswith("request preview") and p["direction"]["performance"]["head_motion"] == "low"
        assert p["request"]["body"]["prompt"] == p["prompt"] and f'"{SCRIPT}"' in p["prompt"]
        assert p["generation_possible"] is False and p["why_not"] == "provider calls are disabled"
        o = c.post("/api/direction/preview", json={**body, "provider": "omni"}).json()
        assert o["request"]["endpoint"] == "POST /llm/generate" and o["why_not"] == "this provider's video request is not verified"
        bad = c.post("/api/direction/preview", json={**body, "direction": {"preset": "custom", "overrides": {"camera": {"zoom": "2x"}}}})
        assert bad.status_code == 400 and bad.json()["detail"]["errors"][0]["field"] == "camera.zoom"
    assert calls == []                                                             # no client was even built


def test_ui_settings_serialise_into_the_project(speech_like_video, tmp_path, monkeypatch):
    from .test_strict import strict_app, upload, wait_for

    app, gw = strict_app(tmp_path, monkeypatch, video=speech_like_video)
    with TestClient(app) as c:
        body = {"reference_id": upload(c, tmp_path), "script": SCRIPT, "voice": "reference.wav",
                "consent": {"attested": True, "by": "QA"},
                "direction": {"preset": "professional_presenter", "overrides": {"visual_constraints": {"realism_priority": "standard"}}}}
        pid = c.post("/api/strict/projects", json=body).json()["id"]
        p, _ = wait_for(c, pid, {"AWAITING_AUDIO_APPROVAL", "FAILED"})
        stored = direction.parse(p["params"]["strict"]["direction"])
        assert stored.preset == "professional_presenter" and stored.visual_constraints.realism_priority == "standard"
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        p, _ = wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        rec = p["params"]["strict"]["direction_record"]
        assert gw.requests[0]["prompt"] == rec["prompt"]                          # what was compiled is what was sent
        assert gw.requests[0]["config"]["negative_prompt"] == rec["negative_prompt"]
        assert (tmp_path / "data" / "projects" / pid / "reports" / "direction.json").exists()
        assert c.post("/api/strict/projects", json={**body, "provider": "omni"}).json()["detail"].startswith("Only Veo")


def test_legacy_requests_without_direction_still_work(speech_like_video, tmp_path, monkeypatch):
    from .test_strict import create, upload, wait_for, strict_app

    app, gw = strict_app(tmp_path, monkeypatch, video=speech_like_video)
    with TestClient(app) as c:
        pid = create(c, upload(c, tmp_path)).json()["id"]                         # no "direction" key at all
        p, _ = wait_for(c, pid, {"AWAITING_AUDIO_APPROVAL"})
        assert p["params"]["strict"]["direction"]["preset"] == "natural_human"
        store = app.state.pipeline.store
        params = store.get(pid)["params"]
        params["strict"].pop("direction")                                         # a project made before this feature
        store.update(pid, params=params)
        c.post(f"/api/projects/{pid}/strict/approve-audio", json={"reviewer": "QA"})
        p, _ = wait_for(c, pid, {"NEEDS_REVIEW", "FAILED"}, timeout=120)
        assert p["state"] == "NEEDS_REVIEW" and "talking-head shot" in gw.requests[0]["prompt"]


def test_gate_blocks_every_default_client(tmp_path):
    from studio.server.config import Settings
    from studio.server.utils_api import ProviderCallsDisabled, gated_client

    with pytest.raises(ProviderCallsDisabled):
        gated_client(Settings(base_url="https://gw.test", api_key="k", provider_calls=False))
    client = gated_client(Settings(base_url="https://gw.test", api_key="k", provider_calls=True))
    client.close()


def test_default_app_refuses_generation_while_gated(tmp_path, monkeypatch, no_network):
    from studio.server.main import create_app

    (tmp_path / "settings.json").write_text(json.dumps({"base_url": "https://gw.test", "api_key": "k"}))
    monkeypatch.delenv("STUDIO_PROVIDER_CALLS", raising=False)
    from studio.server import config as cfg

    monkeypatch.setattr(cfg, "ENV_FILE", tmp_path / "missing.env")
    app = create_app(tmp_path, voice=__import__("studio.tests.conftest", fromlist=["FakeVoice"]).FakeVoice(), recover=False)
    app.state.pipeline.settings._env_file = tmp_path / "missing.env"
    with TestClient(app) as c:
        m = c.get("/api/models").json()
        assert m["offline"] is True and {x["id"] for x in m["models"]} == {"veo-3.1-fast-generate-001", "veo-3.1-generate-001"}
        r = c.post("/api/projects", json={"prompt": "x", "model": "veo-3.1-fast-generate-001", "duration": 4})
        assert r.status_code == 400 and "Provider calls are disabled" in r.json()["detail"]
        assert c.post("/api/settings/test").status_code == 400
        assert c.get("/api/strict/readiness").json()["provider_calls"] is False


# 16-17 existing outputs untouched by this test module (checked by hash in the real data folder)
def test_existing_final_videos_untouched():
    import hashlib
    from studio.server.config import DATA_DIR

    final = DATA_DIR / "projects" / "vinay-voice-conversion-001" / "output" / "final_talking_head.mp4"
    if not final.exists():
        pytest.skip("project not present")
    assert hashlib.sha256(final.read_bytes()).hexdigest().startswith("0fc7954e30a6195e")
