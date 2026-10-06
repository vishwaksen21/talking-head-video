"""HTTP API + static frontend. Run with `python -m studio` from the project root."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

import hashlib
import json
import shutil as _shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import catalog
from .config import DATA_DIR, STUDIO, SettingsStore
from .pipeline import Pipeline
from .store import ProjectStore, State
from .utils_api import UtilsAPIError, UtilsClient
from .voice import ChatterboxVoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# httpx logs request lines at INFO; keep them out (they never include headers, but stay quiet).
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("studio")

WEB = STUDIO / "web"
SERVABLE = re.compile(r"^(generated\.mp4|final\.mp4|original_audio\.wav|cloned_voice\.wav|line_\d{2}\.wav|scene_\d{2}\.mp4"
                      r"|tts\.wav|tts_take_\d{2}\.wav|final_talking_head\.mp4|reference_prepared\.png|frame_\d{2}\.png)$")
# Where each servable name lives inside a project folder.
FILE_DIRS = [(re.compile(r"^scene_\d{2}\.mp4$"), "video"), (re.compile(r"^tts(_take_\d{2})?\.wav$"), "audio"),
             (re.compile(r"^final_talking_head\.mp4$"), "output"), (re.compile(r"^reference_prepared\.png$"), "reference"),
             (re.compile(r"^frame_\d{2}\.png$"), "reports/identity")]
REPORTS = {"dialogue_comparison", "veo_speech_check", "sync_plan", "lipsync_validation", "identity_validation", "results",
           "final_validation"}
MEDIA_TYPES = {".mp4": "video/mp4", ".wav": "audio/wav", ".png": "image/png"}
IMAGE_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a"}
MAX_UPLOAD = 25 * 1024 * 1024


class SettingsIn(BaseModel):
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class SceneIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    use_references: bool = True  # send the project's reference image(s) with this scene


class ProjectIn(BaseModel):
    prompt: Optional[str] = Field(None, max_length=4000)  # single-scene shorthand
    scenes: Optional[list[SceneIn]] = Field(None, max_length=4)
    model: str = Field(min_length=1)
    name: Optional[str] = None
    reference_ids: list[str] = Field(default_factory=list)
    script: Optional[str] = Field(None, max_length=2000)  # approved words, one line per sentence
    end_card: bool = False
    transition: float = Field(0.5, ge=0, le=1.5)
    export_name: Optional[str] = None
    duration: Optional[int] = None
    aspect_ratio: Optional[str] = None
    resolution: Optional[str] = None
    generate_audio: bool = True
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None


class ConsentIn(BaseModel):
    attested: bool = False
    by: str = ""


class DirectionIn(BaseModel):
    preset: str = "natural_human"
    overrides: dict = Field(default_factory=dict)


class PreviewIn(BaseModel):
    provider: str = "veo"
    model: Optional[str] = None
    script: str = ""
    speech_seconds: Optional[float] = None
    reference_id: Optional[str] = None
    reference_mode: str = Field("full", pattern="^(full|face_crop)$")
    voice_method: str = Field("convert", pattern="^(convert|tts)$")
    aspect_ratio: str = "16:9"
    resolution: str = "720p"
    direction: DirectionIn = Field(default_factory=DirectionIn)


class StrictIn(BaseModel):
    provider: str = "veo"
    direction: DirectionIn = Field(default_factory=DirectionIn)
    voice_method: str = Field("convert", pattern="^(convert|tts)$")  # convert: re-voice the video's own audio (default)
    reference_mode: str = Field("full", pattern="^(full|face_crop)$")  # full: the uploaded photo as is (default)
    veo_audio: str = Field("off", pattern="^(off|reference)$")  # tts method only; off: Veo makes a silent video
    reference_id: Optional[str] = None
    script: Optional[str] = None
    voice: Optional[str] = None
    consent: ConsentIn = Field(default_factory=ConsentIn)
    model: str = "veo-3.1-fast-generate-001"
    resolution: str = "720p"
    aspect_ratio: str = "16:9"
    name: Optional[str] = None
    exaggeration: float = Field(0.5, ge=0, le=1)
    cfg_weight: float = Field(0.5, ge=0, le=1)
    temperature: float = Field(0.8, ge=0.1, le=1.5)


class ApproveIn(BaseModel):
    reviewer: str = ""
    override: bool = False


class IdentityReviewIn(BaseModel):
    same_person: bool
    reviewer: str = ""


class TranscriptIn(BaseModel):
    segments: list[dict]
    voice: Optional[str] = None
    max_speedup: float = 1.15
    overflow: str = "freeze"
    exaggeration: float = Field(0.5, ge=0, le=1)
    cfg_weight: float = Field(0.5, ge=0, le=1)
    temperature: float = Field(0.8, ge=0.1, le=1.5)


def create_app(data_dir: Path = DATA_DIR, *, client_factory=None, voice: Optional[ChatterboxVoice] = None,
               recover: bool = True) -> FastAPI:
    settings_store = SettingsStore(data_dir)
    store = ProjectStore(data_dir)
    voice = voice or ChatterboxVoice(settings_store.load().device)
    from .utils_api import ProviderCallsDisabled, gated_client

    factory = client_factory or gated_client
    pipeline = Pipeline(store, settings_store, voice, client_factory=factory)
    if recover:
        pipeline.recover()

    app = FastAPI(title="Voice Studio", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.pipeline = pipeline

    def client():
        try:
            return factory(settings_store.load())
        except UtilsAPIError as exc:
            raise HTTPException(400, str(exc))

    def api_error(exc: UtilsAPIError) -> HTTPException:
        return HTTPException(502 if exc.status is None or exc.status >= 500 else 400, str(exc))

    # ---------------------------------------------------------------- settings
    @app.get("/api/settings")
    def get_settings():
        return settings_store.load().public_view()

    @app.put("/api/settings")
    def put_settings(body: SettingsIn):
        try:
            return settings_store.save(base_url=body.base_url, api_key=body.api_key).public_view()
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/settings/test")
    def test_connection():
        s = settings_store.load()
        with client() as c:
            # Listing models is authenticated, so it proves both the address and the key.
            # (/healthz is not usable: Cloud Run intercepts that path with its own 404.)
            try:
                models = c.list_models(s.video_location)
            except UtilsAPIError as exc:
                raise api_error(exc)
        video = catalog.discover_video_models(models)
        return {"ok": True, "models_listed": len(models), "video_models": len(video), "location": s.video_location}

    def offline_models() -> list[dict]:
        from . import adapters

        return [{"id": m} for m in adapters.PROVIDERS["veo"]["models"]]

    def list_video_models(s) -> tuple[list[dict], bool]:
        """Live listing, or, while the provider gate is off, the verified Veo 3.1 ids (offline)."""
        try:
            c = factory(s)
        except ProviderCallsDisabled:
            return offline_models(), True
        except UtilsAPIError as exc:
            raise HTTPException(400, str(exc))
        with c:
            try:
                return c.list_models(s.video_location), False
            except UtilsAPIError as exc:
                raise api_error(exc)

    @app.get("/api/models")
    def models():
        s = settings_store.load()
        listed, offline = list_video_models(s)
        if offline:
            return {"location": s.video_location, "models": catalog.discover_video_models(listed), "offline": True,
                    "source": "Offline list (provider calls are disabled): the Veo 3.1 models verified in this project."}
        found = catalog.discover_video_models(listed)
        if not found:
            raise HTTPException(404, "The API lists no video generation models for this key.")
        return {"location": s.video_location, "models": found,
                "source": "Model list from the API; options from Google's Veo documentation "
                          "(the API does not publish per-model options)."}

    # ---------------------------------------------------------------- system
    @app.get("/api/system")
    def system():
        ok, why = ChatterboxVoice.installed()
        return {
            "chatterbox": {"installed": ok, "detail": why, "loaded": voice.loaded, "device": voice.device,
                           "load_error": voice.load_error},
            "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
            "voices": pipeline.list_voices(),
            "default_voice": pipeline.default_voice(),
            "provider_calls": settings_store.load().provider_calls,
        }

    @app.get("/api/voices/{voice_id}/audio")
    def voice_audio(voice_id: str):
        try:
            path = pipeline.resolve_voice(voice_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc))
        if path is None:
            raise HTTPException(404, "The built-in voice has no reference clip.")
        return FileResponse(path)

    # ---------------------------------------------------------------- projects
    @app.post("/api/uploads/image")
    async def upload_image(file: UploadFile = File(...)):
        data = await file.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(400, "Images must be under 25 MB.")
        try:
            from PIL import Image
            import io

            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            with Image.open(io.BytesIO(data)) as im:
                width, height, fmt = im.width, im.height, (im.format or "").lower()
        except Exception:
            raise HTTPException(400, "That file is not a readable PNG, JPEG or WebP image.")
        mime = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}.get(fmt)
        if not mime:
            raise HTTPException(400, f"Unsupported image format '{fmt}'. Use PNG, JPEG or WebP.")
        if min(width, height) < 256:
            raise HTTPException(400, "The image is too small; use at least 256 px on each side.")
        digest = hashlib.sha256(data).hexdigest()[:16]
        folder = data_dir / "uploads" / digest
        folder.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "image").name)
        name = Path(name).stem + IMAGE_TYPES[mime]
        (folder / name).write_bytes(data)
        (folder / "meta.json").write_text(json.dumps({"name": name, "mime_type": mime, "width": width, "height": height}))
        return {"id": digest, "name": name, "mime_type": mime, "width": width, "height": height, "bytes": len(data)}

    @app.post("/api/voices")
    async def upload_voice(file: UploadFile = File(...)):
        name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "").name)
        if Path(name).suffix.lower() not in AUDIO_EXT:
            raise HTTPException(400, "Upload a WAV, MP3, FLAC or M4A file.")
        data = await file.read()
        if len(data) > MAX_UPLOAD * 4:
            raise HTTPException(400, "Voice files must be under 100 MB.")
        target = pipeline.voices_dir / name
        if target.exists():
            if target.read_bytes() != data:
                raise HTTPException(400, f"A different voice named '{name}' already exists. Rename the file.")
        else:
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(data)
            try:
                from .media import probe as _probe

                info = _probe(tmp)
                if not info["has_audio"] or info["duration"] < 3:
                    raise ValueError("needs at least 3 seconds of audio")
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                raise HTTPException(400, f"That audio file cannot be used: {exc}")
            tmp.replace(target)
        return next(v for v in pipeline.list_voices() if v["id"] == name)

    @app.post("/api/projects")
    def create_project(body: ProjectIn):
        scene_in = [s for s in body.scenes or [] if s.prompt.strip()]
        scenes = [s.prompt for s in scene_in]
        if not scenes and body.prompt and body.prompt.strip():
            scenes = [body.prompt]
        if not scenes:
            raise HTTPException(400, "Write a prompt for the video.")
        s = settings_store.load()
        with client() as c:
            try:
                listed = c.list_models(s.video_location)
            except UtilsAPIError as exc:
                raise api_error(exc)
        available = {m["id"]: m for m in catalog.discover_video_models(listed)}
        caps = available.get(body.model)
        if caps is None:
            raise HTTPException(400, f"'{body.model}' is not a video model offered by the API.")
        refs = []
        for rid in body.reference_ids:
            folder = data_dir / "uploads" / rid
            if not re.fullmatch(r"[0-9a-f]{16}", rid) or not (folder / "meta.json").exists():
                raise HTTPException(400, "A reference image was not found. Upload it again.")
            refs.append((folder, json.loads((folder / "meta.json").read_text())))
        try:
            config = catalog.build_video_config(
                caps, duration=body.duration, aspect_ratio=body.aspect_ratio, resolution=body.resolution,
                generate_audio=body.generate_audio, negative_prompt=body.negative_prompt, seed=body.seed,
                reference_count=len(refs))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        script = [line.strip() for line in (body.script or "").splitlines() if line.strip()] or None
        export_name = None
        if body.export_name:
            export_name = re.sub(r"[^A-Za-z0-9._-]", "_", body.export_name)
            if not export_name.endswith(".mp4"):
                export_name += ".mp4"
        if refs and scene_in and not any(s.use_references for s in scene_in):
            raise HTTPException(400, "A reference image is attached but no scene uses it. Tick it on a scene or remove it.")
        params = {"video_config": config,
                  "scenes": [{"prompt": s.prompt, "use_references": bool(refs) and s.use_references} for s in scene_in] or None,
                  "script": script, "end_card": body.end_card, "transition": body.transition,
                  "export_name": export_name, "reference_files": []}
        try:
            project = store.create(prompt="\n\n".join(scenes), model=body.model, params=params,
                                   project_id=(body.name or "").strip() or None)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        ref_dir = store.project_dir(project["id"]) / "reference"
        ref_dir.mkdir(exist_ok=True)
        for folder, meta in refs:
            _shutil.copy2(folder / meta["name"], ref_dir / meta["name"])
            params["reference_files"].append({"path": f"reference/{meta['name']}", "mime_type": meta["mime_type"],
                                              "width": meta["width"], "height": meta["height"]})
        store.update(project["id"], params=params)
        pipeline.start_generation(project["id"])
        return present(store.get(project["id"]))

    # ---------------------------------------------------------------- strict talking head
    @app.get("/api/strict/readiness")
    def strict_readiness():
        r = pipeline.strict.readiness()
        gate = settings_store.load().provider_calls
        r["provider_calls"] = gate
        if not gate:
            r["video_api"] = {"ready": False, "detail": "provider calls are disabled (preview only)"}
        # Checking the photo and preparing speech are local, so they stay available with the gate closed.
        r["ready"] = r["chatterbox"]["ready"] and r["speech_check"]["ready"] and r["face_analysis"]["ready"]
        return r

    @app.get("/api/direction/options")
    def direction_options():
        from . import adapters, direction as direction_mod

        return {"schema_version": direction_mod.SCHEMA_VERSION, "presets": direction_mod.PRESET_LABELS,
                "preset_values": direction_mod.PRESETS, "schema": direction_mod.json_schema(),
                "providers": {k: {"label": v["label"], "verified": v["verified"], "models": v["models"]}
                              for k, v in adapters.PROVIDERS.items()},
                "provider_calls": settings_store.load().provider_calls}

    @app.post("/api/direction/preview")
    def direction_preview(body: PreviewIn):
        """Build the request that WOULD be sent. Never calls a provider, whatever the gate."""
        from . import adapters, direction as direction_mod, prompt_compiler

        try:
            d = direction_mod.for_strict(preset=body.direction.preset, overrides=body.direction.overrides,
                                         script=body.script, reference_mode=body.reference_mode,
                                         voice_method=body.voice_method, aspect_ratio=body.aspect_ratio,
                                         resolution=body.resolution, speech_seconds=body.speech_seconds)
            refs = []
            if body.reference_id:
                folder = data_dir / "uploads" / body.reference_id
                if not re.fullmatch(r"[0-9a-f]{16}", body.reference_id) or not (folder / "meta.json").exists():
                    raise HTTPException(400, "The photo was not found. Upload it again.")
                meta = json.loads((folder / "meta.json").read_text())
                refs.append(adapters.ReferenceImage((folder / meta["name"]).read_bytes(), meta["mime_type"]))
            req = adapters.adapter_for(body.provider, body.model).build(d, refs)
            sections = prompt_compiler.compile_sections(d)
        except direction_mod.DirectionError as exc:
            raise HTTPException(400, {"message": f"Video direction: {exc}", "errors": exc.errors})
        gate = settings_store.load().provider_calls
        return {"mode": "request preview (nothing was sent)", "direction": d.model_dump(), "sections": sections,
                "prompt": req.prompt, "negative_prompt": req.negative_prompt, "request": req.preview(),
                "generation_possible": req.generation_allowed and gate,
                "why_not": None if req.generation_allowed and gate else
                ("provider calls are disabled" if req.generation_allowed else "this provider's video request is not verified")}

    @app.post("/api/strict/projects")
    def create_strict(body: StrictIn):
        return present(make_strict(body))

    def make_strict(body: StrictIn, *, project_id: Optional[str] = None, extra_params: Optional[dict] = None,
                    direction_overrides: Optional[dict] = None) -> dict:
        """Create and start one Strict Talking Head project (also used for each segment of a
        long talking head). Raises HTTPException for invalid input."""
        from . import dialogue
        from .strict import NEGATIVE_PROMPT, NEGATIVE_PROMPT_FULL

        if not body.reference_id:
            raise HTTPException(400, "Upload a photo of the person. Strict Talking Head needs one.")
        folder = data_dir / "uploads" / body.reference_id
        if not re.fullmatch(r"[0-9a-f]{16}", body.reference_id) or not (folder / "meta.json").exists():
            raise HTTPException(400, "The photo was not found. Upload it again.")
        try:
            script = dialogue.validate_script(body.script or "")
        except dialogue.ScriptError as exc:
            raise HTTPException(400, str(exc))
        if not body.voice:
            raise HTTPException(400, "Choose the authorised voice to speak the script.")
        if body.voice == "builtin":
            raise HTTPException(400, "The built-in voice cannot be used here; choose an authorised voice.")
        try:
            pipeline.resolve_voice(body.voice)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if not body.consent.attested or not body.consent.by.strip():
            raise HTTPException(400, "Confirm that the person in the photo and the voice owner have authorised this "
                                     "AI video, and enter your name.")
        s = settings_store.load()
        if body.provider != "veo":
            raise HTTPException(400, "Only Veo can generate. Omni is preview-only: its video request is not verified.")
        listed, _offline = list_video_models(s)
        caps = {m["id"]: m for m in catalog.discover_video_models(listed)}.get(body.model)
        if caps is None:
            raise HTTPException(400, f"'{body.model}' is not a video model offered by the API.")
        try:
            config = catalog.build_video_config(caps, duration=8, aspect_ratio=body.aspect_ratio,
                                                resolution=body.resolution,
                                                generate_audio=body.voice_method == "convert" or body.veo_audio == "reference",
                                                negative_prompt=NEGATIVE_PROMPT_FULL if body.reference_mode == "full"
                                                else NEGATIVE_PROMPT, seed=None, reference_count=1)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        from . import direction as direction_mod

        try:
            overrides = body.direction.overrides
            if direction_overrides:
                overrides = direction_mod._merge(overrides or {}, direction_overrides)
            direction = direction_mod.for_strict(
                preset=body.direction.preset, overrides=overrides, script=script,
                reference_mode=body.reference_mode, voice_method=body.voice_method,
                aspect_ratio=body.aspect_ratio, resolution=body.resolution, duration_seconds=8)
        except direction_mod.DirectionError as exc:
            raise HTTPException(400, f"Video direction: {exc}")
        meta = json.loads((folder / "meta.json").read_text())
        params = {"mode": "strict", "video_config": config, "scenes": None, "reference_files": [],
                  "export_name": "final_talking_head.mp4",
                  "strict": {"script": script, "voice": body.voice, "reference_file": meta["name"],
                             "voice_method": body.voice_method, "reference_mode": body.reference_mode,
                             "direction": direction.model_dump(), "provider": body.provider,
                             "veo_audio": "reference" if body.voice_method == "convert" else body.veo_audio,
                             "consent": {"attested": True, "by": body.consent.by.strip(), "at": __import__("time").time(),
                                         "statement": "The person in the photo and the owner of the voice have "
                                                      "authorised this AI-generated talking-head video."},
                             "voice_settings": {"exaggeration": body.exaggeration, "cfg_weight": body.cfg_weight,
                                                "temperature": body.temperature}}}
        params.update(extra_params or {})
        try:
            project = store.create(prompt=f"Strict talking head: \"{script}\"", model=body.model, params=params,
                                   project_id=project_id or (body.name or "").strip() or None)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        ref_dir = store.project_dir(project["id"]) / "reference"
        ref_dir.mkdir(exist_ok=True)
        _shutil.copy2(folder / meta["name"], ref_dir / meta["name"])  # immutable copy of the upload
        store.update(project["id"], state=State.VALIDATING_INPUT, message="Starting")
        pipeline.strict.start(project["id"])
        return store.get(project["id"])

    def strict_action(fn):
        try:
            return present(fn())
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/projects/{project_id}/strict/regenerate-speech")
    def strict_regenerate(project_id: str):
        require(project_id)
        return strict_action(lambda: pipeline.strict.regenerate_speech(project_id))

    @app.post("/api/projects/{project_id}/strict/approve-audio")
    def strict_approve(project_id: str, body: ApproveIn):
        require(project_id)
        return strict_action(lambda: pipeline.strict.approve_audio(project_id, override=body.override, reviewer=body.reviewer))

    @app.post("/api/projects/{project_id}/strict/resync")
    def strict_resync(project_id: str):
        require(project_id)
        return strict_action(lambda: pipeline.strict.resync(project_id))

    @app.post("/api/projects/{project_id}/strict/confirm-identity")
    def strict_confirm(project_id: str, body: IdentityReviewIn):
        require(project_id)
        return strict_action(lambda: pipeline.strict.confirm_identity(project_id, same_person=body.same_person,
                                                                      reviewer=body.reviewer))

    @app.get("/api/projects/{project_id}/reports/{name}")
    def project_report(project_id: str, name: str):
        require(project_id)
        stem = name.removesuffix(".json")
        path = store.project_dir(project_id) / "reports" / f"{stem}.json"
        if stem not in REPORTS or not path.exists():
            raise HTTPException(404, "No such report.")
        return json.loads(path.read_text())

    @app.get("/api/projects")
    def list_projects():
        return [summary(p) for p in store.list() if not p["params"].get("parent")]   # long-video clips show inside their video

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        return present(require(project_id))

    @app.post("/api/projects/{project_id}/transcript")
    def submit_transcript(project_id: str, body: TranscriptIn):
        require(project_id)
        try:
            return present(pipeline.submit_transcript(project_id, body.segments, body.model_dump(exclude={"segments"})))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/projects/{project_id}/cancel")
    def cancel(project_id: str):
        require(project_id)
        return present(pipeline.cancel(project_id))

    @app.post("/api/projects/{project_id}/retry")
    def retry(project_id: str, approve_new_generation: bool = False):
        require(project_id)
        try:
            return present(pipeline.retry(project_id, approve_new_generation=approve_new_generation))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/projects/{project_id}/files/{name}")
    def project_file(project_id: str, name: str, download: bool = False):
        require(project_id)
        if not SERVABLE.match(name):
            raise HTTPException(404, "No such file.")
        sub = next((d for pattern, d in FILE_DIRS if pattern.match(name)), "")
        path = store.project_dir(project_id) / sub / name
        if not path.exists():
            raise HTTPException(404, "That file has not been produced yet.")
        media_type = MEDIA_TYPES[path.suffix]
        filename = f"{project_id}-{name}" if download else None
        return FileResponse(path, media_type=media_type, filename=filename,
                            content_disposition_type="attachment" if download else "inline")

    @app.get("/api/projects/{project_id}/reference/{name}")
    def reference_file(project_id: str, name: str):
        require(project_id)
        refs = {Path(r["path"]).name for r in store.get(project_id)["params"].get("reference_files") or []}
        if name not in refs:
            raise HTTPException(404, "No such reference image.")
        return FileResponse(store.project_dir(project_id) / "reference" / name)

    def require(project_id: str) -> dict:
        project = store.get(project_id)
        if project is None:
            raise HTTPException(404, "Project not found.")
        return project

    def files_for(project: dict) -> dict:
        folder = store.project_dir(project["id"])
        names = ["generated.mp4", "original_audio.wav", "cloned_voice.wav", "final.mp4"]
        names += sorted(p.name for p in folder.glob("line_*.wav"))
        out = {n: f"/api/projects/{project['id']}/files/{n}" for n in names if (folder / n).exists()}
        for sub, pattern in [("video", "scene_*.mp4"), ("audio", "tts*.wav"), ("output", "final_talking_head.mp4"),
                             ("reference", "reference_prepared.png"), ("reports/identity", "frame_*.png")]:
            if (folder / sub).exists():
                out.update({p.name: f"/api/projects/{project['id']}/files/{p.name}"
                            for p in sorted((folder / sub).glob(pattern)) if SERVABLE.match(p.name)})
        out["reports"] = {n: f"/api/projects/{project['id']}/reports/{n}" for n in sorted(REPORTS)
                          if (folder / "reports" / f"{n}.json").exists()}
        return out

    def present(project: dict) -> dict:
        out = {k: v for k, v in project.items() if k != "operation"}
        out["has_operation"] = bool(project.get("operation"))
        out["files"] = files_for(project)
        out["active"] = project["state"] in State.ACTIVE
        return out

    def summary(project: dict) -> dict:
        return {k: project.get(k) for k in ("id", "created_at", "updated_at", "state", "prompt", "model", "error", "message")} \
            | {"has_final": (store.project_dir(project["id"]) / "final.mp4").exists() or
               (store.project_dir(project["id"]) / "output" / "final_long_talking_head.mp4").exists(),
               "mode": project["params"].get("mode")}

    # Standalone Chatterbox (text-to-speech and voice conversion, no video, no provider calls).
    from . import standalone

    app.include_router(standalone.router(data_dir, voice, pipeline, lambda: pipeline.strict.asr))

    # Long talking head: several strict clips (one per script segment) cut together.
    from . import longform as longform_mod

    longform = longform_mod.LongForm(store, pipeline, make_strict, StrictIn)
    app.state.longform = longform
    if recover:
        longform.recover()
    app.include_router(longform_mod.router(longform, StrictIn, settings_store))

    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app
