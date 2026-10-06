"""Provider adapters: CreativeDirection -> each provider's real request body.

Only fields verified for the provider are sent. Anything else is either compiled into the
prompt (listed as "prompt_only") or reported as unsupported and left out. Adapters never
make network calls; they only build and describe requests.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from . import catalog
from .direction import CreativeDirection, DirectionError
from .prompt_compiler import compile_prompt, negative_prompt


@dataclass
class ReferenceImage:
    data: bytes
    mime_type: str


@dataclass
class ProviderRequest:
    provider: str
    endpoint: str
    body: dict
    prompt: str
    negative_prompt: Optional[str]
    verified: bool
    generation_allowed: bool
    prompt_only: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def preview(self) -> dict:
        """The request as it would be sent, with image bytes replaced by a description."""
        return {"provider": self.provider, "endpoint": self.endpoint, "body": _redact(self.body),
                "verified": self.verified, "generation_allowed": self.generation_allowed,
                "prompt_only": self.prompt_only, "unsupported": self.unsupported, "notes": self.notes}


def _redact(node):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in ("image_bytes", "data") and isinstance(v, str) and len(v) > 64:
                raw = base64.b64decode(v)
                out[k] = f"<base64 image, {len(raw)} bytes, sha256 {hashlib.sha256(raw).hexdigest()[:16]}…>"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(node, list):
        return [_redact(x) for x in node]
    return node


class VeoAdapter:
    """POST /llm/videos {model, prompt, config, location}.

    `config` is Google's GenerateVideosConfig, passed through unmodified. Fields used here
    are listed by the gateway and/or verified live in this project:
    - duration_seconds, aspect_ratio, resolution, generate_audio, negative_prompt, number_of_videos
    - reference_images[].{image.image_bytes, image.mime_type, reference_type="asset"}
    """

    name = "veo"
    endpoint = "POST /llm/videos"

    def __init__(self, model: str = "veo-3.1-fast-generate-001", location: str = "us-central1"):
        self.model, self.location = model, location
        self.caps = catalog.capabilities_for(model)

    def build(self, d: CreativeDirection, references: list[ReferenceImage]) -> ProviderRequest:
        prompt, negative = compile_prompt(d), negative_prompt(d)
        prompt_only = ["subject", "scene", "camera", "performance", "speech", "visual_constraints"]
        notes = []
        if d.camera.movement != "locked":
            notes.append("Camera movement is a prompt instruction only; Veo has no camera-motion parameter here.")
        try:
            config = catalog.build_video_config(
                self.caps, duration=d.output.duration_seconds, aspect_ratio=d.output.aspect_ratio,
                resolution=d.output.resolution, generate_audio=d.audio.generated_audio, negative_prompt=negative,
                seed=None, reference_count=len(references))
        except ValueError as exc:
            field_name = ("output.duration_seconds" if "duration" in str(exc) else "output.resolution" if "offer" in str(exc)
                          else "output.aspect_ratio" if "aspect" in str(exc) else "audio.generated_audio"
                          if "audio" in str(exc) else "references")
            raise DirectionError([{"field": field_name, "message": str(exc)}]) from None
        if references:
            config["reference_images"] = catalog.reference_payload([(r.data, r.mime_type) for r in references])
        if d.audio.background_music or d.audio.background_sfx:
            notes.append("Music and sound effects can only be asked for in the prompt.")
        body = {"model": self.model, "prompt": prompt, "config": config, "location": self.location}
        return ProviderRequest(self.name, self.endpoint, body, prompt, negative, verified=True,
                               generation_allowed=True, prompt_only=prompt_only, unsupported=[], notes=notes)


class OmniAdapter:
    """Gemini Omni models (`gemini-omni-flash-preview`, `gemini-omni-1.1-flash-preview`).

    The only gateway route for Gemini models is POST /llm/generate
    {model, contents, system_instruction?, config, location}, where `contents` parts may carry
    `inline_data` images. That envelope is verified from the gateway schema; nothing
    video-specific is:
    - which response modality produces video, and how duration, aspect ratio, resolution and
      audio would be controlled
    - the gateway's response has no video field

    So the request carries only the prompt and the reference image, and is marked unverified
    with generation disabled.
    """

    name = "omni"
    endpoint = "POST /llm/generate"
    MODELS = ("gemini-omni-flash-preview", "gemini-omni-1.1-flash-preview")

    def __init__(self, model: str = "gemini-omni-flash-preview", location: str = "us-central1"):
        if model not in self.MODELS:
            raise DirectionError([{"field": "model", "message": f"'{model}' is not a listed Omni model {self.MODELS}"}])
        self.model, self.location = model, location

    def build(self, d: CreativeDirection, references: list[ReferenceImage]) -> ProviderRequest:
        prompt = compile_prompt(d)
        parts = [{"text": prompt}] + [{"inline_data": {"mime_type": r.mime_type, "data": base64.b64encode(r.data).decode()}}
                                      for r in references]
        body = {"model": self.model, "contents": [{"role": "user", "parts": parts}], "location": self.location}
        unsupported = [
            {"field": "output.duration_seconds", "reason": "no verified Omni control"},
            {"field": "output.aspect_ratio", "reason": "no verified Omni control"},
            {"field": "output.resolution", "reason": "no verified Omni control"},
            {"field": "audio.generated_audio", "reason": "no verified Omni control"},
            {"field": "negative_prompt", "reason": "not a GenerateContentConfig field known to apply"},
            {"field": "video output modality", "reason": "the response_modalities value for video is not verified"},
        ]
        notes = ["Not used anywhere in this project before; request shape verified only up to the /llm/generate envelope.",
                 "The gateway's /llm/generate response has no video field, so a returned video could not be read."]
        return ProviderRequest(self.name, self.endpoint, body, prompt, None, verified=False,
                               generation_allowed=False, prompt_only=["all creative direction"],
                               unsupported=unsupported, notes=notes)


PROVIDERS = {
    "veo": {"label": "Veo (via the Utilities API)", "adapter": VeoAdapter, "verified": True,
            "models": ["veo-3.1-fast-generate-001", "veo-3.1-generate-001"]},
    "omni": {"label": "Gemini Omni (preview only, unverified)", "adapter": OmniAdapter, "verified": False,
             "models": list(OmniAdapter.MODELS)},
}


def adapter_for(provider: str, model: Optional[str] = None):
    if provider not in PROVIDERS:
        raise DirectionError([{"field": "provider", "message": f"unknown provider '{provider}'; use one of {sorted(PROVIDERS)}"}])
    info = PROVIDERS[provider]
    return info["adapter"](model or info["models"][0])
