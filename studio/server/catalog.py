"""Video model discovery.

`GET /llm/models` lists ids only - it publishes no capabilities. So discovery is two steps:

1. The live listing decides WHICH models exist (anything whose id starts with `veo-`).
2. A table of Google's published Veo parameters decides WHAT each accepts, keyed by
   family. A model the table does not know gets the conservative intersection and is
   flagged `capabilities_known: False`, so the UI can say so instead of guessing.

The gateway's own docs note that not every listed Veo id is granted to this project and
that `veo-3.1-*` are the ones known to work; `granted_note` carries that through.
"""

from __future__ import annotations

import re
from typing import Optional

# Google Vertex AI Veo parameter support, per model family.
_FAMILIES = [
    # (pattern, capabilities)
    (re.compile(r"^veo-3\.1-lite"), {
        "durations": [4, 6, 8], "aspect_ratios": ["16:9", "9:16"], "resolutions": ["720p"],
        "audio": True, "known": False, "references": False,
    }),
    (re.compile(r"^veo-3\.1-(fast-)?generate"), {
        "durations": [4, 6, 8], "aspect_ratios": ["16:9", "9:16"], "resolutions": ["720p", "1080p"],
        "audio": True, "known": True, "references": True,
    }),
    (re.compile(r"^veo-3\.0-(fast-)?generate"), {
        "durations": [4, 6, 8], "aspect_ratios": ["16:9", "9:16"], "resolutions": ["720p", "1080p"],
        "audio": True, "known": True, "references": False,
    }),
    (re.compile(r"^veo-2\.0-generate"), {
        "durations": [5, 6, 7, 8], "aspect_ratios": ["16:9", "9:16"], "resolutions": [],
        "audio": False, "known": True, "references": False,
    }),
]
_FALLBACK = {"durations": [8], "aspect_ratios": ["16:9"], "resolutions": [], "audio": False, "known": False,
             "references": False}

# Veo through this gateway (verified live by the sibling platform): reference images go in
# `config.reference_images[]` as {image: {image_bytes, mime_type}, reference_type: "asset"};
# at most 3, all `asset`, and any reference pins the clip to exactly 8 seconds
# ("supported durations are [8] for feature reference_to_video"). There is no image-to-video.
MAX_REFERENCE_IMAGES = 3
REFERENCE_DURATION = 8

# Seen working through this gateway (its own documentation, and the sibling platform).
_VERIFIED_GRANTED = re.compile(r"^veo-3\.1-")


def capabilities_for(model_id: str) -> dict:
    for pattern, caps in _FAMILIES:
        if pattern.search(model_id):
            break
    else:
        caps = _FALLBACK
    return {
        "id": model_id,
        "label": _label(model_id),
        "durations": list(caps["durations"]),
        "default_duration": 8 if 8 in caps["durations"] else caps["durations"][-1],
        "aspect_ratios": list(caps["aspect_ratios"]),
        "resolutions": list(caps["resolutions"]),
        "supports_audio": caps["audio"],
        "supports_reference_images": caps["references"],
        "max_reference_images": MAX_REFERENCE_IMAGES if caps["references"] else 0,
        "reference_duration": REFERENCE_DURATION if caps["references"] else None,
        "capabilities_known": caps["known"],
        "granted_note": None if _VERIFIED_GRANTED.search(model_id) else
            "Listed by the API, but the gateway notes some Veo ids are not enabled for this key.",
    }


def _label(model_id: str) -> str:
    m = re.match(r"^veo-(\d+\.\d+)-(fast-|lite-)?generate", model_id)
    if not m:
        return model_id
    tier = {"fast-": " Fast", "lite-": " Lite"}.get(m.group(2) or "", "")
    return f"Veo {m.group(1)}{tier}"


def discover_video_models(models: list[dict]) -> list[dict]:
    ids = sorted({str(m.get("id")) for m in models if str(m.get("id", "")).startswith("veo-")}, reverse=True)
    return [capabilities_for(i) for i in ids]


def build_video_config(caps: dict, *, duration: Optional[int], aspect_ratio: Optional[str],
                       resolution: Optional[str], generate_audio: bool,
                       negative_prompt: Optional[str], seed: Optional[int], reference_count: int = 0) -> dict:
    """Validate the user's choices against the model and build Google's config object.

    Raises ValueError with a plain message when a choice is not supported, rather than
    silently substituting a different value.
    """
    config: dict = {"number_of_videos": 1}
    if reference_count:
        if not caps["supports_reference_images"]:
            raise ValueError(f"{caps['label']} cannot take a reference image through this API.")
        if reference_count > MAX_REFERENCE_IMAGES:
            raise ValueError(f"Veo accepts at most {MAX_REFERENCE_IMAGES} reference images.")
        if duration is not None and int(duration) != REFERENCE_DURATION:
            raise ValueError(f"With a reference image Veo makes {REFERENCE_DURATION}-second clips only.")
        duration = REFERENCE_DURATION
    if duration is not None:
        if int(duration) not in caps["durations"]:
            raise ValueError(f"{caps['label']} supports durations {caps['durations']} seconds, not {duration}.")
        config["duration_seconds"] = int(duration)
    if aspect_ratio:
        if aspect_ratio not in caps["aspect_ratios"]:
            raise ValueError(f"{caps['label']} supports aspect ratios {caps['aspect_ratios']}, not {aspect_ratio}.")
        config["aspect_ratio"] = aspect_ratio
    if resolution:
        if resolution not in caps["resolutions"]:
            raise ValueError(f"{caps['label']} does not offer a {resolution} option.")
        config["resolution"] = resolution
    if caps["supports_audio"]:
        config["generate_audio"] = bool(generate_audio)
    elif generate_audio:
        raise ValueError(f"{caps['label']} cannot generate audio, so there would be no speech to re-voice.")
    if negative_prompt and negative_prompt.strip():
        config["negative_prompt"] = negative_prompt.strip()
    if seed is not None:
        config["seed"] = int(seed)
    return config


def reference_payload(images: list[tuple[bytes, str]]) -> list[dict]:
    import base64

    return [{"image": {"image_bytes": base64.b64encode(data).decode("ascii"), "mime_type": mime},
             "reference_type": "asset"} for data, mime in images]
