"""Provider-independent creative direction for talking-head video (schema version 1.0).

This is the project's own internal schema. Neither Veo nor Omni accepts it directly: the
provider adapters (`adapters.py`) turn it into each provider's real request. Most fields
become prompt text, because the providers expose few structured controls.

- Validation uses pydantic. Every enum is closed, and unknown fields are rejected with the
  field name rather than silently dropped.
- Presets fill sensible defaults. A user override wins over a preset value, and the
  combination is re-validated.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA_VERSION = "1.0"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Subject(_Strict):
    description: str = "Adult person based on the supplied reference image"
    identity_preservation: Literal["high", "standard"] = "high"
    appearance_preservation: bool = True
    wardrobe_preservation: bool = True


class Scene(_Strict):
    environment: Literal["preserve_reference", "neutral_studio"] = "preserve_reference"
    background: Literal["preserve_reference", "plain_neutral"] = "preserve_reference"
    lighting: Literal["natural_and_consistent", "soft_studio"] = "natural_and_consistent"
    composition: Literal["preserve_reference", "medium_close_up", "head_and_shoulders"] = "preserve_reference"


class Camera(_Strict):
    angle: Literal["eye_level"] = "eye_level"
    movement: Literal["locked", "slow_push_in"] = "locked"
    zoom: Literal["none"] = "none"
    framing: Literal["stable"] = "stable"


# Small, named actions a take can be given at set times (closed vocabulary: free text would
# let anything into the prompt). Hand actions need the hands in frame.
ACTIONS = {
    "slight_head_tilt": "a slight head tilt while making the point",
    "small_nod": "a small single nod on a key word",
    "eyebrow_lift": "a brief lift of the eyebrows with interest",
    "warm_smile": "a warm smile grows while speaking",
    "lean_in": "leans in very slightly toward the camera",
    "weight_shift": "a small, natural shift of the shoulders",
    "glance_away_return": "a brief thoughtful glance to one side, then back to the lens",
    "steady_eye_contact": "steady, engaged eye contact with the lens",
    "open_palm": "one hand rises briefly to chest height, palm open, then settles",
    "both_hands_explain": "both hands come up briefly at chest height in a small explaining gesture, then settle",
    "hand_to_chest": "one hand touches the chest lightly in a sincere gesture, then settles",
}
HAND_ACTIONS = {"open_palm", "both_hands_explain", "hand_to_chest"}
ENDINGS = {
    "closed_mouth": "then stays silent with a relaxed closed mouth",
    "soft_smile": "then pauses with a soft, closed-mouth smile",
    "nod_and_smile": "then finishes with a small nod and a warm closed-mouth smile",
    "glance_and_return": "then pauses, glances down briefly and looks back to the lens with a closed mouth",
}


class Beat(_Strict):
    start_s: float = Field(ge=0, le=60)
    end_s: float = Field(gt=0, le=60)
    action: Literal[tuple(ACTIONS)]


class Performance(_Strict):
    delivery: Literal["natural_conversational", "professional_presenter", "expressive"] = "natural_conversational"
    head_motion: Literal["minimal", "low", "moderate"] = "minimal"
    eye_behavior: Literal["natural_irregular_blinks"] = "natural_irregular_blinks"
    facial_expression: Literal["subtle", "warm", "expressive"] = "subtle"
    posture: Literal["relaxed_upright"] = "relaxed_upright"
    hand_gestures: Literal["avoid_unnecessary_gestures", "occasional_natural"] = "avoid_unnecessary_gestures"
    breathing: Literal["subtle"] = "subtle"
    motion_intensity: Literal["low", "medium"] = "low"
    # Per-take variety (long talking heads). Empty beats and the default ending compile to
    # exactly the prompt this schema produced before they existed.
    beats: list[Beat] = Field(default_factory=list, max_length=4)
    ending: Literal[tuple(ENDINGS)] = "closed_mouth"


class Speech(_Strict):
    script: str = ""
    language: Literal["English"] = "English"
    delivery_style: Literal["calm_and_natural", "confident_and_clear", "warm_and_animated"] = "calm_and_natural"
    exact_script_required: bool = True
    speech_seconds: Optional[float] = Field(None, gt=0, le=60)
    emphasis: Optional[str] = Field(None, min_length=1, max_length=40)   # one word of the script to stress
    # Long videos: the same voice description and speaking speed in every clip's prompt.
    pace_wps: Optional[float] = Field(None, ge=1.5, le=4.0)
    consistent_voice: bool = False
    voice_pitch: Literal["low", "mid", "high"] = "mid"      # follows the authorised voice's measured pitch


class Audio(_Strict):
    generated_audio: bool = True
    background_music: bool = False
    background_sfx: bool = False


class VisualConstraints(_Strict):
    preserve_reference_composition: bool = True
    avoid_exaggerated_motion: bool = True
    avoid_unnecessary_camera_motion: bool = True
    avoid_facial_distortion: bool = True
    avoid_identity_drift: bool = True
    realism_priority: Literal["standard", "high"] = "high"


class Output(_Strict):
    aspect_ratio: Literal["16:9", "9:16"] = "16:9"
    duration_seconds: int = Field(8, ge=1, le=60)
    resolution: Literal["720p", "1080p"] = "720p"


class CreativeDirection(_Strict):
    version: Literal["1.0"] = SCHEMA_VERSION
    task: Literal["realistic_talking_head"] = "realistic_talking_head"
    preset: Literal["natural_human", "professional_presenter", "expressive", "custom"] = "natural_human"
    subject: Subject = Field(default_factory=Subject)
    scene: Scene = Field(default_factory=Scene)
    camera: Camera = Field(default_factory=Camera)
    performance: Performance = Field(default_factory=Performance)
    speech: Speech = Field(default_factory=Speech)
    audio: Audio = Field(default_factory=Audio)
    visual_constraints: VisualConstraints = Field(default_factory=VisualConstraints)
    output: Output = Field(default_factory=Output)


# Presets are partial documents merged over the defaults. "Natural Human" IS the defaults.
PRESETS: dict[str, dict[str, Any]] = {
    "natural_human": {},
    "professional_presenter": {
        "performance": {"delivery": "professional_presenter", "head_motion": "low", "facial_expression": "warm",
                        "hand_gestures": "avoid_unnecessary_gestures", "motion_intensity": "low"},
        "speech": {"delivery_style": "confident_and_clear"},
    },
    "expressive": {
        "performance": {"delivery": "expressive", "head_motion": "moderate", "facial_expression": "expressive",
                        "hand_gestures": "occasional_natural", "motion_intensity": "medium"},
        "speech": {"delivery_style": "warm_and_animated"},
        "visual_constraints": {"avoid_exaggerated_motion": True},
    },
    "custom": {},
}

PRESET_LABELS = {"natural_human": "Natural Human / Professional Talking Head", "professional_presenter": "Professional Presenter",
                 "expressive": "Expressive", "custom": "Custom"}


class DirectionError(ValueError):
    """Invalid creative direction; `errors` lists each bad field with its reason."""

    def __init__(self, errors: list[dict]):
        self.errors = errors
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build(preset: str = "natural_human", overrides: Optional[dict] = None, **known: Any) -> CreativeDirection:
    """Defaults <- preset <- overrides <- known values (script, output settings, ...)."""
    if preset not in PRESETS:
        raise DirectionError([{"field": "preset", "message": f"unknown preset '{preset}'; use one of {sorted(PRESETS)}"}])
    doc = _merge(CreativeDirection().model_dump(), PRESETS[preset])
    doc["preset"] = preset
    doc = _merge(doc, overrides or {})
    doc = _merge(doc, {k: v for k, v in known.items() if v is not None})
    return parse(doc)


def parse(doc: dict) -> CreativeDirection:
    try:
        d = CreativeDirection.model_validate(doc)
    except ValidationError as exc:
        raise DirectionError([{"field": ".".join(str(p) for p in e["loc"]) or "(root)",
                               "message": e["msg"], "type": e["type"]} for e in exc.errors()]) from None
    check_consistency(d)
    return d


def check_consistency(d: "CreativeDirection") -> None:
    """Reject settings that would make the prompt contradict itself."""
    errors = []
    if d.scene.background == "preserve_reference" and d.scene.environment == "neutral_studio":
        errors.append({"field": "scene.environment", "message": "a neutral studio contradicts preserving the reference background"})
    if d.camera.movement != "locked" and d.visual_constraints.preserve_reference_composition and \
            d.scene.composition == "preserve_reference":
        errors.append({"field": "camera.movement", "message": "camera movement contradicts preserving the reference composition"})
    if d.performance.motion_intensity == "low" and d.performance.head_motion == "moderate":
        errors.append({"field": "performance.head_motion", "message": "moderate head motion contradicts low motion intensity"})
    if d.performance.facial_expression == "expressive" and d.performance.motion_intensity == "low" \
            and d.preset != "custom":
        errors.append({"field": "performance.facial_expression", "message": "an expressive face contradicts low motion intensity"})
    last = 0.0
    for i, b in enumerate(d.performance.beats):
        where = f"performance.beats.{i}"
        if b.end_s <= b.start_s:
            errors.append({"field": where, "message": "a beat must end after it starts"})
        if b.start_s < last:
            errors.append({"field": where, "message": "beats must be in order and must not overlap"})
        if b.end_s > d.output.duration_seconds:
            errors.append({"field": where, "message": "a beat cannot run past the end of the clip"})
        if b.action in HAND_ACTIONS and d.performance.hand_gestures != "occasional_natural":
            errors.append({"field": where, "message": "a hand action contradicts avoiding hand gestures"})
        if b.action in HAND_ACTIONS and d.scene.composition != "preserve_reference":
            errors.append({"field": where, "message": "a hand action needs the hands in frame; use the reference framing"})
        last = b.end_s
    if d.speech.emphasis is not None:
        from .dialogue import words as _words

        if " " in d.speech.emphasis.strip() or not _words(d.speech.emphasis):
            errors.append({"field": "speech.emphasis", "message": "emphasis is one word of the script"})
        elif d.speech.script.strip() and _words(d.speech.emphasis)[0] not in _words(d.speech.script):
            errors.append({"field": "speech.emphasis", "message": "the emphasised word is not in the script"})
    if errors:
        raise DirectionError(errors)




def to_json(direction: CreativeDirection) -> str:
    return json.dumps(direction.model_dump(), indent=2, sort_keys=True)


def from_json(text: str) -> CreativeDirection:
    return parse(json.loads(text))


def json_schema() -> dict:
    return CreativeDirection.model_json_schema()


def for_strict(*, preset: str = "natural_human", overrides: Optional[dict] = None, script: str = "",
               reference_mode: str = "full", voice_method: str = "convert", aspect_ratio: str = "16:9",
               resolution: str = "720p", duration_seconds: int = 8, speech_seconds: Optional[float] = None) -> CreativeDirection:
    """The direction a Strict Talking Head project uses. Settings the pipeline depends on win
    over overrides, and a conflict is reported rather than silently changed.

    - the exact script
    - the clip length
    - generated audio when the voice is converted from the video
    - the background implied by the reference mode
    """
    overrides = dict(overrides or {})
    if voice_method == "convert" and overrides.get("audio", {}).get("generated_audio") is False:
        raise DirectionError([{"field": "audio.generated_audio",
                               "message": "voice conversion needs the video's own audio; it cannot be turned off"}])
    scene = {"background": "preserve_reference", "environment": "preserve_reference"} if reference_mode == "full" else \
        {"background": "plain_neutral", "environment": "neutral_studio", "composition": "head_and_shoulders",
         "lighting": "soft_studio"}
    fixed = {"speech": {"script": script, **({"speech_seconds": speech_seconds} if speech_seconds else {})},
             "output": {"aspect_ratio": aspect_ratio, "resolution": resolution, "duration_seconds": duration_seconds},
             "scene": scene}
    if voice_method == "convert":
        fixed["audio"] = {"generated_audio": True}
    return build(preset, _merge(overrides, fixed))
