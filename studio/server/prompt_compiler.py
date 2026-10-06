"""Deterministic compiler: validated CreativeDirection -> natural-language video prompt.

Fixed section order:
1. subject and reference
2. scene and composition
3. camera
4. performance and motion
5. speech
6. lighting and realism
7. constraints
8. audio

The same input always gives byte-identical output. The script is quoted exactly as given,
never paraphrased. Contradictory settings are rejected, and no sentence may appear twice.
"""

from __future__ import annotations

from .direction import ACTIONS, ENDINGS, CreativeDirection, DirectionError, check_consistency

SECTIONS = ("subject", "scene", "camera", "performance", "speech", "realism", "constraints", "audio")

HEAD = {
    "minimal": "Head movement is minimal and incidental: no repeated nodding, no rhythmic swaying and no side-to-side head movement.",
    "low": "Small, natural head movements that follow the speech, without repeated nodding or rhythmic swaying.",
    "moderate": "Natural, moderate head movement that follows the speech, never rhythmic or repetitive.",
}
FACE = {
    "subtle": "Subtle, natural facial expressions, with no exaggerated eyebrow movement and nothing theatrical.",
    "warm": "A warm, friendly expression with gentle, natural smiles, with no exaggerated eyebrow movement and nothing theatrical.",
    "expressive": "A clearly expressive face that matches the words, still natural and never theatrical.",
}
DELIVERY = {
    "calm_and_natural": "calm, natural",
    "confident_and_clear": "confident, clear",
    "warm_and_animated": "warm, animated",
}


def compile_sections(d: CreativeDirection) -> dict[str, list[str]]:
    if not d.speech.script.strip():
        raise DirectionError([{"field": "speech.script", "message": "the approved script is required"}])
    check_consistency(d)
    s: dict[str, list[str]] = {k: [] for k in SECTIONS}

    keep = ["face", "facial structure", "eyes", "nose", "mouth"]
    if d.subject.appearance_preservation:
        keep += ["hairstyle", "skin tone", "glasses"]
    if d.subject.wardrobe_preservation:
        keep += ["clothing"]
    s["subject"].append("One person speaking directly to the camera in a single continuous talking-head shot.")
    s["subject"].append("The person is the person in the reference image: keep their " + ", ".join(keep[:-1]) +
                        f" and {keep[-1]} exactly as in the reference image.")
    if d.subject.identity_preservation == "high":
        s["subject"].append("The face must stay the same person throughout, with no drift in features.")

    if d.scene.background == "preserve_reference":
        s["scene"].append("Keep the background and setting exactly as in the reference image.")
    else:
        s["scene"].append("Plain, neutral light-grey background.")
    s["scene"].append({"preserve_reference": "Keep the framing of the reference image.",
                       "medium_close_up": "Medium close-up.",
                       "head_and_shoulders": "Head-and-shoulders framing."}[d.scene.composition])
    s["scene"].append({"natural_and_consistent": "Natural lighting that stays consistent throughout.",
                       "soft_studio": "Soft, even studio lighting that stays consistent throughout."}[d.scene.lighting])

    s["camera"].append({"locked": "The camera is locked off at eye level: no movement, no zoom and no change in framing.",
                        "slow_push_in": "The camera is at eye level with a very slow, steady push-in and no other movement."}
                       [d.camera.movement])

    p = d.performance
    s["performance"].append("The person stays comfortably upright and naturally centred, with relaxed shoulders.")
    s["performance"].append(HEAD[p.head_motion])
    s["performance"].append(FACE[p.facial_expression])
    s["performance"].append("Natural, irregular blinking and a steady eye line to the camera.")
    s["performance"].append("Subtle, natural breathing.")
    s["performance"].append({"avoid_unnecessary_gestures": "No unnecessary hand gestures.",
                             "occasional_natural": "Occasional small, natural hand gestures."}[p.hand_gestures])
    s["performance"].append("The motion is natural and restrained: alive, not frozen.")
    if p.beats:
        s["performance"].append("Timed actions: " + "; ".join(
            f"[{_ts(b.start_s)}-{_ts(b.end_s)}] {ACTIONS[b.action]}" for b in p.beats) + ".")

    pace = f", taking about {d.speech.speech_seconds:.1f} seconds" if d.speech.speech_seconds else ""
    if d.speech.consistent_voice:
        tone = {"low": "a warm, deep, low-pitched", "mid": "a warm, mid-pitched", "high": "a warm, light, higher-pitched"}
        s["speech"].append(f"The voice is {tone[d.speech.voice_pitch]}, natural speaking voice with the same calm, even "
                           "energy from start to finish: no rising excitement, no dramatic pitch changes.")
    if d.speech.pace_wps:
        s["speech"].append(f"The pace is steady and unhurried, about {d.speech.pace_wps:.1f} words per second, "
                           "never speeding up.")
    stress = f", gently stressing the word '{d.speech.emphasis.strip()}'," if d.speech.emphasis else ""
    s["speech"].append(f"The person speaks in {d.speech.language} with a {DELIVERY[d.speech.delivery_style]} delivery{stress} "
                       "and natural conversational mouth movement." if stress else
                       f"The person speaks in {d.speech.language} with a {DELIVERY[d.speech.delivery_style]} delivery and "
                       "natural conversational mouth movement.")
    if d.speech.exact_script_required:
        s["speech"].append(f"The person says exactly these words and nothing else, starting right away{pace}, "
                           f"{ENDINGS[p.ending]}: \"{d.speech.script}\"")
    else:
        s["speech"].append(f"The person says: \"{d.speech.script}\"")

    if d.visual_constraints.realism_priority == "high":
        s["realism"].append("Photorealistic, with natural skin texture: no beauty filter, no waxy or over-smoothed skin and "
                            "no artificial facial symmetry.")
    else:
        s["realism"].append("Realistic appearance with natural skin texture.")

    s["constraints"].append("No cuts, no scene changes, no other people, no B-roll and no added text or subtitles.")
    if d.visual_constraints.avoid_facial_distortion:
        s["constraints"].append("No facial distortion.")

    a = d.audio
    if not a.generated_audio:
        s["audio"].append("No audio.")
    else:
        extras = [x for x, on in (("no music", not a.background_music), ("no sound effects", not a.background_sfx)) if on]
        s["audio"].append("Audio: only the person's voice" + (", " + " and ".join(extras) if extras else "") + ".")

    flat = [x for k in SECTIONS for x in s[k]]
    if len(flat) != len(set(flat)):
        raise DirectionError([{"field": "(compiled prompt)", "message": "a sentence would appear twice"}])
    return s


def _ts(seconds: float) -> str:
    whole = int(round(seconds))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def compile_prompt(d: CreativeDirection) -> str:
    sections = compile_sections(d)
    return " ".join(x for k in SECTIONS for x in sections[k])


def negative_prompt(d: CreativeDirection) -> str:
    """Terms to avoid, for providers that support a negative prompt. Deterministic order."""
    terms = ["other people", "cuts", "scene changes", "b-roll", "subtitles", "captions", "extra dialogue", "different face"]
    if not d.audio.background_music:
        terms.append("music")
    if d.visual_constraints.avoid_exaggerated_motion:
        terms += ["exaggerated head movement", "repeated nodding", "rhythmic swaying", "theatrical expressions"]
    if d.visual_constraints.realism_priority == "high":
        terms += ["beauty filter", "waxy skin", "over-smoothed skin"]
    if d.visual_constraints.avoid_facial_distortion:
        terms.append("distorted face")
    if d.scene.background == "preserve_reference":
        terms.append("different background")     # the reference's own background may contain text or logos
    else:
        terms += ["text", "logos"]
    if d.visual_constraints.avoid_unnecessary_camera_motion and d.camera.movement == "locked":
        terms.append("camera movement")
    return ", ".join(terms)
