"""Strict Talking Head mode: one uploaded person, one approved script, one voice.

Order (speech first, then video):

  VALIDATING_INPUT -> PREPARING_REFERENCE -> GENERATING_SPEECH -> VALIDATING_DIALOGUE
  -> AWAITING_AUDIO_APPROVAL                         (nothing paid yet; a person listens)
  -> GENERATING_VIDEO                                (one Veo generation, request built here)
  -> SYNCHRONIZING -> COMPOSING
  -> VALIDATING_LIPSYNC -> VALIDATING_IDENTITY -> VALIDATING_OUTPUT
  -> NEEDS_REVIEW -> (a person confirms identity) -> COMPLETED

Policy:
- The approved script is the only source of speech. The final audio is Chatterbox's
  rendering of it. Veo's own soundtrack is kept only as evidence and as the timing
  reference for its mouth movements; it never reaches the output.
- There is no fallback. A failed check stops the run (before the video) or marks it
  NEEDS_REVIEW (after it), naming the check. COMPLETED requires every automated check
  to pass on the current run's final file, plus a person's identity confirmation.
- The provider is Veo through the Utilities API with `config.reference_images`. It
  generates a new face guided by the reference; exact identity is not guaranteed, and the
  UI and reports say so.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from . import catalog, dialogue, identity, lipsync, reference
from .asr import ASRError, LocalASR
from .compose import compose
from .media import MediaError, extract_audio, probe, run_ffmpeg, speech_regions
from .store import State
from .sync import plan, render, change_tempo
from .voice import VoiceError, trim_silence
from .pipeline import Cancelled, StageError

log = logging.getLogger("studio.strict")

PROVIDER = {
    "id": "veo-reference",
    "label": "Veo 3.1 via the Utilities API, with the photo as a reference image",
    "identity": "Approximate. Veo generates a new face guided by the photo; it is not the photo animated.",
    "lip_sync": "Timing-level only. Veo animates a talking mouth; the approved voice is aligned to when the mouth moves "
                "and checked. Mouth shapes are Veo's, not re-animated to the voice. With Veo audio off, timing is a visual "
                "estimate (about ±0.3 s).",
    "clip_seconds": 8,
    "exact_identity": False,   # the gateway forwards only prompt + config; Veo's first-frame `image` input is not reachable
}


@dataclass
class StrictRules:
    max_speech_seconds: float = 7.2     # the cloned speech must fit one 8 s clip with lead-in and tail
    lead_in_seconds: float = 0.3
    max_offset_correction_s: float = 0.35
    conversion_takes: int = 3           # voice conversion is stochastic: keep the take whose word onsets sit closest
    dialogue: dialogue.MatchRules = field(default_factory=dialogue.MatchRules)
    lipsync: lipsync.LipSyncRules = field(default_factory=lipsync.LipSyncRules)
    identity: identity.IdentityRules = field(default_factory=identity.IdentityRules)
    reference: reference.ReferenceRules = field(default_factory=reference.ReferenceRules)


class StrictError(StageError):
    """A named strict-mode failure; the job runner records `stage` as the failed stage."""


def build_prompt(script: str, speech_seconds: float, reference_mode: str = "full") -> str:
    if reference_mode == "full":
        setting = ("Keep the background, setting, framing and lighting exactly as in the reference image. "
                   "Looking straight into the lens with natural eye contact and natural blinking, subtle natural "
                   "expressions and gentle head movement. Static camera. No cuts, no scene changes, no other people, "
                   "no B-roll, no added text or subtitles, no music. ")
    else:
        setting = ("Head-and-shoulders framing, centred, looking straight into the lens with natural eye contact and "
                   "natural blinking, subtle natural expressions and gentle head movement. Static camera, soft even "
                   "studio lighting, plain neutral light-grey background. No cuts, no scene changes, no other people, "
                   "no B-roll, no text, no subtitles, no logos, no music. ")
    return (
        "One person speaking directly to the camera in a single continuous talking-head shot. "
        "The person is the person in the reference image: keep their face, facial structure, eyes, nose, mouth, "
        "hairstyle, skin tone, glasses and clothing exactly as in the reference image. "
        + setting +
        f"The person says exactly these words and nothing else, at a calm natural pace taking about "
        f"{speech_seconds:.1f} seconds, starting right away, then stays silent with a relaxed closed mouth: "
        f"\"{script}\""
    )


NEGATIVE_PROMPT = ("other people, cuts, scene changes, b-roll, text, subtitles, captions, logos, music, "
                   "extra dialogue, different face")
# With the full photo, its own background may contain text or logos; do not fight it.
NEGATIVE_PROMPT_FULL = ("other people, cuts, scene changes, b-roll, subtitles, captions, music, extra dialogue, "
                        "different face, different background")


TAKE_RANK = {"PARTIAL": 0, "FAIL": 1, "UNCERTAIN": 2, "NOT MEASURABLE": 3}


def pick_take(takes: list[dict], anchor_hz: Optional[float], label: str = "clip 1") -> tuple[dict, str]:
    """The conversion take to keep.

    Without an anchor: the best timing (onsets closest to the generated audio).
    With an anchor (a long video's target pitch: the voice recording's, or clip 1's): among the takes whose timing is
    acceptable (aligned within 1 ms, onsets within 20 ms or unmeasurable, never FAIL), the one
    whose median pitch is closest to the anchor. Speed cannot differ between takes: every take
    keeps the generated audio's timeline, so speed is set by the video model's performance."""
    by_timing = min(takes, key=lambda t: (TAKE_RANK.get(t["comp"]["status"], 4), -t["comp"]["measured"],
                                          t["comp"]["max_abs_ms"] if t["comp"]["max_abs_ms"] is not None else 1e9,
                                          abs(t["lag_after"])))
    if not anchor_hz:
        return by_timing, "timing"
    ok = [t for t in takes if t.get("pitch_hz") and t["comp"]["status"] != "FAIL" and abs(t["lag_after"]) <= 24
          and (t["comp"]["max_abs_ms"] is None or t["comp"]["max_abs_ms"] <= 20.0)]
    if not ok:
        return by_timing, "timing (no take with acceptable timing had a measurable pitch)"
    from .timing import semitones

    best = min(ok, key=lambda t: (abs(semitones(t["pitch_hz"], anchor_hz)), abs(t["lag_after"])))
    return best, f"pitch closest to {label} ({anchor_hz:.0f} Hz)"


class StrictTalkingHead:
    def __init__(self, pipeline, asr: Optional[LocalASR] = None, rules: Optional[StrictRules] = None):
        self.p = pipeline
        self.store = pipeline.store
        self.asr = asr or LocalASR()
        self.rules = rules or StrictRules()

    # ------------------------------------------------------------------ public
    def readiness(self) -> dict:
        from . import facekit

        ok_cb, why_cb = self.p.voice.installed()
        ok_asr, why_asr = self.asr.available()
        ok_fk, why_fk = facekit.available()
        settings = self.p.settings.load()
        return {
            "provider": PROVIDER,
            "chatterbox": {"ready": ok_cb, "detail": why_cb},
            "speech_check": {"ready": ok_asr, "detail": f"Whisper base.en, local ({why_asr})"},
            "face_analysis": {"ready": ok_fk, "detail": f"Apple Vision ({why_fk})"},
            "video_api": {"ready": bool(settings.base_url and settings.api_key),
                          "detail": "configured" if settings.base_url and settings.api_key else "set the API in Settings"},
            "ready": ok_cb and ok_asr and ok_fk and bool(settings.base_url and settings.api_key),
        }

    def start(self, project_id: str) -> None:
        self.p._submit(project_id, self._prepare)

    def regenerate_speech(self, project_id: str) -> dict:
        project = self._require(project_id, {State.AWAITING_AUDIO_APPROVAL, State.FAILED})
        if project["state"] == State.FAILED and project.get("failed_stage") not in (
                State.GENERATING_SPEECH, State.VALIDATING_DIALOGUE):
            raise ValueError("Speech can only be regenerated before the video is made.")
        self.store.update(project_id, state=State.GENERATING_SPEECH, error=None, failed_stage=None,
                          message="Generating the approved speech again")
        self.p._submit(project_id, self._speech_only)
        return self.store.get(project_id)

    def approve_audio(self, project_id: str, *, override: bool = False, reviewer: str = "") -> dict:
        project = self._require(project_id, {State.AWAITING_AUDIO_APPROVAL})
        converting = project["params"]["strict"].get("voice_method") == "convert"
        check = project["params"]["strict"].get("dialogue") or ({"passed": True} if converting else None)
        if not check:
            raise ValueError("The speech has not been checked yet.")
        if not check["passed"] and not override:
            raise ValueError("The speech did not match the approved script. Listen to it, then regenerate it, "
                             "or approve it anyway with an explicit override.")
        if not reviewer.strip():
            raise ValueError("Enter your name to approve the audio.")
        params = project["params"]
        strict = params["strict"]
        strict["audio_approval"] = {"by": reviewer.strip(), "at": time.time(), "override": bool(override and not check["passed"]),
                                    "take": strict.get("take")}
        strict["run_id"] = uuid.uuid4().hex[:12]
        speech_s = strict["speech_seconds"]
        if strict.get("direction"):
            record = self.compile_direction(project_id, params, speech_s)
            params["scenes"] = [{"prompt": record["prompt"], "use_references": True}]
            params["video_config"]["negative_prompt"] = record["negative_prompt"]
        else:  # projects created before structured direction existed keep their original prompt
            params["scenes"] = [{"prompt": build_prompt(strict["script"], speech_s, strict.get("reference_mode", "face_crop")),
                                 "use_references": True}]
        self.store.update(project_id, params=params, state=State.GENERATING_VIDEO, message="Sending the video request")
        self.p._submit(project_id, self._video_and_validate)
        return self.store.get(project_id)

    def compile_direction(self, project_id: str, params: dict, speech_seconds: Optional[float]) -> dict:
        """Direction -> prompt, negative prompt and the Veo request, recorded in the project
        and in reports/direction.json. No network call."""
        from . import adapters, direction as direction_mod

        strict = params["strict"]
        doc = dict(strict["direction"])
        doc["speech"] = {**doc.get("speech", {}), "script": strict["script"], "speech_seconds": speech_seconds}
        d = direction_mod.parse(doc)
        folder = self.p._dir(project_id)
        refs = []
        for ref in params.get("reference_files") or []:
            path = folder / ref["path"]
            if path.exists():
                refs.append(adapters.ReferenceImage(path.read_bytes(), ref["mime_type"]))
        req = adapters.VeoAdapter(self.store.get(project_id)["model"]).build(d, refs)
        record = {"schema_version": d.version, "direction": d.model_dump(), "prompt": req.prompt,
                  "negative_prompt": req.negative_prompt, "request_preview": req.preview(), "compiled_at": time.time()}
        strict["direction_record"] = record
        (folder / "reports").mkdir(exist_ok=True)
        (folder / "reports" / "direction.json").write_text(json.dumps(record, indent=2))
        return record

    def confirm_identity(self, project_id: str, *, same_person: bool, reviewer: str) -> dict:
        project = self._require(project_id, {State.NEEDS_REVIEW})
        strict = project["params"]["strict"]
        if not reviewer.strip():
            raise ValueError("Enter your name to record the review.")
        folder = self.p._dir(project_id)
        final = folder / "output" / "final_talking_head.mp4"
        summary = strict.get("validation") or {}
        if summary.get("run_id") != strict.get("run_id") or not final.exists() or \
                reference.sha256(final) != summary.get("final_sha256"):
            raise ValueError("The validation results do not belong to the current final video. Run it again.")
        review = {"same_person": bool(same_person), "by": reviewer.strip(), "at": time.time()}
        strict["identity_review"] = review
        params = project["params"]
        if not same_person:
            self.store.update(project_id, params=params, state=State.FAILED, failed_stage=State.VALIDATING_IDENTITY,
                              error="The reviewer said the person in the video is not the person in the photo.")
            return self.store.get(project_id)
        if summary.get("results") is not None:
            from . import results as results_mod

            failing = results_mod.blocking_failures({"rows": summary["results"]})
        else:  # projects validated before the status table existed
            failing = [name for name, ok in summary.get("checks", {}).items() if ok is False and name != "exact_photo_animated"]
        if failing:
            self.store.update(project_id, params=params)
            raise ValueError("Cannot accept: these checks failed: " + ", ".join(failing) + ".")
        # The provider cannot animate the exact photo, so a human acceptance is a likeness
        # acceptance. COMPLETED is reserved for a provider that animates the photo itself.
        final_state = State.COMPLETED if PROVIDER["exact_identity"] else State.LIKENESS_ACCEPTED
        self.store.update(project_id, params=params, state=final_state, error=None,
                          message="Accepted as a likeness of the photo (not the photo animated)"
                          if final_state == State.LIKENESS_ACCEPTED else "Done")
        return self.store.get(project_id)

    def retry(self, project_id: str) -> dict:
        """Resume from the failed stage. Video generation resumes its stored operation and
        never starts a second one (the quota guard in Pipeline._generate_scene enforces it)."""
        project = self._require(project_id, {State.FAILED, State.CANCELLED})
        stage = project.get("failed_stage")
        before_video = {State.VALIDATING_INPUT, State.PREPARING_REFERENCE, State.GENERATING_SPEECH,
                        State.VALIDATING_DIALOGUE, State.AWAITING_AUDIO_APPROVAL, None}
        self.store.update(project_id, error=None, cancel_requested=0)
        if stage in before_video and not project["params"]["strict"].get("audio_approval"):
            self.store.update(project_id, state=State.VALIDATING_INPUT, message="Starting again")
            self.p._submit(project_id, self._prepare)
        else:
            self.store.update(project_id, state=State.GENERATING_VIDEO, message="Resuming")
            self.p._submit(project_id, self._video_and_validate)
        return self.store.get(project_id)

    def resync(self, project_id: str) -> dict:
        """Re-run synchronisation and every check on the existing clip. No new generation:
        the clip is on disk, and the quota guard refuses to start another."""
        project = self._require(project_id, {State.NEEDS_REVIEW, State.FAILED})
        if not (self.p._dir(project_id) / "video" / "scene_01.mp4").exists():
            raise ValueError("There is no generated clip to synchronise yet.")
        params = project["params"]
        params["strict"]["run_id"] = uuid.uuid4().hex[:12]
        params["strict"].pop("validation", None)
        params["strict"].pop("identity_review", None)
        self.store.update(project_id, params=params, state=State.SYNCHRONIZING, error=None, cancel_requested=0,
                          message="Re-running synchronisation and checks (no new generation)")
        self.p._submit(project_id, self._video_and_validate)
        return self.store.get(project_id)

    def resume_video(self, project_id: str) -> None:
        self.p._submit(project_id, self._video_and_validate)

    # ------------------------------------------------------------------ stage A: no quota
    def _prepare(self, project_id: str) -> None:
        project = self.store.get(project_id)
        strict = project["params"]["strict"]
        folder = self.p._dir(project_id)
        self._stage(project_id, State.VALIDATING_INPUT, "Checking the photo, script and voice")
        try:
            script = dialogue.validate_script(strict["script"])
        except dialogue.ScriptError as exc:
            raise StrictError(State.VALIDATING_INPUT, str(exc))
        if not strict.get("voice") or strict["voice"] == "builtin":
            raise StrictError(State.VALIDATING_INPUT, "Choose an authorised voice; the built-in voice is not allowed here.")
        try:
            self.p.resolve_voice(strict["voice"])
        except ValueError as exc:
            raise StrictError(State.VALIDATING_INPUT, str(exc))
        original = folder / "reference" / strict["reference_file"]
        if not original.exists():
            raise StrictError(State.VALIDATING_INPUT, "The reference photo is missing.")
        if not strict.get("consent", {}).get("attested"):
            raise StrictError(State.VALIDATING_INPUT, "Permission for this likeness and voice has not been confirmed.")

        self._stage(project_id, State.PREPARING_REFERENCE, "Finding the face in the photo")
        try:
            meta = reference.prepare(original, folder / "reference", self.rules.reference,
                                     mode=strict.get("reference_mode", "face_crop"))
        except reference.ReferenceError as exc:
            raise StrictError(State.PREPARING_REFERENCE, str(exc))
        except Exception as exc:
            raise StrictError(State.PREPARING_REFERENCE, f"Face analysis failed: {exc}")
        params = project["params"]
        params["strict"]["script"] = script
        params["strict"]["reference"] = meta
        prep = meta["prepared"]
        params["reference_files"] = [{"path": f"reference/{prep['file']}", "mime_type": prep["mime_type"],
                                      "width": prep["width"], "height": prep["height"]}]
        (folder / "audio").mkdir(exist_ok=True)
        (folder / "audio" / "approved_script.txt").write_text(script + "\n")
        self.store.update(project_id, params=params)
        if strict.get("voice_method") == "convert":
            words = len(dialogue.words(script))
            estimate = round(words / 2.6, 2)  # ~2.6 words/s natural English pace
            params["strict"]["speech_seconds"] = estimate
            self.store.update(project_id, params=params)
            if estimate > self.rules.max_speech_seconds:
                raise StrictError(State.VALIDATING_INPUT,
                                  f"{words} words take about {estimate:.1f} s to say; one 8 s clip fits about "
                                  f"{self.rules.max_speech_seconds:.1f} s. Shorten the script.")
            self.store.update(project_id, state=State.AWAITING_AUDIO_APPROVAL,
                              message="Approve to generate the video (uses 1 generation). Its audio will be converted "
                                      "into the chosen voice.")
            return
        self._speech_only(project_id)

    def _speech_only(self, project_id: str) -> None:
        project = self.store.get(project_id)
        params = project["params"]
        strict = params["strict"]
        folder = self.p._dir(project_id)
        take = int(strict.get("take", 0)) + 1
        self._stage(project_id, State.GENERATING_SPEECH, f"Speaking the approved script (take {take})")
        voice_path = self.p.resolve_voice(strict["voice"])
        try:
            wav = self.p.voice.synthesize(strict["script"], voice_path, **strict.get("voice_settings", {}),
                                          should_stop=lambda: self.store.get(project_id)["cancel_requested"])
        except VoiceError as exc:
            raise StrictError(State.GENERATING_SPEECH, f"Speech generation failed: {exc}")
        sr = self.p.voice.sample_rate
        wav = trim_silence(wav, sr)
        take_path = folder / "audio" / f"tts_take_{take:02d}.wav"
        sf.write(take_path, wav, sr)
        speech_s = round(wav.size / sr, 3)

        self._stage(project_id, State.VALIDATING_DIALOGUE, "Checking the speech against the approved script")
        try:
            heard = self.asr.transcribe(take_path)
        except ASRError as exc:
            raise StrictError(State.VALIDATING_DIALOGUE, str(exc))
        check = dialogue.compare(strict["script"], heard["text"], self.rules.dialogue)
        check.update({"take": take, "audio": take_path.name, "asr_model": heard["model"], "speech_seconds": speech_s})
        (folder / "audio" / f"tts_take_{take:02d}_transcript.json").write_text(json.dumps(heard, indent=2))
        (folder / "reports").mkdir(exist_ok=True)
        (folder / "reports" / "dialogue_comparison.json").write_text(json.dumps(check, indent=2))
        shutil.copy2(take_path, folder / "audio" / "tts.wav")
        strict.update({"take": take, "dialogue": check, "tts_words": heard["words"], "speech_seconds": speech_s})
        self.store.update(project_id, params=params)
        if speech_s > self.rules.max_speech_seconds:
            raise StrictError(State.VALIDATING_DIALOGUE,
                              f"The spoken script lasts {speech_s:.1f} s, but one Veo clip is 8 s "
                              f"(at most {self.rules.max_speech_seconds:.1f} s of speech). Shorten the script.")
        self.store.update(project_id, state=State.AWAITING_AUDIO_APPROVAL,
                          message="Listen to the speech, then approve it to generate the video"
                          if check["passed"] else "The speech does not match the script. Listen and decide")

    # ------------------------------------------------------------------ stage B: one generation + checks
    def _video_and_validate(self, project_id: str) -> None:
        folder = self.p._dir(project_id)
        for sub in ("reports", "audio"):
            (folder / sub).mkdir(exist_ok=True)
        # A new run never reuses an earlier run's outputs or validations.
        # Everything derived from a previous run's video is removed first, so nothing stale can
        # be converted, composed or compared (e.g. an old extracted Veo track when the new
        # clip has no audio).
        for stale in [folder / "final.mp4", folder / "output" / "final_talking_head.mp4", folder / "cloned_voice.wav",
                      folder / "original_audio.wav", folder / "speech_16k.wav",
                      folder / "audio" / "original_veo_audio.wav", folder / "audio" / "converted_voice.wav",
                      folder / "audio" / "final_voice.wav", folder / "audio" / "veo_transcript.json"]:
            stale.unlink(missing_ok=True)
        for name in ("lipsync_validation.json", "identity_validation.json", "final_validation.json", "sync_plan.json",
                     "voice_conversion_timing.json", "veo_speech_check.json", "ffmpeg_compose.log", "results.json"):
            (folder / "reports" / name).unlink(missing_ok=True)

        try:
            self.p._generate(project_id)
        except (StageError, Cancelled):
            raise
        except Exception as exc:
            raise StrictError(State.GENERATING_VIDEO, f"Video generation failed: {exc}")
        project = self.store.get(project_id)
        params, strict = project["params"], project["params"]["strict"]
        video = folder / "generated.mp4"
        vinfo = probe(video)
        checks: dict[str, Optional[bool]] = {}
        notes: list[str] = []

        # Projects created before the setting existed recorded only what was sent to Veo.
        mode = strict.get("veo_audio") or ("reference" if params["video_config"].get("generate_audio") else "off")
        with tempfile.TemporaryDirectory(prefix="strict_frames_") as tmp:
            series = lipsync.frame_series(video, Path(tmp))

        # Veo's own soundtrack. In "off" mode it was not requested; if one came back anyway it is
        # kept as evidence only. In "reference" mode it times the mouth. It never reaches the output.
        self._stage(project_id, State.SYNCHRONIZING, "Checking the generated video")
        native = None
        veo_words, regions = [], []
        veo_check = {"status": "not_measurable", "passed": None,
                     "reasons": ["Veo audio was turned off, so what the mouth says cannot be measured; a person must watch it"]}
        if vinfo["has_audio"]:
            files = extract_audio(video, folder)
            shutil.copy2(files["original"], folder / "audio" / "original_veo_audio.wav")
            if mode == "off":
                notes.append("The video came back with audio although it was turned off. It was discarded.")
            else:
                native = folder / "original_audio.wav"
                regions = speech_regions(files["speech"])
                try:
                    veo_heard = self.asr.transcribe(files["speech"])
                except ASRError as exc:
                    raise StrictError(State.SYNCHRONIZING, str(exc))
                (folder / "audio" / "veo_transcript.json").write_text(json.dumps(veo_heard, indent=2))
                veo_words = veo_heard["words"]
                veo_check = dialogue.compare(strict["script"], veo_heard["text"], self.rules.dialogue)
        elif mode == "reference":
            veo_check = {"status": "no_audio", "passed": False, "reasons": ["the generated video has no audio track"]}
        (folder / "reports" / "veo_speech_check.json").write_text(json.dumps(veo_check, indent=2))
        checks["generated_person_says_the_script"] = veo_check["passed"]
        if veo_check["passed"] is False:
            notes.append("The generated person's mouth does not say the approved script: " + "; ".join(veo_check["reasons"]))

        if strict.get("voice_method", "tts") == "tts":
            # Place the approved speech where the on-screen speaker is talking.
            self._stage(project_id, State.SYNCHRONIZING, "Lining the approved speech up with the mouth")
            sr = self.p.voice.sample_rate
            tts, _ = sf.read(folder / "audio" / "tts.wav", dtype="float32")
            phrases = None
            method = "single take"
            if native is not None and veo_check["passed"]:
                phrases = lipsync.align_by_regions(strict["tts_words"], tts.size / sr, regions)
                method = "measured speech regions (Veo audio, discarded from output)"
                if not phrases:
                    phrases = lipsync.align_phrases(strict["tts_words"], veo_words, regions)
                    method = "recogniser word times"
            elif native is None:
                regions = lipsync.visual_regions(series)
                phrases = lipsync.align_by_regions(strict["tts_words"], tts.size / sr, regions)
                method = "visible mouth movement (estimate, about ±0.3 s)"
                if not regions:
                    notes.append("No talking mouth movement was detected in the video.")
            if phrases:
                # Snap every interior cut to the quietest moment nearby, so no word is split.
                for left, right in zip(phrases, phrases[1:]):
                    cut = lipsync.quietest_point(tts, sr, left["tts_end"])
                    left["tts_end"] = right["tts_start"] = cut
                clips, segments = [], []
                for ph in phrases:
                    a, b = ph["tts_start"], ph["tts_end"]
                    clip = trim_silence(tts[int(a * sr): int(b * sr)], sr)
                    clips.append(clip)
                    segments.append({"start": ph["veo_start"], "text": ph["words"], "target_length": ph["veo_end"] - ph["veo_start"]})
            else:
                onset = regions[0][0] if regions else self.rules.lead_in_seconds
                clips, segments = [tts], [{"start": onset, "text": strict["script"]}]
                notes.append("The speech could not be aligned phrase by phrase; it is placed as one take.")
            placement = plan(segments, [c.size / sr for c in clips], vinfo["duration"], max_speedup=1.15)
            if placement["freeze_seconds"] > 0:
                raise StrictError(State.SYNCHRONIZING, "The approved speech does not fit inside the clip without freezing the "
                                                       "picture. Shorten the script.")
            adjusted = [change_tempo(c, sr, p["tempo"], folder) for c, p in zip(clips, placement["placements"])]
            voice_track = folder / "cloned_voice.wav"
            render(adjusted, sr, placement, voice_track)

            # Correct the residual offset once.
            # - With Veo's own audio (the mouth's timing by construction), compare it with the voice
            #   audio to audio: speech-weighted envelopes at 0.25 ms resolution.
            # - Mouth activity has whole-frame (41.7 ms) resolution: measured, it moved the voice a
            #   full frame off Veo's speech, so it is used only when there is no Veo audio.
            if native is not None:
                import librosa

                from . import timing

                ref_wav, ref_sr = sf.read(native, dtype="float32", always_2d=True)
                ref24 = timing.match_length(librosa.resample(ref_wav.mean(1), orig_sr=ref_sr, target_sr=sr),
                                            int(round(vinfo["duration"] * sr)))
                cur, _ = sf.read(voice_track, dtype="float32")
                lag, r = timing.global_lag_samples(ref24, timing.match_length(cur, ref24.size), max_ms=120, sr=sr)
                offset = {"basis": "voice vs Veo audio, speech-weighted envelope", "lag_samples": lag,
                          "offset_s": round(-lag / sr, 4), "correlation": round(r, 3), "delta_frames": int(lag != 0)}
            else:
                offset = lipsync.measure_offset_visual(series, voice_track)
            if offset:
                shift = offset["offset_s"]
                if offset["delta_frames"] and abs(shift) <= self.rules.max_offset_correction_s:
                    for p in placement["placements"]:
                        p["start"] = round(max(0.0, p["start"] + shift), 3)
                        p["end"] = round(p["end"] + shift, 3)
                    if placement["placements"][-1]["end"] <= vinfo["duration"]:
                        render(adjusted, sr, placement, voice_track)
                        placement["offset_correction_s"] = shift
                    else:
                        notes.append("The measured offset could not be corrected without running past the end of the clip.")
            placement["measured_offset"] = offset
            placement["alignment_method"] = method
            (folder / "reports" / "sync_plan.json").write_text(json.dumps(placement, indent=2))
            params["strict"]["sync"] = {"aligned_phrases": len(phrases or []), "method": method, "placements": placement["placements"],
                                        "offset": offset, "offset_correction_s": placement.get("offset_correction_s", 0.0)}
            self.store.update(project_id, params=params, sync=placement)

        else:
            voice_track = folder / "cloned_voice.wav"
            self._convert_generated_audio(project_id, folder, strict, vinfo, voice_track, checks, notes)
            params = self.store.get(project_id)["params"]

        self._stage(project_id, State.COMPOSING, "Building the final MP4")
        (folder / "output").mkdir(exist_ok=True)
        final = folder / "output" / "final_talking_head.mp4"
        try:
            info = compose(video, voice_track, final, final_duration=vinfo["duration"], freeze_seconds=0,
                           log_path=folder / "reports" / "ffmpeg_compose.log")
        except MediaError as exc:
            raise StrictError(State.COMPOSING, f"FFmpeg could not build the final video: {exc}")
        shutil.copy2(final, folder / "final.mp4")
        shutil.copy2(voice_track, folder / "audio" / "final_voice.wav")
        self.store.merge_media(project_id, "final", info)

        self._stage(project_id, State.VALIDATING_LIPSYNC, "Measuring lip-sync")
        lip = lipsync.validate(final, voice_track, native, self.rules.lipsync, series=series)
        if native is not None:
            checks["lip_sync_timing"] = lip["passed"]
            lip["basis"] = "Veo's own audio as the timing reference"
        else:
            # Without reference audio the voice is graded against the same mouth movement it was
            # aligned to, which cannot reveal a systematic offset. On a real Veo clip this
            # self-check passed while the true offset was 292 ms. So: not measurable.
            checks["lip_sync_timing"] = None
            lip["basis"] = "visible mouth movement only; self-consistent but not verifiable"
            lip["verifiable"] = False
            notes.append("Lip-sync could not be verified: Veo audio was off, so there is no timing reference. "
                         "In a measured real clip, mouth-only timing was 292 ms early.")
        (folder / "reports" / "lipsync_validation.json").write_text(json.dumps(lip, indent=2))

        self._stage(project_id, State.VALIDATING_IDENTITY, "Comparing the face with the photo")
        try:
            ident = identity.validate(final, strict["reference"]["geometry"], folder / "reports" / "identity",
                                      self.rules.identity)
        except Exception as exc:
            ident = {"passed": False, "status": "error", "reasons": [f"face analysis failed: {exc}"],
                     "requires_human_confirmation": True}
        (folder / "reports" / "identity_validation.json").write_text(json.dumps(ident, indent=2))
        checks["face_proportions_similar_to_photo"] = ident["passed"]
        checks["exact_photo_animated"] = False  # Veo regenerates the face; the gateway has no image-to-video input

        self._stage(project_id, State.VALIDATING_OUTPUT, "Checking the final file")
        out = self._validate_output(final, voice_track, native, strict["script"], vinfo)
        (folder / "reports" / "final_validation.json").write_text(json.dumps(out, indent=2))
        checks.update(out["checks"])

        from . import results as results_mod

        assessed = results_mod.assess(folder, self.store.get(project_id)["params"]["strict"])
        (folder / "reports" / "results.json").write_text(json.dumps(assessed, indent=2, default=str))
        summary = {"run_id": strict["run_id"], "final_sha256": reference.sha256(final), "checks": checks,
                   "results": assessed["rows"], "facts": assessed["facts"],
                   "notes": notes, "provider": PROVIDER["id"], "at": time.time()}
        project = self.store.get(project_id)
        params = project["params"]
        params["strict"]["validation"] = summary
        failing = results_mod.blocking_failures(assessed)
        labels = {r["key"]: r["label"] for r in assessed["rows"]}
        not_guaranteed = [labels[r["key"]] for r in assessed["rows"] if r["status"] == results_mod.NOT_GUARANTEED]
        uncertain = [labels[r["key"]] for r in assessed["rows"] if r["status"] == results_mod.UNCERTAIN]
        message = ("No automated check failed." if not failing
                   else "Needs review. Failed: " + "; ".join(labels[k] for k in failing) + ".")
        if uncertain:
            message += " Uncertain: " + "; ".join(uncertain) + "."
        message += " Not guaranteed: " + "; ".join(not_guaranteed) + "."
        self.store.update(project_id, params=params, state=State.NEEDS_REVIEW, message=message)

    def _convert_generated_audio(self, project_id: str, folder: Path, strict: dict, vinfo: dict, voice_track: Path,
                                 checks: dict, notes: list) -> None:
        """The generated video's complete soundtrack, re-spoken in the authorised voice.

        The full extracted track (`audio/original_veo_audio.wav`, PCM decoded from the video,
        not trimmed) goes to Chatterbox voice conversion as one piece. The result is then:
        - forced to the source length to the sample
        - aligned to it with a global offset measured at 0.25 ms resolution
        - scaled by one constant gain, which cannot move any timing
        """
        import librosa

        from . import timing

        self._stage(project_id, State.SYNCHRONIZING, "Converting the generated audio into the chosen voice")
        source = folder / "audio" / "original_veo_audio.wav"
        if not source.exists():
            raise StrictError(State.SYNCHRONIZING, "The generated video has no audio to convert.")
        reference_voice = self.p.resolve_voice(strict["voice"])
        sr = 24000
        src_wav, src_sr = sf.read(source, dtype="float32", always_2d=True)
        src24 = librosa.resample(src_wav.mean(axis=1), orig_sr=src_sr, target_sr=sr)
        n = int(round(src_wav.shape[0] / src_sr * sr))
        src24 = timing.match_length(src24, n)
        takes = []
        for k in range(max(1, self.rules.conversion_takes)):
            self._stage(project_id, State.SYNCHRONIZING,
                        f"Converting the generated audio into the chosen voice (take {k + 1} of {self.rules.conversion_takes})")
            try:
                raw = self.p.voice.convert(source, reference_voice,
                                           should_stop=lambda: self.store.get(project_id)["cancel_requested"])
            except VoiceError as exc:
                raise StrictError(State.SYNCHRONIZING, str(exc))
            conv = timing.match_length(raw, n)
            lag, _ = timing.global_lag_samples(src24, conv, sr=sr)
            conv = timing.shift(conv, lag)
            lag_after, r_after = timing.global_lag_samples(src24, conv, sr=sr)
            comp = timing.onset_comparison(src24, conv, sr)
            takes.append({"take": k + 1, "conv": conv, "raw_diff_ms": round((raw.size - n) / sr * 1000, 3), "lag": lag,
                          "lag_after": lag_after, "r": r_after, "comp": comp, "onset_shift": 0})
        # Alignment is the speech-weighted whole-track offset. A further shift by the median onset
        # difference is applied only when the onset measurements are reliable: at least three
        # measured, none uncertain, and agreeing within 2 ms. Otherwise a few energy thresholds
        # would be steering the whole track.
        for t in takes:
            meas = [p["diff_ms"] for p in t["comp"]["pairs"] if p["status"] == "MEASURED"]
            if len(meas) >= 3 and t["comp"]["uncertain"] == 0 and (max(meas) - min(meas)) <= 2.0:
                t["onset_shift"] = int(round(float(np.median(meas)) / 1000 * sr))
                if t["onset_shift"]:
                    t["conv"] = timing.shift(t["conv"], t["onset_shift"])
                    t["comp"] = timing.onset_comparison(src24, t["conv"], sr)
                    t["lag_after"], t["r"] = timing.global_lag_samples(src24, t["conv"], sr=sr)
        anchor = strict.get("pitch_anchor")          # set for the clips of a long video: clip 1's voice
        if anchor:
            for t in takes:
                t["pitch_hz"] = timing.median_pitch_hz(t["conv"], sr)
        best, chosen_by = pick_take(takes, anchor["pitch_hz"] if anchor else None, (anchor or {}).get("label", "clip 1"))
        conv, lag, lag_after, r_after, comp = best["conv"], best["lag"], best["lag_after"], best["r"], best["comp"]
        raw_diff_ms = best["raw_diff_ms"]
        # One constant gain (speech at about -18 dBFS RMS, peaks under -1.5 dBFS): level only, timing untouched.
        voiced = conv[np.abs(conv) > 1e-3]
        gain = 1.0
        if voiced.size:
            gain = 10 ** (-18 / 20) / max(float(np.sqrt((voiced ** 2).mean())), 1e-6)
            gain = min(gain, 10 ** (-1.5 / 20) / max(float(np.abs(conv).max()), 1e-6))
        conv = (conv * gain).astype(np.float32)
        sf.write(voice_track, conv, sr, subtype="PCM_24")
        shutil.copy2(voice_track, folder / "audio" / "converted_voice.wav")
        report = {
            "method": "Chatterbox voice conversion of the complete generated soundtrack",
            "source": {"file": source.name, "sample_rate": src_sr, "channels": src_wav.shape[1],
                       "samples": int(src_wav.shape[0]), "seconds": round(src_wav.shape[0] / src_sr, 6),
                       "trimmed": False},
            "chatterbox_output_length_diff_ms": raw_diff_ms,
            "final_length_samples_24k": int(conv.size), "source_length_samples_24k": n,
            "length_identical": int(conv.size) == n,
            "global_offset_before_ms": round(lag / sr * 1000, 3), "global_offset_after_ms": round(lag_after / sr * 1000, 3),
            "word_onset_shift_ms": round(best["onset_shift"] / sr * 1000, 3),
            "envelope_correlation": round(r_after, 4), "constant_gain_db": round(20 * np.log10(gain), 2),
            "onsets": comp,
            "takes": [{"take": t["take"], "onset_status": t["comp"]["status"], "measured": t["comp"]["measured"],
                       "uncertain": t["comp"]["uncertain"], "max_abs_ms": t["comp"]["max_abs_ms"],
                       "offset_before_ms": round(t["lag"] / sr * 1000, 3), "offset_after_ms": round(t["lag_after"] / sr * 1000, 3),
                       "word_onset_shift_ms": round(t["onset_shift"] / sr * 1000, 3),
                       "pitch_hz": t.get("pitch_hz")} for t in takes],
            "chosen_take": best["take"], "chosen_by": chosen_by, "pitch_anchor": anchor,
            "note": "Duration is matched to the sample and the whole track is aligned on speech. Voice conversion "
                    "regenerates the speech, so word and phoneme timing inside it can still differ; onsets after "
                    "pauses are measured where reliable and marked UNCERTAIN otherwise.",
        }
        (folder / "reports" / "voice_conversion_timing.json").write_text(json.dumps(report, indent=2))
        if comp["max_abs_ms"] is not None and comp["max_abs_ms"] > 41.7:
            notes.append("Some speech onsets moved by more than one video frame during conversion.")
        placement = {"alignment_method": "voice conversion (timeline identical to the generated audio)",
                     "placements": [{"index": 0, "start": 0.0, "end": round(n / sr, 3), "tempo": 1.0}],
                     "final_duration": round(n / sr, 3), "freeze_seconds": 0.0, "conversion": report}
        (folder / "reports" / "sync_plan.json").write_text(json.dumps(placement, indent=2))
        params = self.store.get(project_id)["params"]
        params["strict"]["sync"] = {"method": placement["alignment_method"], "conversion": report}
        self.store.update(project_id, params=params, sync=placement)

    def _validate_output(self, final: Path, voice_track: Path, native: Optional[Path], script: str, vinfo: dict) -> dict:
        info = probe(final)
        checks, details = {}, {"probe": info}
        checks["video_and_audio_streams"] = info["has_video"] and info["has_audio"]
        checks["duration_matches_video"] = abs(info["duration"] - vinfo["duration"]) <= 0.1
        checks["resolution_preserved"] = (info.get("width"), info.get("height")) == (vinfo.get("width"), vinfo.get("height"))
        black = run_ffmpeg(["-i", str(final), "-vf", "blackdetect=d=0.1:pix_th=0.10", "-an", "-f", "null", "-"],
                           what="checking for black frames")
        details["black_segments"] = black.count("black_start")
        checks["no_black_frames"] = details["black_segments"] == 0
        decoded = _decode(final)
        details["peak"] = round(float(np.abs(decoded).max()), 4) if decoded.size else 0.0
        checks["audio_not_clipped"] = 0 < details["peak"] < 0.999
        # The final soundtrack must be the cloned voice, with no Veo speech under it.
        fin = _decode(final, mono=True, sr=16000)
        voice = _load(voice_track, 16000)
        details["corr_with_cloned_voice"] = _corr(fin, voice)
        checks["audio_is_cloned_voice"] = details["corr_with_cloned_voice"] >= 0.95
        veo_audio = final.parent.parent / "audio" / "original_veo_audio.wav"
        if veo_audio.exists():
            details["corr_with_veo_audio"] = max(abs(_corr(fin, _load(veo_audio, 16000), lag)) for lag in range(-1600, 1601, 160))
            checks["veo_audio_absent"] = details["corr_with_veo_audio"] < 0.2
        else:
            details["corr_with_veo_audio"] = None
            checks["veo_audio_absent"] = True  # Veo generated no audio at all
        final_wav = final.with_name("final_audio_check.wav")
        try:
            run_ffmpeg(["-i", str(final), "-vn", "-ac", "1", "-ar", "16000", str(final_wav)], what="decoding the final audio")
            heard = self.asr.transcribe(final_wav)
            match = dialogue.compare(script, heard["text"], self.rules.dialogue)
        except ASRError as exc:
            match = {"passed": False, "reasons": [str(exc)]}
        finally:
            final_wav.unlink(missing_ok=True)
        details["final_transcript"] = match
        checks["final_speech_matches_script"] = bool(match.get("passed"))
        return {"checks": checks, "details": details}

    # ------------------------------------------------------------------ helpers
    def _stage(self, project_id: str, state: str, message: str) -> None:
        if self.store.get(project_id)["cancel_requested"]:
            raise Cancelled()
        self.store.update(project_id, state=state, message=message)

    def _require(self, project_id: str, states: set) -> dict:
        project = self.store.get(project_id)
        if project is None:
            raise ValueError("Project not found.")
        if (project["params"] or {}).get("mode") != "strict":
            raise ValueError("This is not a Strict Talking Head project.")
        if project["state"] not in states:
            raise ValueError(f"Not possible while the project is {project['state']}.")
        return project


def _decode(path: Path, *, mono: bool = False, sr: int = 48000) -> np.ndarray:
    import subprocess

    from .media import ffmpeg_bin

    raw = subprocess.run([ffmpeg_bin(), "-loglevel", "error", "-i", str(path), "-vn", "-f", "f32le",
                          "-ac", "1" if mono else "2", "-ar", str(sr), "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _load(path: Path, sr: int) -> np.ndarray:
    import librosa

    return librosa.load(path, sr=sr, mono=True)[0].astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray, lag: int = 0) -> float:
    if lag > 0:
        a, b = a[lag:], b
    elif lag < 0:
        a, b = a, b[-lag:]
    n = min(a.size, b.size)
    if n < 100:
        return 0.0
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return round(float((a * b).sum() / den), 4) if den else 0.0
