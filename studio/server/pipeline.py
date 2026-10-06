"""The job manager: runs each project through its states on background threads.

IDLE -> GENERATING_VIDEO -> EXTRACTING_AUDIO -> TRANSCRIBING -> AWAITING_TRANSCRIPT_REVIEW
     -> SYNTHESIZING_VOICE -> SYNCHRONIZING -> COMPOSING -> COMPLETED
Any stage can end in FAILED (with `failed_stage`) or CANCELLED.

The run pauses at AWAITING_TRANSCRIPT_REVIEW on purpose: the transcript is never voiced
until a person has seen it and pressed continue.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from . import catalog
from .compose import compose, join_scenes
from .config import STUDIO, Settings, SettingsStore
from .media import MediaError, extract_audio, mean_volume_db, probe, speech_regions
from .store import ProjectStore, State
from .sync import change_tempo, plan, render
from .transcribe import TranscriptionError, low_confidence, transcribe
from .utils_api import UtilsAPIError, UtilsClient, decode_base64
from .voice import BUILTIN_VOICE, ChatterboxVoice, VoiceError

log = logging.getLogger("studio.pipeline")

SILENT_DB = -50.0
LEAD_IN_S = 0.4
LOGO = STUDIO / "web" / "logo.png"  # copy of the approved ../logo.png


class Cancelled(Exception):
    pass


class StageError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class Pipeline:
    def __init__(self, store: ProjectStore, settings: SettingsStore, voice: ChatterboxVoice,
                 client_factory=None, voices_dir: Optional[Path] = None):
        self.store = store
        self.settings = settings
        self.voice = voice
        self.voices_dir = voices_dir or (store.data_dir / "voices")
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        from .utils_api import gated_client

        self._client_factory = client_factory or gated_client
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="studio-job")
        from .strict import StrictTalkingHead

        self.strict = StrictTalkingHead(self)
        self._running: set[str] = set()
        self._lock = threading.Lock()
        self.sleep = time.sleep

    # ------------------------------------------------------------------ public
    def start_generation(self, project_id: str) -> None:
        self._submit(project_id, self._run_from_generation)

    def submit_transcript(self, project_id: str, segments: list[dict], options: dict) -> dict:
        project = self.store.get(project_id)
        if project is None:
            raise ValueError("Project not found.")
        if project["state"] not in (State.AWAITING_TRANSCRIPT_REVIEW, State.COMPLETED, State.FAILED, State.CANCELLED):
            raise ValueError(f"The transcript cannot be submitted while the project is {project['state']}.")
        if not (self._dir(project_id) / "generated.mp4").exists():
            raise ValueError("This project has no generated video yet.")
        duration = project["media"]["generated"]["duration"]
        clean = validate_segments(segments, duration)
        voice_id = options.get("voice") or self.default_voice()
        self.resolve_voice(voice_id)  # raises if missing
        transcript = project["transcript"] or {"original": [], "has_speech": False}
        transcript["edited"] = clean
        params = {**project["params"], "voice": voice_id,
                  "max_speedup": float(options.get("max_speedup", 1.15)),
                  "overflow": options.get("overflow", "freeze"),
                  "exaggeration": float(options.get("exaggeration", 0.5)),
                  "cfg_weight": float(options.get("cfg_weight", 0.5)),
                  "temperature": float(options.get("temperature", 0.8))}
        if not 1.0 <= params["max_speedup"] <= 1.3:
            raise ValueError("Maximum speed-up must be between 1.0 and 1.3.")
        if params["overflow"] not in ("freeze", "stretch"):
            raise ValueError("Overflow must be 'freeze' or 'stretch'.")
        self.store.update(project_id, transcript=transcript, params=params, error=None, failed_stage=None,
                          cancel_requested=0, state=State.SYNTHESIZING_VOICE, message="Waiting for the voice model")
        self._submit(project_id, self._run_from_synthesis)
        return self.store.get(project_id)

    def retry(self, project_id: str, *, approve_new_generation: bool = False) -> dict:
        """Re-run a failed or cancelled project from the stage that stopped it.

        Video generation resumes polling stored operations and skips finished scenes. A
        scene that needs a brand-new generation (it was refused, or never started) only
        runs within the approved count; `approve_new_generation` approves one more.
        """
        project = self.store.get(project_id)
        if project is None:
            raise ValueError("Project not found.")
        if project["state"] not in (State.FAILED, State.CANCELLED):
            raise ValueError("Only a failed or cancelled project can be retried.")
        if project["params"].get("mode") == "strict":
            if approve_new_generation:
                params = dict(project["params"])
                params["extra_generations_approved"] = int(params.get("extra_generations_approved", 0)) + 1
                self.store.update(project_id, params=params)
            return self.strict.retry(project_id)
        has_video = (self._dir(project_id) / "generated.mp4").exists()
        params = dict(project["params"])
        if approve_new_generation:
            params["extra_generations_approved"] = int(params.get("extra_generations_approved", 0)) + 1
        self.store.update(project_id, error=None, cancel_requested=0, params=params)
        if has_video:
            stage = project.get("failed_stage")
            if stage in (State.SYNTHESIZING_VOICE, State.SYNCHRONIZING, State.COMPOSING) and \
                    (project["transcript"] or {}).get("edited"):
                self.store.update(project_id, state=State.SYNTHESIZING_VOICE, message="Retrying voice synthesis")
                self._submit(project_id, self._run_from_synthesis)
            else:
                self.store.update(project_id, state=State.EXTRACTING_AUDIO, message="Retrying audio processing")
                self._submit(project_id, self._run_from_extraction)
        else:
            self.store.update(project_id, state=State.GENERATING_VIDEO, message="Resuming the video generation")
            self._submit(project_id, self._run_from_generation)
        return self.store.get(project_id)

    def cancel(self, project_id: str) -> dict:
        project = self.store.get(project_id)
        if project is None:
            raise ValueError("Project not found.")
        if project["state"] in State.FINAL or project["state"] in (
                State.AWAITING_TRANSCRIPT_REVIEW, State.AWAITING_AUDIO_APPROVAL, State.NEEDS_REVIEW):
            if project["state"] in (State.AWAITING_TRANSCRIPT_REVIEW, State.AWAITING_AUDIO_APPROVAL):
                return self.store.update(project_id, state=State.CANCELLED, message="Cancelled", failed_stage=project["state"])
            return project
        self.store.update(project_id, cancel_requested=1, message="Cancelling…")
        return self.store.get(project_id)

    def recover(self) -> None:
        """After a restart: resume polling paid generations, fail everything else clearly."""
        gate_open = self.settings.load().provider_calls
        for project in self.store.list(limit=1000):
            strict = project["params"].get("mode") == "strict"
            if project["state"] == State.GENERATING_VIDEO and project.get("operation") and not gate_open:
                self.store.update(project["id"], message="Waiting: provider calls are disabled, so polling did not resume")
                continue
            if project["state"] == State.GENERATING_VIDEO and project.get("operation"):
                log.info("resuming generation poll for %s", project["id"])
                self.strict.resume_video(project["id"]) if strict else self.start_generation(project["id"])
            elif project["state"] in State.ACTIVE or project["state"] == State.IDLE:
                self.store.update(project["id"], state=State.FAILED, failed_stage=project["state"],
                                  error="Interrupted because the server stopped. Press Retry to continue.")

    def default_voice(self) -> str:
        voices = self.list_voices()
        return next((v["id"] for v in voices if v["id"] != BUILTIN_VOICE), BUILTIN_VOICE)

    def list_voices(self) -> list[dict]:
        out = []
        for f in sorted(self.voices_dir.glob("*")):
            if f.suffix.lower() not in (".wav", ".mp3", ".flac", ".m4a"):
                continue
            try:
                info = probe(f)
            except MediaError:
                continue
            out.append({"id": f.name, "label": f.stem.replace("_", " ").title(), "duration": info["duration"],
                        "used_seconds": min(info["duration"], 10.0)})
        out.append({"id": BUILTIN_VOICE, "label": "Chatterbox built-in voice", "duration": None, "used_seconds": None})
        return out

    def resolve_voice(self, voice_id: str) -> Optional[Path]:
        if voice_id == BUILTIN_VOICE:
            return None
        path = (self.voices_dir / voice_id).resolve()
        if path.parent != self.voices_dir.resolve() or not path.exists():
            raise ValueError("The selected reference voice file is missing.")
        return path

    # ------------------------------------------------------------------ runner
    def _submit(self, project_id: str, fn) -> None:
        with self._lock:
            if project_id in self._running:
                raise ValueError("This project is already running.")
            self._running.add(project_id)

        def job():
            try:
                fn(project_id)
            except Cancelled:
                project = self.store.get(project_id)
                self.store.update(project_id, state=State.CANCELLED, failed_stage=project["state"],
                                  message="Cancelled", cancel_requested=0)
            except StageError as exc:
                log.warning("project %s failed at %s: %s", project_id, exc.stage, exc)
                self.store.update(project_id, state=State.FAILED, failed_stage=exc.stage, error=str(exc), message=None)
            except Exception as exc:  # last resort: never leave a job silently stuck
                log.error("project %s crashed: %s", project_id, traceback.format_exc())
                project = self.store.get(project_id)
                self.store.update(project_id, state=State.FAILED, failed_stage=project["state"],
                                  error=f"Unexpected error: {exc.__class__.__name__}: {exc}", message=None)
            finally:
                with self._lock:
                    self._running.discard(project_id)

        self._pool.submit(job)

    def _dir(self, project_id: str) -> Path:
        return self.store.project_dir(project_id)

    def _check_cancel(self, project_id: str) -> None:
        if self.store.get(project_id)["cancel_requested"]:
            raise Cancelled()

    def _stage(self, project_id: str, state: str, message: str) -> None:
        self._check_cancel(project_id)
        self.store.update(project_id, state=state, message=message)

    # ------------------------------------------------------------------ stages
    def _run_from_generation(self, project_id: str) -> None:
        self._generate(project_id)
        self._run_from_extraction(project_id)

    def _run_from_extraction(self, project_id: str) -> None:
        self._extract_and_transcribe(project_id)

    def _run_from_synthesis(self, project_id: str) -> None:
        self._synthesize_and_compose(project_id)

    def _generate(self, project_id: str) -> None:
        """Generate every scene, in order, then join them into `generated.mp4`.

        Each scene's operation id is stored the moment the API accepts it, and a scene
        whose clip is already on disk is never requested again - so Retry, Cancel+Retry
        and a server restart resume polling instead of paying for a second generation.
        """
        project = self.store.get(project_id)
        settings: Settings = self.settings.load()
        stage = State.GENERATING_VIDEO
        scenes = project["params"].get("scenes") or [{"prompt": project["prompt"]}]
        folder = self._dir(project_id)
        (folder / "video").mkdir(exist_ok=True)
        try:
            client = self._client_factory(settings)
        except UtilsAPIError as exc:
            raise StageError(stage, str(exc))
        clips = []
        with client:
            for i, scene in enumerate(scenes):
                clip = folder / "video" / f"scene_{i + 1:02d}.mp4"
                if not clip.exists():
                    self._generate_scene(client, settings, project_id, i, len(scenes), scene["prompt"], clip)
                clips.append(clip)
        if len(clips) == 1 and not project["params"].get("scenes"):
            joined = {"scenes": [{"index": 0, "start": 0.0, "duration": probe(clips[0])["duration"],
                                  "file": clips[0].name}], "transition": 0.0}
            shutil.copy(clips[0], folder / "generated.mp4")
        else:
            self._stage(project_id, stage, f"Joining {len(clips)} scenes")
            try:
                joined = join_scenes(clips, folder / "generated.mp4",
                                     transition=float(project["params"].get("transition", 0.5)))
            except MediaError as exc:
                raise StageError(stage, f"Could not join the scenes: {exc}")
        info = probe(folder / "generated.mp4")
        info["scenes"] = joined["scenes"]
        info["transition"] = joined["transition"]
        self.store.merge_media(project_id, "generated", info)

    def _forget_scene_op(self, project_id: str, index: int) -> None:
        params = dict(self.store.get(project_id)["params"])
        ops = dict(params.get("scene_ops") or {})
        ops.pop(str(index), None)
        params["scene_ops"] = ops
        self.store.update(project_id, params=params, operation=None)

    def _scene_config(self, project_id: str, index: int = 0) -> dict:
        project = self.store.get(project_id)
        config = dict(project["params"]["video_config"])
        refs = project["params"].get("reference_files") or []
        scenes = project["params"].get("scenes") or [{}]
        if refs and scenes[index].get("use_references", True):
            images = []
            for ref in refs:
                path = self._dir(project_id) / ref["path"]
                if not path.exists():
                    raise StageError(State.GENERATING_VIDEO, f"Reference image missing: {ref['path']}")
                images.append((path.read_bytes(), ref["mime_type"]))
            config["reference_images"] = catalog.reference_payload(images)
        return config

    def _generate_scene(self, client, settings: Settings, project_id: str, index: int, total: int,
                        prompt: str, clip: Path) -> None:
        stage = State.GENERATING_VIDEO
        label = f"scene {index + 1} of {total}" if total > 1 else "the video"
        project = self.store.get(project_id)
        ops = dict(project["params"].get("scene_ops") or {})
        # Legacy single-scene projects stored the operation on the row itself.
        operation = ops.get(str(index)) or (project.get("operation") if index == 0 and total == 1 else None)
        op = None
        if not operation:
            params = project["params"]
            allowed = total + int(params.get("extra_generations_approved", 0))
            if int(params.get("generations_started", 0)) >= allowed:
                raise StageError(stage, f"Generating {label} again would use another video generation. "
                                        "Press 'Generate again' to approve one more.")
            self._stage(project_id, stage, f"Sending {label} to the video model")
            try:
                op = client.start_video(model=project["model"], prompt=prompt, config=self._scene_config(project_id, index),
                                        location=settings.video_location)
            except UtilsAPIError as exc:
                raise StageError(stage, f"Video generation could not start for {label}: {exc}")
            operation = op.operation
            project = self.store.get(project_id)
            params = dict(project["params"])
            params["scene_ops"] = {**(params.get("scene_ops") or {}), str(index): operation}
            params.setdefault("scene_started_at", {})[str(index)] = time.time()
            params["generations_started"] = int(params.get("generations_started", 0)) + 1
            self.store.update(project_id, operation=operation, params=params, message=f"Generating {label}")
        else:
            self.store.update(project_id, state=stage, message=f"Generating {label}")

        started = (self.store.get(project_id)["params"].get("scene_started_at") or {}).get(str(index)) or time.time()
        while op is None or not op.done:
            if time.time() - started > settings.max_generation_seconds:
                raise StageError(stage, f"{label.capitalize()} was not ready after "
                                        f"{int(settings.max_generation_seconds / 60)} minutes. Press Retry to keep checking.")
            if self.store.get(project_id)["cancel_requested"]:
                self.store.add_warning(project_id, "Cancelled while generating. The API offers no cancel call, "
                                                   "so the remote job may still finish and count against quota.")
                raise Cancelled()
            self.sleep(settings.poll_interval)
            try:
                op = client.poll_video(operation)
            except UtilsAPIError as exc:
                if exc.retryable:
                    log.warning("poll failed, will retry: %s", exc)
                    op = None
                    continue
                raise StageError(stage, f"Could not check the status of {label}: {exc}")
            self.store.update(project_id, message=f"Generating {label} ({int(time.time() - started)}s so far)")

        if op.status == "failed" or op.error or not op.videos:
            self._forget_scene_op(project_id, index)
        if op.status == "failed" or op.error:
            raise StageError(stage, f"The video model reported a failure for {label}: {op.error or 'no reason given'}")
        if not op.videos:
            if op.rai_filtered_count:
                reasons = "; ".join(op.rai_filtered_reasons or []) or "no reason given"
                raise StageError(stage, f"{label.capitalize()} was blocked by Google's safety filter ({reasons}). "
                                        "Rephrase the prompt and generate again.")
            raise StageError(stage, f"The generation of {label} finished but returned no video.")
        video = op.videos[0]
        if not video.get("data"):
            raise StageError(stage, "The video was returned as a storage link, not inline data; this app cannot download it.")
        try:
            data = decode_base64(video["data"])
        except Exception as exc:
            raise StageError(stage, f"The returned video data could not be decoded: {exc}")
        tmp = clip.with_suffix(".part")
        tmp.write_bytes(data)
        try:
            info = probe(tmp)
        except MediaError as exc:
            raise StageError(stage, f"The downloaded video for {label} is not playable: {exc}")
        if not info["has_video"] or info["duration"] <= 0:
            raise StageError(stage, f"The downloaded file for {label} has no video stream.")
        tmp.replace(clip)
        self.store.merge_media(project_id, f"scene_{index + 1:02d}", info)

    def _extract_and_transcribe(self, project_id: str) -> None:
        stage = State.EXTRACTING_AUDIO
        self._stage(project_id, stage, "Extracting the soundtrack")
        folder = self._dir(project_id)
        video = folder / "generated.mp4"
        try:
            info = probe(video)
            previous = self.store.get(project_id)["media"].get("generated") or {}
            for key in ("scenes", "transition"):  # scene boundaries come from the join, keep them
                if key in previous:
                    info[key] = previous[key]
            self.store.merge_media(project_id, "generated", info)
            if not info["has_audio"]:
                return self._no_speech(project_id, "The generated video has no audio track.")
            files = extract_audio(video, folder)
            audio_info = probe(files["original"])
            volume = mean_volume_db(files["original"])
            audio_info["mean_volume_db"] = volume
            self.store.merge_media(project_id, "original_audio", audio_info)
            if volume is None or volume < SILENT_DB:
                return self._no_speech(project_id, "The generated video's audio is silent.")
            regions = speech_regions(files["speech"])
        except MediaError as exc:
            raise StageError(stage, f"Audio extraction failed: {exc}")

        stage = State.TRANSCRIBING
        self._stage(project_id, stage, "Transcribing the speech")
        settings = self.settings.load()
        try:
            with self._client_factory(settings) as client:
                result = transcribe(client, files["speech"], regions, info["duration"],
                                    model=settings.transcribe_model, location=settings.transcribe_location)
        except (TranscriptionError, UtilsAPIError) as exc:
            raise StageError(stage, str(exc))
        if not result["has_speech"]:
            return self._no_speech(project_id, "No speech was recognised in the audio (it may be music or effects only).")
        segments = result["segments"]
        low = low_confidence(segments)
        transcript = {"model": result["model"], "language": result.get("language"), "has_speech": True,
                      "original": segments, "edited": None, "low_confidence": low, "speech_regions": regions,
                      "timing": result["timing"], "requests": result["requests"]}
        self.store.update(project_id, transcript=transcript)
        if result["timing"] != "measured":
            self.store.add_warning(project_id, "Line start times are the transcription model's estimate, "
                                               "not measured from the audio. Check them before continuing.")
        if low:
            self.store.add_warning(project_id, f"The transcription is unsure about line(s) "
                                               f"{', '.join(str(i + 1) for i in low)}. Please listen and check them.")
        self._propose_script(project_id)
        self.store.update(project_id, state=State.AWAITING_TRANSCRIPT_REVIEW,
                          message="Check the transcript, then create the cloned voice")

    def _no_speech(self, project_id: str, reason: str) -> None:
        self.store.update(project_id, transcript={"has_speech": False, "original": [], "edited": None, "reason": reason})
        if not self.store.get(project_id)["params"].get("script"):
            self.store.add_warning(project_id, f"{reason} Type the words you want spoken instead.")
        self._propose_script(project_id)
        self.store.update(project_id, state=State.AWAITING_TRANSCRIPT_REVIEW,
                          message="No speech found. Type what the voice should say")

    def _propose_script(self, project_id: str) -> None:
        """With an approved script, the script - not Veo's audio - is what gets spoken.

        Lines are spread over the scenes in order (line i goes to scene floor(i*S/N)) and
        spaced evenly inside their scene. What Veo actually said is kept as `original`
        and compared, so a mismatch is visible rather than silently voiced.
        """
        project = self.store.get(project_id)
        script = project["params"].get("script")
        if not script:
            return
        scenes = project["media"]["generated"].get("scenes") or [{"start": 0.0, "duration": project["media"]["generated"]["duration"]}]
        transition = project["media"]["generated"].get("transition", 0.0)
        n, k = len(script), len(scenes)
        by_scene: dict[int, list[int]] = {}
        for i in range(n):
            by_scene.setdefault(min(k - 1, i * k // n), []).append(i)
        proposed = []
        for si, idx in sorted(by_scene.items()):
            sc = scenes[si]
            usable = sc["duration"] - (transition if si + 1 < k else 0.0)
            for j, line_index in enumerate(idx):
                start = sc["start"] + LEAD_IN_S + j * (usable / len(idx))
                proposed.append({"start": round(start, 2), "text": script[line_index], "scene": si + 1})
        transcript = project["transcript"] or {}
        heard = " ".join(seg["text"] for seg in transcript.get("original") or [])
        comparison = compare_to_script(heard, script)
        aligned = align_script_to_speech(script, transcript.get("original") or []) \
            if comparison["similarity"] >= ALIGN_MIN_SIMILARITY else None
        if aligned:
            proposed = aligned
            comparison["placement"] = "aligned_to_speech"
        else:
            comparison["placement"] = "spread_over_scenes"
        transcript.update({"approved_script": script, "proposed": proposed, "comparison": comparison})
        self.store.update(project_id, transcript=transcript)
        if transcript["comparison"]["status"] == "differs":
            self.store.add_warning(project_id, "The generated video's speech differs from the approved script. "
                                               "The approved script will be spoken instead.")

    def _synthesize_and_compose(self, project_id: str) -> None:
        project = self.store.get(project_id)
        params = project["params"]
        segments = project["transcript"]["edited"]
        folder = self._dir(project_id)
        stage = State.SYNTHESIZING_VOICE
        try:
            reference = self.resolve_voice(params["voice"])
        except ValueError as exc:
            raise StageError(stage, str(exc))

        if reference is not None:
            (folder / "reference").mkdir(exist_ok=True)
            kept = folder / "reference" / reference.name
            if not kept.exists():
                shutil.copy2(reference, kept)  # byte-identical copy; the voice file itself is never touched
        clips: list[np.ndarray] = []
        sr = 24000
        for old in folder.glob("line_*.wav"):
            old.unlink()
        for i, seg in enumerate(segments):
            self._stage(project_id, stage, f"Speaking line {i + 1} of {len(segments)} in the target voice")
            try:
                wav = self.voice.synthesize(
                    seg["text"], reference,
                    exaggeration=params["exaggeration"], cfg_weight=params["cfg_weight"],
                    temperature=params["temperature"],
                    should_stop=lambda: self.store.get(project_id)["cancel_requested"],
                )
            except VoiceError as exc:
                if str(exc) == "Cancelled.":
                    raise Cancelled()
                raise StageError(stage, str(exc))
            sr = self.voice.sample_rate
            sf.write(folder / f"line_{i + 1:02d}.wav", wav, sr)
            clips.append(wav)

        stage = State.SYNCHRONIZING
        self._stage(project_id, stage, "Lining the voice up with the video")
        video_info = project["media"]["generated"]
        try:
            placement = plan(segments, [c.size / sr for c in clips], video_info["duration"],
                             max_speedup=params["max_speedup"], overflow=params["overflow"])
            adjusted = [change_tempo(c, sr, p["tempo"], folder) for c, p in zip(clips, placement["placements"])]
            voice_info = render(adjusted, sr, placement, folder / "cloned_voice.wav")
        except (MediaError, ValueError) as exc:
            raise StageError(stage, f"Synchronisation failed: {exc}")
        self.store.merge_media(project_id, "cloned_voice", voice_info)
        self.store.update(project_id, sync=placement)

        stage = State.COMPOSING
        self._stage(project_id, stage, "Building the final MP4")
        try:
            final = compose(folder / "generated.mp4", folder / "cloned_voice.wav", folder / "final.mp4",
                            final_duration=placement["final_duration"], freeze_seconds=placement["freeze_seconds"],
                            end_card_logo=LOGO if params.get("end_card") else None)
            self._export(project_id)
        except MediaError as exc:
            raise StageError(stage, str(exc))
        self.store.merge_media(project_id, "final", final)
        # Sync warnings replace the previous run's, so a re-voice does not accumulate stale notes.
        project = self.store.get(project_id)
        kept = [w for w in project["warnings"] if not w.startswith(("Line ", "The cloned speech"))]
        self.store.update(project_id, warnings=kept + placement["warnings"], state=State.COMPLETED,
                          message="Done", error=None, failed_stage=None)


    def _export(self, project_id: str) -> None:
        """Lay the deliverables out in named folders (copies; the working files stay put)."""
        project = self.store.get(project_id)
        folder = self._dir(project_id)
        for sub in ("reference", "video", "audio", "output", "reports"):
            (folder / sub).mkdir(exist_ok=True)
        if (folder / "original_audio.wav").exists():
            shutil.copy2(folder / "original_audio.wav", folder / "audio" / "original_veo_audio.wav")
        shutil.copy2(folder / "cloned_voice.wav", folder / "audio" / "cloned_voice.wav")
        for line in sorted(folder.glob("line_*.wav")):
            shutil.copy2(line, folder / "audio" / line.name)
        script = project["params"].get("script")
        if script:
            (folder / "audio" / "approved_script.txt").write_text("\n\n".join(script) + "\n")
        spoken = (project["transcript"] or {}).get("edited") or []
        (folder / "audio" / "spoken_lines.json").write_text(json.dumps(spoken, indent=2))
        name = project["params"].get("export_name") or "final_video.mp4"
        shutil.copy2(folder / "final.mp4", folder / "output" / name)


ALIGN_MIN_SIMILARITY = 0.6


def align_script_to_speech(script: list[str], segments: list[dict]) -> Optional[list[dict]]:
    """Time the approved script to where the on-screen speaker actually says it.

    Script words are matched to the transcribed words (difflib), each script word takes
    the speech region of its match (unmatched words take their neighbour's), and runs of
    words in the same region become one line. Lines keep the script's exact wording and
    punctuation; they start at the region's measured start and carry its length as
    `target_length`. Returns None when too little of the script can be located.
    """
    norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower())
    tokens = " ".join(script).split()
    keys = [norm(t) for t in tokens]
    heard, owner = [], []
    for ri, seg in enumerate(segments):
        for w in seg["text"].split():
            if norm(w):
                heard.append(norm(w))
                owner.append(ri)
    if not heard or not tokens:
        return None
    region = [None] * len(tokens)
    matched = 0
    for block in difflib.SequenceMatcher(None, keys, heard, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            region[block.a + k] = owner[block.b + k]
            matched += 1
    if matched < 0.6 * len([k for k in keys if k]):
        return None
    last = next(r for r in region if r is not None)
    for i, r in enumerate(region):  # fill gaps forward, keep it monotonic
        last = max(last, r) if r is not None else last
        region[i] = last
    lines: list[dict] = []
    for tok, r in zip(tokens, region):
        if lines and lines[-1]["_region"] == r:
            lines[-1]["text"] += " " + tok
        else:
            seg = segments[r]
            lines.append({"_region": r, "start": round(float(seg["start"]), 2), "text": tok,
                          "target_length": round(float(seg["end"]) - float(seg["start"]), 2)})
    for line in lines:
        line.pop("_region")
    return lines


def compare_to_script(heard: str, script: list[str]) -> dict:
    words = lambda t: re.findall(r"[a-z0-9']+", t.lower())
    a, b = words(heard), words(" ".join(script))
    if not a:
        return {"status": "no_speech", "similarity": 0.0, "heard": heard}
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return {"status": "matches" if ratio >= 0.9 else "differs", "similarity": round(ratio, 3), "heard": heard}


def validate_segments(segments: list[dict], duration: float) -> list[dict]:
    if not isinstance(segments, list) or not segments:
        raise ValueError("Add at least one line of text to speak.")
    clean = []
    for i, seg in enumerate(segments):
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(seg.get("start", 0))
        except (TypeError, ValueError):
            raise ValueError(f"Line {i + 1} has an invalid start time.")
        if not 0 <= start < duration:
            raise ValueError(f"Line {i + 1} starts at {start:.2f}s, outside the {duration:.2f}s video.")
        item = {"start": round(start, 3), "end": seg.get("end"), "text": text}
        target = seg.get("target_length")
        if target is not None:
            try:
                target = float(target)
            except (TypeError, ValueError):
                raise ValueError(f"Line {i + 1} has an invalid target length.")
            if 0.2 <= target <= duration:
                item["target_length"] = round(target, 3)
        clean.append(item)
    if not clean:
        raise ValueError("Every line is empty. Type the words to speak.")
    clean.sort(key=lambda s: s["start"])
    return clean
