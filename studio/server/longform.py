"""Long talking head: one approved script, spoken across several 8-second clips cut together.

Veo pins a clip to 8 s when it gets a reference photo, and the gateway has no "continue from
the last frame" input. So a 1-1.5 minute talking head is made like an edited interview:

1. The script is split at sentence (then clause) boundaries into segments that each fit one
   clip (at most 17 words, about 6.5 s at 2.6 words/s).
2. Each segment is an ordinary Strict Talking Head project (a "child"): same checks, same
   voice conversion, one generation each. Children are generated one after another.
3. Each take gets its own performance variant: timed actions, shot size (the photo's own
   framing or closer, alternating) and expression. Without this every clip shows Veo's one
   default "presenter" behaviour.
4. The clips are cut together with hard cuts: each clip is trimmed to its speech (cuts on
   whole frames; audio cut on the same instants, 2000 samples per frame at 24 fps), levels
   are matched with one constant gain per clip, and a 10 ms fade removes clicks. Crossfades
   are not used: blending two different head positions looks like ghosting.

The parent project never calls the provider itself; generation happens only through the
children's existing approve step, after an explicit request naming the number of generations.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from . import dialogue
from .media import MediaError, probe, run_ffmpeg, speech_regions
from .store import State

log = logging.getLogger("studio.long")

WORDS_PER_SECOND = 2.6          # the same estimate a strict child uses
MAX_SEGMENT_WORDS = 17          # 17 / 2.6 = 6.5 s, inside a child's 7.2 s limit
MIN_SEGMENT_WORDS = 6
MAX_SEGMENTS = 15               # about 1 min 45 s
MAX_SCRIPT_CHARS = 4000
LEAD_S = 0.12                   # kept before the first word of a clip
TAIL_S = 0.35                   # kept after the last word (breath before the cut)
LAST_TAIL_S = 0.9               # the closing clip keeps its nod and smile
MAX_GAIN_DB = 6.0

READY, GENERATING, STITCHING, MATCHING = "LONG_READY", "LONG_GENERATING", "LONG_STITCHING", "LONG_MATCHING"
RUNNING = {GENERATING, STITCHING, MATCHING}
VOICE = {"pace_wps": 2.6, "consistent_voice": True}
ANCHOR_SOURCES = {"voice": "your voice recording", "clip1": "clip 1"}


def pitch_word(hz: Optional[float]) -> str:
    """How the prompt describes the voice, from a measured pitch."""
    if not hz:
        return "mid"
    return "low" if hz < 115 else "high" if hz > 165 else "mid"


_voice_pitch_cache: dict = {}


def voice_pitch_hz(path: Path) -> Optional[float]:
    """Typical pitch of an authorised voice recording (its first minute)."""
    from . import timing

    key = (str(path), path.stat().st_mtime)
    if key not in _voice_pitch_cache:
        wav, sr = sf.read(path, dtype="float32")
        wav = wav if wav.ndim == 1 else wav.mean(axis=1)
        _voice_pitch_cache[key] = timing.median_pitch_hz(wav[: int(60 * sr)], sr)
    return _voice_pitch_cache[key]
DONE_STATES = {State.NEEDS_REVIEW, State.LIKENESS_ACCEPTED, State.COMPLETED}
PREPARING = {State.IDLE, State.VALIDATING_INPUT, State.PREPARING_REFERENCE}


class LongError(ValueError):
    pass


# ------------------------------------------------------------------ script splitting
def _count(text: str) -> int:
    return len(dialogue.words(text))


def _even_chunks(clause: str, limit: int) -> list[str]:
    tokens = clause.split()
    n = math.ceil(len(tokens) / limit)
    size = math.ceil(len(tokens) / n)
    return [" ".join(tokens[i:i + size]) for i in range(0, len(tokens), size)]


def _pieces(sentence: str) -> list[str]:
    """A sentence, or its clauses when it is too long for one clip, or even word chunks
    when a clause is still too long."""
    if _count(sentence) <= MAX_SEGMENT_WORDS:
        return [sentence]
    out = []
    for clause in re.split(r"(?<=[,;:])\s+", sentence):
        out += [clause] if _count(clause) <= MAX_SEGMENT_WORDS else _even_chunks(clause, MAX_SEGMENT_WORDS)
    return out


def _balanced(pieces: list[str]) -> list[str]:
    """Group consecutive pieces into the fewest clips, then the most even ones (least squared
    slack below the word limit). Every clip costs a generation, so fewer is better."""
    counts = [_count(p) for p in pieces]
    best: list = [(0, 0, -1)] + [None] * len(pieces)      # (clips, slack cost, previous cut)
    for j in range(1, len(pieces) + 1):
        words = 0
        for i in range(j - 1, -1, -1):
            words += counts[i]
            if words > MAX_SEGMENT_WORDS and i < j - 1:
                break
            if best[i] is None:
                continue
            cand = (best[i][0] + 1, best[i][1] + (MAX_SEGMENT_WORDS - words) ** 2, i)
            if best[j] is None or cand[:2] < best[j][:2]:
                best[j] = cand
    cuts, j = [], len(pieces)
    while j > 0:
        i = best[j][2]
        cuts.append(" ".join(pieces[i:j]))
        j = i
    return cuts[::-1]


def split_script(text: str) -> list[str]:
    clean = dialogue.normalize_script(text)
    if not clean or not re.search(r"[A-Za-z]", clean):
        raise LongError("Write the script first.")
    if '"' in clean:
        raise LongError('Remove double quotes (") from the script; they would break the instruction to the video model.')
    if len(clean) > MAX_SCRIPT_CHARS:
        raise LongError(f"The script is {len(clean)} characters; keep it under {MAX_SCRIPT_CHARS}.")
    pieces = [p for s in re.split(r"(?<=[.!?])\s+", clean) for p in _pieces(s) if p.strip()]
    segments = _balanced(pieces)
    if len(segments) > MAX_SEGMENTS:
        raise LongError(f"This script needs {len(segments)} clips; the limit is {MAX_SEGMENTS} (about "
                        f"{MAX_SEGMENTS * 7 // 60} min {MAX_SEGMENTS * 7 % 60} s). Shorten it or make two videos.")
    for seg in segments:
        dialogue.validate_script(seg)
    return segments


# ------------------------------------------------------------------ performance variants
OPENERS = ["slight_head_tilt", "warm_smile", "eyebrow_lift", "lean_in", "steady_eye_contact"]
MIDDLES = ["small_nod", "glance_away_return", "weight_shift", "slight_head_tilt", "eyebrow_lift"]
HANDS = ["open_palm", "both_hands_explain", "hand_to_chest"]
CLOSERS = ["small_nod", "warm_smile", "steady_eye_contact", "weight_shift"]
STOPWORDS = {"about", "after", "again", "their", "there", "these", "those", "which", "while", "would", "could",
             "should", "where", "every", "being", "other", "people", "thank", "hello", "today", "always", "because"}
ACTION_LABELS = {"slight_head_tilt": "head tilt", "small_nod": "nod", "eyebrow_lift": "eyebrows lift",
                 "warm_smile": "smile", "lean_in": "leans in", "weight_shift": "shoulder shift",
                 "glance_away_return": "glance away", "steady_eye_contact": "eye contact",
                 "open_palm": "open-palm gesture", "both_hands_explain": "two-hand gesture",
                 "hand_to_chest": "hand to chest"}


def _pick(pool: list[str], start: int, avoid: set) -> str:
    for k in range(len(pool)):
        cand = pool[(start + k) % len(pool)]
        if cand not in avoid:
            return cand
    return pool[start % len(pool)]


def emphasis_word(text: str) -> Optional[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*[A-Za-z]", text)
    cands = [t for t in tokens if len(t) >= 5 and t.lower() not in STOPWORDS]
    return max(cands, key=len) if cands else None


def variant(text: str, variant_no: int, reference_mode: str, *, last: bool) -> dict:
    """Deterministic per-take direction overrides: the same (text, number) always gives the
    same variant, and neighbouring numbers differ in shot size and actions."""
    speech = _count(text) / WORDS_PER_SECOND
    wide = reference_mode != "full" or variant_no % 2 == 0     # face-crop mode has one fixed framing
    hands = reference_mode == "full" and wide
    a = OPENERS[variant_no % len(OPENERS)]
    b = _pick(HANDS, variant_no // 2, {a}) if hands else _pick(MIDDLES, variant_no, {a})
    c = _pick(CLOSERS, variant_no + 1, {a, b})
    end = min(8, max(3, math.ceil(speech)))
    cut1, cut2 = max(1, round(end / 3)), max(2, round(2 * end / 3))
    cut2 = min(max(cut2, cut1 + 1), end - 1)
    beats = [{"start_s": 0, "end_s": cut1, "action": a}, {"start_s": cut1, "end_s": cut2, "action": b},
             {"start_s": cut2, "end_s": end, "action": c}]
    overrides: dict = {"performance": {
        "beats": beats, "ending": "nod_and_smile" if last else "soft_smile", "head_motion": "low",
        "motion_intensity": "low", "facial_expression": "warm" if variant_no % 2 == 0 else "subtle",
        "hand_gestures": "occasional_natural" if hands else "avoid_unnecessary_gestures"}}
    if reference_mode == "full":
        overrides["scene"] = {"composition": "preserve_reference" if wide else "medium_close_up"}
    overrides["speech"] = dict(VOICE)                       # the same voice and speed in every clip
    word = emphasis_word(text)
    if word:
        overrides["speech"]["emphasis"] = word
    summary = {"framing": "photo framing" if wide else "closer", "actions": [ACTION_LABELS[x] for x in (a, b, c)],
               "ending": "nod and smile" if last else "soft smile", "emphasis": word}
    return {"variant_no": variant_no, "overrides": overrides, "summary": summary}


def plan(script: str, *, reference_mode: str = "full", preset: str = "natural_human",
         user_overrides: Optional[dict] = None, start_variant: int = 0, voice_pitch: str = "mid") -> list[dict]:
    """Segments with their variants and the exact prompt each clip would send. No network."""
    from . import direction as direction_mod
    from .prompt_compiler import compile_prompt

    texts = split_script(script)
    out = []
    for i, text in enumerate(texts):
        v = variant(text, start_variant + i, reference_mode, last=i == len(texts) - 1)
        v["overrides"]["speech"]["voice_pitch"] = voice_pitch
        words = _count(text)
        est = round(words / WORDS_PER_SECOND, 2)
        d = direction_mod.for_strict(preset=preset, overrides=direction_mod._merge(user_overrides or {}, v["overrides"]),
                                     script=text, reference_mode=reference_mode, voice_method="convert",
                                     duration_seconds=8, speech_seconds=est)
        out.append({"index": i, "text": text, "words": words, "est_seconds": est, **v, "prompt": compile_prompt(d)})
    return out


# ------------------------------------------------------------------ stitching
def _ffprobe_frames(path: Path) -> int:
    import subprocess

    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return int((r.stdout.strip() or "0").split(",")[0])


def cut_points(final: Path, work: Path, *, last: bool) -> dict:
    """Where to cut one clip: whole frames from just before its first word to just after its
    last word, plus its speech level."""
    info = probe(final)
    fps = info.get("fps") or 24.0
    frames = _ffprobe_frames(final) or int(round(info["duration"] * fps))
    wav = work / f"cut_{final.parent.parent.name}.wav"
    run_ffmpeg(["-i", str(final), "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(wav)],
               what="reading a clip's audio")
    regions = speech_regions(wav)
    if not regions:
        raise LongError(f"No speech was found in {final.parent.parent.name}.")
    on, off = regions[0][0], regions[-1][1]
    start = max(0, math.floor((on - LEAD_S) * fps))
    end = min(frames, math.ceil((off + (LAST_TAIL_S if last else TAIL_S)) * fps))
    audio, sr = sf.read(wav, dtype="float32")
    speech = audio[int(on * sr):int(off * sr)]
    level = float(20 * np.log10(np.sqrt(np.mean(speech ** 2)) + 1e-9)) if speech.size else -60.0
    return {"fps": fps, "frames": frames, "start_frame": start, "end_frame": end, "speech_s": [round(on, 3), round(off, 3)],
            "level_db": round(level, 2), "width": info.get("width"), "height": info.get("height")}


def join(clips: list[Path], cuts: list[dict], out: Path) -> dict:
    """Hard cuts: trim each clip to its frames, match levels, concatenate, encode once."""
    fps = cuts[0]["fps"]
    w, h = cuts[0]["width"], cuts[0]["height"]
    target = float(np.median([c["level_db"] for c in cuts]))
    spf = 48000 / fps                                           # audio samples per video frame
    filters, chain = [], ""
    for i, c in enumerate(cuts):
        c["gain_db"] = round(float(np.clip(target - c["level_db"], -MAX_GAIN_DB, MAX_GAIN_DB)), 2)
        a0, a1 = int(round(c["start_frame"] * spf)), int(round(c["end_frame"] * spf))
        dur = (a1 - a0) / 48000
        filters.append(f"[{i}:v:0]trim=start_frame={c['start_frame']}:end_frame={c['end_frame']},setpts=PTS-STARTPTS,"
                       f"scale={w}:{h},setsar=1,fps={fps},format=yuv420p[v{i}]")
        filters.append(f"[{i}:a:0]aresample=48000,aformat=channel_layouts=stereo,atrim=start_sample={a0}:end_sample={a1},"
                       f"asetpts=PTS-STARTPTS,volume={c['gain_db']}dB,afade=t=in:st=0:d=0.01,"
                       f"afade=t=out:st={max(0.0, dur - 0.01):.6f}:d=0.01[a{i}]")
        chain += f"[v{i}][a{i}]"
    filters.append(f"{chain}concat=n={len(cuts)}:v=1:a=1[v][a]")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    args = [x for p in clips for x in ("-i", str(p))]
    args += ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18",
             "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
             "-movflags", "+faststart", str(tmp)]
    log_text = run_ffmpeg(args, what="cutting the clips together")
    tmp.replace(out)
    total = sum(c["end_frame"] - c["start_frame"] for c in cuts)
    return {"fps": fps, "frames_expected": total, "seconds_expected": round(total / fps, 3), "target_level_db": round(target, 2),
            "ffmpeg": " ".join(["ffmpeg", *args[:2 * len(clips)], "-filter_complex", "<see filters>", *args[2 * len(clips) + 2:]]),
            "filters": filters, "log": (log_text or "")[-2000:]}


# ------------------------------------------------------------------ orchestration
class LongForm:
    def __init__(self, store, pipeline, make_strict, strict_in):
        self.store, self.p, self.make_strict, self.strict_in = store, pipeline, make_strict, strict_in
        self._stop: set[str] = set()
        self._running: set[str] = set()
        self._lock = threading.Lock()
        self.poll_s = 2.0

    # ---------------------------------------------------------------- create
    def create(self, template: dict, script: str, *, name: Optional[str] = None) -> dict:
        """`template` is a StrictIn body without the script; one child per segment."""
        vpath = self.p.resolve_voice(template["voice"]) if template.get("voice") else None
        segs = plan(script, reference_mode=template.get("reference_mode", "full"),
                    voice_pitch=pitch_word(voice_pitch_hz(vpath)) if vpath else "mid",
                    preset=(template.get("direction") or {}).get("preset", "natural_human"),
                    user_overrides=(template.get("direction") or {}).get("overrides"))
        clean = dialogue.normalize_script(script)
        params = {"mode": "long", "long": {
            "script": clean, "template": template,
            "segments": [{"index": s["index"], "text": s["text"], "words": s["words"], "est_seconds": s["est_seconds"],
                          "takes": [], "current": None, "variant_no": s["variant_no"], "summary": s["summary"]}
                         for s in segs]}}
        parent = self.store.create(prompt=f"Long talking head: {len(segs)} clips", model=template.get("model", ""),
                                   params=params, project_id=name or None)
        pid = parent["id"]
        try:
            for s in segs:
                self._new_take(pid, s["index"], s["variant_no"], s["overrides"])
        except Exception as exc:
            self.store.update(pid, state=State.FAILED, error=f"Could not prepare the clips: {getattr(exc, 'detail', exc)}")
            raise
        self.store.update(pid, state=READY, message=f"Preparing {len(segs)} clips")
        return self.view(pid)

    def _new_take(self, pid: str, index: int, variant_no: int, overrides: dict) -> str:
        parent = self.store.get(pid)
        lf = parent["params"]["long"]
        seg = lf["segments"][index]
        take = len(seg["takes"]) + 1
        cid = f"{pid}-s{index + 1:02d}" + (f"-t{take}" if take > 1 else "")
        body = self.strict_in(**{**lf["template"], "script": seg["text"], "voice_method": "convert", "name": None})
        child = self.make_strict(body, project_id=cid, extra_params={"parent": pid, "segment": index},
                                 direction_overrides=overrides)
        parent = self.store.get(pid)                      # re-read: never overwrite a concurrent update
        seg = parent["params"]["long"]["segments"][index]
        seg["takes"].append(child["id"])
        seg["current"] = child["id"]
        seg["variant_no"] = variant_no
        self.store.update(pid, params=parent["params"])
        return child["id"]

    # ---------------------------------------------------------------- view
    def view(self, pid: str) -> dict:
        parent = self._get(pid)
        lf = parent["params"]["long"]
        segs, n_done, n_waiting, failing_clips = [], 0, 0, []
        from . import results as results_mod

        for seg in lf["segments"]:
            child = self.store.get(seg["current"]) if seg["current"] else None
            row = {k: seg[k] for k in ("index", "text", "words", "est_seconds", "summary", "takes")}
            if child:
                folder = self.store.project_dir(child["id"])
                v = (child["params"].get("strict") or {}).get("validation") or {}
                fails = results_mod.blocking_failures({"rows": v["results"]}) if v.get("results") else []
                labels = [r["label"] for r in v.get("results") or [] if r["key"] in fails]
                has_final = (folder / "output" / "final_talking_head.mp4").exists()
                row.update({"project_id": child["id"], "state": child["state"], "message": child.get("message"),
                            "error": child.get("error"), "active": child["state"] in State.ACTIVE,
                            "failed_checks": labels, "ready": child["state"] in DONE_STATES and has_final,
                            "final_url": f"/api/projects/{child['id']}/files/final_talking_head.mp4" if has_final else None,
                            "prompt": ((child["params"].get("strict") or {}).get("direction_record") or {}).get("prompt")})
                n_done += row["ready"]
                n_waiting += child["state"] in PREPARING | {State.AWAITING_AUDIO_APPROVAL}
                if row["ready"] and labels:
                    failing_clips.append(seg["index"] + 1)
            segs.append(row)
        stitch = lf.get("stitch")
        out = {k: parent.get(k) for k in ("id", "created_at", "updated_at", "state", "message", "error", "model")}
        out.update({"mode": "long", "script": lf["script"], "segments": segs, "clips": len(segs), "clips_ready": n_done,
                    "clips_to_generate": n_waiting, "clips_with_failed_checks": failing_clips,
                    "can_stitch": n_done == len(segs) and parent["state"] not in RUNNING,
                    "running": parent["state"] in RUNNING, "anchor_source": lf.get("anchor_source", "voice"),
                    "anchor": self._anchor_view(pid, lf),
                    "can_match": parent["state"] not in RUNNING and (
                        n_done >= 1 if lf.get("anchor_source", "voice") == "voice"
                        else n_done >= 2 and bool(segs and segs[0].get("ready"))),
                    "est_seconds": round(sum(s["est_seconds"] + LEAD_S + TAIL_S for s in lf["segments"]), 1),
                    "stitch": stitch, "review": lf.get("review"),
                    "final_url": f"/api/long/{pid}/final.mp4" if stitch and (self._final(pid)).exists() else None})
        return out

    # ---------------------------------------------------------------- generate
    def generate(self, pid: str, *, reviewer: str, generations: int) -> dict:
        parent = self._get(pid)
        if parent["state"] in RUNNING:
            raise LongError("This video is already running.")
        if not reviewer.strip():
            raise LongError("Enter your name to approve the generations.")
        self._require_gate()
        pending = self._pending(pid)
        if not pending:
            raise LongError("Every clip has already been generated. Redo a clip to make a new take.")
        if generations != len(pending):
            raise LongError(f"This uses {len(pending)} generation{'s' if len(pending) != 1 else ''}; "
                            f"confirm that number to start.")
        with self._lock:
            if pid in self._running:
                raise LongError("This video is already running.")
            self._running.add(pid)
            self._stop.discard(pid)
        self.store.update(pid, state=GENERATING, error=None, message=f"Generating clip 1 of {len(pending)}")
        threading.Thread(target=self._generate_all, args=(pid, pending, reviewer.strip()), daemon=True,
                         name=f"long-{pid}").start()
        return self.view(pid)

    def _pending(self, pid: str) -> list[int]:
        lf = self._get(pid)["params"]["long"]
        out = []
        for seg in lf["segments"]:
            child = self.store.get(seg["current"]) if seg["current"] else None
            if child and child["state"] in PREPARING | {State.AWAITING_AUDIO_APPROVAL}:
                out.append(seg["index"])
        return out

    def _require_gate(self) -> None:
        from .utils_api import ProviderCallsDisabled

        try:
            client = self.p._client_factory(self.p.settings.load())
        except ProviderCallsDisabled:
            raise LongError("Video generation is switched off (STUDIO_PROVIDER_CALLS). Nothing was generated.")
        close = getattr(client, "close", None)
        if close:
            close()

    def _wait(self, cid: str, until, timeout: float) -> dict:
        t0 = time.time()
        while True:
            child = self.store.get(cid)
            if until(child) or time.time() - t0 > timeout:
                return child
            time.sleep(self.poll_s)

    def _generate_all(self, pid: str, pending: list[int], reviewer: str) -> None:
        done, problems = 0, []
        try:
            for k, index in enumerate(pending):
                if pid in self._stop:
                    break
                cid = self._get(pid)["params"]["long"]["segments"][index]["current"]
                self.store.update(pid, message=f"Clip {k + 1} of {len(pending)}: preparing")
                child = self._wait(cid, lambda c: c["state"] not in PREPARING, timeout=300)
                if child["state"] != State.AWAITING_AUDIO_APPROVAL:
                    problems.append(f"clip {index + 1}: {child.get('error') or child['state']}")
                    continue
                self._prime(pid, index, cid)
                try:
                    self.p.strict.approve_audio(cid, reviewer=reviewer)
                except ValueError as exc:
                    problems.append(f"clip {index + 1}: {exc}")
                    continue
                self.store.update(pid, message=f"Clip {k + 1} of {len(pending)}: generating")
                child = self._wait(cid, lambda c: c["state"] not in State.ACTIVE, timeout=3600)
                if child["state"] in DONE_STATES:
                    done += 1
                else:
                    problems.append(f"clip {index + 1}: {child.get('error') or child['state']}")
                    self.store.update(pid, message=f"Clip {index + 1} stopped: {child.get('error') or child['state']}")
        except Exception as exc:   # never leave the parent stuck in GENERATING
            log.exception("long video %s", pid)
            problems.append(str(exc))
        finally:
            with self._lock:
                self._running.discard(pid)
                stopped = pid in self._stop
                self._stop.discard(pid)
            view = self.view(pid)
            msg = (f"Stopped. {view['clips_ready']} of {view['clips']} clips are ready." if stopped else
                   "All clips are ready. Cut them together." if view["clips_ready"] == view["clips"] else
                   f"{view['clips_ready']} of {view['clips']} clips are ready.")
            self.store.update(pid, state=READY, message=msg, error="; ".join(problems) or None)

    # ---------------------------------------------------------------- one voice across clips
    def anchor(self, pid: str) -> Optional[dict]:
        """The target pitch for the clips: the authorised voice recording's (default), or clip 1's
        converted voice."""
        from . import timing

        parent = self._get(pid)
        lf = parent["params"]["long"]
        if lf.get("anchor_source", "voice") == "voice":
            path = self.p.resolve_voice(lf["template"]["voice"])
            pitch = voice_pitch_hz(path) if path else None
            return {"source": "voice", "label": ANCHOR_SOURCES["voice"], "pitch_hz": pitch,
                    "file": path.name} if pitch else None
        first = lf["segments"][0]["current"]
        child = self.store.get(first) if first else None
        path = self.store.project_dir(first) / "audio" / "converted_voice.wav" if first else None
        if not child or child["state"] not in DONE_STATES or not path.exists():
            return None
        stamp = path.stat().st_mtime
        cached = lf.get("anchor")
        if cached and cached.get("project") == first and cached.get("mtime") == stamp:
            return cached
        wav, sr = sf.read(path, dtype="float32")
        pitch = timing.median_pitch_hz(wav if wav.ndim == 1 else wav.mean(axis=1), sr)
        if not pitch:
            return None
        anchor = {"source": "clip1", "label": ANCHOR_SOURCES["clip1"], "project": first, "pitch_hz": pitch, "mtime": stamp}
        parent = self.store.get(pid)
        parent["params"]["long"]["anchor"] = anchor
        self.store.update(pid, params=parent["params"])
        return anchor

    def _anchor_view(self, pid: str, lf: dict) -> Optional[dict]:
        try:
            a = self.anchor(pid)
        except Exception:      # a missing voice file must not break the page
            return None
        return {k: a[k] for k in ("source", "label", "pitch_hz")} if a else None

    def set_anchor_source(self, pid: str, source: str) -> dict:
        if source not in ANCHOR_SOURCES:
            raise LongError("Choose your voice recording or clip 1.")
        parent = self._get(pid)
        if parent["state"] in RUNNING:
            raise LongError("Wait until the current run finishes.")
        parent["params"]["long"]["anchor_source"] = source
        parent["params"]["long"].pop("anchor", None)
        self.store.update(pid, params=parent["params"])
        return self.view(pid)

    def _prime(self, pid: str, index: int, cid: str) -> None:
        """Before a clip is generated or re-converted: the fixed voice description (its pitch word
        from the target), the speed, and the target pitch for choosing its conversion take."""
        child = self.store.get(cid)
        params = child["params"]
        strict = params["strict"]
        source = self._get(pid)["params"]["long"].get("anchor_source", "voice")
        anchor = self.anchor(pid) if (source == "voice" or index > 0) else None
        if strict.get("direction"):
            speech = strict["direction"].setdefault("speech", {})
            speech.update(VOICE)
            if source == "voice" and anchor:
                speech["voice_pitch"] = pitch_word(anchor["pitch_hz"])
        if anchor:
            strict["pitch_anchor"] = {"pitch_hz": anchor["pitch_hz"], "label": anchor["label"],
                                      "from": anchor.get("project") or anchor.get("file")}
        else:
            strict.pop("pitch_anchor", None)
        self.store.update(cid, params=params)

    def match_voices(self, pid: str) -> dict:
        """Re-convert the clips against the target pitch. Uses the clips already on disk: the
        quota guard refuses any new generation, so this costs nothing but local time."""
        view = self.view(pid)
        if view["running"]:
            raise LongError("Wait until the current run finishes.")
        if not view["can_match"]:
            raise LongError("Clip 1 and at least one other clip must be ready first." if view["anchor_source"] == "clip1"
                            else "At least one clip must be ready first.")
        first = 0 if view["anchor_source"] == "voice" else 1         # clip 1 is the target itself otherwise
        if not view["anchor"]:
            raise LongError("The target pitch could not be measured.")
        targets = [s["index"] for s in view["segments"][first:] if s.get("ready") and s["state"] == State.NEEDS_REVIEW]
        if not targets:
            raise LongError("No clip can be re-matched (accepted clips are left as they are).")
        with self._lock:
            if pid in self._running:
                raise LongError("This video is already running.")
            self._running.add(pid)
        self.store.update(pid, state=MATCHING, error=None, message=f"Matching voices to {view['anchor']['label']}")
        threading.Thread(target=self._match_all, args=(pid, targets), daemon=True, name=f"match-{pid}").start()
        return self.view(pid)

    def _match_all(self, pid: str, targets: list[int]) -> None:
        problems = []
        try:
            for index in targets:
                cid = self._get(pid)["params"]["long"]["segments"][index]["current"]
                self.store.update(pid, message=f"Matching clip {index + 1} (no new generation)")
                self._prime(pid, index, cid)
                try:
                    self.p.strict.resync(cid)
                except ValueError as exc:
                    problems.append(f"clip {index + 1}: {exc}")
                    continue
                child = self._wait(cid, lambda c: c["state"] not in State.ACTIVE, timeout=1800)
                if child["state"] not in DONE_STATES:
                    problems.append(f"clip {index + 1}: {child.get('error') or child['state']}")
        except Exception as exc:
            log.exception("match %s", pid)
            problems.append(str(exc))
        finally:
            with self._lock:
                self._running.discard(pid)
            parent = self.store.get(pid)
            parent["params"]["long"].pop("review", None)
            self.store.update(pid, params=parent["params"], state=READY, error="; ".join(problems) or None,
                              message="Voices matched. Cut the clips together again."
                              if not problems else "Some clips could not be matched.")

    def stop(self, pid: str) -> dict:
        self._get(pid)
        with self._lock:
            if pid in self._running:
                self._stop.add(pid)
        self.store.update(pid, message="Stopping after the current clip")
        return self.view(pid)

    # ---------------------------------------------------------------- redo one clip
    def redo(self, pid: str, index: int) -> dict:
        parent = self._get(pid)
        if parent["state"] in RUNNING:
            raise LongError("Wait until the current run finishes.")
        segs = parent["params"]["long"]["segments"]
        if not 0 <= index < len(segs):
            raise LongError("No such clip.")
        seg = segs[index]
        child = self.store.get(seg["current"]) if seg["current"] else None
        if child and child["state"] in State.ACTIVE:
            raise LongError("This clip is still being made.")
        if child and child["state"] == State.AWAITING_AUDIO_APPROVAL:
            self.p.cancel(child["id"])             # not generated yet: nothing is lost
        used = max([s["variant_no"] for s in segs]) + 1
        lf = parent["params"]["long"]
        v = variant(seg["text"], used, lf["template"].get("reference_mode", "full"), last=index == len(segs) - 1)
        seg_summary = v["summary"]
        self._new_take(pid, index, v["variant_no"], v["overrides"])
        parent = self.store.get(pid)
        parent["params"]["long"]["segments"][index]["summary"] = seg_summary
        parent["params"]["long"].pop("review", None)
        self.store.update(pid, params=parent["params"], message=f"Clip {index + 1} has a new take to generate")
        return self.view(pid)

    # ---------------------------------------------------------------- stitch
    def stitch(self, pid: str, *, draft: bool = False) -> dict:
        view = self.view(pid)
        if view["running"]:
            raise LongError("Wait until the current run finishes.")
        missing = [s["index"] + 1 for s in view["segments"] if not s.get("ready")]
        if missing:
            raise LongError("These clips are not ready yet: " + ", ".join(map(str, missing)) + ".")
        if view["clips_with_failed_checks"] and not draft:
            raise LongError("These clips failed a check: " + ", ".join(map(str, view["clips_with_failed_checks"])) +
                            ". Redo them, or make a draft to preview the cuts.")
        with self._lock:
            if pid in self._running:
                raise LongError("This video is already running.")
            self._running.add(pid)
        self.store.update(pid, state=STITCHING, error=None, message="Cutting the clips together")
        threading.Thread(target=self._stitch, args=(pid, draft), daemon=True, name=f"stitch-{pid}").start()
        return self.view(pid)

    def _final(self, pid: str) -> Path:
        return self.store.project_dir(pid) / "output" / "final_long_talking_head.mp4"

    def _stitch(self, pid: str, draft: bool) -> None:
        from . import reference

        try:
            view = self.view(pid)
            folder = self.store.project_dir(pid)
            clips = [self.store.project_dir(s["project_id"]) / "output" / "final_talking_head.mp4" for s in view["segments"]]
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                cuts = [cut_points(p, work, last=i == len(clips) - 1) for i, p in enumerate(clips)]
                out = self._final(pid)
                joined = join(clips, cuts, out)
                rows = self._check(out, cuts, joined, view, work)
            failing = [r["label"] for r in rows if r["status"] == "FAIL"]
            record = {"at": time.time(), "draft": draft, "clips": [s["project_id"] for s in view["segments"]],
                      "cuts": cuts, "join": {k: v for k, v in joined.items() if k != "log"}, "results": rows,
                      "final_sha256": reference.sha256(out), "seconds": joined["seconds_expected"]}
            (folder / "reports").mkdir(exist_ok=True)
            (folder / "reports" / "long_form.json").write_text(json.dumps(record, indent=2, default=str))
            parent = self.store.get(pid)
            parent["params"]["long"]["stitch"] = record
            parent["params"]["long"].pop("review", None)
            self.store.update(pid, params=parent["params"], state=State.NEEDS_REVIEW,
                              message=("Draft: some clips failed their checks" if draft else
                                       "Check failed: " + ", ".join(failing) if failing else "Ready to review"))
        except Exception as exc:
            log.exception("stitch %s", pid)
            self.store.update(pid, state=READY, error=f"Cutting the clips together failed: {exc}", message=None)
        finally:
            with self._lock:
                self._running.discard(pid)

    def _check(self, out: Path, cuts: list[dict], joined: dict, view: dict, work: Path) -> list[dict]:
        rows = []
        info = probe(out)
        frames = _ffprobe_frames(out)
        exp = joined["frames_expected"]
        rows.append({"key": "frames", "label": "Every cut frame is in the video, in order",
                     "status": "PASS" if frames == exp else "FAIL", "evidence": f"{frames} frames, expected {exp}"})
        wav = work / "final_48k.wav"
        run_ffmpeg(["-i", str(out), "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(wav)], what="reading the final audio")
        n = sf.info(wav).frames
        diff_ms = (n / 48000 - exp / joined["fps"]) * 1000
        rows.append({"key": "audio_length", "label": "Audio runs exactly as long as the video",
                     "status": "PASS" if abs(diff_ms) <= 1000 / joined["fps"] else "FAIL",
                     "evidence": f"audio {n / 48000:.3f} s, video {exp / joined['fps']:.3f} s ({diff_ms:+.1f} ms)"})
        audio, _ = sf.read(wav, dtype="float32")
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        rows.append({"key": "clipping", "label": "Audio not clipped", "status": "PASS" if peak < 0.999 else "FAIL",
                     "evidence": f"peak {peak:.3f}"})
        # What each cut piece of the FINAL file says, against that clip's approved words.
        heard, wrong, pos = [], [], 0
        for i, c in enumerate(cuts):
            length = (c["end_frame"] - c["start_frame"]) / joined["fps"]
            piece = work / f"piece_{i:02d}.wav"
            run_ffmpeg(["-ss", f"{pos:.6f}", "-t", f"{length:.6f}", "-i", str(wav), "-ar", "16000", str(piece)],
                       what="cutting a piece for the speech check")
            pos += length
            try:
                text = self.p.strict.asr.transcribe(piece)["text"]
            except Exception as exc:
                rows.append({"key": "script", "label": "The video says the full script", "status": "UNCERTAIN",
                             "evidence": f"speech check unavailable: {exc}"})
                break
            heard.append(text)
            if not dialogue.compare(view["segments"][i]["text"], text)["passed"]:
                wrong.append(i + 1)
        else:
            m = dialogue.compare(view["script"], " ".join(heard))
            rows.append({"key": "script", "label": "The video says the full script",
                         "status": "PASS" if m["passed"] and not wrong else "FAIL",
                         "evidence": f"word match {m['word_accuracy']:.0%}" +
                                     (f"; check clip {', '.join(map(str, wrong))}" if wrong else "")})
        failed = view["clips_with_failed_checks"]
        rows.append({"key": "clips", "label": "Every clip passed its own checks", "status": "FAIL" if failed else "PASS",
                     "evidence": ("clips " + ", ".join(map(str, failed)) + " failed") if failed else
                     f"{len(cuts)} clips"})
        return rows

    # ---------------------------------------------------------------- review
    def accept(self, pid: str, *, reviewer: str, override: bool = False) -> dict:
        from . import reference

        parent = self._get(pid)
        lf = parent["params"]["long"]
        record = lf.get("stitch")
        if parent["state"] != State.NEEDS_REVIEW or not record:
            raise LongError("Cut the clips together first.")
        if not reviewer.strip():
            raise LongError("Enter your name to record the review.")
        if record["draft"]:
            raise LongError("A draft cannot be accepted. Redo the clips that failed, then cut again.")
        out = self._final(pid)
        if not out.exists() or reference.sha256(out) != record["final_sha256"]:
            raise LongError("The final video changed after it was checked. Cut it again.")
        failing = [r for r in record["results"] if r["status"] == "FAIL"]
        if any(r["key"] != "script" for r in failing) or (failing and not override):
            raise LongError("Cannot accept: " + ", ".join(r["label"] for r in failing) + ".")
        lf["review"] = {"by": reviewer.strip(), "at": time.time(), "override": bool(failing)}
        self.store.update(pid, params=parent["params"], state=State.LIKENESS_ACCEPTED,
                          message="Accepted as a likeness of the photo (not the photo animated)")
        return self.view(pid)

    # ---------------------------------------------------------------- restart
    def recover(self) -> None:
        for p in self.store.list(limit=1000):
            if p["params"].get("mode") == "long" and p["state"] in RUNNING:
                self.store.update(p["id"], state=READY, message="Stopped when the server restarted. Start it again to continue.")

    def _get(self, pid: str) -> dict:
        p = self.store.get(pid)
        if not p or p["params"].get("mode") != "long":
            raise KeyError(pid)
        return p


# ------------------------------------------------------------------ HTTP
from pydantic import BaseModel, Field, ValidationError  # noqa: E402


class PlanIn(BaseModel):
    script: str = Field(max_length=MAX_SCRIPT_CHARS + 500)
    reference_mode: str = Field("full", pattern="^(full|face_crop)$")
    preset: str = "natural_human"
    voice: Optional[str] = None


class AnchorIn(BaseModel):
    source: str = Field(pattern="^(voice|clip1)$")


class GenerateIn(BaseModel):
    reviewer: str = ""
    generations: int = Field(ge=1, le=MAX_SEGMENTS)


class AcceptIn(BaseModel):
    reviewer: str = ""
    override: bool = False


def router(longform: LongForm, strict_in, settings_store):
    from fastapi import APIRouter, Body, HTTPException
    from fastapi.responses import FileResponse

    r = APIRouter(prefix="/api/long")

    def call(fn):
        try:
            return fn()
        except KeyError:
            raise HTTPException(404, "No such long video.")
        except (LongError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @r.post("/plan")
    def make_plan(body: PlanIn):
        def go():
            path = longform.p.resolve_voice(body.voice) if body.voice and body.voice != "builtin" else None
            segs = plan(body.script, reference_mode=body.reference_mode, preset=body.preset,
                        voice_pitch=pitch_word(voice_pitch_hz(path)) if path else "mid")
            est = sum(s["est_seconds"] + LEAD_S + TAIL_S for s in segs)
            return {"segments": segs, "clips": len(segs), "generations": len(segs), "est_seconds": round(est, 1),
                    "provider_calls": settings_store.load().provider_calls}
        return call(go)

    @r.post("/projects")
    def create(body: dict = Body(...)):
        """The Strict Talking Head fields, with the full script; validated like a strict request."""
        name = (body.get("name") or "").strip() or None
        if name and len(name) > 48:
            raise HTTPException(400, "Project name: at most 48 characters.")
        try:
            parsed = strict_in.model_validate({**body, "name": None})
        except ValidationError as exc:
            raise HTTPException(400, "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()))
        template = parsed.model_dump(exclude={"script", "name"})
        return call(lambda: longform.create(template, parsed.script or "", name=name))

    @r.get("/{pid}")
    def get(pid: str):
        return call(lambda: longform.view(pid))

    @r.post("/{pid}/generate")
    def generate(pid: str, body: GenerateIn):
        return call(lambda: longform.generate(pid, reviewer=body.reviewer, generations=body.generations))

    @r.post("/{pid}/voice-target")
    def voice_target(pid: str, body: AnchorIn):
        return call(lambda: longform.set_anchor_source(pid, body.source))

    @r.post("/{pid}/match-voices")
    def match(pid: str):
        return call(lambda: longform.match_voices(pid))

    @r.post("/{pid}/stop")
    def stop(pid: str):
        return call(lambda: longform.stop(pid))

    @r.post("/{pid}/segments/{index}/redo")
    def redo(pid: str, index: int):
        return call(lambda: longform.redo(pid, index))

    @r.post("/{pid}/stitch")
    def stitch(pid: str, draft: bool = False):
        return call(lambda: longform.stitch(pid, draft=draft))

    @r.post("/{pid}/accept")
    def accept(pid: str, body: AcceptIn):
        return call(lambda: longform.accept(pid, reviewer=body.reviewer, override=body.override))

    @r.get("/{pid}/final.mp4")
    def final(pid: str, download: bool = False):
        call(lambda: longform._get(pid))
        path = longform._final(pid)
        if not path.exists():
            raise HTTPException(404, "Not cut together yet.")
        return FileResponse(path, media_type="video/mp4", filename=f"{pid}.mp4" if download else None,
                            content_disposition_type="attachment" if download else "inline")

    return r
