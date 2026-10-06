"""Lip-sync measurement and speech-timing alignment.

**Validation: what is measured.** Per frame, Apple Vision locates the mouth (outer-lip
landmarks). The mean absolute pixel change of a mouth-sized crop between consecutive frames
is "mouth activity", smoothed over 5 frames. It is cross-correlated with the speech
loudness envelope:
- of the final voice
- of Veo's own soundtrack: the "native" pair, which is in sync by construction

**Why this metric.** It was calibrated on a real Veo talking head
(`reports/talking_head_root_cause.md`, "Lip-sync measurement"):

| Audio | Lag at peak |
|---|---|
| Native | 0 frames (r=0.61) |
| The same voice delayed 300 ms | -4 frames (moved by 7 frames ≈ 292 ms) |
| Delayed 500 ms | -9 frames |
| Unrelated audio | -12 frames (edge of the search) |

Vision's lip landmarks were tried first: inner-lip height is only 7-10 px at 720p, and it
correlated r=0.07 even with the native audio, so it is not used.

**What passes.** The voice's lag must be within `max_lag_delta_frames` of the native lag,
with a correlation of at least max(`min_correlation`, `min_ratio_to_native` × native).
This measures *timing* (the mouth moves when the voice speaks), not phoneme-level lip shapes.

**Alignment.** Both soundtracks are transcribed with word timestamps and matched word to
word. Each cloned phrase is placed where the on-screen speaker says it, then fitted to its
length by `sync.plan`, which allows at most 15% change.
"""

from __future__ import annotations

import difflib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import facekit
from .dialogue import words as norm_words
from .media import probe, run_ffmpeg

SMOOTH = 5


@dataclass
class LipSyncRules:
    max_lag_delta_frames: int = 2        # ±83 ms at 24 fps, relative to the native pair
    min_correlation: float = 0.30
    min_ratio_to_native: float = 0.60
    search_frames: int = 16
    max_yaw_deg: float = 25.0
    min_single_face_fraction: float = 0.95


def _smooth(v) -> np.ndarray:
    return np.convolve(np.asarray(v, float), np.ones(SMOOTH) / SMOOTH, mode="same")


def frame_series(video: Path, work: Path) -> dict:
    """Per-frame face count, head yaw and mouth activity, at the video's own frame rate."""
    from PIL import Image

    info = probe(video)
    fps = info["fps"] or 24
    frames = work / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(video), str(frames / "%05d.png")], what="extracting frames")
    files = sorted(frames.glob("*.png"))
    results = facekit.analyse(files)
    activity, counts, yaw, prev = [], [], [], None
    for path, r in zip(files, results):
        faces = r.get("faces") or []
        counts.append(len(faces))
        main = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3]) if faces else None
        iod = facekit.interocular(main) if main else None
        if not main or not iod or not main.get("outer_lips"):
            activity.append(np.nan)
            yaw.append(None)
            prev = None
            continue
        yaw.append(abs(main.get("yaw", 0.0)) * 57.2958)
        lips = np.array(main["outer_lips"])
        cx, cy = lips[:, 0].mean(), lips[:, 1].mean()
        with Image.open(path) as im:
            crop = im.convert("L").crop((int(cx - iod * 0.45), int(cy - iod * 0.3), int(cx + iod * 0.45), int(cy + iod * 0.3)))
            cur = np.asarray(crop.resize((64, 40)), float)
        activity.append(float(np.abs(cur - prev).mean()) if prev is not None else np.nan)
        prev = cur
    return {"fps": fps, "frames": len(files), "activity": activity, "face_counts": counts, "yaw_deg": yaw}


def envelope(audio: Path, fps: float, n: int) -> np.ndarray:
    import librosa

    wav, sr = librosa.load(audio, sr=16000, mono=True)
    hop = sr / fps
    env = np.zeros(n)
    for i in range(n):
        seg = wav[int(i * hop): int((i + 1) * hop)]
        env[i] = float(np.sqrt((seg ** 2).mean())) if seg.size else 0.0
    return env


def correlate(activity, env: np.ndarray, lag: int) -> Optional[float]:
    m = _smooth(np.nan_to_num(np.asarray(activity, float), nan=np.nanmean(np.asarray(activity, float))))
    e = _smooth(env)
    a, b = (m[lag:], e[: len(e) - lag]) if lag >= 0 else (m[: len(m) + lag], e[-lag:])
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def best_lag(activity, env: np.ndarray, search: int) -> tuple[float, int]:
    """Highest correlation over lags -search..search (positive: the mouth moves after the sound)."""
    best = (-1.0, 0)
    for lag in range(-search, search + 1):
        r = correlate(activity, env, lag)
        if r is not None and r > best[0]:
            best = (r, lag)
    return best


def visual_regions(series: dict, *, merge_gap_s: float = 0.35, min_region_s: float = 0.2) -> list[tuple[float, float]]:
    """Where the mouth is visibly talking, from mouth activity alone.

    Used when the generated video has no audio of its own. The threshold is halfway between
    the 10th and 90th percentile of smoothed activity.

    Measured against the audio of a real Veo clip:
    - visual onsets came about 0.25-0.3 s before the sound
    - one 0.44 s pause was not detected
    Treat timings from this as estimates (about ±0.3 s).
    """
    a = np.asarray(series["activity"], float)
    if not np.isfinite(a).any():
        return []
    a = _smooth(np.nan_to_num(a, nan=float(np.nanmedian(a))))
    lo, hi = np.percentile(a, 10), np.percentile(a, 90)
    if hi - lo <= 1e-6:
        return []
    on = a > lo + 0.5 * (hi - lo)
    fps, regions = series["fps"], []
    for i, v in enumerate(on):
        if not v:
            continue
        t0, t1 = i / fps, (i + 1) / fps
        if regions and t0 - regions[-1][1] <= merge_gap_s:
            regions[-1][1] = t1
        else:
            regions.append([t0, t1])
    return [(round(x, 3), round(y, 3)) for x, y in regions if y - x >= min_region_s]


def measure_offset_visual(series: dict, audio: Path, search: int = 8) -> dict:
    """Best lag between the voice and mouth activity, with no native audio to compare against."""
    fps, n = series["fps"], series["frames"]
    r, lag = best_lag(series["activity"], envelope(audio, fps, n), search)
    return {"lag_frames": lag, "native_lag_frames": None, "delta_frames": lag, "offset_s": round(lag / fps, 3),
            "correlation": round(r, 3), "native_correlation": None, "basis": "visual (no generated audio)"}


def measure_offset(series: dict, audio: Path, native: Path, search: int = 16) -> dict:
    """How far `audio` is from the timing the video's own audio has, in seconds.
    Positive: the voice is early (it should be delayed)."""
    fps, n = series["fps"], series["frames"]
    r_a, lag_a = best_lag(series["activity"], envelope(audio, fps, n), search)
    r_n, lag_n = best_lag(series["activity"], envelope(native, fps, n), search)
    return {"lag_frames": lag_a, "native_lag_frames": lag_n, "delta_frames": lag_a - lag_n,
            "offset_s": round((lag_a - lag_n) / fps, 3), "correlation": round(r_a, 3), "native_correlation": round(r_n, 3)}


def validate(video: Path, final_audio: Path, native_audio: Optional[Path], rules: LipSyncRules = LipSyncRules(),
             series: Optional[dict] = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="lipsync_") as tmp:
        s = series or frame_series(video, Path(tmp))
    n, fps = s["frames"], s["fps"]
    corr, lag = best_lag(s["activity"], envelope(final_audio, fps, n), rules.search_frames)
    native = best_lag(s["activity"], envelope(native_audio, fps, n), rules.search_frames) if native_audio else None
    delta = lag - (native[1] if native else 0)
    single = sum(1 for c in s["face_counts"] if c == 1) / max(1, n)
    facing = [y for y in s["yaw_deg"] if y is not None]
    facing_ok = sum(1 for y in facing if y <= rules.max_yaw_deg) / max(1, len(facing))
    floor = max(rules.min_correlation, rules.min_ratio_to_native * native[0]) if native else rules.min_correlation
    reasons = []
    if abs(delta) > rules.max_lag_delta_frames:
        reasons.append(f"the voice is {abs(delta) / fps * 1000:.0f} ms {'ahead of' if delta > 0 else 'behind'} the mouth "
                       f"({delta:+d} frames; allowed ±{rules.max_lag_delta_frames})")
    if corr < floor:
        reasons.append(f"mouth movement follows the voice weakly (r={corr:.2f}, needed {floor:.2f})")
    if single < rules.min_single_face_fraction:
        reasons.append(f"exactly one face in only {single:.0%} of frames")
    if facing_ok < 0.9:
        reasons.append(f"the speaker faces the camera in only {facing_ok:.0%} of frames")
    return {
        "frames": n, "fps": fps,
        "correlation": round(corr, 3), "lag_frames": lag, "lag_delta_frames": delta,
        "offset_ms": round(delta / fps * 1000),
        "native_correlation": round(native[0], 3) if native else None,
        "native_lag_frames": native[1] if native else None,
        "single_face_fraction": round(single, 3), "facing_camera_fraction": round(facing_ok, 3),
        "passed": not reasons, "reasons": reasons, "rules": rules.__dict__,
        "measures": "timing of mouth activity vs voice loudness; not phoneme-level lip shapes",
    }


# ------------------------------------------------------------------ alignment


def align_phrases(tts_words: list[dict], veo_words: list[dict], veo_regions: list[tuple[float, float]],
                  *, gap_s: float = 0.25, min_coverage: float = 0.8) -> Optional[list[dict]]:
    """Match the cloned-voice words to the on-screen speaker's words and group them into
    phrases. Returns [{tts_start, tts_end, veo_start, veo_end, words}], or None when too
    few words match to align safely."""
    def flat(ws):
        out = []
        for w in ws:
            for t in norm_words(w["word"]):
                out.append((t, w["start"], w["end"]))
        return out

    a, b = flat(tts_words), flat(veo_words)
    if not a or not b:
        return None
    pairs = []
    for blk in difflib.SequenceMatcher(None, [x[0] for x in a], [x[0] for x in b], autojunk=False).get_matching_blocks():
        pairs += [(a[blk.a + k], b[blk.b + k]) for k in range(blk.size)]
    if len(pairs) < min_coverage * len(a):
        return None

    def region_end(t):
        for s, e in veo_regions:
            if s - 0.05 <= t <= e + 0.05:
                return e
        return None

    phrases = [[pairs[0]]]
    for prev, cur in zip(pairs, pairs[1:]):
        if cur[1][1] - prev[1][2] >= gap_s or cur[0][1] - prev[0][2] >= gap_s:
            phrases.append([cur])
        else:
            phrases[-1].append(cur)
    out = []
    for ph in phrases:
        veo_start = ph[0][1][1]
        veo_end = ph[-1][1][2]
        snapped = region_end(ph[-1][1][1])  # Whisper word ends run into silence; use the measured sound
        if snapped is not None:
            veo_end = min(veo_end, snapped)
        out.append({"tts_start": ph[0][0][1], "tts_end": ph[-1][0][2], "veo_start": round(veo_start, 3),
                    "veo_end": round(max(veo_end, veo_start + 0.2), 3), "words": " ".join(p[0][0] for p in ph)})
    return out


def align_by_regions(tts_words: list[dict], tts_duration: float, veo_regions: list[tuple[float, float]],
                     *, snap_s: float = 0.25) -> Optional[list[dict]]:
    """Align on measured sound rather than recogniser timestamps.

    Whisper's word times on generated audio can be far off (measured: "Prudential" at 0.0 s
    when the sound starts at 0.9 s). The speech regions from the energy detector are
    reliable. The cloned speech is split so that each part's share of the cloned speech
    equals its region's share of the on-screen speech. Each cut is snapped to the nearest
    word boundary of the cloned speech, so no word is cut. Returns
    [{tts_start, tts_end, veo_start, veo_end}] or None.
    """
    regions = [(float(a), float(b)) for a, b in veo_regions if b - a > 0.05]
    if not regions or not tts_words or tts_duration <= 0:
        return None
    total = sum(b - a for a, b in regions)
    boundaries = sorted({round((w1["end"] + w2["start"]) / 2, 3) for w1, w2 in zip(tts_words, tts_words[1:])})
    cuts, acc = [0.0], 0.0
    for a, b in regions[:-1]:
        acc += b - a
        target = acc / total * tts_duration
        near = [t for t in boundaries if abs(t - target) <= snap_s and t > cuts[-1] + 0.2]
        if not near:
            continue  # no word boundary nearby: keep this region joined to the next one
        cuts.append(min(near, key=lambda t: abs(t - target)))
    cuts.append(tts_duration)
    # Merge regions that did not get their own cut.
    pieces, ri = [], 0
    groups: list[list[tuple[float, float]]] = [[] for _ in range(len(cuts) - 1)]
    acc = 0.0
    for a, b in regions:
        mid_share = (acc + (b - a) / 2) / total * tts_duration
        idx = max(i for i in range(len(cuts) - 1) if cuts[i] <= mid_share + 1e-9)
        groups[idx].append((a, b))
        acc += b - a
    for (ta, tb), grp in zip(zip(cuts, cuts[1:]), groups):
        if not grp:
            continue
        pieces.append({"tts_start": round(ta, 3), "tts_end": round(tb, 3), "veo_start": round(grp[0][0], 3),
                       "veo_end": round(grp[-1][1], 3), "words": ""})
    return pieces or None


def quietest_point(wav: np.ndarray, sr: int, t: float, window_s: float = 0.08) -> float:
    """Move a cut to the quietest 10 ms within ±window_s of `t`. Recogniser word boundaries
    are approximate; cutting at them can split a word (measured: "health" heard as
    "help with" after a cut)."""
    hop = max(1, int(sr * 0.01))
    lo, hi = max(0, int((t - window_s) * sr)), min(wav.size, int((t + window_s) * sr))
    if hi - lo < 2 * hop:
        return t
    best, best_e = t, float("inf")
    for i in range(lo, hi - hop, hop):
        e = float(np.mean(wav[i:i + hop] ** 2))
        if e < best_e:
            best, best_e = (i + hop / 2) / sr, e
    return round(best, 3)


def cut_points(phrases: list[dict], tts_duration: float) -> list[tuple[float, float]]:
    """Slice the cloned voice between phrases at the middle of the pause, never inside a word."""
    bounds = [0.0]
    for left, right in zip(phrases, phrases[1:]):
        bounds.append((left["tts_end"] + right["tts_start"]) / 2)
    bounds.append(tts_duration)
    return list(zip(bounds, bounds[1:]))
