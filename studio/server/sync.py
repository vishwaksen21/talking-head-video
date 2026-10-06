"""Sentence-level audio/video synchronisation.

Each transcript segment knows when its words started in the original video. The cloned
line is placed at that same moment, so the pauses of the original performance survive.

When a cloned line is longer than the time it has (up to the next line, or the end of the
video), it is sped up - but only as far as `max_speedup`, because beyond ~15% speech
starts to sound rushed. Whatever still does not fit pushes the following lines later.

If the speech then runs past the end of the video, the overflow strategy decides:
  * "freeze":  hold the last video frame until the last word has been said (default)
  * "stretch": allow a stronger speed-up (up to `stretch_cap`), then freeze any remainder
The last sentence is never cut, and the video is never padded with black frames.

`plan()` is pure arithmetic on durations, so it is unit-tested without audio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .media import run_ffmpeg

MIN_GAP_S = 0.10        # minimum silence between two pushed lines
TAIL_S = 0.30           # breath after the last word when the frame is frozen
DRIFT_WARN_S = 0.35     # a line starting this much later than the original is reported
TARGET_RMS_DB = -18.0   # speech RMS; the final lands near -16 LUFS integrated (spoken web video)
CEILING_DB = -1.5
FADE_S = 0.008


MIN_TEMPO = 1 / 1.15  # slowing down is capped at the same 15% as speeding up


def plan(segments: list[dict], lengths: list[float], video_duration: float, *,
         max_speedup: float = 1.15, overflow: str = "freeze", stretch_cap: float = 1.30) -> dict:
    """A segment may carry `target_length`: how long the on-screen speaker took to say it.
    The cloned line is then stretched toward that length (within 15% either way), so it
    starts and stops with the mouth. Lip shapes cannot be matched by timing alone."""
    if len(segments) != len(lengths):
        raise ValueError("every segment needs a synthesised length")
    if overflow not in ("freeze", "stretch"):
        raise ValueError("overflow must be 'freeze' or 'stretch'")

    def layout(cap: float) -> list[dict]:
        cursor, placed = 0.0, []
        for i, (seg, length) in enumerate(zip(segments, lengths)):
            start = max(float(seg["start"]), cursor)
            limit = float(segments[i + 1]["start"]) if i + 1 < len(segments) else video_duration
            available = max(limit - start, 0.05)
            tempo = 1.0
            target = seg.get("target_length")
            if target:
                tempo = min(max(length / float(target), MIN_TEMPO), cap)
            if length / tempo > available:
                tempo = min(length / available, cap)
            out_len = length / tempo
            placed.append({
                "index": i,
                "original_start": round(float(seg["start"]), 3),
                "start": round(start, 3),
                "end": round(start + out_len, 3),
                "synth_length": round(length, 3),
                "tempo": round(tempo, 4),
            })
            cursor = start + out_len + MIN_GAP_S
        return placed

    placed = layout(max_speedup)
    speech_end = placed[-1]["end"] if placed else 0.0
    if overflow == "stretch" and speech_end > video_duration:
        placed = layout(max(stretch_cap, max_speedup))
        speech_end = placed[-1]["end"] if placed else 0.0

    freeze = 0.0
    final_duration = video_duration
    if speech_end > video_duration:
        final_duration = round(speech_end + TAIL_S, 3)
        freeze = round(final_duration - video_duration, 3)

    warnings = []
    for p, seg in zip(placed, segments):
        if seg.get("target_length"):
            p["target_length"] = round(float(seg["target_length"]), 3)
            if abs(p["synth_length"] / p["tempo"] - p["target_length"]) > 0.25:
                warnings.append(f"Line {p['index'] + 1} lasts {p['end'] - p['start']:.1f}s; the speaker on screen "
                                f"takes {p['target_length']:.1f}s.")
        elif p["tempo"] > 1.0:
            warnings.append(f"Line {p['index'] + 1} was sped up {round((p['tempo'] - 1) * 100)}% to fit.")
        drift = p["start"] - p["original_start"]
        if drift > DRIFT_WARN_S:
            warnings.append(f"Line {p['index'] + 1} starts {drift:.1f}s later than in the original.")
    if freeze:
        warnings.append(
            f"The cloned speech is {freeze:.1f}s longer than the video, so the last frame is held for {freeze:.1f}s."
        )
    big = [p for p in placed if abs(p["tempo"] - 1) > 0.005 and not p.get("target_length")]
    status = "in_sync" if not warnings and not big else ("extended" if freeze else "adjusted")
    return {
        "video_duration": round(video_duration, 3),
        "speech_end": round(speech_end, 3),
        "final_duration": round(final_duration, 3),
        "freeze_seconds": freeze,
        "max_speedup": max_speedup,
        "overflow": overflow,
        "placements": placed,
        "warnings": warnings,
        "status": status,
    }


def change_tempo(wav: np.ndarray, sr: int, tempo: float, work: Path) -> np.ndarray:
    """Pitch-preserving speed change with FFmpeg's atempo (valid range 0.5-2.0)."""
    if abs(tempo - 1.0) < 1e-3:
        return wav
    src, dst = work / "_tempo_in.wav", work / "_tempo_out.wav"
    sf.write(src, wav, sr)
    run_ffmpeg(["-i", str(src), "-af", f"atempo={tempo:.4f}", "-ar", str(sr), "-ac", "1", str(dst)],
               what="adjusting speech speed")
    out, _ = sf.read(dst, dtype="float32")
    src.unlink(missing_ok=True)
    dst.unlink(missing_ok=True)
    return out


def _fade(wav: np.ndarray, sr: int) -> np.ndarray:
    n = min(int(sr * FADE_S), wav.size // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        wav = wav.copy()
        wav[:n] *= ramp
        wav[-n:] *= ramp[::-1]
    return wav


def render(clips: list[np.ndarray], sr: int, placement: dict, out_path: Path) -> dict:
    """Lay the (already tempo-adjusted) clips on a silent track of the final length."""
    total = int(round(placement["final_duration"] * sr))
    track = np.zeros(total, dtype=np.float32)
    for clip, p in zip(clips, placement["placements"]):
        start = int(round(p["start"] * sr))
        # atempo can return a few ms more than start..end; trim first so the fade lands on the
        # real end and the next gap stays digital silence.
        planned = int(round((p["end"] - p["start"]) * sr))
        clip = _fade(clip[:planned] if planned > 0 else clip, sr)
        end = min(total, start + clip.size)
        track[start:end] += clip[: end - start]

    voiced = track[np.abs(track) > 1e-4]
    if voiced.size:
        rms = float(np.sqrt((voiced ** 2).mean()))
        gain = 10 ** (TARGET_RMS_DB / 20) / max(rms, 1e-6)
        peak = float(np.abs(track).max()) * gain
        ceiling = 10 ** (CEILING_DB / 20)
        if peak > ceiling:
            gain *= ceiling / peak
        track *= gain
    sf.write(out_path, track, sr, subtype="PCM_16")
    return {"file": out_path.name, "duration": round(total / sr, 3), "sample_rate": sr}
