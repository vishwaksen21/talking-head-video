"""Sample-exact timing between a source recording and its voice-converted version.

Four different properties, measured separately because they have different certainty:

1. **Duration.** Sample counts, compared exactly (not container durations).
2. **Global alignment.** One offset for the whole track.
   - Found by cross-correlating speech-weighted log-loudness envelopes (2 ms RMS, 0.25 ms hop).
   - Each envelope is clipped 40 dB below its own speech peak and z-normalised, so different
     noise floors (voice conversion raises breaths and room noise: measured -52 vs -57 dBFS)
     and different overall loudness do not drive the result.
   - Sign convention: positive = the converted audio is late.
3. **Speech onsets after pauses.** Energy-based, multi-threshold, may be UNCERTAIN.
   - This is NOT a word-boundary or phoneme detector. It finds where sound resumes after a
     pause of at least `min_pause_s`, measured at three levels below each signal's own speech peak.
   - Rises that do not stay loud for `sustain_s` (breaths, clicks) are ignored.
   - A pair whose time difference changes by more than `max_spread_ms` across the three
     levels, or that cannot be paired, is reported as UNCERTAIN instead of given a number.
4. **Phoneme timing.** Not measured here.
   - Voice conversion regenerates the speech, so internal word and phoneme timing can change
     even when duration and global offset match exactly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

HOP = 6          # samples at 24 kHz = 0.25 ms
SMOOTH = 48      # 2 ms RMS window
LEVELS_DB = (-35.0, -30.0, -25.0)   # below each signal's speech peak


def _env_db(a: np.ndarray) -> np.ndarray:
    e = np.sqrt(np.convolve(a.astype(np.float64) ** 2, np.ones(SMOOTH) / SMOOTH, mode="same"))[::HOP]
    return 20 * np.log10(e + 1e-9)


def _speech_peak_db(db: np.ndarray) -> float:
    return float(np.percentile(db, 99.5))


def _speech_weighted(a: np.ndarray, floor_below_peak_db: float = 40.0) -> np.ndarray:
    db = _env_db(a)
    peak = _speech_peak_db(db)
    x = np.clip(db, peak - floor_below_peak_db, None)
    sd = float(x.std())
    return (x - x.mean()) / sd if sd > 0 else x * 0.0


def global_lag_samples(source: np.ndarray, converted: np.ndarray, max_ms: float = 40.0, sr: int = 24000) -> tuple[int, float]:
    """Offset of `converted` relative to `source` in samples (positive: converted is late),
    and the correlation at that offset (on speech-weighted envelopes)."""
    a, b = _speech_weighted(source), _speech_weighted(converted)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    max_k = int(max_ms / 1000 * sr / HOP)
    best, best_r = 0, -2.0
    for k in range(-max_k, max_k + 1):
        x = a[max(0, -k): n - max(0, k)]
        y = b[max(0, k): n + min(0, k)]
        if len(x) < 100 or x.std() == 0 or y.std() == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best_r:
            best, best_r = k, r
    return best * HOP, best_r


def shift(a: np.ndarray, samples: int) -> np.ndarray:
    """Move audio earlier (positive `samples`) or later (negative), keeping the length."""
    if samples == 0:
        return a
    if samples > 0:
        return np.concatenate([a[samples:], np.zeros(samples, a.dtype)])
    return np.concatenate([np.zeros(-samples, a.dtype), a[:samples]])


def match_length(a: np.ndarray, n: int) -> np.ndarray:
    return a[:n] if a.size >= n else np.concatenate([a, np.zeros(n - a.size, a.dtype)])


def onsets(a: np.ndarray, sr: int = 24000, *, levels_db=LEVELS_DB, min_pause_s: float = 0.2,
           sustain_s: float = 0.04) -> list[dict]:
    """Speech onsets after pauses. Each item: {"times": {level: seconds}} for every level at
    which the onset was found. A pause is a run below the lowest level of at least
    `min_pause_s`. A rise must stay above the level for `sustain_s` to count."""
    if a.size == 0:
        return []
    db = _env_db(a)
    peak = _speech_peak_db(db)
    lowest = peak + min(levels_db)
    min_quiet = int(min_pause_s * sr / HOP)
    sustain = max(1, int(sustain_s * sr / HOP))
    quiet = db < lowest
    starts, run = [], 0
    for i, q in enumerate(quiet):
        if q:
            run += 1
            continue
        if run >= min_quiet:
            window = ~quiet[i:i + sustain]
            if window.size == sustain and window.mean() >= 0.9:   # sustained sound, not a breath or click
                starts.append(i)
        run = 0
    found = []
    for s in starts:
        times = {}
        limit = min(len(db), s + int(0.12 * sr / HOP))
        for lv in levels_db:
            above = np.nonzero(db[s:limit] >= peak + lv)[0]
            if above.size:
                times[lv] = round((s + above[0]) * HOP / sr, 5)
        found.append({"times": times})
    return found


def onset_comparison(source: np.ndarray, converted: np.ndarray, sr: int = 24000, *, window_s: float = 0.06,
                     max_spread_ms: float = 5.0, tolerance_ms: float = 20.0) -> dict:
    """Pair source and converted onsets and report differences with their certainty."""
    src, conv = onsets(source, sr), onsets(converted, sr)
    mid = LEVELS_DB[1]
    pairs, used = [], set()
    for o in src:
        t = o["times"].get(mid)
        if t is None:
            pairs.append({"source_s": None, "status": "UNCERTAIN", "reason": "onset not resolved at all levels"})
            continue
        cands = [(j, c) for j, c in enumerate(conv) if j not in used and c["times"].get(mid) is not None
                 and abs(c["times"][mid] - t) <= window_s]
        if not cands:
            pairs.append({"source_s": t, "status": "UNCERTAIN", "reason": "no matching onset in the converted audio"})
            continue
        j, c = min(cands, key=lambda jc: abs(jc[1]["times"][mid] - t))
        used.add(j)
        diffs = [round((c["times"][lv] - o["times"][lv]) * 1000, 2) for lv in LEVELS_DB
                 if lv in c["times"] and lv in o["times"]]
        spread = max(diffs) - min(diffs) if len(diffs) == len(LEVELS_DB) else None
        if spread is None or spread > max_spread_ms:
            pairs.append({"source_s": t, "converted_s": c["times"][mid], "diffs_ms": diffs, "status": "UNCERTAIN",
                          "reason": f"difference changes by {spread} ms across levels" if spread is not None
                          else "not found at every level"})
        else:
            pairs.append({"source_s": t, "converted_s": c["times"][mid], "diff_ms": float(np.median(diffs)),
                          "diffs_ms": diffs, "status": "MEASURED"})
    unpaired = len(conv) - len(used)
    measured = [p["diff_ms"] for p in pairs if p["status"] == "MEASURED"]
    uncertain = sum(1 for p in pairs if p["status"] == "UNCERTAIN") + unpaired
    if not pairs:
        status = "NOT MEASURABLE"
    elif uncertain or len(measured) < 2:
        status = "UNCERTAIN"
    elif max(abs(d) for d in measured) > tolerance_ms:
        status = "FAIL"
    else:
        status = "PARTIAL"  # only onsets after pauses are measurable; never all words
    return {
        "basis": "energy-based speech onsets after pauses, three levels below each signal's speech peak; "
                 "not word boundaries, not phonemes",
        "pairs": pairs,
        "measured": len(measured),
        "uncertain": uncertain,
        "unpaired_converted_onsets": unpaired,
        "mean_abs_ms": round(float(np.mean(np.abs(measured))), 2) if measured else None,
        "max_abs_ms": round(float(np.max(np.abs(measured))), 2) if measured else None,
        "tolerance_ms": tolerance_ms,
        "status": status,
    }


# Backwards-compatible helper used by older reports: measured pairs only.
def onset_differences(source: np.ndarray, converted: np.ndarray, sr: int = 24000, window_s: float = 0.06) -> list[dict]:
    comp = onset_comparison(source, converted, sr, window_s=window_s)
    return [{"source_s": p["source_s"], "converted_s": p["converted_s"], "diff_ms": p["diff_ms"]}
            for p in comp["pairs"] if p["status"] == "MEASURED"]


def median_pitch_hz(a: np.ndarray, sr: int = 24000) -> Optional[float]:
    """Median fundamental frequency of the voiced frames (pYIN at 16 kHz), or None when the
    signal has no reliably voiced speech. Used to keep one speaker's clips at the same pitch."""
    import librosa

    if a.size < sr // 2:
        return None
    y = librosa.resample(np.asarray(a, dtype=np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else a
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=16000, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    return round(float(np.median(f0)), 2) if f0.size >= 10 else None


def semitones(a: float, b: float) -> float:
    return 12.0 * float(np.log2(a / b))
