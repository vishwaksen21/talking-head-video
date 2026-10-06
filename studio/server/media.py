"""FFmpeg/ffprobe helpers: probing, audio extraction, speech-region detection."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class MediaError(Exception):
    pass


def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise MediaError("FFmpeg is not installed or not on PATH (brew install ffmpeg).")
    return path


def ffprobe_bin() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise MediaError("ffprobe is not installed or not on PATH (it ships with FFmpeg).")
    return path


def run_ffmpeg(args: list[str], *, what: str) -> str:
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
        raise MediaError(f"FFmpeg failed while {what}: {tail}")
    return proc.stderr


def probe(path: Path) -> dict:
    """Duration and stream facts for a media file."""
    proc = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise MediaError(f"Could not read {path.name}: {proc.stderr.strip()[:300]}")
    data = json.loads(proc.stdout)
    fmt = data.get("format", {})
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    info = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "format": fmt.get("format_name"),
        "duration": round(float(fmt.get("duration") or 0), 3),
        "has_video": video is not None,
        "has_audio": audio is not None,
    }
    if video:
        rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
        num, _, den = rate.partition("/")
        info.update({
            "video_codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": round(float(num) / float(den), 3) if den and float(den) else None,
        })
    if audio:
        info.update({
            "audio_codec": audio.get("codec_name"),
            "sample_rate": int(audio.get("sample_rate") or 0),
            "channels": audio.get("channels"),
        })
    return info


def mean_volume_db(path: Path) -> Optional[float]:
    err = run_ffmpeg(["-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"], what="measuring loudness")
    m = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", err)
    if not m or m.group(1) == "-inf":
        return None
    return float(m.group(1))


def extract_audio(video: Path, out_dir: Path) -> dict:
    """Extract the soundtrack twice:

    * `original_audio.wav` - PCM at the source rate and channel count (the decode of the
      AAC track, no further loss), kept for preview and timing analysis.
    * `speech_16k.wav`     - 16 kHz mono PCM for transcription (small enough to send
      inline; 16 kHz is what speech recognisers use).
    """
    original = out_dir / "original_audio.wav"
    speech = out_dir / "speech_16k.wav"
    run_ffmpeg(["-i", str(video), "-vn", "-map", "0:a:0", "-c:a", "pcm_s16le", str(original)], what="extracting audio")
    run_ffmpeg(["-i", str(original), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(speech)],
               what="converting audio for transcription")
    return {"original": original, "speech": speech}


def speech_regions(path: Path, *, frame_s: float = 0.02, merge_gap_s: float = 0.35,
                   min_region_s: float = 0.20) -> list[tuple[float, float]]:
    """Where sound is: [(start, end), ...] in seconds.

    The threshold adapts to the clip: 15 dB above its noise floor (10th percentile of
    20 ms frame levels), kept between -45 and -25 dBFS. Generated videos often carry a
    room-tone bed, so a fixed threshold either hears speech everywhere or nowhere.
    Gaps shorter than `merge_gap_s` are treated as part of one utterance.
    """
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = wav.mean(axis=1)
    frame = max(1, int(sr * frame_s))
    n = mono.size // frame
    if n == 0:
        return []
    rms = np.sqrt((mono[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-9)
    threshold = float(np.clip(np.percentile(db, 10) + 15.0, -45.0, -25.0))
    loud = db > threshold
    regions: list[list[float]] = []
    for i, on in enumerate(loud):
        if not on:
            continue
        t0, t1 = i * frame_s, (i + 1) * frame_s
        if regions and t0 - regions[-1][1] <= merge_gap_s:
            regions[-1][1] = t1
        else:
            regions.append([t0, t1])
    return [(round(a, 3), round(b, 3)) for a, b in regions if b - a >= min_region_s]
