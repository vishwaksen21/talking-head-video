"""Final composition: the original pictures + the cloned voice, as a browser-safe MP4.

The original soundtrack is replaced, not mixed: only the video stream of the source and
the audio stream of the cloned track are mapped. When the plan holds the last frame, the
video is re-encoded with `tpad` (clone mode - the final frame, never black); otherwise
the video stream is copied bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .media import MediaError, mean_volume_db, probe, run_ffmpeg


END_CARD_FADE = 0.6


def compose(video: Path, voice: Path, out_path: Path, *, final_duration: float, freeze_seconds: float,
            end_card_logo: Optional[Path] = None, end_card_seconds: float = 2.0, log_path: Optional[Path] = None) -> dict:
    """Mux the cloned voice over the original pictures.

    With `end_card_logo`, the last `end_card_seconds` fade to a white card carrying the
    real logo file, centred at a third of the frame width. The model is never asked to
    draw a logo; the approved asset is composited here.
    """
    tmp = out_path.with_suffix(".partial.mp4")
    args = ["-i", str(video), "-i", str(voice)]
    filters = []
    if freeze_seconds > 0:
        filters.append(f"[0:v]tpad=stop_mode=clone:stop_duration={freeze_seconds:.3f}[base]")
    else:
        filters.append("[0:v]null[base]")
    if end_card_logo is not None:
        if not end_card_logo.exists():
            raise MediaError(f"The logo file for the end card is missing: {end_card_logo}")
        info = probe(video)
        w, h = info["width"], info["height"]
        start = max(0.0, final_duration - end_card_seconds)
        args += ["-loop", "1", "-t", f"{final_duration:.3f}", "-i", str(end_card_logo)]
        filters += [
            f"color=c=white:s={w}x{h}:d={final_duration:.3f},format=rgba,"
            f"fade=t=in:st={start:.3f}:d={END_CARD_FADE}:alpha=1[card]",
            f"[2:v]scale={int(w * 0.34)}:-1,format=rgba,fade=t=in:st={start + 0.25:.3f}:d={END_CARD_FADE}:alpha=1[logo]",
            "[base][card]overlay=shortest=1[carded]",
            "[carded][logo]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p[v]",
        ]
    else:
        filters.append("[base]format=yuv420p[v]")
    reencode = freeze_seconds > 0 or end_card_logo is not None
    if reencode:
        args += ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", "1:a:0",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    else:
        args += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
    # A mono voice is copied to both channels at full level. Plain "-ac 2" applies a -3 dB pan
    # law (measured: the final came out 3.04 dB below the voice track).
    voice_channels = probe(voice).get("channels") or 2
    if voice_channels == 1:
        args += ["-af", "pan=stereo|c0=c0|c1=c0"]
    args += [
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-t", f"{final_duration:.3f}", "-movflags", "+faststart", str(tmp),
    ]
    try:
        stderr = run_ffmpeg(args, what="composing the final video")
    except MediaError as exc:
        if log_path:
            log_path.write_text("ffmpeg " + " ".join(args) + "\n\nFAILED: " + str(exc) + "\n")
        raise
    if log_path:
        log_path.write_text("ffmpeg " + " ".join(args) + "\n\n" + stderr)
    info = verify(tmp, expected_duration=final_duration)
    tmp.replace(out_path)
    info["file"] = out_path.name
    return info


def verify(path: Path, *, expected_duration: float, tolerance: float = 0.15) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise MediaError("The final video file was not written.")
    info = probe(path)
    if not info["has_video"]:
        raise MediaError("The final video has no video stream.")
    if not info["has_audio"]:
        raise MediaError("The final video has no audio stream.")
    if abs(info["duration"] - expected_duration) > tolerance:
        raise MediaError(
            f"The final video is {info['duration']:.2f}s long, expected {expected_duration:.2f}s."
        )
    volume = mean_volume_db(path)
    if volume is None or volume < -60:
        raise MediaError("The final video's audio is silent.")
    info["mean_volume_db"] = volume
    return info


def join_scenes(clips: list[Path], out_path: Path, *, transition: float = 0.5) -> dict:
    """Join scene clips in order with a crossfade (video `xfade`, audio `acrossfade`).

    Every clip is conformed to the first clip's size and frame rate first, because xfade
    refuses mismatched inputs. A clip with no audio gets silence so the soundtrack stays
    continuous. Returns where each scene starts in the joined timeline.
    """
    infos = [probe(c) for c in clips]
    if len(clips) == 1:
        import shutil

        shutil.copy(clips[0], out_path)
        return {"scenes": [{"index": 0, "start": 0.0, "duration": infos[0]["duration"], "file": clips[0].name}],
                "transition": 0.0}
    w, h, fps = infos[0]["width"], infos[0]["height"], infos[0]["fps"] or 24
    for info in infos:
        if info["duration"] <= transition * 2:
            raise MediaError(f"{info['file']} is too short ({info['duration']}s) to crossfade.")
    args, filters = [], []
    for i, (clip, info) in enumerate(zip(clips, infos)):
        args += ["-i", str(clip)]
        filters.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                       f"fps={fps},format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS[v{i}]")
        if info["has_audio"]:
            filters.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[a{i}]")
        else:
            filters.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={info['duration']:.3f}[a{i}]")
    starts, offset, vprev, aprev = [0.0], 0.0, "v0", "a0"
    for i in range(1, len(clips)):
        offset += infos[i - 1]["duration"] - transition
        starts.append(round(offset, 3))
        filters.append(f"[{vprev}][v{i}]xfade=transition=fade:duration={transition}:offset={offset:.3f}[vx{i}]")
        filters.append(f"[{aprev}][a{i}]acrossfade=d={transition}[ax{i}]")
        vprev, aprev = f"vx{i}", f"ax{i}"
    run_ffmpeg([*args, "-filter_complex", ";".join(filters), "-map", f"[{vprev}]", "-map", f"[{aprev}]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out_path)],
               what="joining the scenes")
    return {"scenes": [{"index": i, "start": s, "duration": infos[i]["duration"], "file": clips[i].name}
                       for i, s in enumerate(starts)], "transition": transition}
