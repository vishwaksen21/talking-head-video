"""Evidence-based status table for a Strict Talking Head output.

Every requirement is measured from the files on disk: the generated clip, the selected voice
track, the final MP4 and the FFmpeg log. Each gets exactly one status:

  PASS            objectively verified
  FAIL            objectively disproven
  PARTIAL         some but not all of the requirement verified
  NOT TESTED      the test was not performed (e.g. no data for it)
  NOT GUARANTEED  the architecture cannot guarantee it
  UNCERTAIN       a measurement was made but is not reliable enough to decide

There is deliberately no "overall passed". `assess` only reads files; it never modifies the
project, so it can audit existing outputs without touching their final MP4.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from . import dialogue, timing
from .media import ffmpeg_bin, ffprobe_bin

PASS, FAIL, PARTIAL = "PASS", "FAIL", "PARTIAL"
NOT_TESTED, NOT_GUARANTEED, UNCERTAIN = "NOT TESTED", "NOT GUARANTEED", "UNCERTAIN"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(path: Path, sr: int, channels: int) -> np.ndarray:
    raw = subprocess.run([ffmpeg_bin(), "-loglevel", "error", "-i", str(path), "-vn", "-f", "f32le",
                          "-ac", str(channels), "-ar", str(sr), "-"], capture_output=True).stdout
    a = np.frombuffer(raw, np.float32)
    return a.reshape(-1, channels) if channels > 1 else a


def streams(path: Path) -> list[dict]:
    out = subprocess.run([ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                         capture_output=True, text=True).stdout
    data = json.loads(out or "{}")
    return data.get("streams", []), data.get("format", {})


def frame_hashes(path: Path, *, decoded: bool) -> list[str]:
    args = [ffmpeg_bin(), "-loglevel", "error", "-i", str(path), "-map", "0:v:0"]
    args += ([] if decoded else ["-c", "copy"]) + ["-f", "framemd5", "-"]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    return [l.split(",")[-1].strip() for l in out.splitlines() if l and not l.startswith("#")]


def _row(key, label, status, evidence, **data):
    return {"key": key, "label": label, "status": status, "evidence": evidence, **data}


def assess(folder: Path, strict: dict, *, asr=None, final: Optional[Path] = None, voice: Optional[Path] = None,
           compose_log: Optional[Path] = None, derived: Optional[Path] = None) -> dict:
    """Measure every requirement for the project in `folder`. Read-only.

    `final`, `voice`, `compose_log` and `derived` default to the project's own files. They
    can point elsewhere to grade an alternative output (e.g. a re-composition written next
    to the original final)."""
    method = strict.get("voice_method", "tts")
    video = folder / "generated.mp4"
    final = final or folder / "output" / "final_talking_head.mp4"
    voice = voice or folder / "cloned_voice.wav"
    compose_log = compose_log or folder / "reports" / "ffmpeg_compose.log"
    veo = folder / "audio" / "original_veo_audio.wav"
    reports = folder / "reports"
    rows, facts = [], {}
    if not final.exists() or not video.exists() or not voice.exists():
        missing = [p.name for p in (final, video, voice) if not p.exists()]
        return {"rows": [_row("files", "Required files", FAIL, f"missing: {', '.join(missing)}")], "facts": {}}

    import soundfile as sf

    v_streams, _ = streams(video)
    f_streams, f_format = streams(final)
    v_video = next((s for s in v_streams if s["codec_type"] == "video"), {})
    v_audio = next((s for s in v_streams if s["codec_type"] == "audio"), None)
    f_v = [s for s in f_streams if s["codec_type"] == "video"]
    f_a = [s for s in f_streams if s["codec_type"] == "audio"]
    video_duration = float(v_video.get("duration") or 0)
    voice_info = sf.info(voice)
    voice_wav, voice_sr = sf.read(voice, dtype="float32")
    facts.update({"video_duration_s": video_duration, "voice_track": {"samples": voice_info.frames, "sample_rate": voice_info.samplerate,
                  "channels": voice_info.channels, "sha256": sha256(voice)}})

    # 1. complete source audio passed to voice conversion
    conv_report = _json(reports / "voice_conversion_timing.json")
    if method == "convert":
        if not veo.exists() or v_audio is None:
            rows.append(_row("complete_source_to_vc", "Complete generated audio sent to voice conversion", FAIL,
                             "the generated clip has no audio track, or the extracted track is missing"))
        else:
            src_info = sf.info(veo)
            native = decode(video, int(v_audio["sample_rate"]), 1)
            recorded = (conv_report or {}).get("source", {})
            ok = (src_info.frames == native.size and recorded.get("samples") == src_info.frames
                  and not recorded.get("trimmed", True))
            facts["source_audio"] = {"extracted_samples": src_info.frames, "video_audio_samples": int(native.size),
                                     "sample_rate": src_info.samplerate, "channels": src_info.channels,
                                     "samples_recorded_as_sent": recorded.get("samples")}
            rows.append(_row("complete_source_to_vc", "Complete generated audio sent to voice conversion", PASS if ok else FAIL,
                             f"extracted {src_info.frames} samples = clip's audio {native.size} samples at "
                             f"{src_info.samplerate} Hz; conversion report records {recorded.get('samples')} sent, "
                             f"trimmed={recorded.get('trimmed')}"))
    else:
        rows.append(_row("complete_source_to_vc", "Complete generated audio sent to voice conversion", NOT_TESTED,
                         "text-to-speech method: the voice comes from the approved script, not from the video's audio"))

    # 2. the selected audio is what the final MP4 contains
    fin48 = decode(final, 48000, 2) if f_a else np.zeros((0, 2), np.float32)
    selected_ok, corr, gain_db = False, None, None
    if fin48.size:
        import librosa

        ref48 = librosa.resample(voice_wav if voice_wav.ndim == 1 else voice_wav.mean(1), orig_sr=voice_sr, target_sr=48000)
        left = fin48[:, 0]
        n = min(left.size, ref48.size)
        best = max(range(-480, 481, 4), key=lambda k: float(np.dot(left[max(0, k):n + min(0, k)], ref48[max(0, -k):n - max(0, k)])))
        x, y = left[max(0, best):n + min(0, best)], ref48[max(0, -best):n - max(0, best)]
        corr = float(np.corrcoef(x, y)[0, 1])
        gain_db = float(20 * np.log10(abs(np.dot(x, y) / max(np.dot(y, y), 1e-12)) + 1e-12))
        selected_ok = corr >= 0.99 and abs(best) <= 4 and abs(gain_db) <= 0.5
        facts["final_vs_voice_track"] = {"offset_samples_48k": best, "correlation": round(corr, 5), "gain_db": round(gain_db, 2)}
    derived = derived or folder / "audio" / ("converted_voice.wav" if method == "convert" else "final_voice.wav")
    same_file = derived.exists() and sha256(derived) == facts["voice_track"]["sha256"]
    rows.append(_row("selected_audio_in_final", "The selected voice track is the final audio",
                     PASS if selected_ok and same_file else FAIL,
                     f"final audio vs voice track: correlation {corr if corr is None else round(corr, 5)}, "
                     f"offset {facts.get('final_vs_voice_track', {}).get('offset_samples_48k')} samples, gain "
                     f"{gain_db if gain_db is None else round(gain_db, 2)} dB; voice track sha256 "
                     f"{'matches' if same_file else 'does NOT match'} {derived.name}. Transformations after "
                     f"Chatterbox: {', '.join(_transforms(method, conv_report))}"))

    # 3. original Veo audio excluded
    if veo.exists() and fin48.size:
        import librosa

        veo_wav, veo_sr = sf.read(veo, dtype="float32", always_2d=True)
        veo48 = librosa.resample(veo_wav.mean(1), orig_sr=veo_sr, target_sr=48000)
        left = fin48[:, 0]
        n = min(left.size, veo48.size)
        c = max(abs(float(np.corrcoef(left[max(0, k):n + min(0, k)], veo48[max(0, -k):n - max(0, k)])[0, 1]))
                for k in range(-960, 961, 48))
        log = compose_log.read_text() if compose_log.exists() else ""
        mapped_only_voice = "-map 1:a:0" in log and "-map 0:a" not in log and not re.search(r"amix|amerge", log)
        rows.append(_row("veo_audio_excluded", "Original Veo audio excluded from the final",
                         PASS if c < 0.2 and mapped_only_voice else FAIL,
                         f"max waveform correlation with Veo audio {c:.3f} (±20 ms); FFmpeg command "
                         f"{'maps only the voice track (-map 1:a:0), no mixing' if mapped_only_voice else 'not verified from the log'}"))
    elif not veo.exists():
        rows.append(_row("veo_audio_excluded", "Original Veo audio excluded from the final", PASS,
                         "the generated clip had no audio of its own"))

    # 4. exact duration (decoded sample counts, not container metadata)
    expected48 = int(round(video_duration * 48000))
    expected_voice = int(round(video_duration * voice_sr))
    dur_ok = fin48.shape[0] == expected48 and voice_info.frames == expected_voice
    facts["durations"] = {"final_audio_samples_48k": int(fin48.shape[0]), "expected_48k": expected48,
                          "voice_track_samples": voice_info.frames, "expected_voice": expected_voice,
                          "final_container_s": float(f_format.get("duration") or 0)}
    rows.append(_row("duration_exact", "Exact duration (sample counts)", PASS if dur_ok else FAIL,
                     f"voice track {voice_info.frames} samples at {voice_sr} Hz (expected {expected_voice}); final audio "
                     f"decodes to {fin48.shape[0]} samples at 48 kHz (expected {expected48} for {video_duration:.3f} s of video)"))

    # 5. global alignment and 6. speech onsets (voice vs the generated audio's timeline)
    if veo.exists():
        import librosa

        veo_wav, veo_sr = sf.read(veo, dtype="float32", always_2d=True)
        src24 = timing.match_length(librosa.resample(veo_wav.mean(1), orig_sr=veo_sr, target_sr=24000), expected_voice)
        v24 = voice_wav if voice_sr == 24000 else librosa.resample(voice_wav, orig_sr=voice_sr, target_sr=24000)
        lag, r = timing.global_lag_samples(src24, v24)
        comp = timing.onset_comparison(src24, v24)
        facts["global_offset_ms"] = round(lag / 24, 3)
        facts["onsets"] = comp
        if method == "convert":
            g_status = PASS if abs(lag) <= 24 and r >= 0.8 else (UNCERTAIN if r < 0.8 else FAIL)
            rows.append(_row("global_alignment", "Global audio alignment", g_status,
                             f"offset {lag / 24:+.2f} ms against the generated audio (speech-weighted envelope, "
                             f"r={r:.3f}); tolerance ±1 ms"))
        else:
            rows.append(_row("global_alignment", "Global audio alignment", NOT_TESTED,
                             "text-to-speech is placed phrase by phrase; there is no single global timeline to match"))
        ev = (f"{comp['measured']} onset(s) measured, {comp['uncertain']} uncertain; mean |Δ| {comp['mean_abs_ms']} ms, "
              f"max |Δ| {comp['max_abs_ms']} ms (tolerance ±{comp['tolerance_ms']} ms). {comp['basis']}.")
        rows.append(_row("speech_onsets", "Speech onset comparison (after pauses)", comp["status"], ev))
    else:
        rows.append(_row("global_alignment", "Global audio alignment", NOT_TESTED, "no generated audio to align against"))
        rows.append(_row("speech_onsets", "Speech onset comparison (after pauses)", "NOT MEASURABLE",
                         "no generated audio to compare against"))
    rows.append(_row("phoneme_timing", "Phoneme timing preservation", NOT_GUARANTEED,
                     "not measured; voice conversion regenerates speech, so internal word and phoneme timing can change "
                     "even when duration and global offset match"))

    # 7. stream mapping and audio metadata
    log = compose_log.read_text() if compose_log.exists() else ""
    a = f_a[0] if f_a else {}
    map_ok = (len(f_v) == 1 and len(f_a) == 1 and len(f_streams) == 2 and f_v[0].get("codec_name") == v_video.get("codec_name")
              and a.get("codec_name") == "aac" and int(a.get("sample_rate", 0)) == 48000 and int(a.get("channels", 0)) == 2
              and "-map 0:v:0" in log and "-map 1:a:0" in log)
    rows.append(_row("stream_mapping", "Final MP4 stream mapping", PASS if map_ok else FAIL,
                     f"{len(f_v)} video ({f_v[0].get('codec_name') if f_v else '-'}) + {len(f_a)} audio "
                     f"({a.get('codec_name')}, {a.get('sample_rate')} Hz, {a.get('channels')} ch); command maps "
                     f"{'0:v:0 and 1:a:0' if '-map 0:v:0' in log and '-map 1:a:0' in log else 'not verified'}; "
                     f"audio is resampled to 48 kHz stereo AAC by design"))

    # 8. video frames preserved
    src_p, fin_p = frame_hashes(video, decoded=False), frame_hashes(final, decoded=False)
    if src_p and src_p == fin_p:
        rows.append(_row("video_frames_preserved", "Video frames preserved", PASS,
                         f"all {len(src_p)} video packets byte-identical (framemd5 of the copied stream)"))
    else:
        src_d, fin_d = frame_hashes(video, decoded=True), frame_hashes(final, decoded=True)
        same = src_d and src_d == fin_d
        rows.append(_row("video_frames_preserved", "Video frames preserved", PASS if same else FAIL,
                         f"packets differ; decoded frames {'identical' if same else 'differ'} ({len(src_d)} vs {len(fin_d)})"))

    # 9. A/V compatibility, clipping, black frames
    f_dur = float(f_v[0].get("duration") or 0) if f_v else 0
    rows.append(_row("av_duration", "Video/audio duration compatibility",
                     PASS if abs(f_dur - video_duration) < 1e-3 and dur_ok else FAIL,
                     f"final video stream {f_dur:.3f} s vs source {video_duration:.3f} s; audio decodes to exactly the video length"))
    peak = float(np.abs(fin48).max()) if fin48.size else 0.0
    rows.append(_row("audio_not_clipped", "Audio not clipped", PASS if 0 < peak < 0.999 else FAIL,
                     f"decoded peak {peak:.3f} (full scale 1.0)"))
    black = subprocess.run([ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(final), "-vf", "blackdetect=d=0.1:pix_th=0.10",
                            "-an", "-f", "null", "-"], capture_output=True, text=True).stderr.count("black_start")
    rows.append(_row("no_black_frames", "No black frames", PASS if black == 0 else FAIL, f"{black} black segment(s)"))

    # 10. speech content
    veo_check = _json(reports / "veo_speech_check.json")
    if veo_check and veo_check.get("passed") is not None:
        rows.append(_row("generated_speech_matches_script", "Generated person says the approved script",
                         PASS if veo_check["passed"] else FAIL,
                         f"Whisper on the generated audio: “{veo_check.get('transcript', '')}” "
                         f"(word accuracy {veo_check.get('word_accuracy')})"))
    else:
        rows.append(_row("generated_speech_matches_script", "Generated person says the approved script", NOT_TESTED,
                         "the generated clip has no audio to transcribe"))
    if asr is not None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "final16k.wav"
            subprocess.run([ffmpeg_bin(), "-loglevel", "error", "-y", "-i", str(final), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
            heard = asr.transcribe(wav)
        match = dialogue.compare(strict.get("script", ""), heard["text"])
    else:
        match = (_json(reports / "final_validation.json") or {}).get("details", {}).get("final_transcript")
    if match:
        rows.append(_row("final_speech_matches_script", "Final audio says the approved script",
                         PASS if match.get("passed") else FAIL,
                         f"Whisper on the final MP4: “{match.get('transcript', '')}” (word accuracy {match.get('word_accuracy')}); "
                         "speech recognition can mishear names"))
    else:
        rows.append(_row("final_speech_matches_script", "Final audio says the approved script", NOT_TESTED, "no transcript"))

    # 11. lip sync and identity
    lip = _json(reports / "lipsync_validation.json")
    if not lip:
        rows.append(_row("visual_lip_sync", "Visual lip-sync", NOT_TESTED, "no lip-sync measurement"))
    elif lip.get("verifiable") is False or lip.get("native_correlation") is None:
        rows.append(_row("visual_lip_sync", "Visual lip-sync", UNCERTAIN,
                         "no reference audio: timing graded against the same mouth movement it was aligned to"))
    else:
        rows.append(_row("visual_lip_sync", "Visual lip-sync", PARTIAL if lip.get("passed") else FAIL,
                         f"mouth-movement timing vs voice: offset {lip.get('offset_ms')} ms, r={lip.get('correlation')} "
                         f"(Veo's own audio r={lip.get('native_correlation')}). Timing only: mouth shapes are the "
                         "video model's, not re-animated to the voice"))
    ident = _json(reports / "identity_validation.json")
    geo = ""
    if ident:
        geo = f"face geometry {ident.get('status')} (median {ident.get('median_distance')}); "
    rows.append(_row("identity", "Exact identity preservation",
                     FAIL if ident and ident.get("passed") is False else NOT_GUARANTEED,
                     geo + "the video model generates a new face guided by the photo; it does not animate the photo. "
                     "Similarity is not identity"))
    rows.append(_row("photo_animated", "Uploaded photo animated exactly", NOT_GUARANTEED,
                     "the gateway has no image-to-video input; the photo is only a reference image"))
    return {"rows": rows, "facts": facts}


def blocking_failures(result: dict) -> list[str]:
    """Any FAIL blocks acceptance. NOT GUARANTEED and UNCERTAIN do not block, but are shown
    to the reviewer."""
    return [r["key"] for r in result["rows"] if r["status"] == FAIL]


def _transforms(method: str, conv: Optional[dict]) -> list[str]:
    steps = []
    if method == "convert" and conv:
        steps.append(f"length matched to {conv.get('final_length_samples_24k')} samples "
                     f"(Chatterbox returned {conv.get('chatterbox_output_length_diff_ms')} ms different)")
        steps.append(f"whole-track shift {conv.get('global_offset_before_ms')} ms"
                     + (f" + onset shift {conv.get('word_onset_shift_ms')} ms" if conv.get("word_onset_shift_ms") else ""))
        steps.append(f"constant gain {conv.get('constant_gain_db')} dB")
    elif method == "tts":
        steps.append("phrases placed and time-fitted (≤15%), speech level normalised")
    steps.append("FFmpeg: mono copied to both channels at full level, resampled to 48 kHz, AAC 192 kb/s")
    return steps


def _json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
