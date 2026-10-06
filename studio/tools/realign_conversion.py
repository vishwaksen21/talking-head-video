"""Re-compose a voice-conversion project with the corrected speech-only alignment.

    .venv/bin/python -m studio.tools.realign_conversion <project_id>

Local only: no network, no generation, no Chatterbox run.
- **Input:** the take already selected (`audio/converted_voice.wav`).
- **Measurement:** its offset against the generated audio, measured ONCE with
  `timing.global_lag_samples` (speech-weighted).
- **Correction:** one whole-sample shift, applied once. Gain and content unchanged.
- **Output:** a NEW final MP4 next to the original, built from the copied video stream.
- **Untouched:** the original final (also backed up), the video, the source audio and the
  selected take are never modified.
- **Report:** a fresh read-only validation goes to `reports/realign_<stamp>.json`.
"""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from studio.server import results, timing
from studio.server.compose import compose
from studio.server.config import DATA_DIR
from studio.server.store import ProjectStore

SR = 24000


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(pid: str) -> int:
    store = ProjectStore(DATA_DIR)
    project = store.get(pid)
    folder = store.project_dir(pid)
    strict = project["params"]["strict"]
    if strict.get("voice_method") != "convert":
        print("not a voice-conversion project"); return 2
    video, source = folder / "generated.mp4", folder / "audio" / "original_veo_audio.wav"
    selected, final = folder / "audio" / "converted_voice.wav", folder / "output" / "final_talking_head.mp4"
    for p in (video, source, selected, final):
        if not p.exists():
            print(f"missing required artifact: {p.relative_to(folder)}"); return 2
    protected = {str(p.relative_to(folder)): sha(p) for p in (video, folder / "video" / "scene_01.mp4", source, selected, final)}
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # 1. backup of the original final (a copy; the original stays in place, untouched)
    backup = folder / "output" / "backup" / f"final_talking_head__before_realign_{stamp}.mp4"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final, backup)
    assert sha(backup) == protected["output/final_talking_head.mp4"]

    # 2. measure once
    src, src_sr = sf.read(source, dtype="float32", always_2d=True)
    n = int(round(src.shape[0] / src_sr * SR))
    src24 = timing.match_length(librosa.resample(src.mean(1), orig_sr=src_sr, target_sr=SR), n)
    conv, conv_sr = sf.read(selected, dtype="float32")
    info = sf.info(selected)
    if conv_sr != SR or conv.ndim != 1 or conv.size != n:
        print(f"selected take is not {n} mono samples at {SR} Hz"); return 2
    lag, r = timing.global_lag_samples(src24, conv)
    measured = {"lag_samples": lag, "offset_ms": round(lag / SR * 1000, 3), "correlation": round(r, 4),
                "basis": "speech-weighted log-loudness envelopes, 0.25 ms resolution; positive = voice late"}
    print("measured before correction:", measured)
    if r < 0.8:
        print("UNCERTAIN: alignment correlation too low to correct reliably; nothing composed")
        return 3

    # 3. one whole-sample shift; check what leaves the window is silence
    dropped = conv[-abs(lag):] if lag < 0 else conv[:lag]
    dropped_peak = float(np.abs(dropped).max()) if dropped.size else 0.0
    corrected = timing.shift(conv, lag)
    assert corrected.size == n
    out_voice = folder / "audio" / f"converted_voice_realigned_{stamp}.wav"
    sf.write(out_voice, corrected, SR, subtype=info.subtype)

    # 4. compose a NEW final from the copied video stream and the corrected voice
    out_final = folder / "output" / f"final_talking_head_realigned_{stamp}.mp4"
    log = folder / "reports" / f"ffmpeg_compose_realigned_{stamp}.log"
    duration = float(results.streams(video)[0][0]["duration"])
    compose(video, out_voice, out_final, final_duration=duration, freeze_seconds=0, log_path=log)

    # 5. fresh read-only validation of the new output
    assessed = results.assess(folder, strict, final=out_final, voice=out_voice, compose_log=log, derived=out_voice)
    lag_after, r_after = timing.global_lag_samples(src24, corrected)
    fin = results.decode(out_final, SR, 2)[:, 0]
    lag_final, r_final = timing.global_lag_samples(src24, timing.match_length(fin, n))
    after = {k: sha(folder / k) for k in protected}
    report = {
        "at": time.time(), "project": pid, "network_calls": 0, "chatterbox_runs": 0, "generations": 0,
        "original_final": {"file": "output/final_talking_head.mp4", "sha256": protected["output/final_talking_head.mp4"]},
        "backup": {"file": str(backup.relative_to(folder)), "sha256": sha(backup)},
        "selected_take": {"file": "audio/converted_voice.wav", "sha256": protected["audio/converted_voice.wav"],
                          "chosen_take": json.loads((folder / "reports" / "voice_conversion_timing.json").read_text()).get("chosen_take")},
        "correction": {"measured_before": measured, "applied_shift_samples": lag, "applied_shift_ms": round(lag / SR * 1000, 3),
                       "applied_times": 1, "dropped_samples": int(dropped.size), "dropped_peak": round(dropped_peak, 6),
                       "gain_changed": False},
        "corrected_voice": {"file": str(out_voice.relative_to(folder)), "sha256": sha(out_voice), "samples": int(corrected.size)},
        "new_final": {"file": str(out_final.relative_to(folder)), "sha256": sha(out_final), "compose_log": str(log.relative_to(folder))},
        "independent_offsets": {
            "corrected_voice_vs_source_ms": round(lag_after / SR * 1000, 3), "corrected_voice_r": round(r_after, 4),
            "new_final_audio_vs_source_ms": round(lag_final / SR * 1000, 3), "new_final_r": round(r_final, 4)},
        "protected_files_unchanged": after == protected,
        "validation": assessed,
    }
    path = folder / "reports" / f"realign_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    params = store.get(pid)["params"]
    params["strict"]["realignment"] = {k: report[k] for k in ("at", "original_final", "backup", "correction", "corrected_voice",
                                                              "new_final", "independent_offsets", "protected_files_unchanged")}
    params["strict"]["realignment"]["report"] = str(path.relative_to(folder))
    params["strict"]["realignment"]["rows"] = assessed["rows"]
    store.update(pid, params=params)
    print(json.dumps({k: report[k] for k in ("original_final", "backup", "correction", "corrected_voice", "new_final",
                                              "independent_offsets", "protected_files_unchanged")}, indent=1))
    for r in assessed["rows"]:
        print(f"  {r['status']:15s} {r['label']} | {r['evidence'][:150]}")
    print("report:", path.relative_to(folder))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
