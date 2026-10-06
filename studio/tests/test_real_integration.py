"""Real integration: live gateway transcription + the installed Chatterbox + FFmpeg.

Opt-in, because it calls the real API (a few thousand tokens, no video generation) and
loads the real models:

    STUDIO_REAL=1 STUDIO_FIXTURE_VIDEO=/path/to/veo_clip.mp4 .venv/bin/python -m pytest studio/tests/test_real_integration.py -s

STUDIO_FIXTURE_VIDEO should be an existing video with speech (e.g. an earlier Veo render).
It is copied, never modified. The API credentials come from the project's .env.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("STUDIO_REAL") != "1", reason="set STUDIO_REAL=1 to run")

ROOT = Path(__file__).resolve().parents[2]


def _embedding(path: Path) -> np.ndarray:
    """CAMPPlus speaker embedding, the same scorer the repo's app.py uses."""
    import librosa
    from chatterbox.vc import ChatterboxVC

    global _VC
    try:
        _VC
    except NameError:
        from studio.server.voice import pick_device

        _VC = ChatterboxVC.from_pretrained(pick_device())
    wav, _ = librosa.load(path, sr=24000)
    ref = _VC.s3gen.embed_ref(wav[:240000], 24000, device=_VC.device)
    return ref["embedding"].squeeze().detach().cpu().numpy()


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_real_pipeline_on_existing_video(tmp_path):
    from studio.server.config import SettingsStore
    from studio.server.media import probe
    from studio.server.pipeline import Pipeline
    from studio.server.store import ProjectStore, State
    from studio.server.voice import ChatterboxVoice

    fixture = Path(os.environ["STUDIO_FIXTURE_VIDEO"])
    reference = ROOT / "studio" / "data" / "voices" / "reference.wav"
    assert reference.exists(), "copy a reference voice to studio/data/voices/reference.wav"

    data = tmp_path / "data"
    (data / "voices").mkdir(parents=True)
    shutil.copy(reference, data / "voices" / "reference.wav")
    store = ProjectStore(data)
    pipeline = Pipeline(store, SettingsStore(data), ChatterboxVoice())

    project = store.create(prompt="(existing video, no generation)", model="fixture", params={"video_config": {}})
    pid = project["id"]
    shutil.copy(fixture, store.project_dir(pid) / "generated.mp4")
    store.merge_media(pid, "generated", probe(store.project_dir(pid) / "generated.mp4"))
    store.update(pid, state=State.FAILED, failed_stage=State.EXTRACTING_AUDIO)
    pipeline.retry(pid)  # runs extraction + live transcription

    def wait(states, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            p = store.get(pid)
            if p["state"] in states:
                return p
            time.sleep(0.5)
        raise AssertionError("timeout")

    p = wait({State.AWAITING_TRANSCRIPT_REVIEW, State.FAILED})
    assert p["state"] == State.AWAITING_TRANSCRIPT_REVIEW, p["error"]
    t = p["transcript"]
    print("\nTRANSCRIPT", [(s["start"], s["text"], s["confidence"]) for s in t["original"]], t["timing"])
    assert t["has_speech"] and t["timing"] == "measured"

    pipeline.submit_transcript(pid, t["original"], {"voice": "reference.wav"})
    p = wait({State.COMPLETED, State.FAILED})
    assert p["state"] == State.COMPLETED, p["error"]
    folder = store.project_dir(pid)
    final = p["media"]["final"]
    print("SYNC", p["sync"]["status"], p["sync"]["warnings"], "FINAL", final["duration"], final["mean_volume_db"])
    assert final["has_audio"] and abs(final["duration"] - p["sync"]["final_duration"]) < 0.15

    # The new soundtrack is the reference speaker, not the video's original speaker.
    cloned = _embedding(folder / "cloned_voice.wav")
    to_ref = _cos(cloned, _embedding(data / "voices" / "reference.wav"))
    to_orig = _cos(cloned, _embedding(folder / "original_audio.wav"))
    print(f"SIMILARITY cloned~reference={to_ref:.3f} cloned~original={to_orig:.3f}")
    assert to_ref > to_orig
