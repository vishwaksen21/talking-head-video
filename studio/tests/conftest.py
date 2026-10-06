"""Shared fixtures. Synthetic media is made with FFmpeg so tests need no network."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(autouse=True)
def _no_real_env(monkeypatch):
    """Tests never read the developer's .env (API key, provider-call gate): every setting a
    test needs is given explicitly."""
    from studio.server import config

    monkeypatch.setattr(config, "_read_env_file", lambda path: {})
    monkeypatch.delenv("STUDIO_PROVIDER_CALLS", raising=False)
    monkeypatch.delenv("UTILS_API_KEY", raising=False)
    monkeypatch.delenv("UTILS_BASE_URL", raising=False)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def tone_bursts(path: Path, bursts: list[tuple[float, float]], duration: float, sr: int = 24000) -> None:
    """A WAV that is silent except for 220 Hz bursts at the given (start, end) times."""
    t = np.arange(int(duration * sr)) / sr
    wav = np.zeros_like(t, dtype=np.float32)
    for a, b in bursts:
        mask = (t >= a) & (t < b)
        wav[mask] = 0.3 * np.sin(2 * np.pi * 220 * t[mask])
    sf.write(path, wav, sr)


@pytest.fixture
def speech_like_video(tmp_path) -> Path:
    """4 s 16:9 video whose soundtrack has 'speech' at 0.5-1.5 s and 2.2-3.4 s."""
    audio = tmp_path / "bursts.wav"
    tone_bursts(audio, [(0.5, 1.5), (2.2, 3.4)], 4.0)
    out = tmp_path / "speech.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x180:rate=24:duration=4", "-i", str(audio),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out))
    return out


@pytest.fixture
def silent_video(tmp_path) -> Path:
    """4 s video with no audio stream at all."""
    out = tmp_path / "silent.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x180:rate=24:duration=4", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", str(out))
    return out


class FakeVoice:
    """Stands in for Chatterbox in isolated tests: a tone whose length scales with the text."""

    sample_rate = 24000
    loaded = True
    load_error = None
    device = "cpu"

    def __init__(self, seconds_per_char: float = 0.06, convert_delay: int = 0, convert_extra: int = 0, tones=(137,)):
        self.seconds_per_char = seconds_per_char
        self.tones = list(tones)      # conversion pitch, cycling per call (takes can differ in pitch)
        self.calls: list[tuple[str, object]] = []
        self.converted: list[tuple[object, object]] = []
        self.convert_delay, self.convert_extra = convert_delay, convert_extra

    @staticmethod
    def installed():
        return True, "ok"

    def convert(self, source, reference, **kwargs):
        """Stand-in for Chatterbox voice conversion: the same timeline in a 'different voice'
        (the source's loudness envelope on a different tone, so the waveform itself is new),
        optionally late by `convert_delay` samples and longer by `convert_extra`."""
        import librosa

        self.converted.append((source, reference))
        wav, _ = librosa.load(source, sr=24000, mono=True)
        env = np.sqrt(np.convolve(wav.astype(np.float64) ** 2, np.ones(48) / 48, mode="same"))
        t = np.arange(wav.size) / 24000
        tone = self.tones[(len(self.converted) - 1) % len(self.tones)]
        voiced = (env * 1.4 * np.sin(2 * np.pi * tone * t)).astype(np.float32)
        return np.concatenate([np.zeros(self.convert_delay, np.float32), voiced,
                               np.zeros(self.convert_extra, np.float32)])

    def synthesize(self, text, reference, **kwargs):
        self.calls.append((text, reference))
        n = int(len(text) * self.seconds_per_char * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
