"""Server-side configuration.

Defaults come from the project's `.env`. The base URL and API key entered in the UI are
saved to `studio/data/settings.json` (mode 600) and override `.env`. The key is never
returned to the browser: `public_view()` reports only whether one is set.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]  # chatterbox-video/
STUDIO = ROOT / "studio"
DATA_DIR = Path(os.environ.get("STUDIO_DATA_DIR", STUDIO / "data"))
ENV_FILE = ROOT / ".env"


def _read_env_file(path: Path) -> dict:
    values: dict = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def normalize_base_url(url: str) -> str:
    """Accept the docs page URL people copy from the browser and return the API root.

    `https://host/docs#/` -> `https://host`. Only the scheme and host are meaningful for
    this gateway, but a deliberate path prefix (e.g. `/api`) is kept.
    """
    url = (url or "").strip()
    if not url:
        return ""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("Base URL must start with http:// or https://")
    path = parts.path.rstrip("/")
    for suffix in ("/docs", "/redoc", "/openapi.json"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


@dataclass
class Settings:
    base_url: str = ""
    api_key: str = ""
    video_location: str = "us-central1"
    transcribe_model: str = "gemini-2.5-flash"
    transcribe_location: str = "global"
    request_timeout: float = 120.0
    poll_interval: float = 5.0
    max_generation_seconds: float = 900.0
    # Chatterbox device: auto picks cuda > mps > cpu, like the repo's app.py.
    device: str = "auto"
    # Hard gate for every Utilities API call (model listing, generation, polling, Gemini).
    # OFF by default: set STUDIO_PROVIDER_CALLS=on in .env to allow the explicit generation
    # workflow. Request previews never call a provider either way.
    provider_calls: bool = False
    extra: dict = field(default_factory=dict)

    def public_view(self) -> dict:
        return {
            "base_url": self.base_url,
            "api_key_set": bool(self.api_key),
            "video_location": self.video_location,
            "transcribe_model": self.transcribe_model,
            "provider_calls": self.provider_calls,
        }


class SettingsStore:
    def __init__(self, data_dir: Path = DATA_DIR, env_file: Path = ENV_FILE):
        self._path = data_dir / "settings.json"
        self._env_file = env_file
        self._lock = threading.Lock()
        data_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Settings:
        env = {**_read_env_file(self._env_file), **{k: v for k, v in os.environ.items() if k.startswith(("UTILS_", "STUDIO_"))}}
        s = Settings(
            base_url=normalize_base_url(env.get("UTILS_BASE_URL", "")) if env.get("UTILS_BASE_URL") else "",
            api_key=env.get("UTILS_API_KEY", ""),
            video_location=env.get("UTILS_VIDEO_LOCATION", "us-central1"),
            transcribe_model=env.get("STUDIO_TRANSCRIBE_MODEL", "gemini-2.5-flash"),
            device=env.get("STUDIO_DEVICE", "auto"),
            provider_calls=env.get("STUDIO_PROVIDER_CALLS", "off").strip().lower() in ("on", "1", "true", "yes"),
        )
        if self._path.exists():
            saved = json.loads(self._path.read_text())
            s.base_url = saved.get("base_url") or s.base_url
            s.api_key = saved.get("api_key") or s.api_key
        return s

    def save(self, *, base_url: str | None, api_key: str | None) -> Settings:
        with self._lock:
            saved = json.loads(self._path.read_text()) if self._path.exists() else {}
            if base_url is not None:
                saved["base_url"] = normalize_base_url(base_url)
            if api_key:  # an empty field means "keep the current key"
                saved["api_key"] = api_key.strip()
            tmp = self._path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(saved, f)
            os.replace(tmp, self._path)
            os.chmod(self._path, 0o600)
        return self.load()
