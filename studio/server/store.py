"""Project history: one SQLite row per project, files under data/projects/<id>/."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional


class State:
    IDLE = "IDLE"
    GENERATING_VIDEO = "GENERATING_VIDEO"
    EXTRACTING_AUDIO = "EXTRACTING_AUDIO"
    TRANSCRIBING = "TRANSCRIBING"
    AWAITING_TRANSCRIPT_REVIEW = "AWAITING_TRANSCRIPT_REVIEW"
    SYNTHESIZING_VOICE = "SYNTHESIZING_VOICE"
    SYNCHRONIZING = "SYNCHRONIZING"
    COMPOSING = "COMPOSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    # Strict Talking Head mode
    VALIDATING_INPUT = "VALIDATING_INPUT"
    PREPARING_REFERENCE = "PREPARING_REFERENCE"
    GENERATING_SPEECH = "GENERATING_SPEECH"
    VALIDATING_DIALOGUE = "VALIDATING_DIALOGUE"
    AWAITING_AUDIO_APPROVAL = "AWAITING_AUDIO_APPROVAL"
    VALIDATING_LIPSYNC = "VALIDATING_LIPSYNC"
    VALIDATING_IDENTITY = "VALIDATING_IDENTITY"
    VALIDATING_OUTPUT = "VALIDATING_OUTPUT"
    NEEDS_REVIEW = "NEEDS_REVIEW"   # output exists, but checks failed or a person must confirm identity
    LIKENESS_ACCEPTED = "LIKENESS_ACCEPTED"  # a person accepted the output as a likeness; NOT the exact photo animated

    ACTIVE = {GENERATING_VIDEO, EXTRACTING_AUDIO, TRANSCRIBING, SYNTHESIZING_VOICE, SYNCHRONIZING, COMPOSING,
              VALIDATING_INPUT, PREPARING_REFERENCE, GENERATING_SPEECH, VALIDATING_DIALOGUE, VALIDATING_LIPSYNC,
              VALIDATING_IDENTITY, VALIDATING_OUTPUT}
    FINAL = {COMPLETED, FAILED, CANCELLED, LIKENESS_ACCEPTED}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    state TEXT NOT NULL,
    prompt TEXT NOT NULL,
    model TEXT NOT NULL,
    params TEXT NOT NULL,          -- JSON: the config sent to the API + sync options
    operation TEXT,                -- gateway operation name while generating
    message TEXT,                  -- current step, in plain words
    error TEXT,
    failed_stage TEXT,
    warnings TEXT NOT NULL DEFAULT '[]',
    media TEXT NOT NULL DEFAULT '{}',        -- JSON: probe results per asset
    transcript TEXT,               -- JSON: {original: [...], edited: [...], ...}
    sync TEXT,                     -- JSON: the synchronisation plan + report
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
"""

_JSON_FIELDS = ("params", "warnings", "media", "transcript", "sync")


class ProjectStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.projects_dir = data_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = data_dir / "studio.db"
        self._lock = threading.RLock()
        with self._connect() as db:
            db.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._db_path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def project_dir(self, project_id: str) -> Path:
        path = self.projects_dir / project_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create(self, *, prompt: str, model: str, params: dict, project_id: Optional[str] = None) -> dict:
        """`project_id` may be a chosen name (lowercase letters, digits, dashes). It must be new:
        an existing project or folder is never reused or overwritten."""
        now = time.time()
        if project_id:
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", project_id):
                raise ValueError("Project name: 3-64 lowercase letters, digits or dashes.")
            if self.get(project_id) or (self.projects_dir / project_id).exists():
                raise ValueError(f"A project named '{project_id}' already exists. Choose another name.")
        else:
            project_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO projects (id, created_at, updated_at, state, prompt, model, params, message)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (project_id, now, now, State.IDLE, prompt, model, json.dumps(params), "Queued"),
            )
        self.project_dir(project_id)
        return self.get(project_id)

    def get(self, project_id: str) -> Optional[dict]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return self._decode(row) if row else None

    def list(self, limit: int = 100) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM projects ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._decode(r) for r in rows]

    def update(self, project_id: str, **fields) -> dict:
        if not fields:
            return self.get(project_id)
        fields["updated_at"] = time.time()
        for key in _JSON_FIELDS:
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key])
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock, self._connect() as db:
            db.execute(f"UPDATE projects SET {cols} WHERE id = ?", (*fields.values(), project_id))
        return self.get(project_id)

    def add_warning(self, project_id: str, text: str) -> None:
        with self._lock:
            project = self.get(project_id)
            warnings = [w for w in project["warnings"] if w != text] + [text]
            self.update(project_id, warnings=warnings)

    def merge_media(self, project_id: str, key: str, info: Optional[dict]) -> None:
        with self._lock:
            media = self.get(project_id)["media"]
            if info is None:
                media.pop(key, None)
            else:
                media[key] = info
            self.update(project_id, media=media)

    def delete(self, project_id: str) -> None:
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        out = dict(row)
        for key in _JSON_FIELDS:
            if out.get(key):
                out[key] = json.loads(out[key])
            elif key in ("warnings",):
                out[key] = []
            elif key in ("media",):
                out[key] = {}
        out["cancel_requested"] = bool(out["cancel_requested"])
        return out
