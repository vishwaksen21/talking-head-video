"""Apple Vision face analysis through a small Swift helper (studio/tools/facekit.swift).

Vision is built into macOS, and the helper is compiled on first use with the Xcode Command
Line Tools already present. Nothing is downloaded. It reports, per image:
- face boxes with detection confidence
- Vision's face-capture quality
- head pose
- landmarks (lips, eyes, nose, contour)

What this can and cannot measure:
- Face boxes, pose and geometry are measurements of what is on screen.
- Geometry is NOT a face-recognition embedding. It can catch a grossly different face; it
  cannot tell a close lookalike from the same person. See reports/talking_head_root_cause.md.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

from .config import STUDIO

SOURCE = STUDIO / "tools" / "facekit.swift"
BINARY = STUDIO / "bin" / "facekit"
_lock = threading.Lock()


class FaceKitError(Exception):
    pass


def available() -> tuple[bool, str]:
    if BINARY.exists() and BINARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return True, "ready"
    if not shutil.which("swiftc"):
        return False, "swiftc not found (install the Xcode Command Line Tools: xcode-select --install)"
    return True, "will be compiled on first use"


def _ensure_binary() -> Path:
    with _lock:
        if BINARY.exists() and BINARY.stat().st_mtime >= SOURCE.stat().st_mtime:
            return BINARY
        swiftc = shutil.which("swiftc")
        if not swiftc:
            raise FaceKitError("swiftc not found; install the Xcode Command Line Tools (xcode-select --install).")
        BINARY.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run([swiftc, "-O", str(SOURCE), "-o", str(BINARY)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise FaceKitError(f"Could not build the face helper: {proc.stderr.strip()[-400:]}")
        return BINARY


def analyse(images: list[Path], *, batch: int = 64) -> list[dict]:
    """Run Vision on each image; returns one dict per image, in order."""
    binary = _ensure_binary()
    results: list[dict] = []
    for i in range(0, len(images), batch):
        chunk = [str(p) for p in images[i:i + batch]]
        proc = subprocess.run([str(binary), *chunk], capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise FaceKitError(f"Face analysis failed: {proc.stderr.strip()[-300:]}")
        lines = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
        if len(lines) != len(chunk):
            raise FaceKitError("Face analysis returned an unexpected number of results.")
        results.extend(lines)
    return results


def segment_person(image: Path, out: Path) -> float:
    """Person on a plain light-grey background (Vision person segmentation). Returns the
    fraction of the image kept as person. The input is not modified."""
    binary = _ensure_binary()
    proc = subprocess.run([str(binary), "--segment", str(image), str(out)], capture_output=True, text=True, timeout=120)
    result = json.loads(proc.stdout.strip() or "{}") if proc.stdout.strip() else {}
    if proc.returncode != 0 or "error" in result:
        raise FaceKitError(f"Person segmentation failed: {result.get('error') or proc.stderr.strip()[-200:]}")
    return float(result["coverage"])


def _centre(pts: list) -> tuple[float, float]:
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def interocular(face: dict) -> Optional[float]:
    if not face.get("left_eye") or not face.get("right_eye"):
        return None
    return _dist(_centre(face["left_eye"]), _centre(face["right_eye"])) or None


def geometry(face: dict) -> Optional[list[float]]:
    """Four expression-stable, scale-free proportions over the eye distance:
    eyes-to-nose-tip, nose width, jaw width and mean eye width.

    Mouth-dependent proportions (nose-to-mouth, mouth-to-chin, mouth width) are left out
    on purpose: speech moves them as much as a change of person does.
    """
    try:
        le, re_ = _centre(face["left_eye"]), _centre(face["right_eye"])
        nose, contour = face["nose"], face["face_contour"]
        if not (nose and contour and face["left_eye"] and face["right_eye"]):
            return None
    except KeyError:
        return None
    iod = _dist(le, re_)
    if not iod:
        return None
    eyes = ((le[0] + re_[0]) / 2, (le[1] + re_[1]) / 2)
    tip = max(nose, key=lambda p: p[1])
    span = lambda pts: max(p[0] for p in pts) - min(p[0] for p in pts)
    return [_dist(eyes, tip) / iod, span(nose) / iod, span(contour) / iod,
            (span(face["left_eye"]) + span(face["right_eye"])) / 2 / iod]


def geometry_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
