"""Reference-versus-output face check (advisory).

No face-recognition model is installed, by the project owner's choice. This uses Apple
Vision's landmarks: four expression-stable proportions (`facekit.geometry`), compared between
the reference and frames sampled across the video.

**What it can and cannot tell.** Calibration on real Veo clips (`reports/talking_head_root_cause.md`):

| Compared with the reference | Mean geometry distance |
|---|---|
| The same person (reference frame vs. the same video) | 0.062 (max 0.101) |
| Different people | 0.104 – 0.199 |
| Veo's lookalike generated from the reference | 0.064 |

The last row is the important limit.
- **A pass means "not grossly different"; it is not proof of identity, and never proof of consent.**
- **A fail is a strong signal that the face changed.**

The result therefore always requires a person's confirmation before the project can be
marked complete. Thresholds are configurable (`IdentityRules`).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from . import facekit
from .media import probe, run_ffmpeg


@dataclass
class IdentityRules:
    samples: int = 8
    max_median_distance: float = 0.09     # calibration: same person 0.062, different people 0.104-0.199
    max_frame_to_frame: float = 0.15      # same person, frames 1 s apart: max 0.107
    min_confidence: float = 0.80
    min_single_face_fraction: float = 1.0


def validate(video: Path, reference_geometry: list[float], work: Path, rules: IdentityRules = IdentityRules()) -> dict:
    info = probe(video)
    work.mkdir(parents=True, exist_ok=True)
    times = [info["duration"] * (i + 0.5) / rules.samples for i in range(rules.samples)]
    frames = []
    for i, t in enumerate(times):
        out = work / f"frame_{i + 1:02d}.png"
        run_ffmpeg(["-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(out)], what="sampling frames")
        frames.append(out)
    results = facekit.analyse(frames)
    rows, geos = [], []
    for t, path, r in zip(times, frames, results):
        faces = r.get("faces") or []
        row = {"time": round(t, 2), "frame": path.name, "faces": len(faces)}
        if len(faces) == 1:
            f = faces[0]
            g = facekit.geometry(f)
            row.update({"confidence": round(f["confidence"], 3), "quality": round(f.get("quality", -1), 3),
                        "face_px": round(min(f["bbox"][2], f["bbox"][3])),
                        "distance": round(facekit.geometry_distance(reference_geometry, g), 3) if g else None})
            if g and f["confidence"] >= rules.min_confidence:
                geos.append(g)
        rows.append(row)
    single = sum(1 for r in rows if r["faces"] == 1) / len(rows)
    dists = [r["distance"] for r in rows if r.get("distance") is not None]
    consistency = [facekit.geometry_distance(a, b) for a, b in zip(geos, geos[1:])]
    median = statistics.median(dists) if dists else None
    reasons = []
    if single < rules.min_single_face_fraction:
        reasons.append(f"exactly one face in only {single:.0%} of sampled frames")
    if median is None:
        reasons.append("no usable face could be measured in the video")
    elif median > rules.max_median_distance:
        reasons.append(f"face proportions differ from the reference (median {median:.2f} > {rules.max_median_distance})")
    if consistency and max(consistency) > rules.max_frame_to_frame:
        reasons.append(f"the face changes between frames (max {max(consistency):.2f} > {rules.max_frame_to_frame})")
    return {
        "frames": rows,
        "single_face_fraction": round(single, 3),
        "median_distance": round(median, 3) if median is not None else None,
        "max_frame_to_frame": round(max(consistency), 3) if consistency else None,
        "status": "plausible_match" if not reasons else "mismatch",
        "passed": not reasons,
        "reasons": reasons,
        "requires_human_confirmation": True,
        "method": "Apple Vision landmark geometry (4 expression-stable proportions); not face recognition",
        "rules": rules.__dict__,
    }
