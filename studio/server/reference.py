"""Reference image preparation for Strict Talking Head mode.

The uploaded photo is an immutable input. It is copied byte for byte to
`reference/<name>` and never altered. A *derived* face-centred crop is what goes to the
video model, so that the face, not a branded backdrop or the clothing, dominates the
reference. Both files are hashed into `reference/metadata.json`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import facekit


@dataclass
class ReferenceRules:
    min_confidence: float = 0.80      # Vision face-detection confidence
    min_face_px: int = 128            # smallest face box side, in source pixels
    min_quality: float = 0.30         # Vision face-capture quality, 0..1
    max_yaw_deg: float = 25.0         # the subject should face the camera
    crop_scale: float = 2.2           # crop side as a multiple of the face box (head and shoulders)
    plain_background: bool = True     # replace the backdrop with plain grey in the derived reference
    min_face_pixels_identical: float = 0.98  # share of face-box pixels that must be unchanged by segmentation


class ReferenceError(Exception):
    """The photo cannot be used; the message says why in plain words."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def prepare(original: Path, out_dir: Path, rules: ReferenceRules = ReferenceRules(), mode: str = "full") -> dict:
    """Validate the photo's face, then decide what goes to the video model.

    mode "full" (default): the uploaded file itself, byte for byte, with its own background,
    framing and resolution. Nothing is cropped, resized, re-encoded or segmented.
    mode "face_crop": a derived head-and-shoulders crop on a plain grey background.
    """
    from PIL import Image

    try:
        with Image.open(original) as im:
            im.verify()
        with Image.open(original) as im:
            width, height, fmt = im.width, im.height, (im.format or "").upper()
    except Exception as exc:
        raise ReferenceError("The reference file is not a readable image.") from exc
    if fmt not in ("PNG", "JPEG", "WEBP"):
        raise ReferenceError(f"Unsupported image format {fmt or 'unknown'}; use PNG, JPEG or WebP.")

    result = facekit.analyse([original])[0]
    if result.get("error"):
        raise ReferenceError(f"The image could not be analysed: {result['error']}")
    faces = result.get("faces") or []
    if not faces:
        raise ReferenceError("No face was detected in the reference image.")
    if len(faces) > 1:
        raise ReferenceError(f"{len(faces)} faces were detected; the reference must show exactly one person.")
    face = faces[0]
    x, y, w, h = face["bbox"]
    problems = []
    if face["confidence"] < rules.min_confidence:
        problems.append(f"face detection confidence {face['confidence']:.2f} is below {rules.min_confidence}")
    if min(w, h) < rules.min_face_px:
        problems.append(f"the face is only {int(min(w, h))} px; at least {rules.min_face_px} px is needed")
    if face.get("quality", -1) >= 0 and face["quality"] < rules.min_quality:
        problems.append(f"face capture quality {face['quality']:.2f} is below {rules.min_quality} (blur, lighting or pose)")
    yaw = abs(face.get("yaw", 0.0)) * 57.2958
    if yaw > rules.max_yaw_deg:
        problems.append(f"the head is turned {yaw:.0f}° from the camera (max {rules.max_yaw_deg:.0f}°)")
    if problems:
        raise ReferenceError("The reference face is not usable: " + "; ".join(problems) + ".")
    if facekit.geometry(face) is None:
        raise ReferenceError("Facial landmarks could not be measured on the reference image.")

    face_meta = {"confidence": round(face["confidence"], 3), "bbox": [round(v, 1) for v in face["bbox"]],
                 "quality": round(face.get("quality", -1), 3), "yaw_deg": round(yaw, 1)}
    if mode == "full":
        digest = sha256(original)
        meta = {
            "mode": "full",
            "original": {"file": original.name, "sha256": digest, "width": width, "height": height,
                         "format": fmt, "bytes": original.stat().st_size},
            "prepared": {"file": original.name, "sha256": digest, "width": width, "height": height,
                         "mime_type": MIME[fmt], "background": "original (as uploaded)", "resized": False,
                         "derived_from": None, "note": "the uploaded file itself, unchanged"},
            "face": face_meta,
            "geometry": [round(v, 4) for v in facekit.geometry(face)],
            "rules": rules.__dict__,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return meta

    side = max(w, h) * rules.crop_scale
    cx, cy = x + w / 2, y + h / 2 + h * 0.25  # shift down a little to include shoulders
    left = max(0, min(width - side, cx - side / 2))
    top = max(0, min(height - side, cy - side / 2))
    side = min(side, width, height)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = out_dir / "reference_prepared.png"
    cropped = out_dir / "_crop.png"
    with Image.open(original) as im:
        im.convert("RGB").crop((int(left), int(top), int(left + side), int(top + side))).save(cropped)
    background = "original"
    coverage = None
    face_identical = 1.0
    if rules.plain_background:
        # Replace the backdrop (e.g. branded walls) with plain grey. The face itself must come
        # through untouched: measured below, and the plain crop is used if it did not.
        try:
            coverage = facekit.segment_person(cropped, prepared)
            face_identical = _face_pixels_identical(cropped, prepared, (x - left, y - top, w, h))
            if face_identical < rules.min_face_pixels_identical:
                shutil.copy(cropped, prepared)
                background = f"original (segmentation altered {1 - face_identical:.1%} of face pixels)"
                face_identical = 1.0
            else:
                background = "plain grey (Vision person segmentation)"
        except facekit.FaceKitError:
            shutil.copy(cropped, prepared)
    else:
        shutil.copy(cropped, prepared)
    cropped.unlink(missing_ok=True)

    meta = {
        "mode": "face_crop",
        "original": {"file": original.name, "sha256": sha256(original), "width": width, "height": height,
                     "format": fmt, "bytes": original.stat().st_size},
        "prepared": {"file": prepared.name, "sha256": sha256(prepared), "size": int(side),
                     "crop": [int(left), int(top), int(side), int(side)], "derived_from": original.name,
                     "background": background, "person_coverage": round(coverage, 3) if coverage is not None else None,
                     "face_pixels_identical": round(face_identical, 4), "resized": False, "format": "PNG (lossless)",
                     "mime_type": "image/png", "width": int(side), "height": int(side)},
        "face": face_meta,
        "geometry": [round(v, 4) for v in facekit.geometry(face)],
        "rules": rules.__dict__,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def _face_pixels_identical(before: Path, after: Path, box: tuple) -> float:
    """Share of pixels inside the face box (inset 10%) that are unchanged (within 2 levels)."""
    import numpy as np
    from PIL import Image

    x, y, w, h = box
    inset = (x + w * 0.1, y + h * 0.1, x + w * 0.9, y + h * 0.9)
    a = np.asarray(Image.open(before).convert("RGB").crop(tuple(int(v) for v in inset)), dtype=np.int16)
    b = np.asarray(Image.open(after).convert("RGB").crop(tuple(int(v) for v in inset)), dtype=np.int16)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float((np.abs(a - b).max(axis=2) <= 2).mean())
