"""Read-only audit of existing Strict Talking Head outputs.

    .venv/bin/python -m studio.tools.audit_project            # every strict project with a final MP4
    .venv/bin/python -m studio.tools.audit_project <id> ...   # specific projects

For each project it measures the final MP4, voice track and generated clip with
`results.assess`.
- It writes the outcome to a NEW file, `reports/audit_<timestamp>.json`.
- It records it under `params.strict.audit`, so the Studio shows it.
- It never modifies, regenerates or deletes media, and it makes no network calls.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from studio.server import results
from studio.server.config import DATA_DIR
from studio.server.store import ProjectStore


def media_hashes(folder: Path) -> dict:
    out = {}
    for rel in ("output/final_talking_head.mp4", "cloned_voice.wav", "generated.mp4"):
        p = folder / rel
        if p.exists():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*")
    args = parser.parse_args()
    store = ProjectStore(DATA_DIR)
    projects = [store.get(i) for i in args.ids] if args.ids else store.list(limit=1000)
    for project in projects:
        if not project or project["params"].get("mode") != "strict":
            continue
        folder = store.project_dir(project["id"])
        if not (folder / "output" / "final_talking_head.mp4").exists():
            print(f"{project['id']}: no final MP4, skipped")
            continue
        before = media_hashes(folder)
        assessed = results.assess(folder, project["params"]["strict"])
        after = media_hashes(folder)
        assert before == after, "audit must not modify media"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        record = {"at": time.time(), "media_sha256": after, **assessed,
                  "note": "Read-only audit with the current assessment code; media untouched."}
        (folder / "reports" / f"audit_{stamp}.json").write_text(json.dumps(record, indent=2, default=str))
        params = project["params"]
        params["strict"]["audit"] = {k: record[k] for k in ("at", "rows", "facts", "note", "media_sha256")}
        store.update(project["id"], params=params)
        print(f"{project['id']}:")
        for r in assessed["rows"]:
            print(f"  {r['status']:15s} {r['label']}")


if __name__ == "__main__":
    main()
