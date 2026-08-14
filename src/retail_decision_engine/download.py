from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .config import RAW_DIR, ensure_directories, load_sources


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path, force: bool = False) -> dict[str, object]:
    if target.exists() and not force:
        return {
            "path": str(target.relative_to(RAW_DIR.parent.parent)),
            "url": url,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "status": "existing",
        }

    partial = target.with_suffix(target.suffix + ".part")
    if force:
        partial.unlink(missing_ok=True)
    resume_at = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "retail-decision-engine/0.2 academic research"}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = resume_at > 0 and response.status == 206
        mode = "ab" if resumed else "wb"
        if resume_at and not resumed:
            resume_at = 0
        with partial.open(mode) as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)

    if target.suffix == ".zip" and not zipfile.is_zipfile(partial):
        raise ValueError(f"Downloaded archive failed ZIP integrity check: {partial}")
    partial.replace(target)
    return {
        "path": str(target.relative_to(RAW_DIR.parent.parent)),
        "url": url,
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
        "status": "downloaded",
    }


def download_categories(categories: list[str], force: bool = False) -> Path:
    ensure_directories()
    sources = load_sources()
    unknown = sorted(set(categories) - set(sources))
    if unknown:
        raise ValueError(f"Unknown categories: {', '.join(unknown)}")

    records: list[dict[str, object]] = []
    for category in categories:
        source = sources[category]
        category_dir = RAW_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        records.append(
            {
                "category": category,
                "kind": "upc",
                **_download(source["upc_url"], category_dir / "upc.csv", force),
            }
        )
        records.append(
            {
                "category": category,
                "kind": "movement",
                **_download(source["movement_url"], category_dir / "movement.zip", force),
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_owner": "Kilts Center for Marketing, University of Chicago Booth",
        "files": records,
    }
    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
