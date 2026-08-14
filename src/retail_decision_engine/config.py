from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
SOURCE_CONFIG = PROJECT_ROOT / "config" / "sources.json"


def ensure_directories() -> None:
    for path in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, ARTIFACT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_sources() -> dict[str, dict[str, str]]:
    with SOURCE_CONFIG.open(encoding="utf-8") as handle:
        return json.load(handle)
