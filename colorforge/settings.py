"""Impostazioni persistenti dell'applicazione (JSON in %APPDATA%/ColorForge)."""
from __future__ import annotations

import json
import os
from pathlib import Path


def app_dir() -> Path:
    base = os.environ.get("APPDATA") or os.path.join(Path.home(), ".config")
    p = Path(base) / "ColorForge"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load() -> dict:
    try:
        return json.loads((app_dir() / "settings.json").read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def save(d: dict) -> None:
    p = app_dir() / "settings.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), "utf-8")
    os.replace(tmp, p)


def bambu_template_path() -> Path | None:
    p = load().get("bambu_template")
    return Path(p) if p and Path(p).exists() else None
