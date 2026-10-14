"""Database locale dei filamenti (JSON), con import/export e storico calibrazioni."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from importlib import resources
from pathlib import Path
from typing import Iterable

from colorforge.filaments.model import Filament

FORMAT_VERSION = 1


def default_user_db_path() -> Path:
    base = os.environ.get("APPDATA") or os.path.join(Path.home(), ".config")
    return Path(base) / "ColorForge" / "filaments.json"


class FilamentDatabase:
    def __init__(self, filaments: Iterable[Filament] = (), calibration_history: list | None = None):
        self._items: dict[str, Filament] = {}
        self.calibration_history: list[dict] = list(calibration_history or [])
        for f in filaments:
            self.add(f)

    # --- accesso -------------------------------------------------------
    def add(self, f: Filament, overwrite: bool = False) -> None:
        if f.id in self._items and not overwrite:
            raise KeyError(f"Filamento già presente: {f.id}")
        self._items[f.id] = f

    def remove(self, fid: str) -> None:
        self._items.pop(fid)

    def get(self, fid: str) -> Filament:
        return self._items[fid]

    def __contains__(self, fid: str) -> bool:
        return fid in self._items

    def __iter__(self):
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def owned(self) -> list[Filament]:
        return [f for f in self if f.owned_spools > 0]

    # --- persistenza ---------------------------------------------------
    def to_dict(self) -> dict:
        return {"format": "colorforge-filaments", "version": FORMAT_VERSION,
                "filaments": [f.to_dict() for f in self],
                "calibration_history": self.calibration_history}

    @classmethod
    def from_dict(cls, d: dict) -> "FilamentDatabase":
        if d.get("format") != "colorforge-filaments":
            raise ValueError("File non riconosciuto come database filamenti ColorForge")
        if int(d.get("version", 0)) > FORMAT_VERSION:
            raise ValueError("Database creato con una versione più recente di ColorForge")
        return cls((Filament.from_dict(x) for x in d.get("filaments", [])),
                   d.get("calibration_history", []))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # scrittura atomica

    @classmethod
    def load(cls, path: str | Path) -> "FilamentDatabase":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def load_default(cls) -> "FilamentDatabase":
        user = default_user_db_path()
        if user.exists():
            return cls.load(user)
        return cls.builtin()

    @classmethod
    def builtin(cls) -> "FilamentDatabase":
        text = resources.files("colorforge.data").joinpath("filaments_default.json").read_text("utf-8")
        return cls.from_dict(json.loads(text))

    def merge(self, other: "FilamentDatabase", overwrite: bool = False) -> int:
        n = 0
        for f in other:
            if overwrite or f.id not in self:
                self.add(f, overwrite=True)
                n += 1
        return n
