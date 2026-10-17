"""Provino di calibrazione: una scala di spessori per ogni filamento.

Layout (vista dall'alto, mm): piastra di base del filamento chiaro di
riferimento; una riga per filamento; colonna 0 = base scoperta (riferimento
bianco), colonne successive = 1, 2, 3, ... strati del filamento sopra la base.
Un'aletta nell'angolo in alto a sinistra indica l'orientamento.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from colorforge.geometry.mesh import Mesh, interval_mesh
from colorforge.optimize.search import LayerPlan

DEFAULT_STEPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]


@dataclass
class SpecimenSpec:
    base_id: str
    filament_ids: list[str]
    steps: list[int] = field(default_factory=lambda: list(DEFAULT_STEPS))
    patch_mm: float = 10.0
    gap_mm: float = 2.0
    margin_mm: float = 3.0
    layer_mm: float = 0.08
    first_layer_mm: float = 0.16
    base_mm: float = 0.64
    id: str = field(default_factory=lambda: "CF-" + uuid.uuid4().hex[:8].upper())
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))
    printer_id: str = ""
    notes: str = ""

    @property
    def plan(self) -> LayerPlan:
        top = self.base_mm + max(self.steps) * self.layer_mm
        return LayerPlan.from_settings(self.first_layer_mm, self.layer_mm, self.base_mm, top + 1e-6)

    @property
    def columns(self) -> list[int]:
        return [0] + list(self.steps)

    @property
    def size_mm(self) -> tuple[float, float]:
        n_c, n_r = len(self.columns), len(self.filament_ids)
        w = 2 * self.margin_mm + n_c * self.patch_mm + (n_c - 1) * self.gap_mm
        h = 2 * self.margin_mm + n_r * self.patch_mm + (n_r - 1) * self.gap_mm
        return w, h

    def patch_rect(self, row: int, col: int) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) in mm, origine in alto a sinistra della piastra, y verso il basso."""
        x0 = self.margin_mm + col * (self.patch_mm + self.gap_mm)
        y0 = self.margin_mm + row * (self.patch_mm + self.gap_mm)
        return x0, y0, x0 + self.patch_mm, y0 + self.patch_mm

    def to_dict(self) -> dict:
        return {"format": "colorforge-specimen", "version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "SpecimenSpec":
        if d.get("format") != "colorforge-specimen":
            raise ValueError("Il file non è un provino ColorForge")
        d = {k: v for k, v in d.items() if k not in ("format", "version")}
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SpecimenSpec":
        return cls.from_dict(json.loads(Path(path).read_text("utf-8")))


def specimen_objects(spec: SpecimenSpec, pixel_mm: float = 0.5) -> list[tuple[str, str, Mesh]]:
    """[(nome, id filamento, mesh)] : piastra di base + un oggetto per filamento."""
    plan = spec.plan
    B = plan.base_layers
    w, h = spec.size_mm
    tab = 6.0  # aletta di orientamento in alto a sinistra
    cols, rows = int(round(w / pixel_mm)), int(round((h + tab) / pixel_mm))
    toff = int(round(tab / pixel_mm))
    lo0 = np.zeros((rows, cols), dtype=np.int64)
    base_hi = np.zeros((rows, cols), dtype=np.int64)
    base_hi[toff:, :] = B
    base_hi[:toff, :toff] = B  # aletta
    out = [(f"Base {spec.base_id}", spec.base_id, interval_mesh(lo0, base_hi, pixel_mm, plan.z_of_level))]
    for r, fid in enumerate(spec.filament_ids):
        lo = np.zeros((rows, cols), dtype=np.int64)
        hi = np.zeros((rows, cols), dtype=np.int64)
        for c, n in enumerate(spec.columns):
            if n == 0:
                continue
            x0, y0, x1, y1 = spec.patch_rect(r, c)
            ys, ye = toff + int(round(y0 / pixel_mm)), toff + int(round(y1 / pixel_mm))
            xs, xe = int(round(x0 / pixel_mm)), int(round(x1 / pixel_mm))
            lo[ys:ye, xs:xe] = B
            hi[ys:ye, xs:xe] = B + n
        out.append((f"Riga {r + 1} - {fid}", fid, interval_mesh(lo, hi, pixel_mm, plan.z_of_level)))
    return out


def export_specimen(spec: SpecimenSpec, path_3mf: str | Path, db, slots: dict[str, int] | None = None):
    from colorforge.color.spaces import srgb_to_hex
    from colorforge.export.threemf import write_3mf
    slots = dict(slots or {})
    order = [spec.base_id] + [f for f in spec.filament_ids if f != spec.base_id]
    for f in order:
        slots.setdefault(f, len(set(slots.values()) | {0}) if slots else 1)
    auto = {}
    for f in order:
        auto[f] = slots.get(f) or len(auto) + 1
    parts = []
    for name, fid, mesh in specimen_objects(spec):
        f = db.get(fid)
        parts.append((f"slot {auto[fid]} - {name} {f.label}", srgb_to_hex(f.srgb), auto[fid], mesh))
    write_3mf(parts, path_3mf, {"Title": f"Provino {spec.id}", "Application": "ColorForge AI"})
    spec.save(Path(path_3mf).with_suffix(".cfcal"))
    return auto
