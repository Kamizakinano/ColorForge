"""Profili stampante e di stampa."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources


@dataclass
class ChangerSystem:
    id: str
    name: str
    slots: int
    kind: str  # "ams" | "manual"
    max_units: int = 1

    @property
    def capacity(self) -> int:
        """Numero massimo di filamenti caricabili contemporaneamente (0 = illimitato, manuale)."""
        return 0 if self.kind == "manual" else self.slots * self.max_units


@dataclass
class PrinterProfile:
    id: str
    name: str
    build_volume_mm: tuple[float, float, float]
    nozzle_mm: float
    changers: list[ChangerSystem]
    slicer: str = ""

    def changer(self, cid: str) -> ChangerSystem:
        for c in self.changers:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def fits(self, w: float, h: float, z: float) -> bool:
        bx, by, bz = self.build_volume_mm
        return (w <= bx and h <= by or w <= by and h <= bx) and z <= bz


@dataclass
class PrintSettings:
    layer_height_mm: float = 0.08
    first_layer_height_mm: float = 0.16
    base_thickness_mm: float = 0.64
    max_thickness_mm: float = 2.4
    min_feature_mm: float = 0.4
    max_color_changes: int = 8
    max_layers: int = 0  # 0 = derivato da max_thickness


def load_printers() -> list[PrinterProfile]:
    data = json.loads(resources.files("colorforge.data").joinpath("printers.json").read_text("utf-8"))
    out = []
    for p in data["printers"]:
        out.append(PrinterProfile(
            id=p["id"], name=p["name"], build_volume_mm=tuple(p["build_volume_mm"]),
            nozzle_mm=p["nozzle_mm"], slicer=p.get("slicer", ""),
            changers=[ChangerSystem(**c) for c in p["changers"]]))
    return out


def get_printer(pid: str) -> PrinterProfile:
    for p in load_printers():
        if p.id == pid:
            return p
    raise KeyError(f"Stampante sconosciuta: {pid}")
