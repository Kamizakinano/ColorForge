"""Scheda filamento e livelli di affidabilità dei dati ottici."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from colorforge.color.spaces import hex_to_srgb, lab_to_srgb, srgb_to_lab, srgb_to_linear


class Reliability(str, Enum):
    """Affidabilità di un dato. Non confondere mai 'measured' con le altre."""
    MEASURED = "measured"        # misurato con strumento su provino stampato
    PHOTO = "photo"              # misurato da fotografia con riferimento cromatico
    DECLARED = "declared"        # dichiarato dal produttore / dalla community
    ESTIMATED = "estimated"      # stima generica del software
    UNKNOWN = "unknown"

    @property
    def weight(self) -> float:
        return {"measured": 1.0, "photo": 0.7, "declared": 0.5,
                "estimated": 0.25, "unknown": 0.0}[self.value]


@dataclass
class Measurement:
    """Misura su provino: `thickness_mm` di questo filamento sopra uno sfondo noto."""
    thickness_mm: float
    background_lab: list[float]
    measured_lab: list[float]
    method: str = "spectrophotometer"
    specimen_id: str = ""
    notes: str = ""


@dataclass
class Filament:
    id: str
    brand: str
    name: str
    material: str = "PLA"
    color_name: str = ""
    product_code: str = ""
    # Colore del materiale stampato a spessore "infinito" (opaco), CIELAB D65.
    lab: list[float] = field(default_factory=lambda: [50.0, 0.0, 0.0])
    color_reliability: Reliability = Reliability.ESTIMATED
    # Transmission Distance (mm): spessore oltre il quale il colore sottostante
    # non è più distinguibile. None = sconosciuto: NON viene inventato, si usa
    # un default esplicitamente marcato come stimato dal modello ottico.
    td_mm: Optional[float] = None
    td_reliability: Reliability = Reliability.UNKNOWN
    # Coefficiente di scattering Kubelka-Munk [1/mm] ottenuto da calibrazione.
    km_scattering: Optional[float] = None
    reflectance_spectrum: Optional[dict] = None  # {"wavelengths_nm": [...], "values": [...]}
    density_g_cm3: float = 1.24
    print_temp_c: Optional[int] = None
    print_params: dict = field(default_factory=dict)
    calibration_notes: str = ""
    source: str = ""
    measurement_method: str = ""
    measurements: list[Measurement] = field(default_factory=list)
    owned_spools: int = 0

    @classmethod
    def from_hex(cls, id: str, brand: str, name: str, hex_color: str, **kw) -> "Filament":
        return cls(id=id, brand=brand, name=name,
                   lab=srgb_to_lab(hex_to_srgb(hex_color)).tolist(), **kw)

    @property
    def srgb(self) -> np.ndarray:
        return lab_to_srgb(np.array(self.lab, dtype=float))

    @property
    def linear_rgb(self) -> np.ndarray:
        return srgb_to_linear(self.srgb)

    @property
    def reliability(self) -> float:
        """Affidabilità complessiva [0..1] dei dati ottici di questo filamento."""
        td_w = self.td_reliability.weight
        if self.km_scattering is not None:
            td_w = max(td_w, Reliability.MEASURED.weight if self.measurements else td_w)
        return round(0.5 * self.color_reliability.weight + 0.5 * td_w, 3)

    @property
    def label(self) -> str:
        return f"{self.brand} {self.name}".strip()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["color_reliability"] = self.color_reliability.value
        d["td_reliability"] = self.td_reliability.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Filament":
        d = dict(d)
        if "hex" in d and "lab" not in d:
            d["lab"] = srgb_to_lab(hex_to_srgb(d.pop("hex"))).tolist()
        d.pop("hex", None)
        d["color_reliability"] = Reliability(d.get("color_reliability", "estimated"))
        d["td_reliability"] = Reliability(d.get("td_reliability", "unknown"))
        d["measurements"] = [Measurement(**m) for m in d.get("measurements", [])]
        lab = d.get("lab")
        if not (isinstance(lab, list) and len(lab) == 3):
            raise ValueError(f"Filamento {d.get('id')}: campo 'lab' non valido")
        if d.get("td_mm") is not None and not (0 < float(d["td_mm"]) < 100):
            raise ValueError(f"Filamento {d.get('id')}: td_mm fuori intervallo")
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})
