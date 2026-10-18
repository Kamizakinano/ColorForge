"""Modello ottico degli strati: Kubelka-Munk a due flussi, per banda.

Ogni filamento è descritto da:
  * R_inf  - riflettanza per banda di uno strato otticamente infinito
             (il colore del materiale stampato spesso);
  * S      - coefficiente di scattering [1/mm], ipotizzato indipendente dalla banda.
Il rapporto K/S per banda deriva da R_inf: K/S = (1 - R_inf)^2 / (2 R_inf).

La riflettanza di uno strato di spessore d sopra uno sfondo di riflettanza Rg è
la soluzione chiusa di Kubelka (1948):

    R = (1 - Rg (a - b coth(b S d))) / (a - Rg + b coth(b S d)),
    a = 1 + K/S,  b = sqrt(a^2 - 1)

Lo stack si calcola dal basso verso l'alto: ogni strato usa come sfondo la
riflettanza dello stack sottostante. Questo NON è una sovrapposizione RGB
alfa: la non linearità tra spessore, assorbimento e scattering è esplicita.

Limiti noti (documentati): bande = canali sRGB lineari (non spettrale),
nessuna correzione di Saunderson per la riflessione superficiale, nessuna
diffusione laterale. Il modello è pensato per essere *calibrato*: S viene
stimato dalla TD se non c'è una calibrazione, ed è marcato come stimato.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colorforge.filaments.model import Filament, Reliability

#: TD generica usata quando il dato reale manca. È un'ipotesi del software,
#: viene sempre segnalata all'utente e riduce l'affidabilità della previsione.
DEFAULT_TD_MM = 2.0
#: Stima della TD dalla luminosità (L*) quando il dato manca: i filamenti scuri
#: sono molto coprenti, quelli chiari lasciano passare più luce. Ordine di
#: grandezza tipico dei PLA pigmentati; resta una STIMA da sostituire con la
#: calibrazione o con il valore dichiarato.
TD_DARK_MM, TD_LIGHT_MM = 0.4, 3.0


def estimated_td(lab_l: float) -> float:
    t = min(max(lab_l / 100.0, 0.0), 1.0)
    return TD_DARK_MM + (TD_LIGHT_MM - TD_DARK_MM) * t * t
#: Contrasto residuo (sfondo bianco vs nero) che definisce la TD nel modello.
TD_CONTRAST = 0.02

_R_MIN, _R_MAX = 0.002, 0.995


def _coth(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 1e-9)
    return np.where(x > 20, 1.0, 1.0 / np.tanh(np.minimum(x, 20)))


def km_layer(r_inf: np.ndarray, s: float, d: float, rg: np.ndarray) -> np.ndarray:
    """Riflettanza di uno strato (spessore d mm) sopra uno sfondo rg. Broadcast su bande."""
    r_inf = np.clip(r_inf, _R_MIN, _R_MAX)
    if d <= 0:
        return np.asarray(rg, dtype=float)
    ks = (1 - r_inf) ** 2 / (2 * r_inf)
    a = 1 + ks
    b = np.sqrt(a * a - 1)
    c = b * _coth(b * s * d)
    return (1 - rg * (a - c)) / (a - rg + c)


def blend_layer(r_inf: np.ndarray, td_mm: float, d: float, rg: np.ndarray) -> np.ndarray:
    """Strato sottile come in HueForge: il colore sotto si attenua in modo esponenziale
    con lo spessore e a spessore TD ne resta solo TD_CONTRAST. Rispetto a Kubelka-Munk
    con S ricavata dalla TD, i primi strati coprono molto meno (un azzurro sottile sul
    nero dà un blu scuro, non subito l'azzurro pieno), come si osserva sulle stampe."""
    if d <= 0:
        return np.asarray(rg, dtype=float)
    k = np.log(1.0 / TD_CONTRAST) / max(td_mm, 1e-3)
    t = np.exp(-k * d)
    return r_inf + (np.asarray(rg, dtype=float) - r_inf) * t


def scattering_from_td(r_inf: np.ndarray, td_mm: float) -> float:
    """S tale che a spessore TD il contrasto bianco/nero medio scenda a TD_CONTRAST."""
    lo, hi = 1e-3, 1e5  # su x = S * d
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        contrast = np.mean(km_layer(r_inf, mid, 1.0, 1.0) - km_layer(r_inf, mid, 1.0, 0.0))
        if contrast > TD_CONTRAST:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi) / td_mm)


@dataclass
class LayerOptics:
    filament_id: str
    r_inf: np.ndarray
    s: float
    reliability: float
    s_source: str  # "calibration" | "td" | "default"
    td_mm: float | None = None


class OpticalModel:
    """Modello ottico calibrabile per un insieme di filamenti."""

    def __init__(self, filaments: list[Filament], background: float = 0.0):
        self.background = float(background)
        self.layers: dict[str, LayerOptics] = {}
        self.warnings: list[str] = []
        for f in filaments:
            r_inf = f.linear_rgb
            td = None
            if f.km_scattering is not None:
                s, src = float(f.km_scattering), "calibration"
            elif f.td_mm is not None:
                td = float(f.td_mm)
                s, src = scattering_from_td(r_inf, td), "td"
            else:
                td = estimated_td(float(f.lab[0]))
                s, src = scattering_from_td(r_inf, td), "default"
                self.warnings.append(
                    f"{f.label}: TD sconosciuta, stimata {td:.1f} mm dalla luminosità del colore "
                    "(previsione poco affidabile: inserisci la TD o calibra il filamento).")
            rel = f.reliability if src != "default" else min(f.reliability, Reliability.ESTIMATED.weight)
            self.layers[f.id] = LayerOptics(f.id, r_inf, s, rel, src, td)

    def layer(self, fid: str, d: float, rg: np.ndarray) -> np.ndarray:
        lo = self.layers[fid]
        if lo.td_mm is not None:
            return blend_layer(lo.r_inf, lo.td_mm, d, rg)
        return km_layer(lo.r_inf, lo.s, d, rg)

    def stack_lut(self, filament_per_layer: list[str], layer_heights: np.ndarray) -> np.ndarray:
        """Colore (sRGB lineare) visto dall'alto per ogni numero di strati 0..N.

        `filament_per_layer[i]` è il filamento dello strato i (dal basso);
        `layer_heights[i]` il suo spessore. Ritorna array (N+1, 3); la riga 0
        è lo sfondo (assenza di materiale).
        """
        n = len(filament_per_layer)
        out = np.empty((n + 1, 3))
        r = np.full(3, self.background)
        out[0] = r
        for i in range(n):
            r = self.layer(filament_per_layer[i], float(layer_heights[i]), r)
            out[i + 1] = r
        return out

    def reliability(self, fids: list[str]) -> float:
        return float(min(self.layers[f].reliability for f in fids)) if fids else 0.0
