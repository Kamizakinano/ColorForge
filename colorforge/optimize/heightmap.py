"""Assegnazione per pixel della quota (numero di strati) e correzione artefatti."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from colorforge.color.spaces import delta_e2000


def assign_levels(target_lab: np.ndarray, lut_lab: np.ndarray, base_layers: int,
                  chunk: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Per ogni pixel la quota (strati) con ΔE2000 minimo. Ritorna (livelli, ΔE)."""
    h, w = target_lab.shape[:2]
    x = target_lab.reshape(-1, 3)
    cand = lut_lab[base_layers:]
    levels = np.empty(len(x), dtype=np.int32)
    err = np.empty(len(x))
    # Dedup: molti pixel condividono lo stesso colore quantizzato a 0.5 unità Lab
    q = np.round(x * 2).astype(np.int32)
    uq, inv = np.unique(q, axis=0, return_inverse=True)
    ux = uq / 2.0
    ul = np.empty(len(ux), dtype=np.int32)
    ue = np.empty(len(ux))
    for i in range(0, len(ux), chunk):
        de = delta_e2000(ux[i:i + chunk, None, :], cand[None, :, :])
        ul[i:i + chunk] = de.argmin(axis=1)
        ue[i:i + chunk] = de.min(axis=1)
    inv = inv.reshape(-1)
    levels[:] = ul[inv] + base_layers
    err[:] = delta_e2000(x, lut_lab[levels])
    return levels.reshape(h, w), err.reshape(h, w)


def smooth_levels(levels: np.ndarray, target_lab: np.ndarray, lut_lab: np.ndarray,
                  max_increase: float = 0.05) -> tuple[np.ndarray, np.ndarray, str]:
    """Filtro mediano 3x3 sulla mappa quote: riduce pixel isolati e gradini
    inutili. Accettato solo se l'errore medio cresce meno di `max_increase`
    (relativo): la correzione locale non deve peggiorare il risultato globale."""
    err0 = delta_e2000(target_lab, lut_lab[levels])
    sm = ndimage.median_filter(levels, size=3, mode="nearest")
    err1 = delta_e2000(target_lab, lut_lab[sm])
    if err1.mean() <= err0.mean() * (1 + max_increase):
        return sm, err1, f"Livellamento applicato (ΔE medio {err0.mean():.2f} → {err1.mean():.2f})."
    return levels, err0, "Livellamento scartato: peggiorava la fedeltà complessiva."
