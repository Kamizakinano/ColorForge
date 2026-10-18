"""Accessori opzionali: cornice, gancio da appendere, piedini d'appoggio.

Tutto è costruito con `column_mesh`/`interval_mesh` su griglie in mm, quindi le
mesh sono chiuse e manifold. Coordinate: il pannello occupa [0, W] x [0, H] mm
(y verso l'alto), come le mesh del pannello.
"""
from __future__ import annotations

import numpy as np

from colorforge.geometry.mesh import Mesh, interval_mesh

STEP = 0.5  # mm per cella degli accessori


def _shift(m: Mesh, dx: float, dy: float) -> Mesh:
    return Mesh(m.vertices + np.array([dx, dy, 0.0]), m.faces)


def _z_lin(step_mm):
    return lambda l: np.asarray(l, dtype=float) * step_mm


def frame(width_mm: float, height_mm: float, border_mm: float, top_mm: float) -> Mesh:
    """Cornice ad anello intorno al pannello, alta `top_mm` (sopra il pannello)."""
    b = int(round(border_mm / STEP))
    cols = int(round(width_mm / STEP)) + 2 * b
    rows = int(round(height_mm / STEP)) + 2 * b
    ring = np.ones((rows, cols), bool)
    ring[b:rows - b, b:cols - b] = False
    hz = int(round(top_mm / 0.1))
    lo = np.zeros((rows, cols), np.int64)
    hi = np.where(ring, hz, 0)
    m = interval_mesh(lo, hi, STEP, _z_lin(0.1))
    return _shift(m, -b * STEP, -b * STEP)


def hook(width_mm: float, height_mm: float, offset_mm: float, thick_mm: float = 2.0,
         tab_w: float = 18.0, tab_h: float = 12.0, hole_d: float = 5.0) -> Mesh:
    """Linguetta con foro sopra il bordo superiore, al centro: per appendere a un chiodo."""
    cols, rows = int(round(tab_w / STEP)), int(round(tab_h / STEP))
    yy, xx = np.mgrid[0:rows, 0:cols]
    cx, cy = (cols - 1) / 2, rows * 0.45
    body = np.ones((rows, cols), bool)
    # angoli superiori arrotondati
    r = cols / 2
    body &= ~((yy < r) & ((xx - cx) ** 2 + (yy - r) ** 2 > r ** 2))
    body &= (xx - cx) ** 2 + (yy - cy) ** 2 > (hole_d / 2 / STEP) ** 2
    hz = int(round(thick_mm / 0.1))
    m = interval_mesh(np.zeros_like(yy), np.where(body, hz, 0), STEP, _z_lin(0.1))
    # la riga 0 è in alto: la linguetta si appoggia al bordo superiore del pannello
    return _shift(m, width_mm / 2 - tab_w / 2, height_mm + offset_mm - 0.5)


def stand_foot(panel_thick_mm: float, length_mm: float = 30.0, depth_mm: float = 14.0,
               height_mm: float = 10.0, clearance: float = 0.3) -> Mesh:
    """Piedino con fessura: il pannello si infila in verticale. Si stampa a parte."""
    cols, rows = int(round(length_mm / STEP)), int(round(depth_mm / STEP))
    slot = int(np.ceil((panel_thick_mm + clearance) / STEP))
    s0 = (rows - slot) // 2
    hz = int(round(height_mm / 0.1))
    floor = int(round(2.0 / 0.1))
    hi = np.full((rows, cols), hz, np.int64)
    hi[s0:s0 + slot, :] = floor          # la fessura lascia un fondo di 2 mm
    return interval_mesh(np.zeros_like(hi), hi, STEP, _z_lin(0.1))


def accessories_for(width_mm: float, height_mm: float, panel_top_mm: float, add_frame: bool,
                    add_hook: bool, add_stand: bool, border_mm: float = 5.0) -> list[tuple[str, Mesh]]:
    out = []
    off = border_mm if add_frame else 0.0
    if add_frame:
        out.append(("cornice", frame(width_mm, height_mm, border_mm, panel_top_mm + 1.0)))
    if add_hook:
        out.append(("gancio", hook(width_mm, height_mm, off)))
    if add_stand:
        f1 = stand_foot(panel_top_mm)
        # due piedini accanto al pannello, sul piatto
        out.append(("piedino 1", _shift(f1, 0, -off - 20)))
        out.append(("piedino 2", _shift(f1, max(width_mm - 30, 35), -off - 20)))
    return out
