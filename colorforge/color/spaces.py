"""Conversioni di spazio colore e metriche percettive.

Spazio di lavoro interno: sRGB lineare (primari sRGB, D65) in [0, 1].
Le metriche usano CIELAB (D65, osservatore 2°) e CIEDE2000.
"""
from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def linear_to_lab(lin: np.ndarray) -> np.ndarray:
    """sRGB lineare (..., 3) -> CIELAB (..., 3)."""
    return rgb2lab(linear_to_srgb(lin), channel_axis=-1)


def srgb_to_lab(srgb: np.ndarray) -> np.ndarray:
    return rgb2lab(np.clip(srgb, 0, 1), channel_axis=-1)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    return np.clip(lab2rgb(lab, channel_axis=-1), 0, 1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return deltaE_ciede2000(lab1, lab2, channel_axis=-1)


def hex_to_srgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Colore esadecimale non valido: {hex_color!r}")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def srgb_to_hex(srgb: np.ndarray) -> str:
    v = np.clip(np.round(np.asarray(srgb) * 255), 0, 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*v)
