"""Preparazione adattiva e reversibile dell'immagine per la stampa a strati."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from colorforge.color.spaces import lab_to_srgb, srgb_to_lab
from colorforge.imaging.analysis import estimate_noise


@dataclass
class PrepOptions:
    denoise: bool = True
    sharpen: bool = True
    auto_contrast: bool = True
    shadow_highlight_recovery: bool = True
    remove_background: bool = False
    background_fill: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def to_dict(self) -> dict:
        return asdict(self)


def resize_to_grid(srgb: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Ridimensiona con filtro di area (Lanczos) a risoluzione di stampa."""
    img = Image.fromarray((np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    img = img.resize((cols, rows), Image.Resampling.LANCZOS, reducing_gap=3.0)
    return np.asarray(img, dtype=np.float64) / 255.0


def prepare(srgb: np.ndarray, opts: PrepOptions, alpha: np.ndarray | None = None,
            subject: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Ritorna (immagine preparata, elenco operazioni applicate). Non altera l'originale."""
    log: list[str] = []
    lab = srgb_to_lab(srgb).copy()
    L = lab[..., 0]
    if opts.denoise:
        sigma_n = estimate_noise(L / 100.0)
        if sigma_n > 0.01:
            s = min(1.2, 20 * sigma_n)
            for c in range(3):
                lab[..., c] = ndimage.gaussian_filter(lab[..., c], s if c else s * 0.6)
            log.append(f"Riduzione rumore gaussiana sigma={s:.2f} (rumore stimato {sigma_n:.3f}).")
    if opts.auto_contrast:
        lo, hi = np.percentile(lab[..., 0], [0.5, 99.5])
        if hi - lo > 5 and (lo > 3 or hi < 97):
            # Stiramento moderato: al massimo il 50% verso [0,100]
            nlo, nhi = lo * 0.5, hi + (100 - hi) * 0.5
            lab[..., 0] = np.clip((lab[..., 0] - lo) / (hi - lo) * (nhi - nlo) + nlo, 0, 100)
            log.append(f"Contrasto adattato: L* [{lo:.0f},{hi:.0f}] -> [{nlo:.0f},{nhi:.0f}].")
    if opts.shadow_highlight_recovery:
        Lb = ndimage.gaussian_filter(lab[..., 0], 8)
        lift = np.clip((25 - Lb) / 25, 0, 1) * 6 - np.clip((Lb - 85) / 15, 0, 1) * 4
        if np.abs(lift).max() > 0.5:
            lab[..., 0] = np.clip(lab[..., 0] + lift, 0, 100)
            log.append("Recupero moderato di ombre e luci (max ±6 L*).")
    if opts.sharpen:
        blur = ndimage.gaussian_filter(lab[..., 0], 1.0)
        lab[..., 0] = np.clip(lab[..., 0] + 0.4 * (lab[..., 0] - blur), 0, 100)
        log.append("Nitidezza moderata (unsharp mask 0.4, raggio 1).")
    out = lab_to_srgb(lab)
    fill = np.array(opts.background_fill)
    if alpha is not None:
        out = out * alpha[..., None] + fill * (1 - alpha[..., None])
        log.append("Aree trasparenti riempite con il colore di sfondo.")
    if opts.remove_background and subject is not None:
        out = np.where(subject[..., None], out, fill)
        log.append("Sfondo sostituito (separazione soggetto euristica).")
    return out, log
