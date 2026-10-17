"""Analisi automatica della fotografia (sola lettura, nessuna modifica)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

from colorforge.color.spaces import srgb_to_lab


@dataclass
class ImageReport:
    width: int
    height: int
    mean_lightness: float
    contrast_std_l: float
    mean_chroma: float
    overexposed_pct: float
    underexposed_pct: float
    noise_estimate: float
    detail_energy: float
    distinct_colors: int
    subject_bbox: tuple[int, int, int, int] | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_noise(gray: np.ndarray) -> float:
    """Stima sigma del rumore (Immerkaer 1996) su immagine in [0,1]."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    conv = ndimage.convolve(gray, k, mode="reflect")
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def subject_mask(srgb: np.ndarray) -> np.ndarray:
    """Maschera grezza del soggetto: distanza cromatica dal colore del bordo.

    Euristica classica (non AI): funziona su sfondi uniformi; altrimenti
    l'utente viene avvisato che la separazione non è affidabile.
    """
    lab = srgb_to_lab(srgb)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=-1)
    spread = np.median(np.linalg.norm(border - bg, axis=-1))
    thr = max(12.0, 3 * spread)
    m = ndimage.binary_opening(dist > thr, iterations=2)
    m = ndimage.binary_fill_holes(m)
    lbl, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lbl, range(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def analyze(srgb: np.ndarray, print_pixel_mm: float | None = None, max_pixels: int = 1_000_000) -> ImageReport:
    h, w = srgb.shape[:2]
    step = int(np.ceil(np.sqrt(h * w / max_pixels)))
    if step > 1:  # le statistiche non richiedono la piena risoluzione
        srgb = srgb[::step, ::step]
    lab = srgb_to_lab(srgb)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    gray = L / 100.0
    over = float(np.mean(np.all(srgb > 0.98, axis=-1)) * 100)
    under = float(np.mean(np.all(srgb < 0.02, axis=-1)) * 100)
    noise = estimate_noise(gray)
    lap = ndimage.laplace(ndimage.gaussian_filter(gray, 1.0))
    detail = float(np.mean(np.abs(lap)))
    q = (np.clip(srgb, 0, 1) * 31).astype(np.int32)
    distinct = int(np.unique(q[..., 0] * 1024 + q[..., 1] * 32 + q[..., 2]).size)

    mask = subject_mask(srgb)
    bbox = None
    if 0.02 < mask.mean() < 0.95:
        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    warnings = []
    if over > 5:
        warnings.append(f"{over:.1f}% dell'immagine è sovraesposta: i dettagli nelle luci sono persi.")
    if under > 5:
        warnings.append(f"{under:.1f}% dell'immagine è sottoesposta: i dettagli nelle ombre sono persi.")
    if noise > 0.03:
        warnings.append("Rumore elevato: verrà applicata una riduzione moderata.")
    if L.std() < 10:
        warnings.append("Contrasto molto basso: il rilievo cromatico risulterà piatto.")
    if min(h, w) < 200:
        warnings.append("Risoluzione bassa: la stampa potrebbe mostrare pixel evidenti.")
    return ImageReport(w, h, float(L.mean()), float(L.std()), float(chroma.mean()), over, under,
                       noise, detail, distinct, bbox, warnings)


def detail_loss_map(srgb: np.ndarray, pixel_mm: float, min_feature_mm: float) -> np.ndarray:
    """Energia dei dettagli più piccoli della risoluzione stampabile (0..1)."""
    gray = srgb_to_lab(srgb)[..., 0] / 100.0
    sigma = max(min_feature_mm / max(pixel_mm, 1e-6) / 2, 0.5)
    fine = np.abs(gray - ndimage.gaussian_filter(gray, sigma))
    return np.clip(fine / (fine.max() + 1e-9), 0, 1)


def importance_map(srgb: np.ndarray, detail: float = 1.0, center: float = 0.5) -> np.ndarray:
    """Peso per pixel (media 1): più alto su contorni/dettagli e verso il soggetto.
    Euristica locale, verificabile: bordo (gradiente di L*) + soggetto stimato + centro."""
    lab = srgb_to_lab(srgb)
    L = ndimage.gaussian_filter(lab[..., 0], 0.8)
    g = np.hypot(ndimage.sobel(L, 0), ndimage.sobel(L, 1))
    g = ndimage.maximum_filter(g, 3)
    g = g / (np.percentile(g, 99) + 1e-6)
    h, w = L.shape
    yy, xx = np.mgrid[0:h, 0:w]
    c = 1 - np.hypot((yy - h / 2) / h, (xx - w / 2) / w) * 1.4
    wmap = 1 + detail * np.clip(g, 0, 1) + center * np.clip(c, 0, 1)
    try:
        subj = subject_mask(srgb)
        if 0.05 < subj.mean() < 0.9:
            wmap = wmap + 0.5 * subj
    except Exception:  # la stima del soggetto è facoltativa
        pass
    return wmap / wmap.mean()
