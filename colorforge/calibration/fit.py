"""Lettura del provino (foto o colorimetro) e stima dei parametri ottici."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from skimage.transform import ProjectiveTransform

from colorforge.color.spaces import (delta_e2000, lab_to_srgb, linear_to_lab, srgb_to_lab,
                                     srgb_to_linear)
from colorforge.calibration.specimen import SpecimenSpec
from colorforge.filaments.model import Measurement, Reliability
from colorforge.optics.km import TD_CONTRAST, km_layer


def sample_photo(srgb: np.ndarray, corners_px: list[tuple[float, float]], spec: SpecimenSpec,
                 inner: float = 0.5) -> np.ndarray:
    """Colori sRGB medi (mediana) di ogni quadratino. `corners_px` = angoli della
    piastra (senza aletta) in ordine: alto-sx, alto-dx, basso-dx, basso-sx.
    Ritorna (righe, colonne, 3)."""
    w, h = spec.size_mm
    t = ProjectiveTransform()
    if not t.estimate(np.array([[0, 0], [w, 0], [w, h], [0, h]], float), np.asarray(corners_px, float)):
        raise ValueError("Angoli non validi: impossibile raddrizzare la foto.")
    H, W = srgb.shape[:2]
    out = np.zeros((len(spec.filament_ids), len(spec.columns), 3))
    for r in range(len(spec.filament_ids)):
        for c in range(len(spec.columns)):
            x0, y0, x1, y1 = spec.patch_rect(r, c)
            cx, cy, hw = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * inner / 2
            g = np.linspace(-hw, hw, 9)
            gx, gy = np.meshgrid(cx + g, cy + g)
            pts = t(np.stack([gx.ravel(), gy.ravel()], 1))
            px = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
            py = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
            out[r, c] = np.median(srgb[py, px], axis=0)
    return out


def white_balance(measured_lin: np.ndarray, base_ref_lin: np.ndarray) -> np.ndarray:
    """Normalizza per canale così che la base scoperta (colonna 0) valga il
    colore di riferimento della base (von Kries in RGB lineare)."""
    base = np.median(measured_lin[:, 0], axis=0)
    gain = base_ref_lin / np.maximum(base, 1e-4)
    return np.clip(measured_lin * gain, 0, 1)


@dataclass
class FitResult:
    filament_id: str
    r_inf_lab: np.ndarray
    scattering: float
    td_mm: float
    rms_de: float
    n_points: int
    reliable: bool
    message: str


def td_from_scattering(r_inf: np.ndarray, s: float) -> float:
    lo, hi = 1e-3, 1e5
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        c = np.mean(km_layer(r_inf, mid, 1.0, 1.0) - km_layer(r_inf, mid, 1.0, 0.0))
        lo, hi = (mid, hi) if c > TD_CONTRAST else (lo, mid)
    return float(np.sqrt(lo * hi) / s)


def fit_filament(thick_mm: np.ndarray, measured_lab: np.ndarray, background_lin: np.ndarray,
                 init_lin: np.ndarray, init_s: float = 3.0) -> tuple[np.ndarray, float, float]:
    """Stima R∞ (sRGB lineare) e S minimizzando ΔE00 tra modello KM e misure."""
    def unpack(p):
        return 1 / (1 + np.exp(-p[:3])), float(np.exp(p[3]))

    def resid(p):
        r, s = unpack(p)
        pred = np.array([km_layer(r, s, d, background_lin) for d in thick_mm])
        return delta_e2000(linear_to_lab(pred), measured_lab)

    r0 = np.clip(init_lin, 0.01, 0.99)
    p0 = np.concatenate([np.log(r0 / (1 - r0)), [np.log(init_s)]])
    sol = least_squares(resid, p0, method="trf", loss="soft_l1", max_nfev=400)
    r, s = unpack(sol.x)
    return r, s, float(np.sqrt(np.mean(resid(sol.x) ** 2)))


def fit_specimen(spec: SpecimenSpec, patch_lab: np.ndarray) -> list[FitResult]:
    """`patch_lab` (righe, colonne, 3) in CIELAB (già bilanciato). Una stima per riga."""
    plan = spec.plan
    out = []
    for r, fid in enumerate(spec.filament_ids):
        bg = srgb_to_linear(lab_to_srgb(patch_lab[r, 0]))
        steps = np.array(spec.columns[1:])
        thick = steps * spec.layer_mm
        meas = patch_lab[r, 1:]
        init = srgb_to_linear(lab_to_srgb(meas[-1]))
        rr, s, rms = fit_filament(thick, meas, bg, init)
        td = td_from_scattering(rr, s)
        # affidabilità: contrasto sufficiente tra primi e ultimi gradini e residuo basso
        contrast = float(delta_e2000(patch_lab[r, 0], meas[-1]))   # base scoperta vs spessore massimo
        ok = rms < 4.0 and contrast > 3.0
        if not ok:
            msg = ("Il filamento è indistinguibile dalla base: misura poco informativa" if contrast <= 3.0
                   else f"Errore di adattamento alto (ΔE {rms:.1f}): controlla foto e angoli")
        elif float(delta_e2000(meas[0], meas[-1])) < 2.0:
            msg = f"OK – molto coprente (opaco già con {spec.columns[1]} strato)"
        else:
            msg = "OK"
        out.append(FitResult(fid, linear_to_lab(rr), s, td, rms, len(thick), ok, msg))
    return out


def apply_fits(db, spec: SpecimenSpec, patch_lab: np.ndarray, fits: list[FitResult],
               method: str, accept_unreliable: bool = False) -> list[str]:
    """Aggiorna il database. Le stime non affidabili si applicano solo se richiesto."""
    rel = Reliability.MEASURED if method == "colorimeter" else Reliability.PHOTO
    done = []
    for r, fit in enumerate(fits):
        if not fit.reliable and not accept_unreliable:
            continue
        f = db.get(fit.filament_id)
        f.lab = [float(x) for x in fit.r_inf_lab]
        f.color_reliability = rel
        f.km_scattering = float(fit.scattering)
        f.td_mm = round(float(fit.td_mm), 3)
        f.td_reliability = rel
        f.measurement_method = method
        bg = patch_lab[r, 0].tolist()
        for c, n in enumerate(spec.columns[1:], start=1):
            f.measurements.append(Measurement(thickness_mm=n * spec.layer_mm, background_lab=bg,
                                              measured_lab=patch_lab[r, c].tolist(), method=method,
                                              specimen_id=spec.id))
        f.calibration_notes = f"Calibrato {time.strftime('%Y-%m-%d')} provino {spec.id} ({method}), ΔE fit {fit.rms_de:.2f}"
        done.append(fit.filament_id)
    db.calibration_history.append({
        "date": time.strftime("%Y-%m-%d %H:%M"), "specimen": spec.id, "method": method,
        "results": [{"id": x.filament_id, "td_mm": round(x.td_mm, 3), "rms_de": round(x.rms_de, 2),
                     "applied": x.filament_id in done, "message": x.message} for x in fits]})
    return done
