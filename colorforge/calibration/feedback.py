"""Correzione del modello dal confronto con la stampa reale.

Si fotografa la stampa, si raddrizza sulla griglia del progetto e, in zone
uniformi, si confronta il colore reale con quello previsto. I parametri ottici
(R∞, S) dei filamenti usati vengono riadattati insieme, con una
regolarizzazione verso i valori attuali, così da spiegare i colori osservati.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from skimage.transform import ProjectiveTransform, warp

from colorforge.color.spaces import delta_e2000, linear_to_lab, srgb_to_lab, srgb_to_linear
from colorforge.filaments.model import Reliability
from colorforge.optics.km import km_layer


def rectify(photo_srgb: np.ndarray, corners_px, shape) -> np.ndarray:
    """Raddrizza la foto della stampa sulla griglia (righe, colonne) del progetto.
    Angoli in ordine: alto-sx, alto-dx, basso-dx, basso-sx dell'immagine stampata."""
    rows, cols = shape
    t = ProjectiveTransform()
    if not t.estimate(np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]], float), np.asarray(corners_px, float)):
        raise ValueError("Angoli non validi")
    return warp(photo_srgb, t, output_shape=(rows, cols), order=1)


def stacks_for(res) -> tuple[list[str], np.ndarray]:
    """Per ogni pixel la pila di strati come segmenti (indice filamento, spessore mm)
    dal basso. Ritorna (ids, array (H, W, S, 2))."""
    plan = res.plan
    z = plan.z_of_level
    if res.zone is not None:
        rs, idx = res.zone["recipes"], res.zone["idx"]
        B = plan.base_layers
        fids = list(rs.fids)
        tb = float(z(B))
        fa, a, fb, b = rs.fa[idx], rs.a[idx], rs.fb[idx], rs.b[idx]
        seg = np.zeros(idx.shape + (3, 2))
        seg[..., 0, :] = [0, tb]
        seg[..., 1, 0] = np.maximum(fa, 0)
        seg[..., 1, 1] = np.where(fa >= 0, z(B + a) - tb, 0)
        seg[..., 2, 0] = np.maximum(fb, 0)
        seg[..., 2, 1] = np.where(fb >= 0, z(B + np.where(fa >= 0, a, 0) + b) - z(B + np.where(fa >= 0, a, 0)), 0)
        return fids, seg
    seq = res.solution.sequence
    fids = []
    for f, _ in seq:
        if f not in fids:
            fids.append(f)
    lv = res.levels
    seg = np.zeros(lv.shape + (len(seq), 2))
    for i, (f, start) in enumerate(seq):
        end = seq[i + 1][1] if i + 1 < len(seq) else plan.max_layers
        top = np.clip(lv, start, end)
        seg[..., i, 0] = fids.index(f)
        seg[..., i, 1] = z(top) - z(np.full_like(lv, start))
    return fids, seg


def pick_samples(pred_lab: np.ndarray, n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pixel in zone uniformi (bassa varianza locale della previsione), ben distribuiti."""
    var = _local_std(pred_lab)
    ok = var < np.percentile(var, 40)
    ys, xs = np.nonzero(ok)
    rng = np.random.default_rng(seed)
    if len(ys) > n:
        k = rng.choice(len(ys), n, replace=False)
        ys, xs = ys[k], xs[k]
    return ys, xs


def _local_std(lab):
    m = ndimage.uniform_filter(lab, (5, 5, 1))
    m2 = ndimage.uniform_filter(lab ** 2, (5, 5, 1))
    return np.sqrt(np.maximum(m2 - m ** 2, 0)).sum(-1)


def _predict(r_inf, s, segs, bg):
    out = np.empty((len(segs), 3))
    for i, sg in enumerate(segs):
        r = np.full(3, bg)
        for fi, t in sg:
            if t > 0:
                r = km_layer(r_inf[int(fi)], s[int(fi)], t, r)
        out[i] = r
    return out


def refit_from_print(res, db, photo_srgb, corners_px, n_samples: int = 500, reg: float = 0.5):
    """Ritorna dict con parametri nuovi e ΔE prima/dopo sui campioni."""
    rows, cols = res.levels.shape
    rect = rectify(photo_srgb, corners_px, (rows, cols))
    rect = ndimage.uniform_filter(rect, (3, 3, 1))
    pred_lab = linear_to_lab(srgb_to_linear(np.clip(res.simulated_srgb, 0, 1)))
    ys, xs = pick_samples(pred_lab, n_samples)
    fids, seg = stacks_for(res)
    segs = seg[ys, xs]
    obs_lin = srgb_to_linear(rect[ys, xs])
    pred_lin0 = srgb_to_linear(res.simulated_srgb[ys, xs])
    # bilanciamento: guadagno per canale che allinea in media osservato e previsto
    gain = np.median(pred_lin0, 0) / np.maximum(np.median(obs_lin, 0), 1e-4)
    obs_lab = linear_to_lab(np.clip(obs_lin * gain, 0, 1))
    r0 = np.array([db.get(f).linear_rgb for f in fids])
    from colorforge.optics.km import OpticalModel
    m = OpticalModel([db.get(f) for f in fids], background=res.settings.background_reflectance)
    s0 = np.array([m.layers[f].s for f in fids])
    bg = res.settings.background_reflectance
    nF = len(fids)
    r0c = np.clip(r0, 0.01, 0.99)

    def unpack(p):
        r = 1 / (1 + np.exp(-p[:3 * nF].reshape(nF, 3)))
        s = np.exp(p[3 * nF:])
        return r, s

    p0 = np.concatenate([np.log(r0c / (1 - r0c)).ravel(), np.log(s0)])

    def resid(p):
        r, s = unpack(p)
        de = delta_e2000(linear_to_lab(_predict(r, s, segs, bg)), obs_lab)
        return np.concatenate([de / np.sqrt(len(de)), reg * (p - p0) / np.sqrt(len(p))])

    before = float(delta_e2000(linear_to_lab(_predict(r0c, s0, segs, bg)), obs_lab).mean())
    sol = least_squares(resid, p0, method="trf", loss="soft_l1", max_nfev=60)
    r1, s1 = unpack(sol.x)
    after = float(delta_e2000(linear_to_lab(_predict(r1, s1, segs, bg)), obs_lab).mean())
    return {"fids": fids, "r_inf": r1, "s": s1, "before": before, "after": after,
            "n": len(ys), "rectified": rect}


def apply_refit(db, fit: dict, project_note: str = "") -> list[str]:
    from colorforge.calibration.fit import td_from_scattering
    done = []
    if fit["after"] >= fit["before"] - 0.2:
        return done  # nessun miglioramento significativo: non si tocca il database
    for f_id, r, s in zip(fit["fids"], fit["r_inf"], fit["s"]):
        f = db.get(f_id)
        f.lab = [float(x) for x in linear_to_lab(r)]
        f.km_scattering = float(s)
        f.td_mm = round(td_from_scattering(r, float(s)), 3)
        if f.td_reliability.weight < Reliability.PHOTO.weight:
            f.td_reliability = Reliability.PHOTO
        if f.color_reliability.weight < Reliability.PHOTO.weight:
            f.color_reliability = Reliability.PHOTO
        f.calibration_notes = (f"Corretto dalla stampa reale {time.strftime('%Y-%m-%d')} "
                               f"(ΔE {fit['before']:.1f} → {fit['after']:.1f}) {project_note}").strip()
        done.append(f_id)
    db.calibration_history.append({"date": time.strftime("%Y-%m-%d %H:%M"), "method": "print_feedback",
                                   "before": round(fit["before"], 2), "after": round(fit["after"], 2),
                                   "filaments": done})
    return done
