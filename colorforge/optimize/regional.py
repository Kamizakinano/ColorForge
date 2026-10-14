"""Modalità "multicolore per zona" (AMS): filamenti diversi nello stesso strato.

Ogni colonna di pixel è una "ricetta":
    base   filamento f0 per B strati (comune a tutto il pannello)
    medio  filamento fa per a strati (0..M)
    sopra  filamento fb per b strati (0..M-a)
Il colore visto dall'alto è calcolato con il modello ottico (stack KM). Per ogni
pixel si sceglie la ricetta con ΔE2000 minimo (+ piccola penalità di spessore).
Gli oggetti 3D sono uno per (ruolo, filamento): in ogni colonna occupano un solo
intervallo di quote, quindi si costruiscono con `interval_mesh`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from colorforge.color.spaces import delta_e2000, linear_to_lab
from colorforge.geometry.mesh import interval_mesh
from colorforge.optics.km import OpticalModel
from colorforge.optimize.search import LayerPlan, representative_colors


@dataclass
class RecipeSet:
    fids: list[str]
    base: int                 # indice del filamento base in fids
    fa: np.ndarray            # (N,) indice filamento medio, -1 = nessuno
    a: np.ndarray             # (N,) strati medi
    fb: np.ndarray            # (N,) indice filamento superiore, -1 = nessuno
    b: np.ndarray
    lin: np.ndarray           # (N,3) colore previsto sRGB lineare
    lab: np.ndarray           # (N,3)

    def __len__(self):
        return len(self.a)


def build_recipes(model: OpticalModel, plan: LayerPlan, fids: list[str], base: int,
                  step: int = 1) -> RecipeSet:
    B = plan.base_layers
    M = plan.max_layers - B
    lh = plan.heights
    # stack base
    r0 = np.full(3, model.background)
    for i in range(B):
        r0 = model.layer(fids[base], lh[i], r0)
    F = len(fids)
    counts = list(range(0, M + 1, step))
    if counts[-1] != M:
        counts.append(M)
    out_fa, out_a, out_fb, out_b, out_lin = [], [], [], [], []
    for ia in range(-1, F):
        if ia == base:
            continue  # = base più spessa: rappresentata da fb==base con fa==-1
        a_list = [0] if ia < 0 else [c for c in counts if c > 0]
        for a in a_list:
            r = r0
            for k in range(a):
                r = model.layer(fids[ia], lh[B + k], r)
            for ib in range(-1, F):
                if ib == ia:
                    continue
                b_list = [0] if ib < 0 else [c for c in counts if 0 < c <= M - a]
                rb = r
                done = 0
                for b in b_list:
                    for k in range(done, b):
                        rb = model.layer(fids[ib], lh[B + a + k], rb)
                    done = b
                    out_fa.append(ia); out_a.append(a); out_fb.append(ib); out_b.append(b); out_lin.append(rb)
                if ib < 0:
                    out_fa.append(ia); out_a.append(a); out_fb.append(-1); out_b.append(0); out_lin.append(r)
    lin = np.array(out_lin)
    lin, idx = np.unique(np.round(lin, 7), axis=0, return_index=True)
    sel = np.sort(idx)
    arr = lambda x: np.asarray(x)[sel]
    lin = np.array(out_lin)[sel]
    return RecipeSet(fids, base, arr(out_fa), arr(out_a), arr(out_fb), arr(out_b), lin, linear_to_lab(lin))


def _thickness(rs: RecipeSet) -> np.ndarray:
    return rs.a + rs.b


def nearest_recipes(target_lab: np.ndarray, rs: RecipeSet, k: int = 24,
                    thick_penalty: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Per ogni colore target (P,3) indice di ricetta e ΔE00."""
    tree = cKDTree(rs.lab)
    k = min(k, len(rs))
    q = np.round(target_lab * 2) / 2
    uq, inv = np.unique(q.reshape(-1, 3), axis=0, return_inverse=True)
    _, cand = tree.query(uq, k=k)
    cand = cand.reshape(len(uq), k)
    de = delta_e2000(uq[:, None, :], rs.lab[cand])
    cost = de + thick_penalty * _thickness(rs)[cand]
    j = cost.argmin(axis=1)
    best = cand[np.arange(len(uq)), j]
    idx = best[inv.reshape(-1)].reshape(target_lab.shape[:-1])
    return idx, delta_e2000(target_lab, rs.lab[idx])


def choose_filaments(model, plan, target_lab, fids, capacity, progress=None, seed=0):
    """Sceglie base e sottoinsieme di al più `capacity` filamenti (greedy sul ΔE medio)."""
    targets, w = representative_colors(target_lab, 64, seed)
    L = np.array([model.layers[f].r_inf.mean() for f in fids])
    base = int(np.argmax(L))  # base chiara: massima riflettanza, sfondo per le velature
    chosen = [base]
    remaining = [i for i in range(len(fids)) if i != base]

    def score(idx_list):
        sub = [fids[i] for i in idx_list]
        rs = build_recipes(model, plan, sub, 0, step=2)
        _, de = nearest_recipes(targets, rs)
        return float(np.sum(w * de))

    cur = score(chosen)
    while remaining and len(chosen) < max(1, capacity):
        trials = [(score(chosen + [i]), i) for i in remaining]
        s, i = min(trials)
        if progress:
            progress(0.2 + 0.5 * len(chosen) / max(capacity, 1), f"Scelta filamenti: {len(chosen) + 1}")
        if s > cur - 0.05:
            break
        cur = s
        chosen.append(i)
        remaining.remove(i)
    return [fids[i] for i in chosen], cur


def objects_intervals(rs: RecipeSet, idx: np.ndarray, B: int):
    """{(ruolo, filamento): (lo, hi)} per ogni oggetto, quote in strati assoluti."""
    fa, a, fb, b = rs.fa[idx], rs.a[idx], rs.fb[idx], rs.b[idx]
    objs = {}
    zero = np.zeros(idx.shape, dtype=np.int64)
    base_hi = np.full(idx.shape, B, dtype=np.int64)
    objs[("base", rs.fids[rs.base])] = (zero, base_hi)
    for j, f in enumerate(rs.fids):
        m = fa == j
        if m.any():
            objs[("medio", f)] = (np.where(m, B, 0), np.where(m, B + a, 0))
        m = fb == j
        if m.any():
            lo = B + np.where(fa >= 0, a, 0)
            objs[("sopra", f)] = (np.where(m, lo, 0), np.where(m, lo + b, 0))
    return objs


def clean_recipes(idx: np.ndarray, rs: RecipeSet, B: int, target_lab: np.ndarray,
                  passes: int = 2) -> tuple[np.ndarray, int, int]:
    """Rimuove i pixel isolati (non stampabili) assegnando la ricetta del vicino
    con ΔE minore. La validità geometrica è garantita da `interval_mesh`.
    Ritorna (indici, celle modificate, 0)."""
    idx = idx.copy()
    changed = 0
    for _ in range(passes):
        idx, n = _clean_isolated(idx, rs, target_lab)
        changed += n
    return idx, changed, 0


def _clean_isolated(idx, rs, target_lab):
    changed = 0
    # 1) pixel isolati (nessun vicino a 4 con la stessa ricetta)
    P = np.pad(idx, 1, mode="edge")
    c = P[1:-1, 1:-1]
    nb = [P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]]
    iso = np.all([n != c for n in nb], axis=0)
    if iso.any():
        best = nb[0].copy()
        bestde = delta_e2000(target_lab, rs.lab[nb[0]])
        for n in nb[1:]:
            d = delta_e2000(target_lab, rs.lab[n])
            better = d < bestde
            best[better], bestde[better] = n[better], d[better]
        idx[iso] = best[iso]
        changed += int(iso.sum())
    return idx, changed


def build_objects(rs: RecipeSet, idx: np.ndarray, plan: LayerPlan, pixel_mm: float):
    """Mesh chiusa e manifold per ogni (ruolo, filamento)."""
    return {k: interval_mesh(lo, hi, pixel_mm, plan.z_of_level)
            for k, (lo, hi) in objects_intervals(rs, idx, plan.base_layers).items()}
