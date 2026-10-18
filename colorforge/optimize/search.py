"""Ricerca automatica di sequenza filamenti e quote di cambio colore.

Modello di stampa (come HueForge con cambi per strato): la quota z di un
pixel determina quanti strati ha; il filamento di ogni strato è globale e
cambia a quote fissate. Il colore di un pixel con n strati è quindi la
riflettanza dello stack 0..n calcolata dal modello ottico (LUT per quota).

Ricerca: beam search su (filamento, strato d'inizio) seguita da un
affinamento a discesa coordinata. Obiettivo: CIEDE2000 medio pesato dei
colori rappresentativi dell'immagine (k-means in CIELAB), più una piccola
penalità per ogni cambio colore che dipende dal profilo di qualità.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.cluster.vq import kmeans2

from colorforge.color.spaces import delta_e2000, linear_to_lab
from colorforge.optics.km import OpticalModel


@dataclass
class QualityProfile:
    name: str
    beam_width: int
    position_step: int
    clusters: int
    change_penalty: float   # ΔE equivalente per cambio colore
    refine_rounds: int

    @classmethod
    def get(cls, name: str) -> "QualityProfile":
        return QUALITY[name]


QUALITY = {
    "massima": QualityProfile("massima", 10, 1, 96, 0.0, 4),
    "bilanciata": QualityProfile("bilanciata", 6, 2, 64, 0.15, 2),
    "veloce": QualityProfile("veloce", 3, 3, 32, 0.4, 1),
}


@dataclass
class LayerPlan:
    """Geometria degli strati: spessori e numero di strati della base."""
    first_layer_mm: float
    layer_mm: float
    base_layers: int
    max_layers: int

    @property
    def heights(self) -> np.ndarray:
        h = np.full(self.max_layers, self.layer_mm)
        h[0] = self.first_layer_mm
        return h

    def z_of_level(self, level: np.ndarray | int) -> np.ndarray:
        level = np.asarray(level, dtype=float)
        return np.where(level <= 0, 0.0, self.first_layer_mm + (level - 1) * self.layer_mm)

    @classmethod
    def from_settings(cls, first, layer, base_mm, max_mm) -> "LayerPlan":
        base = max(1, int(round((base_mm - first) / layer)) + 1)
        top = int(round((max_mm - first) / layer)) + 1
        if top <= base:
            raise ValueError("Spessore massimo inferiore o uguale alla base")
        return cls(first, layer, base, top)


@dataclass
class Solution:
    sequence: list[tuple[str, int]]  # (filamento, strato iniziale 0-based)
    score: float
    mean_de: float
    lut_lin: np.ndarray = field(repr=False, default=None)
    lut_lab: np.ndarray = field(repr=False, default=None)

    @property
    def filaments(self) -> list[str]:
        return [f for f, _ in self.sequence]

    @property
    def changes(self) -> int:
        return len(self.sequence) - 1


def representative_colors(lab_pixels: np.ndarray, k: int, seed: int = 0,
                          pixel_weights: np.ndarray | None = None, balance: float = 0.0):
    """k colori rappresentativi (CIELAB) con pesi; deterministico.
    `pixel_weights` dà più importanza a zone (dettagli, soggetto)."""
    x = lab_pixels.reshape(-1, 3)
    pw = None if pixel_weights is None else pixel_weights.reshape(-1)
    rng = np.random.default_rng(seed)
    if len(x) > 40000:
        sel = rng.choice(len(x), 40000, replace=False)
        x = x[sel]
        pw = None if pw is None else pw[sel]
    k = min(k, len(np.unique(np.round(x, 1), axis=0)))
    cent, lbl = kmeans2(x, k, minit="++", seed=seed, iter=20)
    w = np.bincount(lbl, weights=pw, minlength=k).astype(float)
    if balance > 0:  # 0 = peso proporzionale ai pixel; 1 = ogni colore pesa uguale
        w = np.where(w > 0, w ** (1 - balance), 0)
    keep = w > 0
    return cent[keep], w[keep] / w.sum()


class Evaluator:
    def __init__(self, model: OpticalModel, plan: LayerPlan, targets: np.ndarray,
                 weights: np.ndarray, change_penalty: float):
        self.model, self.plan = model, plan
        self.targets, self.weights = targets, weights
        self.penalty = change_penalty
        self.cache: dict[tuple, Solution] = {}
        self.evaluations = 0

    def layer_fids(self, seq) -> list[str]:
        fids = []
        for i, (f, start) in enumerate(seq):
            end = seq[i + 1][1] if i + 1 < len(seq) else self.plan.max_layers
            fids += [f] * (end - start)
        return fids

    def evaluate(self, seq) -> Solution:
        key = tuple(seq)
        if key in self.cache:
            return self.cache[key]
        self.evaluations += 1
        lut = self.model.stack_lut(self.layer_fids(seq), self.plan.heights)
        lab = linear_to_lab(lut)
        cand = lab[self.plan.base_layers:]  # quote ammesse
        de = delta_e2000(self.targets[:, None, :], cand[None, :, :])
        mean_de = float(np.sum(self.weights * de.min(axis=1)))
        sol = Solution(list(seq), mean_de + self.penalty * (len(seq) - 1), mean_de, lut, lab)
        self.cache[key] = sol
        return sol


def optimize(model: OpticalModel, plan: LayerPlan, target_lab: np.ndarray,
             filament_ids: list[str], max_colors: int, quality: QualityProfile,
             progress: Callable[[float, str], None] | None = None,
             cancelled: Callable[[], bool] | None = None,
             seed: int = 0, monotonic: bool = False, balance: float = 0.0) -> tuple[Solution, list[Solution], Evaluator]:
    """Ritorna (migliore, candidate migliori distinte, valutatore).
    `monotonic`: come HueForge, ogni filamento sopra è più chiaro di quello sotto."""
    lum = {f: float(np.mean(model.layers[f].r_inf)) for f in filament_ids}
    targets, weights = representative_colors(target_lab, quality.clusters, seed, balance=balance)
    ev = Evaluator(model, plan, targets, weights, quality.change_penalty)
    progress = progress or (lambda p, m: None)
    cancelled = cancelled or (lambda: False)
    max_colors = max(1, min(max_colors, len(filament_ids)))

    starts = filament_ids
    if monotonic:  # base: i più scuri (la luce si costruisce verso l'alto)
        starts = sorted(filament_ids, key=lambda f: lum[f])[:max(2, len(filament_ids) // 2)]
    beam = sorted((ev.evaluate([(f, 0)]) for f in starts), key=lambda s: s.score)
    pool = list(beam)
    beam = beam[:quality.beam_width]
    for depth in range(1, max_colors):
        if cancelled():
            break
        progress(0.1 + 0.6 * depth / max_colors, f"Ricerca combinazioni: {depth + 1} colori")
        nxt = []
        for sol in beam:
            last_f, last_s = sol.sequence[-1]
            lo = max(plan.base_layers, last_s + 1)
            for start in range(lo, plan.max_layers, quality.position_step):
                for f in filament_ids:
                    if f != last_f and (not monotonic or (lum[f] > lum[last_f] and
                                                          f not in [x for x, _ in sol.sequence])):
                        nxt.append(ev.evaluate(sol.sequence + [(f, start)]))
        if not nxt:
            break
        uniq = {tuple(s.sequence): s for s in nxt}
        nxt = sorted(uniq.values(), key=lambda s: s.score)
        pool += nxt[:quality.beam_width * 3]
        # diversità: al massimo 2 varianti (quote diverse) per la stessa sequenza di filamenti,
        # altrimenti una buona combinazione parziale riempie il fascio e blocca le alternative
        seen, beam = {}, []
        for c in nxt:
            k = tuple(f for f, _ in c.sequence)
            if seen.get(k, 0) < 2:
                beam.append(c); seen[k] = seen.get(k, 0) + 1
            if len(beam) >= quality.beam_width:
                break

    pool = sorted({tuple(s.sequence): s for s in pool}.values(), key=lambda s: s.score)
    best = pool[0]
    progress(0.75, "Affinamento quote e filamenti")
    for _ in range(quality.refine_rounds):
        if cancelled():
            break
        improved = False
        for i in range(len(best.sequence)):
            for cand in _neighbours(best.sequence, i, filament_ids, plan):
                if monotonic and any(lum[cand[k + 1][0]] <= lum[cand[k][0]] for k in range(len(cand) - 1)):
                    continue
                s = ev.evaluate(cand)
                if s.score < best.score - 1e-6:
                    best, improved = s, True
        if not improved:
            break
    pool = sorted({tuple(s.sequence): s for s in pool + [best]}.values(), key=lambda s: s.score)
    return best, pool[:8], ev


def _neighbours(seq, i, fids, plan):
    f, s = seq[i]
    out = []
    if i > 0:
        lo = max(plan.base_layers, seq[i - 1][1] + 1)
        hi = seq[i + 1][1] - 1 if i + 1 < len(seq) else plan.max_layers - 1
        for d in (-3, -2, -1, 1, 2, 3):
            if lo <= s + d <= hi:
                out.append(seq[:i] + [(f, s + d)] + seq[i + 1:])
    for g in fids:
        if g != f and (i == 0 or seq[i - 1][0] != g) and (i + 1 >= len(seq) or seq[i + 1][0] != g):
            out.append(seq[:i] + [(g, s)] + seq[i + 1:])
    return out


def optimize_monotonic(model: OpticalModel, plan: LayerPlan, target_lab: np.ndarray,
                       filament_ids: list[str], max_colors: int, quality: QualityProfile,
                       progress=None, cancelled=None, seed: int = 0, balance: float = 0.5):
    """Ricerca completa per lo stile HueForge. L'ordine è fissato (dal più scuro al più
    chiaro), quindi si prova OGNI sottoinsieme di filamenti e per ciascuno si ottimizzano
    le quote dei cambi (partenza uniforme + discesa coordinata)."""
    from itertools import combinations
    targets, weights = representative_colors(target_lab, quality.clusters, seed, balance=balance)
    ev = Evaluator(model, plan, targets, weights, quality.change_penalty)
    progress = progress or (lambda p, m: None)
    cancelled = cancelled or (lambda: False)
    lum = {f: float(np.mean(model.layers[f].r_inf)) for f in filament_ids}
    order = sorted(filament_ids, key=lambda f: lum[f])
    B, top = plan.base_layers, plan.max_layers
    subsets = [c for k in range(2, min(max_colors, len(order)) + 1) for c in combinations(order, k)]
    results = []
    for n, sub in enumerate(subsets):
        if cancelled():
            break
        if n % 8 == 0:
            progress(0.1 + 0.65 * n / max(len(subsets), 1), f"Combinazioni di filamenti: {n + 1}/{len(subsets)}")
        k = len(sub)
        starts = [0] + [int(round(B + (top - B) * i / k)) for i in range(1, k)]
        for i in range(2, k):
            starts[i] = max(starts[i], starts[i - 1] + 1)
        cur = ev.evaluate(list(zip(sub, starts)))
        for _ in range(4 if quality.name == "massima" else 2):
            improved = False
            for i in range(1, k):
                for d in (-3, -2, -1, 1, 2, 3):
                    st = list(starts)
                    st[i] += d
                    lo = max(B, st[i - 1] + 1)
                    hi = (st[i + 1] - 1) if i + 1 < k else top - 1
                    if not lo <= st[i] <= hi:
                        continue
                    c = ev.evaluate(list(zip(sub, st)))
                    if c.score < cur.score - 1e-6:
                        cur, starts, improved = c, st, True
            if not improved:
                break
        results.append(cur)
    results.sort(key=lambda c: c.score)
    return results[0], results[:60], ev
