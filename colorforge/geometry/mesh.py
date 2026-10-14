"""Mesh a colonne (terrazzata) da una mappa di quote intere.

Ogni pixel è una colonna di base `px` × `px` mm alta `level` strati. Le pareti
verticali sono suddivise a ogni livello, così ogni spigolo è condiviso da
esattamente due triangoli (nessuna giunzione a T): la mesh è chiusa e
orientata in modo coerente. È esatta rispetto alla stampa a strati, perché
le quote sono già quantizzate all'altezza di strato.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray  # (V,3) float mm
    faces: np.ndarray     # (F,3) int

    @property
    def volume(self) -> float:
        v = self.vertices[self.faces]
        return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)

    @property
    def bounds(self) -> np.ndarray:
        return np.array([self.vertices.min(0), self.vertices.max(0)])


def column_mesh(levels: np.ndarray, pixel_mm: float, z_of_level, level_offset: int = 0) -> Mesh:
    """`levels[r, c]` = strati sopra `level_offset` (0 = vuoto)."""
    H = np.pad(np.asarray(levels, dtype=np.int64), 1)
    R, C = H.shape
    L = int(H.max()) + 1
    if L <= 1:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))

    def vid(r, c, l):
        return (r * (C + 1) + c) * L + l

    tris = []
    # superfici superiori e inferiori
    r, c = np.nonzero(H > 0)
    h = H[r, c]
    A, B, Cc, D = (r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)
    for lv, top in ((h, True), (np.zeros_like(h), False)):
        a, b, cc, d = vid(*A, lv), vid(*B, lv), vid(*Cc, lv), vid(*D, lv)
        if top:
            tris += [np.stack([a, d, cc], 1), np.stack([a, cc, b], 1)]
        else:
            tris += [np.stack([a, b, cc], 1), np.stack([a, cc, d], 1)]

    def walls(h1, h2, p, q, pos_normal_first):
        """Pareti tra celle adiacenti: p,q = estremi dello spigolo condiviso."""
        out = []
        for hi_, lo_, flip in ((h1, h2, False), (h2, h1, True)):
            d = hi_ - lo_
            m = d > 0
            if not m.any():
                continue
            d = d[m]
            idx = np.repeat(np.arange(d.size), d)
            lvl = np.repeat(lo_[m], d) + (np.arange(d.sum()) - np.repeat(np.cumsum(d) - d, d))
            pr, pc = p[0][m][idx], p[1][m][idx]
            qr, qc = q[0][m][idx], q[1][m][idx]
            P1, P2 = vid(pr, pc, lvl), vid(qr, qc, lvl)
            P3, P4 = vid(qr, qc, lvl + 1), vid(pr, pc, lvl + 1)
            if pos_normal_first != flip:
                out += [np.stack([P1, P3, P2], 1), np.stack([P1, P4, P3], 1)]
            else:
                out += [np.stack([P1, P2, P3], 1), np.stack([P1, P3, P4], 1)]
        return out

    # pareti tra (r,c) e (r,c+1): spigolo x=c+1, da riga r a r+1. Normale +x se sinistra più alta.
    rr, cc_ = np.mgrid[0:R, 0:C - 1]
    rr, cc_ = rr.ravel(), cc_.ravel()
    tris += walls(H[rr, cc_], H[rr, cc_ + 1], (rr, cc_ + 1), (rr + 1, cc_ + 1), True)
    # pareti tra (r,c) e (r+1,c): spigolo y=r+1, da colonna c a c+1. Normale -y se sopra più alta.
    rr, cc_ = np.mgrid[0:R - 1, 0:C]
    rr, cc_ = rr.ravel(), cc_.ravel()
    tris += walls(H[rr, cc_], H[rr + 1, cc_], (rr + 1, cc_), (rr + 1, cc_ + 1), False)

    faces = np.concatenate(tris)
    used, inv = np.unique(faces, return_inverse=True)
    faces = inv.reshape(-1, 3)
    l = used % L
    rc = used // L
    vr, vc = rc // (C + 1), rc % (C + 1)
    x = (vc - 1) * pixel_mm
    y = (R - 2 - (vr - 1)) * pixel_mm  # riga 0 dell'immagine in alto (y massimo)
    z = z_of_level(l + level_offset)
    verts = np.stack([x, y, z], 1).astype(np.float64)
    return Mesh(verts, faces.astype(np.int64))


def fix_saddles(levels: np.ndarray, max_iter: int = 200) -> tuple[np.ndarray, int]:
    """Elimina le configurazioni "a scacchiera" 2x2 (due celle piene in diagonale
    allo stesso livello), che produrrebbero spigoli verticali condivisi da 4
    facce (non-manifold). Alza di uno strato la cella più bassa della coppia
    vuota: modifica minima, il colore previsto viene poi ricalcolato.
    Ritorna (livelli corretti, numero di celle modificate)."""
    H = np.pad(np.asarray(levels, dtype=np.int64).copy(), 1)
    changed = 0
    for _ in range(max_iter):
        a, b = H[:-1, :-1], H[:-1, 1:]
        c, d = H[1:, :-1], H[1:, 1:]
        # checkerboard a qualche livello: min(a,d) > max(b,c) oppure min(b,c) > max(a,d)
        m1 = np.minimum(a, d) > np.maximum(b, c)
        m2 = np.minimum(b, c) > np.maximum(a, d)
        if not (m1.any() or m2.any()):
            break
        for mask, (p, q, op, oq) in ((m1, ((0, 1), (1, 0), b, c)), (m2, ((0, 0), (1, 1), a, d))):
            rr, cc = np.nonzero(mask)
            if rr.size == 0:
                continue
            lowp = op[rr, cc] <= oq[rr, cc]
            dr = np.where(lowp, p[0], q[0])
            dc = np.where(lowp, p[1], q[1])
            tr, tc = rr + dr, cc + dc
            # celle esterne (padding) non si toccano: alza invece l'altra
            border = (tr == 0) | (tc == 0) | (tr == H.shape[0] - 1) | (tc == H.shape[1] - 1)
            tr = np.where(border, rr + np.where(lowp, q[0], p[0]), tr)
            tc = np.where(border, cc + np.where(lowp, q[1], p[1]), tc)
            ok = ~((tr == 0) | (tc == 0) | (tr == H.shape[0] - 1) | (tc == H.shape[1] - 1))
            tr, tc = np.unique(np.stack([tr[ok], tc[ok]], 1), axis=0).T if ok.any() else (tr[:0], tc[:0])
            H[tr, tc] += 1
            changed += tr.size
        # celle di bordo in saddle con il padding: impossibile (padding = 0 è sempre "basso")
    return H[1:-1, 1:-1], changed


def validate(mesh: Mesh) -> list[str]:
    """Controlli di integrità: chiusura (ogni spigolo in 2 facce con versi
    opposti), triangoli degeneri, volume positivo. Ritorna i problemi."""
    problems = []
    f = mesh.faces
    if len(f) == 0:
        return ["Mesh vuota"]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    directed = e[:, 0] * (len(mesh.vertices) + 1) + e[:, 1]
    _, cnt = np.unique(directed, return_counts=True)
    if (cnt > 1).any():
        problems.append(f"{int((cnt > 1).sum())} spigoli orientati duplicati (orientamento incoerente)")
    und = np.sort(e, axis=1)
    _, cu = np.unique(und[:, 0] * (len(mesh.vertices) + 1) + und[:, 1], return_counts=True)
    if (cu != 2).any():
        problems.append(f"{int((cu != 2).sum())} spigoli non condivisi da due facce (mesh non chiusa)")
    v = mesh.vertices[f]
    area = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    if (area < 1e-12).any():
        problems.append(f"{int((area < 1e-12).sum())} triangoli degeneri")
    if mesh.volume <= 0:
        problems.append("Volume non positivo (normali invertite)")
    return problems
