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


def _component_lut() -> np.ndarray:
    """LUT[pattern 8 bit, voxel 0..7] -> etichetta della componente connessa per
    facce del voxel nel blocco 2x2x2 attorno a un vertice della griglia."""
    lut = np.zeros((256, 8), dtype=np.int64)
    for p in range(256):
        parent = list(range(8))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a in range(8):
            for bit in (1, 2, 4):
                b = a ^ bit
                if a < b and p >> a & 1 and p >> b & 1:
                    parent[find(b)] = find(a)
        for a in range(8):
            lut[p, a] = min(x for x in range(8) if find(x) == find(a)) if p >> a & 1 else 0
    return lut


_LUT = _component_lut()


def interval_mesh(lo: np.ndarray, hi: np.ndarray, pixel_mm: float, z_of_level) -> Mesh:
    """Mesh di colonne con intervallo variabile [lo, hi) per pixel (hi <= lo = vuoto).

    Superficie di un insieme di voxel (cella, strato). Le facce sono suddivise a
    ogni livello (nessuna giunzione a T) e ogni vertice della griglia è sdoppiato
    per componente connessa dei voxel che lo toccano (blocco 2x2x2): anche dove
    due colonne si toccano solo lungo uno spigolo la mesh resta 2-manifold.
    """
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    present = hi > lo
    LO = np.pad(np.where(present, lo, 0), 1)
    HI = np.pad(np.where(present, hi, 0), 1)
    P = np.pad(present, 1)
    R, C = HI.shape
    L = int(HI.max()) + 1
    if not present.any():
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))

    tri_corners = []   # (n, 3, 3) angoli (r, c, l) della griglia
    tri_owner = []     # (n, 3) voxel proprietario (r, c, l)

    def emit(corners, owner):
        tri_corners.append(np.stack([np.stack(k, -1) for k in corners], 1))
        tri_owner.append(np.stack(owner, -1))

    r, c = np.nonzero(P)
    th, bl = HI[r, c], LO[r, c]
    for lv, own, top in ((th, th - 1, True), (bl, bl, False)):
        A, B, Cc, D = (r, c, lv), (r, c + 1, lv), (r + 1, c + 1, lv), (r + 1, c, lv)
        o = (r, c, own)
        if top:
            emit((A, D, Cc), o); emit((A, Cc, B), o)
        else:
            emit((A, B, Cc), o); emit((A, Cc, D), o)

    def segs(lo1, hi1, p1, lo2, hi2, p2):
        e1 = np.where(p2, np.minimum(hi1, lo2), hi1)
        s2 = np.where(p2, np.maximum(lo1, hi2), hi1)
        return ((lo1, np.where(p1, np.maximum(e1 - lo1, 0), 0)),
                (s2, np.where(p1 & p2, np.maximum(hi1 - s2, 0), 0)))

    def wall(start, cnt, p, q, own, outward_first):
        m = cnt > 0
        if not m.any():
            return
        d = cnt[m]
        idx = np.repeat(np.arange(d.size), d)
        lvl = np.repeat(start[m], d) + (np.arange(d.sum()) - np.repeat(np.cumsum(d) - d, d))
        pr, pc = p[0][m][idx], p[1][m][idx]
        qr, qc = q[0][m][idx], q[1][m][idx]
        o = (own[0][m][idx], own[1][m][idx], lvl)
        P1, P2, P3, P4 = (pr, pc, lvl), (qr, qc, lvl), (qr, qc, lvl + 1), (pr, pc, lvl + 1)
        if outward_first:
            emit((P1, P3, P2), o); emit((P1, P4, P3), o)
        else:
            emit((P1, P2, P3), o); emit((P1, P3, P4), o)

    for horiz in (True, False):
        if horiz:
            rr, cc = (x.ravel() for x in np.mgrid[0:R, 0:C - 1])
            i1, i2 = (rr, cc), (rr, cc + 1)
            p, q = (rr, cc + 1), (rr + 1, cc + 1)
        else:
            rr, cc = (x.ravel() for x in np.mgrid[0:R - 1, 0:C])
            i1, i2 = (rr, cc), (rr + 1, cc)
            p, q = (rr + 1, cc), (rr + 1, cc + 1)
        l1, h1, q1 = LO[i1], HI[i1], P[i1]
        l2, h2, q2 = LO[i2], HI[i2], P[i2]
        for st, n in segs(l1, h1, q1, l2, h2, q2):
            wall(st, n, p, q, i1, horiz)
        for st, n in segs(l2, h2, q2, l1, h1, q1):
            wall(st, n, p, q, i2, not horiz)

    corners = np.concatenate(tri_corners)           # (n,3,3)
    owner = np.concatenate(tri_owner)                # (n,3)
    vr, vc, vl = corners[..., 0], corners[..., 1], corners[..., 2]

    def occ(r_, c_, l_):
        ok = (r_ >= 0) & (r_ < R) & (c_ >= 0) & (c_ < C)
        r2, c2 = np.clip(r_, 0, R - 1), np.clip(c_, 0, C - 1)
        return ok & P[r2, c2] & (LO[r2, c2] <= l_) & (l_ < HI[r2, c2])

    pattern = np.zeros(vr.shape, dtype=np.int64)
    for bit in range(8):
        dr, dc, dl = bit >> 2 & 1, bit >> 1 & 1, bit & 1
        pattern |= occ(vr - 1 + dr, vc - 1 + dc, vl - 1 + dl).astype(np.int64) << bit
    obit = ((owner[:, None, 0] - (vr - 1)) << 2) | ((owner[:, None, 1] - (vc - 1)) << 1) | (owner[:, None, 2] - (vl - 1))
    label = _LUT[pattern, obit]
    key = (((vr * (C + 1) + vc) * L + vl) << 3) | label
    used, inv = np.unique(key.ravel(), return_inverse=True)
    faces = inv.reshape(-1, 3)
    g = used >> 3
    l = g % L
    rc = g // L
    gr, gc = rc // (C + 1), rc % (C + 1)
    x = (gc - 1) * pixel_mm
    y = (R - 2 - (gr - 1)) * pixel_mm
    verts = np.stack([x, y, z_of_level(l)], 1).astype(np.float64)
    # voxel vuoto verso cui guarda ogni faccia (proprietario + normale)
    tv = verts[faces]
    nrm = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    ax = np.abs(nrm).argmax(1)
    sg = np.sign(nrm[np.arange(len(nrm)), ax]).astype(np.int64)
    d_r = np.where(ax == 1, -sg, 0)  # y cresce verso le righe più basse dell'immagine
    d_c = np.where(ax == 0, sg, 0)
    d_l = np.where(ax == 2, sg, 0)
    facing = ((owner[:, 0] + d_r) * (C + 2) + owner[:, 1] + d_c) * (L + 2) + owner[:, 2] + d_l
    verts, faces = _split_nonmanifold(verts, faces.astype(np.int64), facing)
    return Mesh(verts, faces)


def _split_nonmanifold(verts, faces, owner):
    """Separa le superfici agli spigoli condivisi da 4 facce (voxel in diagonale):
    attorno ai vertici coinvolti le facce vengono raggruppate in "ombrelli"
    (facce unite da spigoli accoppiati; a 4 facce si accoppiano quelle rivolte
    verso lo stesso voxel vuoto) e ogni ombrello riceve una copia del vertice."""
    n = len(verts) + 1
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    fidx = np.tile(np.arange(len(faces)), 3)
    und = np.sort(e, axis=1)
    key = und[:, 0] * n + und[:, 1]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    bad = cnt[inv] > 2
    if not bad.any():
        return verts, faces
    bad_v = np.unique(und[bad])
    bad_set = set(bad_v.tolist())
    # facce incidenti ai vertici problematici
    inc = {}
    sel = np.nonzero(np.any(np.isin(faces, bad_v), axis=1))[0]
    for f_i in sel.tolist():
        for v in faces[f_i].tolist():
            if v in bad_set:
                inc.setdefault(v, []).append(f_i)
    verts = list(map(tuple, verts))
    faces = faces.copy()
    for v, fl in inc.items():
        parent = {f: f for f in fl}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        # spigoli (v, w) e facce che li usano
        by_edge = {}
        for f in fl:
            for w in faces[f].tolist():
                if w != v:
                    by_edge.setdefault(w, []).append(f)
        for w, fs in by_edge.items():
            if len(fs) == 2:
                parent[find(fs[0])] = find(fs[1])
            elif len(fs) == 4:
                for i in range(4):
                    for j in range(i + 1, 4):
                        if owner[fs[i]] == owner[fs[j]]:
                            parent[find(fs[i])] = find(fs[j])
            else:
                for f in fs[1:]:
                    parent[find(f)] = find(fs[0])
        groups = {}
        for f in fl:
            groups.setdefault(find(f), []).append(f)
        for g in list(groups.values())[1:]:
            nv = len(verts)
            verts.append(verts[v])
            for f in g:
                row = faces[f]
                row[row == v] = nv
    return np.array(verts, dtype=np.float64), faces


def interval_saddles(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Blocchi 2x2 (indice in alto a sinistra) che a qualche livello hanno
    occupazione a scacchiera: produrrebbero spigoli non-manifold."""
    pres = hi > lo
    LO = np.pad(np.where(pres, lo, 0), 1)
    HI = np.pad(np.where(pres, hi, 0), 1)

    def ov(a, b):  # intervalli (lo,hi) che si sovrappongono
        return np.minimum(a[1], b[1]) > np.maximum(a[0], b[0])

    c = [(LO[:-1, :-1], HI[:-1, :-1]), (LO[:-1, 1:], HI[:-1, 1:]),
         (LO[1:, :-1], HI[1:, :-1]), (LO[1:, 1:], HI[1:, 1:])]
    out = np.zeros(LO[:-1, :-1].shape, dtype=bool)
    for (d1, d2), (o1, o2) in (((0, 3), (1, 2)), ((1, 2), (0, 3))):
        # livelli in cui d1 e d2 sono entrambi pieni
        s = np.maximum(c[d1][0], c[d2][0])
        e = np.minimum(c[d1][1], c[d2][1])
        has = e > s
        # o1 e o2 entrambi vuoti in almeno un livello di [s,e): basta verificare che
        # l'unione delle coperture di o1,o2 non copra tutto [s,e)
        cov = np.zeros_like(has)
        a1, b1 = np.clip(c[o1][0], s, e), np.clip(c[o1][1], s, e)
        a2, b2 = np.clip(c[o2][0], s, e), np.clip(c[o2][1], s, e)
        # ordinati per inizio
        first_a = np.minimum(a1, a2)
        sw = a2 < a1
        fa, fb = np.where(sw, a2, a1), np.where(sw, b2, b1)
        ga, gb = np.where(sw, a1, a2), np.where(sw, b1, b2)
        cov = (fa <= s) & ((fb >= e) | ((ga <= fb) & (np.maximum(fb, gb) >= e)))
        out |= has & ~cov
        del first_a
    return out


def interval_touching(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coppie di celle adiacenti i cui intervalli si toccano solo in un piano
    (hi di una = lo dell'altra): spigolo condiviso da 4 facce.
    Ritorna (maschera orizzontale (R, C-1), maschera verticale (R-1, C))."""
    pres = hi > lo

    def t(l1, h1, p1, l2, h2, p2):
        return p1 & p2 & ((h1 == l2) | (h2 == l1))
    hm = t(lo[:, :-1], hi[:, :-1], pres[:, :-1], lo[:, 1:], hi[:, 1:], pres[:, 1:])
    vm = t(lo[:-1], hi[:-1], pres[:-1], lo[1:], hi[1:], pres[1:])
    return hm, vm


def bad_edge_points(mesh: Mesh) -> np.ndarray:
    """Coordinate (x, y) dei vertici degli spigoli non condivisi da esattamente
    due facce con verso opposto."""
    f = mesh.faces
    if len(f) == 0:
        return np.zeros((0, 2))
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    n = len(mesh.vertices) + 1
    und = np.sort(e, axis=1)
    ku = und[:, 0] * n + und[:, 1]
    kd = e[:, 0] * n + e[:, 1]
    _, iu, cu = np.unique(ku, return_inverse=True, return_counts=True)
    _, idd, cd = np.unique(kd, return_inverse=True, return_counts=True)
    bad = (cu[iu] != 2) | (cd[idd] > 1)
    return mesh.vertices[np.unique(e[bad])][:, :2]
