"""Scrittura STL binario."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from colorforge.geometry.mesh import Mesh


def write_stl(mesh: Mesh, path: str | Path, name: str = "ColorForge") -> None:
    v = mesh.vertices[mesh.faces].astype(np.float32)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(v), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, v
    with open(path, "wb") as fh:
        fh.write(name.encode("ascii", "replace")[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(v)))
        fh.write(rec.tobytes())
