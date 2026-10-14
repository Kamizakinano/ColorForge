import numpy as np
import trimesh

from colorforge.export.stl import write_stl
from colorforge.export.threemf import read_3mf_meshes, write_3mf
from colorforge.geometry.mesh import column_mesh, fix_saddles, validate


def z(l):
    return np.asarray(l) * 0.1


def test_random_heightmap_is_watertight():
    for seed in range(3):
        lv = np.random.default_rng(seed).integers(0, 8, (25, 31))
        lv, _ = fix_saddles(lv)
        m = column_mesh(lv, 0.4, z)
        assert validate(m) == []
        t = trimesh.Trimesh(m.vertices, m.faces, process=False)
        assert t.is_watertight and t.is_winding_consistent
        assert abs(m.volume - lv.sum() * 0.4 * 0.4 * 0.1) < 1e-6


def test_dimensions():
    m = column_mesh(np.full((10, 20), 5), 0.5, z)
    (x0, y0, z0), (x1, y1, z1) = m.bounds
    assert np.allclose([x1 - x0, y1 - y0, z1 - z0], [10, 5, 0.5])


def test_stl_and_3mf_roundtrip(tmp_path):
    m = column_mesh(np.full((4, 4), 3), 1.0, z)
    write_stl(m, tmp_path / "a.stl")
    t = trimesh.load(tmp_path / "a.stl")
    assert t.is_watertight and abs(t.volume - 4.8) < 1e-4
    write_3mf([("base", "#000000", 1, m), ("top", "#FFFFFF", 2, m)], tmp_path / "a.3mf")
    objs = read_3mf_meshes(tmp_path / "a.3mf")
    assert [o[0] for o in objs] == ["base", "top"] and len(objs[0][2]) == len(m.faces)
