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


def test_bambu_project_from_user_template(tmp_path):
    """Progetto Bambu generato dal modello: struttura come Bambu Studio 2.8.2.60."""
    import json
    import zipfile
    import pytest
    from colorforge.export.bambu import BambuTemplate, map_to_template_slots, write_bambu_project
    settings = {"version": "02.08.02.60", "printer_model": "Bambu Lab X2D", "layer_height": "0.1",
                "filament_colour": ["#FFFFFF", "#000000", "#F72323"], "filament_type": ["PLA"] * 3,
                "printable_area": ["0x0", "256x0", "256x256", "0x256"]}
    t = BambuTemplate("x.3mf", settings, "02.08.02.60")
    m = column_mesh(np.full((6, 8), 4), 1.0, z)
    out = tmp_path / "b.3mf"
    write_bambu_project([("bianco", 1, m), ("rosso", 3, m)], out, t, 0.08, 0.16, np.ones((5, 5, 3)))
    zf = zipfile.ZipFile(out)
    ps = json.loads(zf.read("Metadata/project_settings.config"))
    assert ps["layer_height"] == "0.08" and ps["sparse_infill_density"] == "100%"
    ms = zf.read("Metadata/model_settings.config").decode()
    assert ms.count('key="extruder" value="3"') == 1 and "<plate>" in ms
    model = zf.read("3D/3dmodel.model").decode()
    assert "BambuStudio-02.08.02.60" in model and "<components>" in model
    assert {o[0] for o in read_3mf_meshes(out)} >= {None}
    mp = map_to_template_slots([("ams-slot2-000000", "#000000"), ("img-F82020", "#F82020")], t)
    assert mp["ams-slot2-000000"][0] == 2 and mp["img-F82020"][0] == 3
    with pytest.raises(ValueError):
        write_bambu_project([("x", 9, m)], out, t, 0.08, 0.16)
