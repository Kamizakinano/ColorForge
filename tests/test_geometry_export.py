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


def test_purge_cost_and_slice_info(db, tmp_path):
    import zipfile
    from colorforge.export.bambu import BambuTemplate, cost_estimate, purge_estimate, read_slice_info
    from colorforge.pipeline import JobSettings, run
    from tests.conftest import make_image
    fids = ["generic-pla-white", "generic-pla-red", "generic-pla-blue"]
    res = run(make_image("saturated"), db, JobSettings(filament_ids=fids, width_mm=20, pixel_mm=0.5, quality="veloce"))
    mat = [0, 100, 200, 100, 0, 300, 200, 300, 0]
    st = {"filament_colour": ["#FFFFFF", "#FF0000", "#0000FF"], "flush_volumes_matrix": [str(x) for x in mat],
          "filament_cost": ["20", "25", "30"], "filament_density": ["1.24"] * 3, "nozzle_diameter": ["0.4"]}
    t1 = BambuTemplate("t", st, "02.08.02.60")
    slots = {f: i + 1 for i, f in enumerate(fids)}
    one = purge_estimate(res, t1, slots)
    two = purge_estimate(res, BambuTemplate("t", {**st, "nozzle_diameter": ["0.4", "0.4"]}, ""), slots)
    assert one["spurgo_g"] > 0 and two["spurgo_g"] < one["spurgo_g"] and two["ugello_secondario_slot"]
    c = cost_estimate(res, t1, slots, db)
    assert c["costo_eur"] > 0
    p = tmp_path / "s.3mf"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("Metadata/slice_info.config", '<config><plate><metadata key="index" value="1"/>'
                   '<metadata key="prediction" value="3600"/><metadata key="weight" value="12.5"/>'
                   '<filament id="2" type="PLA" color="#FF0000" used_m="1.5" used_g="4.0"/></plate></config>')
        z.writestr("Metadata/project_settings.config", '{"filament_cost": ["20", "25"]}')
    info = read_slice_info(p)
    assert info["piatti"][0]["tempo_s"] == 3600 and info["piatti"][0]["filamenti"][0]["costo_eur"] == 0.1


def test_bambu_export_never_mixes_materials(db, tmp_path):
    import pytest
    from colorforge.export.bambu import BambuTemplate, map_to_template_slots
    from colorforge.pipeline import JobSettings, export_3mf, palette_filaments, run
    from colorforge.filaments.database import FilamentDatabase
    from tests.conftest import make_image
    st = {"filament_colour": ["#FFFFFF", "#F72323", "#1E88E5", "#000000"],
          "filament_type": ["PLA", "PLA", "PETG", "PLA"], "printable_area": ["0x0", "256x0", "256x256", "0x256"]}
    t = BambuTemplate("t", st, "02.08.02.60")
    # un blu dell'immagine non deve finire sullo slot PETG (3) se la stampa è PLA
    assert map_to_template_slots([("img-1E88E5", "#1E88E5")], t, material="PLA")["img-1E88E5"][0] != 3
    fil = palette_filaments(["#FFFFFF", "#F72323", "#1E88E5", "#000000"], "PLA")
    pdb = FilamentDatabase(fil)
    res = run(make_image("saturated"), pdb, JobSettings(filament_ids=[f.id for f in fil], width_mm=20,
                                                        pixel_mm=0.5, quality="veloce", material="PLA"))
    export_3mf(res, tmp_path / "ok.3mf", template=t)          # nessun PETG usato
    with pytest.raises(ValueError, match="Nessun filamento PETG"):
        run(make_image("saturated"), pdb, JobSettings(filament_ids=[f.id for f in fil], material="PETG"))
    from colorforge.export.bambu import write_bambu_project
    from colorforge.geometry.mesh import column_mesh
    m = column_mesh(np.full((4, 4), 3), 1.0, z)
    with pytest.raises(ValueError, match="materiali diversi"):
        write_bambu_project([("a", 1, m), ("b", 3, m)], tmp_path / "bad.3mf", t, 0.08, 0.16)
