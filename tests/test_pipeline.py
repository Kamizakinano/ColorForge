import numpy as np
import pytest
from PIL import Image

from colorforge.geometry.mesh import validate
from colorforge.imaging.importer import ImageImportError, from_pil, load_image
from colorforge.pipeline import JobSettings, export_all, run
from tests.conftest import make_image

FIDS = ["generic-pla-black", "generic-pla-white", "generic-pla-red", "generic-pla-blue", "generic-pla-yellow"]


def settings(**kw):
    d = dict(filament_ids=FIDS, width_mm=30, pixel_mm=0.5, max_colors=4, quality="veloce")
    d.update(kw)
    return JobSettings(**d)


@pytest.mark.parametrize("mode", ["zone", "layers"])
@pytest.mark.parametrize("kind", ["gradient", "bw", "saturated", "many"])
def test_full_pipeline(db, kind, mode, tmp_path):
    res = run(make_image(kind), db, settings(mode=mode))
    s = res.settings
    assert res.mesh_problems == []
    assert res.levels.min() >= res.plan.base_layers and res.levels.max() <= res.plan.max_layers
    assert res.levels.shape == (40, 60) or res.levels.shape[1] == 60
    assert res.metrics["tipo"].startswith("previsione")
    assert res.reliability < 0.5 and any("TD sconosciuta" in w for w in res.warnings)
    assert len({f for f in res.solution.filaments}) <= s.max_colors
    files = export_all(res, tmp_path)
    assert all(__import__("os").path.getsize(p) > 0 for p in files.values())


def test_zone_mode_reproduces_distinct_colors(db):
    """Rosso, verde e blu affiancati: per zona devono uscire tutti, a strati no."""
    fids = ["generic-pla-white", "generic-pla-red", "generic-pla-green", "generic-pla-blue"]
    z = run(make_image("saturated"), db, settings(filament_ids=fids, mode="zone"))
    l = run(make_image("saturated"), db, settings(filament_ids=fids, mode="layers"))
    assert z.metrics["delta_e2000_medio"] < l.metrics["delta_e2000_medio"]
    assert z.mesh_problems == [] and len(z.objects) >= 3


def test_gradient_black_white_is_accurate(db):
    res = run(make_image("gradient"), db, settings(filament_ids=FIDS[:2], quality="bilanciata"))
    assert res.metrics["delta_e2000_medio"] < 6


def test_repeatable(db):
    a = run(make_image("many"), db, settings())
    b = run(make_image("many"), db, settings())
    assert a.solution.sequence == b.solution.sequence and np.array_equal(a.levels, b.levels)


def test_too_large_for_printer(db):
    with pytest.raises(ValueError, match="volume"):
        run(make_image("bw"), db, settings(width_mm=400))


def test_transparency_and_bad_file(tmp_path):
    a = np.zeros((10, 10, 4), np.uint8); a[..., 3] = 0; a[:5, :, 3] = 255
    img = from_pil(Image.fromarray(a, "RGBA"))
    assert img.alpha is not None and img.alpha.min() == 0
    bad = tmp_path / "rotta.png"; bad.write_bytes(b"non un png")
    with pytest.raises(ImageImportError):
        load_image(bad)


def test_palette_and_palette_filaments(db, tmp_path):
    from colorforge.filaments.database import FilamentDatabase
    from colorforge.pipeline import export_3mf, image_palette, palette_filaments
    from colorforge.export.threemf import read_3mf_meshes
    img = make_image("saturated")
    pal = image_palette(img.srgb, k=6)
    assert abs(sum(w for _, w in pal) - 1) < 1e-6 and pal[0][1] >= pal[-1][1]
    fil = palette_filaments([h for h, _ in pal[:3]] + ["#FFFFFF"])
    pdb = FilamentDatabase(fil)
    res = run(img, pdb, settings(filament_ids=[f.id for f in fil], max_colors=4))
    assert set(res.solution.filaments) <= {f.id for f in fil}
    probs = export_3mf(res, tmp_path / "x.3mf")
    assert all(v == [] for v in probs.values())
    assert len(read_3mf_meshes(tmp_path / "x.3mf")) == len(probs)


def test_suggest_purchases_finds_missing_color(db):
    import numpy as np
    from colorforge.color.spaces import srgb_to_lab
    from colorforge.imaging.preprocess import resize_to_grid
    from colorforge.optics.km import OpticalModel
    from colorforge.optimize.regional import suggest_purchases
    from colorforge.optimize.search import LayerPlan
    img = make_image("saturated").srgb  # rosso, verde, blu
    owned = ["generic-pla-white", "generic-pla-red", "generic-pla-blue"]
    cands = ["generic-pla-green", "generic-pla-yellow", "generic-pla-brown"]
    fil = [db.get(f) for f in owned + cands]
    m = OpticalModel(fil)
    plan = LayerPlan.from_settings(0.16, 0.08, 0.64, 2.4)
    base, cur, sug = suggest_purchases(m, plan, srgb_to_lab(img), owned, cands, capacity=8)
    assert sug and sug[0][0] == "generic-pla-green" and sug[0][3] > 1


def test_relief_raises_subject_and_stays_manifold(db):
    import numpy as np
    from PIL import Image
    from colorforge.imaging.importer import from_pil
    a = np.full((50, 50, 3), 235, np.uint8)
    yy, xx = np.mgrid[0:50, 0:50]
    a[(yy - 25) ** 2 + (xx - 25) ** 2 < 15 ** 2] = (200, 30, 40)   # disco rosso su sfondo chiaro
    img = from_pil(Image.fromarray(a))
    flat = run(img, db, settings(width_mm=25, mode="zone"))
    rel = run(img, db, settings(width_mm=25, mode="zone", relief_mm=3.0))
    assert rel.mesh_problems == []
    assert rel.levels[25, 25] - flat.levels[25, 25] >= int(2.5 / 0.08)   # centro rialzato
    assert abs(int(rel.levels[2, 2]) - int(flat.levels[2, 2])) <= 1        # sfondo invariato
    assert np.allclose(rel.simulated_srgb, flat.simulated_srgb)            # colori identici


def test_accessories_only_when_requested(db, tmp_path):
    from colorforge.export.threemf import read_3mf_meshes
    from colorforge.pipeline import export_3mf
    base = run(make_image("many"), db, settings())
    n0 = len(read_3mf_meshes(_x(export_3mf, base, tmp_path / "a.3mf")))
    acc = run(make_image("many"), db, settings(add_frame=True, add_hook=True, add_stand=True))
    probs = export_3mf(acc, tmp_path / "b.3mf")
    names = [o[0] for o in read_3mf_meshes(tmp_path / "b.3mf")]
    assert len(names) == n0 + 4 and any("cornice" in n for n in names)
    assert all(v == [] for k, v in probs.items() if not k.startswith("_"))


def _x(fn, res, path):
    fn(res, path)
    return path
