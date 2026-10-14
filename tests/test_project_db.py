import numpy as np
import pytest

from colorforge.filaments.database import FilamentDatabase
from colorforge.filaments.model import Filament, Reliability
from colorforge.pipeline import JobSettings, run
from colorforge.project.session import load_project, save_project
from tests.conftest import make_image


def test_project_roundtrip(db, tmp_path):
    s = JobSettings(filament_ids=["generic-pla-black", "generic-pla-white"], width_mm=20, pixel_mm=0.5,
                    max_colors=2, quality="veloce")
    res = run(make_image("gradient"), db, s)
    p = tmp_path / "p.cfproj"
    save_project(p, res, db)
    save_project(p, res, db)
    assert (tmp_path / "p.cfproj.bak").exists()
    r2, fdb = load_project(p)
    assert np.array_equal(r2.levels, res.levels) and r2.solution.sequence == res.solution.sequence
    assert len(fdb) == 2 and r2.settings.quality == "veloce"


def test_custom_filament_db(tmp_path):
    db = FilamentDatabase()
    f = Filament.from_hex("mio-rosa", "Bambu", "PLA Basic Rosa", "#F55A74", td_mm=3.2,
                          td_reliability=Reliability.DECLARED)
    db.add(f)
    db.save(tmp_path / "f.json")
    db2 = FilamentDatabase.load(tmp_path / "f.json")
    g = db2.get("mio-rosa")
    assert g.td_mm == 3.2 and g.td_reliability is Reliability.DECLARED
    assert np.allclose(g.lab, f.lab)
    with pytest.raises(KeyError):
        db2.add(f)


def test_invalid_import_rejected():
    with pytest.raises(ValueError):
        FilamentDatabase.from_dict({"format": "altro"})
    with pytest.raises(ValueError):
        FilamentDatabase.from_dict({"format": "colorforge-filaments", "version": 1,
                                    "filaments": [{"id": "x", "brand": "b", "name": "n", "lab": [1, 2]}]})
