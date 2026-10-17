import numpy as np
from skimage.transform import ProjectiveTransform, warp

from colorforge.calibration.fit import apply_fits, fit_specimen, sample_photo, white_balance
from colorforge.calibration.specimen import SpecimenSpec, export_specimen, specimen_objects
from colorforge.color.spaces import linear_to_lab, linear_to_srgb, srgb_to_linear
from colorforge.export.threemf import read_3mf_meshes
from colorforge.filaments.model import Reliability
from colorforge.geometry.mesh import validate
from colorforge.optics.km import km_layer

TRUTH = {"generic-pla-red": (np.array([0.55, 0.04, 0.04]), 5.0),
         "generic-pla-blue": (np.array([0.03, 0.08, 0.40]), 12.0)}
BASE = "generic-pla-white"


def synthetic_photo(spec, base_lin, cast=(1.0, 0.9, 0.75), px_per_mm=8):
    """Render del provino visto in prospettiva con dominante di colore."""
    w, h = spec.size_mm
    W, H = int(w * px_per_mm), int(h * px_per_mm)
    flat = np.ones((H, W, 3)) * base_lin
    for r, fid in enumerate(spec.filament_ids):
        R, S = TRUTH[fid]
        for c, n in enumerate(spec.columns):
            x0, y0, x1, y1 = (v * px_per_mm for v in spec.patch_rect(r, c))
            col = base_lin if n == 0 else km_layer(R, S, n * spec.layer_mm, base_lin)
            flat[int(y0):int(y1), int(x0):int(x1)] = col
    flat = flat * np.array(cast)
    out_h, out_w = int(H * 1.4), int(W * 1.4)
    dst = np.array([[0.12 * out_w, 0.15 * out_h], [0.9 * out_w, 0.08 * out_h],
                    [0.95 * out_w, 0.9 * out_h], [0.05 * out_w, 0.85 * out_h]])
    t = ProjectiveTransform()
    t.estimate(np.array([[0, 0], [W, 0], [W, H], [0, H]], float), dst)
    img = warp(linear_to_srgb(flat), t.inverse, output_shape=(out_h, out_w), cval=0.2)
    corners = [tuple(p / px_per_mm * px_per_mm) for p in dst]
    return img, corners


def test_specimen_geometry_and_export(db, tmp_path):
    spec = SpecimenSpec(BASE, list(TRUTH))
    for _, _, m in specimen_objects(spec):
        assert validate(m) == []
    export_specimen(spec, tmp_path / "p.3mf", db)
    assert len(read_3mf_meshes(tmp_path / "p.3mf")) == 3
    assert SpecimenSpec.load(tmp_path / "p.cfcal").id == spec.id


def test_photo_calibration_recovers_parameters(db):
    spec = SpecimenSpec(BASE, list(TRUTH))
    base_lin = db.get(BASE).linear_rgb
    img, corners = synthetic_photo(spec, base_lin)
    srgb = sample_photo(img, corners, spec)
    lin = white_balance(srgb_to_linear(srgb), base_lin)
    fits = fit_specimen(spec, linear_to_lab(lin))
    for fit in fits:
        R, S = TRUTH[fit.filament_id]
        assert fit.reliable, fit.message
        assert abs(fit.scattering - S) / S < 0.1
        assert np.linalg.norm(fit.r_inf_lab - linear_to_lab(R)) < 2.0
    done = apply_fits(db, spec, linear_to_lab(lin), fits, "photo")
    f = db.get("generic-pla-red")
    assert set(done) == set(TRUTH) and f.td_reliability is Reliability.PHOTO and f.km_scattering
    assert len(f.measurements) == len(spec.steps) and db.calibration_history


def test_unreliable_fit_not_applied(db):
    spec = SpecimenSpec(BASE, ["generic-pla-red"])
    lab = np.tile(np.array([50.0, 0, 0]), (1, len(spec.columns), 1))  # nessun contrasto
    fits = fit_specimen(spec, lab)
    assert not fits[0].reliable
    assert apply_fits(db, spec, lab, fits, "photo") == []


def test_print_feedback_corrects_wrong_parameters(db):
    """Stampa 'reale' simulata con TD vere diverse da quelle nel database:
    la correzione deve ridurre l'errore e avvicinare i parametri al vero."""
    import copy
    from colorforge.calibration.feedback import apply_refit, refit_from_print, stacks_for, _predict
    from colorforge.filaments.database import FilamentDatabase
    from colorforge.pipeline import JobSettings, run
    from tests.conftest import make_image
    fids = ["generic-pla-white", "generic-pla-red", "generic-pla-blue", "generic-pla-yellow"]
    res = run(make_image("many", 80, 60), db, JobSettings(filament_ids=fids, width_mm=40, pixel_mm=0.5,
                                                          max_colors=4, quality="veloce"))
    # mondo reale: stessi colori, scattering diverso (TD reale ≠ stimata)
    truth = FilamentDatabase([copy.deepcopy(db.get(f)) for f in fids])
    from colorforge.optics.km import OpticalModel
    m = OpticalModel([db.get(f) for f in fids])
    used, seg = stacks_for(res)
    r = np.array([truth.get(f).linear_rgb for f in used])
    s_true = np.array([m.layers[f].s * 2.5 for f in used])
    H, W = res.levels.shape
    real_lin = _predict(r, s_true, seg.reshape(-1, seg.shape[2], 2), 0.0).reshape(H, W, 3)
    photo = linear_to_srgb(real_lin)
    corners = [(0, 0), (W, 0), (W, H), (0, H)]
    fit = refit_from_print(res, db, photo, corners, n_samples=300)
    assert fit["after"] < fit["before"] * 0.5
    done = apply_refit(db, fit)
    assert done and all(db.get(f).km_scattering for f in done)
