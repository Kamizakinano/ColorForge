import numpy as np

from colorforge.color.spaces import delta_e2000, linear_to_srgb, srgb_to_lab, srgb_to_linear
from colorforge.optics.km import TD_CONTRAST, OpticalModel, km_layer, scattering_from_td


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 50)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-9)


def test_de2000_reference_pair():
    # Sharma et al. (2005), coppia 1: ΔE00 = 2.0425
    d = delta_e2000(np.array([50, 2.6772, -79.7751]), np.array([50, 0, -82.7485]))
    assert abs(float(d) - 2.0425) < 1e-3


def test_km_limits():
    r = np.array([0.8, 0.3, 0.05])
    assert np.allclose(km_layer(r, 5.0, 0.0, 0.5), 0.5)          # spessore nullo = sfondo
    assert np.allclose(km_layer(r, 5.0, 100.0, 0.0), r, atol=1e-6)  # spessore infinito = R_inf
    thin_white = km_layer(r, 5.0, 0.1, 1.0)
    thin_black = km_layer(r, 5.0, 0.1, 0.0)
    assert np.all(thin_white > thin_black)                         # lo sfondo traspare


def test_km_monotone_towards_rinf():
    r = np.array([0.6, 0.6, 0.6])
    vals = [km_layer(r, 3.0, d, 0.0)[0] for d in np.linspace(0.05, 3, 30)]
    assert np.all(np.diff(vals) >= -1e-12) and vals[-1] <= 0.6 + 1e-9


def test_td_definition():
    r = np.array([0.5, 0.2, 0.7])
    s = scattering_from_td(r, 1.5)
    c = np.mean(km_layer(r, s, 1.5, 1.0) - km_layer(r, s, 1.5, 0.0))
    assert abs(c - TD_CONTRAST) < 1e-4


def test_unknown_td_is_flagged(db):
    m = OpticalModel(list(db))
    assert m.warnings and all(l.s_source == "default" for l in m.layers.values())
    assert m.reliability(list(m.layers)) <= 0.25


def test_stack_not_rgb_average(db):
    m = OpticalModel([db.get("generic-pla-black"), db.get("generic-pla-white")])
    lut = m.stack_lut(["generic-pla-black"] * 4 + ["generic-pla-white"] * 20, np.full(24, 0.08))
    lab = srgb_to_lab(linear_to_srgb(lut))[:, 0]
    assert np.all(np.diff(lab[5:]) > 0)  # più bianco sopra = più chiaro, in modo non lineare
    assert abs(np.diff(lab[5:]).std()) > 0.01
