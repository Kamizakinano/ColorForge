import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_main_window_runs_job(app, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from colorforge.gui.main_window import MainWindow
    from tests.conftest import make_image
    from PIL import Image
    import numpy as np
    p = tmp_path / "g.png"
    Image.fromarray((make_image("many").srgb * 255).astype(np.uint8)).save(p)
    w = MainWindow()
    w.load_image_path(str(p))
    w.sp_width.setValue(20); w.sp_pixel.setValue(0.5); w.sp_colors.setValue(3)
    w.cb_quality.setCurrentIndex(2)
    s = w.settings()
    assert s.printer_id == "bambu-x2d" and s.changer_id == "ams2pro-x2" and s.filament_ids
    w.start()
    assert _wait(app, lambda: w.result is not None and not w.thread.isRunning())
    assert w.bt_export.isEnabled()


def test_filament_editor_manual_entry(app, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from colorforge.filaments.database import FilamentDatabase
    from colorforge.gui.filament_editor import FilamentEditor
    dlg = FilamentEditor(FilamentDatabase(), {})
    dlg.add_row()
    dlg.table.item(0, 3).setText("#F55A74")
    dlg.table.item(0, 5).setText("PLA Rosa")
    dlg.table.cellWidget(0, 7).setValue(3.5)
    dlg.table.cellWidget(0, 1).setValue(5)
    dlg.accept()
    f = dlg.db.get("utente-1")
    assert f.name == "PLA Rosa" and f.td_mm == 3.5 and dlg.slots["utente-1"] == 5
    assert (tmp_path / "ColorForge" / "filaments.json").exists()


def _wait(app, cond, timeout=60):
    import time
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    return cond()


def test_palette_selection_updates_preview_and_exports_3mf(app, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from PySide6.QtCore import Qt
    from PIL import Image
    import numpy as np
    from colorforge.gui.main_window import MainWindow
    from colorforge.export.threemf import read_3mf_meshes
    from tests.conftest import make_image
    p = tmp_path / "s.png"
    Image.fromarray((make_image("saturated").srgb * 255).astype(np.uint8)).save(p)
    w = MainWindow()
    w.sp_width.setValue(20); w.sp_pixel.setValue(0.5)
    w.load_image_path(str(p))
    assert w.palette.list.count() > 0
    for i in range(min(3, w.palette.list.count())):
        w.palette.list.item(i).setCheckState(Qt.CheckState.Checked)
    assert _wait(app, lambda: w.result is not None and not (w.thread and w.thread.isRunning()))
    chosen = {f"img-{h.lstrip('#')}" for h in w.palette.selected()}
    assert set(w.result.solution.filaments) <= chosen
    # esportazione 3MF in background
    from colorforge.pipeline import export_3mf
    out = tmp_path / "x.3mf"
    done = []
    from PySide6.QtCore import QThread
    main_thread = QThread.currentThread()
    seen = []
    orig = w._on_progress
    w._on_progress = lambda p_, m: (seen.append(QThread.currentThread() is main_thread), orig(p_, m))
    w._run_task(lambda prog: (export_3mf(w.result, out, progress=prog), str(out))[1], done.append, "Esportazione 3MF")
    assert _wait(app, lambda: bool(done))
    assert out.exists() and len(read_3mf_meshes(out)) >= 1
    # l'interfaccia deve essere aggiornata solo dal thread principale (crash su Windows altrimenti)
    assert seen and all(seen)
    assert not list(tmp_path.glob("*.part"))


def test_calibration_dialog_photo_flow(app, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from colorforge.calibration.specimen import SpecimenSpec
    from colorforge.filaments.database import FilamentDatabase
    from colorforge.gui.calibration_dialog import CalibrationDialog
    from colorforge.gui.main_window import ZoomView
    from tests.test_calibration import BASE, TRUTH, synthetic_photo
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    db = FilamentDatabase.builtin()
    dlg = CalibrationDialog(db, {}, None, ZoomView)
    spec = SpecimenSpec(BASE, list(TRUTH))
    dlg.load_spec(spec)
    img, corners = synthetic_photo(spec, db.get(BASE).linear_rgb)
    dlg.photo = img
    dlg.click.view.set_image(img)
    for x, y in corners:
        dlg.add_corner(x, y)
    dlg.compute()
    assert dlg.fits and all(f.reliable for f in dlg.fits)
    dlg.save()
    assert db.get("generic-pla-red").km_scattering is not None
    assert (tmp_path / "ColorForge" / "filaments.json").exists()


def test_guided_three_solutions(app, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from PySide6.QtWidgets import QDialog
    import numpy as np
    from PIL import Image
    from colorforge.gui.main_window import MainWindow
    from tests.conftest import make_image
    shown = []
    monkeypatch.setattr(QDialog, "exec", lambda self: shown.append(self) or 0)
    p = tmp_path / "m.png"
    Image.fromarray((make_image("many").srgb * 255).astype(np.uint8)).save(p)
    w = MainWindow()
    w.sp_width.setValue(20); w.sp_pixel.setValue(0.5); w.sp_colors.setValue(4)
    w.ck_auto.setChecked(False)
    w.load_image_path(str(p))
    w.guided()
    assert _wait(app, lambda: bool(shown), 120)
