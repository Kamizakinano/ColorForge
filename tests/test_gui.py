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
    w.thread.wait(60000)
    app.processEvents()
    assert w.result is not None and w.bt_export.isEnabled()


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
