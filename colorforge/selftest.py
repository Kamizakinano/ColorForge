"""Autodiagnosi dell'eseguibile: `ColorForge.exe --selftest <cartella>`.

Esegue l'intero flusso su immagini sintetiche, verifica geometria ed export,
apre la finestra principale e scrive un report. Codice di uscita 0 = tutto ok.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


def run_selftest(out_dir: str | None = None) -> int:
    out = Path(out_dir or tempfile.mkdtemp(prefix="colorforge_selftest_"))
    out.mkdir(parents=True, exist_ok=True)
    report = {"checks": []}

    def check(name, fn):
        try:
            detail = fn()
            report["checks"].append({"name": name, "ok": True, "detail": detail})
        except Exception:
            report["checks"].append({"name": name, "ok": False, "error": traceback.format_exc()})

    from colorforge.filaments.database import FilamentDatabase
    from colorforge.geometry.mesh import validate
    from colorforge.imaging.importer import load_image
    from colorforge.pipeline import JobSettings, export_all, run
    from colorforge.printers.profiles import get_printer

    def data_files():
        db = FilamentDatabase.builtin()
        p = get_printer("bambu-x2d")
        assert len(db) >= 2 and p.changer("ams2pro-x2").capacity == 8
        return f"{len(db)} filamenti, {p.name}"

    def pipeline():
        y, x = np.mgrid[0:60, 0:80] / np.array([60, 80])[:, None, None]
        img = np.stack([x, y, 1 - x * y], -1)
        src = out / "test.png"
        Image.fromarray((img * 255).astype(np.uint8)).save(src)
        loaded = load_image(src)
        db = FilamentDatabase.builtin()
        s = JobSettings(filament_ids=[f.id for f in db][:5], width_mm=40, pixel_mm=0.5,
                        max_colors=4, quality="veloce")
        res = run(loaded, db, s)
        assert res.mesh_problems == [], res.mesh_problems
        files = export_all(res, out / "export")
        for f in files.values():
            assert Path(f).stat().st_size > 0, f
        return {"dE": round(res.metrics["delta_e2000_medio"], 2), "files": list(files)}

    def gui():
        from PySide6.QtWidgets import QApplication
        from colorforge.gui.main_window import MainWindow
        app = QApplication.instance() or QApplication([])
        w = MainWindow()
        w.show()
        app.processEvents()
        w.grab().save(str(out / "gui.png"))
        w.close()
        return "finestra principale creata"

    check("dati incorporati", data_files)
    check("flusso completo", pipeline)
    check("interfaccia grafica", gui)
    ok = all(c["ok"] for c in report["checks"])
    report["ok"] = ok
    (out / "selftest_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run_selftest(sys.argv[1] if len(sys.argv) > 1 else None))
