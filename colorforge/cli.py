"""Interfaccia a riga di comando: python -m colorforge.cli foto.jpg -o uscita/"""
from __future__ import annotations

import argparse
import sys

from colorforge.filaments.database import FilamentDatabase
from colorforge.imaging.importer import ImageImportError, load_image
from colorforge.pipeline import JobSettings, export_all, run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ColorForge AI: foto -> stampa 3D multicolore")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default="output")
    ap.add_argument("--db", help="database filamenti JSON")
    ap.add_argument("--filaments", help="id separati da virgola (default: tutti quelli posseduti)")
    ap.add_argument("--width", type=float, default=100.0)
    ap.add_argument("--pixel", type=float, default=0.3)
    ap.add_argument("--colors", type=int, default=8)
    ap.add_argument("--quality", default="massima", choices=["massima", "bilanciata", "veloce"])
    ap.add_argument("--printer", default="bambu-x2d")
    ap.add_argument("--changer", default="ams2pro-x2")
    a = ap.parse_args(argv)
    db = FilamentDatabase.load(a.db) if a.db else FilamentDatabase.load_default()
    fids = a.filaments.split(",") if a.filaments else [f.id for f in (db.owned() or list(db))]
    s = JobSettings(printer_id=a.printer, changer_id=a.changer, filament_ids=fids,
                    width_mm=a.width, pixel_mm=a.pixel, max_colors=a.colors, quality=a.quality)
    try:
        img = load_image(a.image)
    except ImageImportError as e:
        print(f"Errore: {e}", file=sys.stderr)
        return 2
    res = run(img, db, s, progress=lambda p, m: print(f"[{p * 100:5.1f}%] {m}", flush=True))
    files = export_all(res, a.out)
    print(open(files["istruzioni"], encoding="utf-8").read())
    for k, v in files.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
