"""Avvio: python -m colorforge"""
import logging
import os
import sys
from pathlib import Path


def main() -> int:
    log_dir = Path(os.environ.get("APPDATA") or Path.home() / ".config") / "ColorForge"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / "colorforge.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from PySide6.QtWidgets import QApplication
    from colorforge.gui.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName("ColorForge AI")
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.load_image_path(sys.argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
