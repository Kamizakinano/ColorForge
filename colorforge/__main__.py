"""Avvio: python -m colorforge"""
import logging
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        if not os.environ.get("QT_QPA_PLATFORM"):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
        try:
            from colorforge.selftest import run_selftest
            return run_selftest(sys.argv[2] if len(sys.argv) > 2 else None)
        except BaseException:  # mai finestre di errore bloccanti durante il collaudo
            import traceback
            Path(sys.argv[2] if len(sys.argv) > 2 else ".").mkdir(parents=True, exist_ok=True)
            (Path(sys.argv[2] if len(sys.argv) > 2 else ".") / "selftest_crash.txt").write_text(
                traceback.format_exc(), "utf-8")
            return 3
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
