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
    # crash nativi (es. Qt) e eccezioni non gestite finiscono nel log
    import faulthandler
    global _crash_log
    _crash_log = open(log_dir / "colorforge_crash.log", "a", encoding="utf-8")
    faulthandler.enable(_crash_log, all_threads=True)

    def _hook(et, ev, tb):
        import traceback
        logging.getLogger("colorforge").error("Eccezione non gestita:\n%s", "".join(traceback.format_exception(et, ev, tb)))
    sys.excepthook = _hook
    from PySide6.QtWidgets import QApplication
    from colorforge.gui.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName("ColorForge AI")
    # pulsanti standard di Qt in italiano (Salva, Annulla, Sì, No…)
    from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator
    tr = QTranslator(app)
    if tr.load(QLocale("it_IT"), "qtbase", "_", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)):
        app.installTranslator(tr)
    from importlib import resources
    from PySide6.QtGui import QIcon
    app.setWindowIcon(QIcon(str(resources.files("colorforge.data").joinpath("icon.png"))))
    from colorforge import settings as appset
    from colorforge.gui.theme import apply_theme
    cfg = appset.load()
    apply_theme(app, bool(cfg.get("dark", True)))
    w = MainWindow()
    w.show()
    if not cfg.get("guide_shown"):
        cfg["guide_shown"] = True
        appset.save(cfg)
        w.show_guide()
    if cfg.get("check_updates", True):
        w.check_updates(False)
    if len(sys.argv) > 1:
        w.load_image_path(sys.argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
