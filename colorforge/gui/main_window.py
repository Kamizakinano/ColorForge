"""Finestra principale di ColorForge AI."""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QSlider, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from colorforge.color.spaces import srgb_to_hex, srgb_to_lab
from colorforge.filaments.database import FilamentDatabase
from colorforge.gui.filament_editor import FilamentEditor
from colorforge.imaging.importer import ImageImportError, LoadedImage, load_image
from colorforge.gui.palette import PalettePanel
from colorforge.pipeline import (Cancelled, JobResult, JobSettings, build_report, export_3mf, export_all,
                                 palette_filaments, run)
from colorforge.printers.profiles import get_printer, load_printers
from colorforge.project.session import load_project, save_project

log = logging.getLogger("colorforge")
QUALITY_LABELS = [("Qualità massima", "massima"), ("Bilanciata", "bilanciata"), ("Veloce", "veloce")]


def to_pixmap(arr: np.ndarray) -> QPixmap:
    a = np.ascontiguousarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
    h, w, _ = a.shape
    return QPixmap.fromImage(QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


class ZoomView(QGraphicsView):
    """Vista immagine con zoom (rotellina) e trascinamento."""

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.scene().addItem(self.item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_image(self, arr: np.ndarray | None):
        self.item.setPixmap(to_pixmap(arr) if arr is not None else QPixmap())
        self.scene().setSceneRect(self.item.boundingRect())
        self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)


class Worker(QObject):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, image, db, settings):
        super().__init__()
        self.image, self.db, self.settings = image, db, settings
        self.cancel = False

    def run(self):
        try:
            res = run(self.image, self.db, self.settings, progress=self.progress.emit,
                      cancelled=lambda: self.cancel)
            self.done.emit(res)
        except Cancelled:
            self.failed.emit("Elaborazione annullata.")
        except ValueError as e:
            self.failed.emit(str(e))
        except Exception as e:  # errore inatteso: log tecnico, messaggio leggibile
            log.error("Errore elaborazione\n%s", traceback.format_exc())
            self.failed.emit(f"Errore imprevisto durante l'elaborazione ({type(e).__name__}). "
                             "Dettagli nel file di log.")


class TaskWorker(QObject):
    """Esegue una funzione lunga (es. esportazione) fuori dal thread dell'interfaccia."""
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit))
        except OSError as e:
            self.failed.emit(f"Impossibile scrivere il file: {e}")
        except Exception as e:
            log.error("Errore\n%s", traceback.format_exc())
            self.failed.emit(f"Errore imprevisto ({type(e).__name__}): {e}")


class Relay(QObject):
    """Riceve i segnali di un worker nel thread dell'interfaccia.

    I segnali collegati a lambda o funzioni semplici verrebbero eseguiti nel
    thread del worker (toccare i widget da lì fa chiudere l'app su Windows):
    collegandoli agli slot di un QObject del thread principale Qt li accoda
    correttamente.
    """

    def __init__(self, on_progress, on_done, on_failed, parent=None):
        super().__init__(parent)
        self._p, self._d, self._f = on_progress, on_done, on_failed

    @Slot(float, str)
    def progress(self, p, m):
        self._p(p, m)

    @Slot(object)
    def done(self, r):
        self._d(r)

    @Slot(str)
    def failed(self, m):
        self._f(m)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ColorForge AI")
        self.resize(1400, 860)
        self.db = FilamentDatabase.load_default()
        self.slots: dict[str, int] = {}
        self.image: LoadedImage | None = None
        self.result: JobResult | None = None
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self.pending_auto = False
        self.auto_run = False
        self.auto_timer = QTimer(self)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.setInterval(700)
        self.auto_timer.timeout.connect(lambda: self.start(auto=True))
        self._bg: list = []
        self.printers = load_printers()
        self._build()
        self.refresh_filaments()

    # --- costruzione UI ---------------------------------------------------
    def _build(self):
        side = QWidget()
        sl = QVBoxLayout(side)
        b = QPushButton("📷  Carica fotografia…")
        b.clicked.connect(self.open_image)
        sl.addWidget(b)

        g = QGroupBox("Stampante")
        f = QFormLayout(g)
        self.cb_printer = QComboBox()
        for p in self.printers:
            self.cb_printer.addItem(p.name, p.id)
        self.cb_printer.currentIndexChanged.connect(self._printer_changed)
        self.cb_changer = QComboBox()
        f.addRow("Modello", self.cb_printer)
        f.addRow("Cambio filamento", self.cb_changer)
        sl.addWidget(g)

        g = QGroupBox("Dimensioni e qualità")
        f = QFormLayout(g)
        self.sp_width = QDoubleSpinBox(); self.sp_width.setRange(10, 400); self.sp_width.setValue(100); self.sp_width.setSuffix(" mm")
        self.lb_height = QLabel("-")
        self.sp_pixel = QDoubleSpinBox(); self.sp_pixel.setRange(0.1, 2); self.sp_pixel.setSingleStep(0.05); self.sp_pixel.setValue(0.4); self.sp_pixel.setSuffix(" mm")
        self.sp_layer = QDoubleSpinBox(); self.sp_layer.setRange(0.04, 0.32); self.sp_layer.setSingleStep(0.02); self.sp_layer.setValue(0.08); self.sp_layer.setSuffix(" mm")
        self.sp_base = QDoubleSpinBox(); self.sp_base.setRange(0.2, 5); self.sp_base.setValue(0.64); self.sp_base.setSuffix(" mm")
        self.sp_max = QDoubleSpinBox(); self.sp_max.setRange(0.5, 10); self.sp_max.setValue(2.4); self.sp_max.setSuffix(" mm")
        self.sp_colors = QSpinBox(); self.sp_colors.setRange(1, 16); self.sp_colors.setValue(8)
        self.cb_quality = QComboBox()
        for lbl, k in QUALITY_LABELS:
            self.cb_quality.addItem(lbl, k)
        self.cb_mode = QComboBox()
        self.cb_mode.addItem("Multicolore per zona (AMS)", "zone")
        self.cb_mode.addItem("Cambi per strato (stile HueForge)", "layers")
        self.cb_mode.setToolTip("Per zona: filamenti diversi nello stesso strato, molto più fedele ma con più "
                                "cambi filamento.\nPer strato: un colore per strato su tutto il pannello, "
                                "pochi cambi, colori limitati.")
        self.cb_mode.currentIndexChanged.connect(lambda _i: self._palette_changed())
        self.sl_econ = QSlider(Qt.Orientation.Horizontal)
        self.sl_econ.setRange(0, 100); self.sl_econ.setValue(30)
        self.sl_econ.setToolTip("Sinistra: massima fedeltà (più filamenti e cambi).\n"
                                "Destra: meno cambi filamento (stampa più rapida, meno spurgo).")
        self.lb_econ = QLabel("")
        self.sl_econ.valueChanged.connect(self._econ_label)
        self._econ_label(30)
        econ_box = QWidget(); eb = QHBoxLayout(econ_box); eb.setContentsMargins(0, 0, 0, 0)
        eb.addWidget(QLabel("Fedeltà")); eb.addWidget(self.sl_econ, 1); eb.addWidget(QLabel("Meno cambi"))
        self.ck_imp = QCheckBox("Priorità a contorni e soggetto"); self.ck_imp.setChecked(True)
        self.sp_width.valueChanged.connect(self._update_height)
        for lbl, w in (("Larghezza", self.sp_width), ("Altezza (auto)", self.lb_height),
                       ("Risoluzione", self.sp_pixel), ("Altezza strato", self.sp_layer),
                       ("Spessore base", self.sp_base), ("Spessore massimo", self.sp_max),
                       ("Colori massimi", self.sp_colors), ("Modalità colore", self.cb_mode),
                       ("Qualità", self.cb_quality), ("Equilibrio", econ_box), ("", self.lb_econ),
                       ("", self.ck_imp)):
            f.addRow(lbl, w)
        sl.addWidget(g)

        g = QGroupBox("Filamenti disponibili")
        v = QVBoxLayout(g)
        self.lst_fil = QListWidget()
        self.lst_fil.setMinimumHeight(110)
        v.addWidget(self.lst_fil)
        b = QPushButton("🎨  I miei filamenti…")
        b.clicked.connect(self.edit_filaments)
        v.addWidget(b)
        b = QPushButton("💡  Suggerisci filamenti da acquistare")
        b.setToolTip("Calcola quali filamenti in più migliorerebbero di più questa immagine")
        b.clicked.connect(self.suggest_purchases)
        v.addWidget(b)
        sl.addWidget(g, 1)

        g = QGroupBox("Preparazione immagine")
        v = QVBoxLayout(g)
        self.ck_denoise = QCheckBox("Riduzione rumore"); self.ck_denoise.setChecked(True)
        self.ck_sharp = QCheckBox("Nitidezza moderata"); self.ck_sharp.setChecked(True)
        self.ck_contrast = QCheckBox("Contrasto adattivo"); self.ck_contrast.setChecked(True)
        for w in (self.ck_denoise, self.ck_sharp, self.ck_contrast):
            v.addWidget(w)
        sl.addWidget(g)

        g = QGroupBox("Rilievo e accessori (facoltativi)")
        v = QVBoxLayout(g)
        rl = QHBoxLayout()
        self.ck_relief = QCheckBox("Rilievo 3D del soggetto")
        self.sp_relief = QDoubleSpinBox(); self.sp_relief.setRange(0.5, 15); self.sp_relief.setValue(4.0); self.sp_relief.setSuffix(" mm")
        self.sp_relief.setEnabled(False)
        self.ck_relief.toggled.connect(self.sp_relief.setEnabled)
        rl.addWidget(self.ck_relief); rl.addWidget(self.sp_relief)
        v.addLayout(rl)
        self.ck_frame = QCheckBox("Aggiungi cornice")
        self.ck_hook = QCheckBox("Aggiungi gancio per appendere")
        self.ck_stand = QCheckBox("Aggiungi piedini d'appoggio")
        for w in (self.ck_frame, self.ck_hook, self.ck_stand):
            v.addWidget(w)
        v.addWidget(QLabel("<small>Il rilievo richiede la modalità 'Multicolore per zona'. Gli accessori "
                           "vengono aggiunti al 3MF solo se spuntati.</small>"))
        sl.addWidget(g)

        self.bt_run = QPushButton("▶  Elabora automaticamente")
        self.bt_run.setStyleSheet("font-weight:bold; padding:8px;")
        self.bt_run.clicked.connect(self.start)
        self.bt_guided = QPushButton("✨  Proponi 3 soluzioni")
        self.bt_guided.setToolTip("Calcola una versione di massima qualità, una equilibrata e una con pochi cambi, "
                                  "con tempi e consumi, e ti fa scegliere")
        self.bt_guided.clicked.connect(self.guided)
        self.bt_mini = QPushButton("🔬  Miniatura di prova (40 mm)…")
        self.bt_mini.setToolTip("Stessa soluzione in piccolo, per verificare i colori prima della stampa grande")
        self.bt_mini.clicked.connect(self.export_miniature)
        self.bt_cancel = QPushButton("Annulla"); self.bt_cancel.setEnabled(False)
        self.bt_cancel.clicked.connect(self.cancel)
        self.bt_3mf = QPushButton("💾  Esporta 3MF per Bambu Studio…"); self.bt_3mf.setEnabled(False)
        self.bt_3mf.clicked.connect(self.export_3mf_file)
        self.bt_export = QPushButton("📁  Esporta tutto (STL, 3MF, istruzioni)…"); self.bt_export.setEnabled(False)
        self.bt_export.clicked.connect(self.export)
        for w in (self.bt_run, self.bt_guided, self.bt_cancel, self.bt_3mf, self.bt_mini, self.bt_export):
            sl.addWidget(w)

        self.tabs = QTabWidget()
        self.v_orig, self.v_prep, self.v_sim, self.v_err = (ZoomView() for _ in range(4))
        self.palette = PalettePanel(to_pixmap)
        self.palette.nearest = self._nearest_owned
        self.palette.changed.connect(self._palette_changed)
        self.palette.bt_add.clicked.connect(self.palette_to_db)
        self.ck_auto = QCheckBox("Aggiorna automaticamente l'anteprima quando cambio i colori")
        self.ck_auto.setChecked(True)
        self.lb_source = QLabel("")
        main = QWidget()
        ml = QVBoxLayout(main)
        ml.setContentsMargins(0, 0, 0, 0)
        cmp_split = QSplitter()
        for w, title in ((self.v_orig, "Originale"), (self.palette, None), (self.v_sim, "Risultato previsto")):
            if title:
                box = QWidget(); bl = QVBoxLayout(box); bl.setContentsMargins(0, 0, 0, 0)
                bl.addWidget(QLabel(f"<b>{title}</b>")); bl.addWidget(w)
                cmp_split.addWidget(box)
            else:
                cmp_split.addWidget(w)
        cmp_split.setSizes([430, 270, 430])
        ml.addWidget(cmp_split, 1)
        row = QHBoxLayout(); row.addWidget(self.ck_auto); row.addStretch(1); row.addWidget(self.lb_source)
        ml.addLayout(row)
        self.v_cmp = main
        for v, t in ((main, "Originale · colori · risultato"), (self.v_prep, "Preparata"),
                     (self.v_err, "Mappa errori ΔE")):
            self.tabs.addTab(v, t)
        self.txt = QTextEdit(); self.txt.setReadOnly(True)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.tabs)
        right.addWidget(self.txt)
        right.setSizes([620, 200])
        self.txt.setMinimumHeight(60)
        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(340)
        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setSizes([360, 1040])
        self.setCentralWidget(split)

        self.progress = QProgressBar(); self.progress.setRange(0, 1000)
        self.lb_phase = QLabel("Pronto")
        self.statusBar().addWidget(self.lb_phase, 1)
        self.statusBar().addPermanentWidget(self.progress)

        cm = self.menuBar().addMenu("&Calibrazione")
        a = QAction("Calibra i filamenti (provino)…", self)
        a.triggered.connect(self.open_calibration)
        cm.addAction(a)
        a = QAction("Confronta con la stampa reale…", self)
        a.triggered.connect(self.open_print_feedback)
        cm.addAction(a)
        m = self.menuBar().addMenu("&File")
        a2 = QAction("Leggi tempi e costi reali dallo slicer…", self)
        a2.triggered.connect(self.read_slicer_estimate)
        m.addAction(a2)
        a = QAction("Imposta progetto Bambu di riferimento (i tuoi slot AMS)…", self)
        a.triggered.connect(self.set_bambu_template)
        m.addAction(a)
        for text, fn in (("Apri fotografia…", self.open_image), ("Apri progetto…", self.open_project),
                         ("Salva progetto…", self.save_project), ("Esporta…", self.export)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            m.addAction(a)
        self._printer_changed()

    # --- stato -------------------------------------------------------------
    def _econ_label(self, v):
        txt = ("massima fedeltà" if v < 10 else "equilibrato" if v < 50 else
               "pochi cambi" if v < 85 else "minimo dei cambi")
        self.lb_econ.setText(f"<small>{txt} ({v}%)</small>")

    def _printer_changed(self):
        p = self.printers[self.cb_printer.currentIndex()]
        self.cb_changer.clear()
        for c in p.changers:
            self.cb_changer.addItem(c.name, c.id)

    def _update_height(self):
        if self.image is not None:
            h, w = self.image.srgb.shape[:2]
            self.lb_height.setText(f"{self.sp_width.value() * h / w:.1f} mm")

    def refresh_filaments(self):
        self.lst_fil.clear()
        for f in self.db:
            it = QListWidgetItem(f"{f.label}  {srgb_to_hex(f.srgb)}" + (f"  · slot {self.slots[f.id]}" if f.id in self.slots else ""))
            it.setData(Qt.ItemDataRole.UserRole, f.id)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if f.owned_spools > 0 else Qt.CheckState.Unchecked)
            pm = to_pixmap(np.ones((16, 16, 3)) * f.srgb)
            it.setIcon(pm)
            self.lst_fil.addItem(it)

    def selected_filaments(self) -> list[str]:
        return [self.lst_fil.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.lst_fil.count())
                if self.lst_fil.item(i).checkState() == Qt.CheckState.Checked]

    def job_filaments(self) -> tuple[FilamentDatabase, list[str], str]:
        """Filamenti da usare: i colori spuntati nella tavolozza, se ce ne sono,
        altrimenti i filamenti selezionati a sinistra."""
        chosen = self.palette.selected()
        if chosen:
            fil = palette_filaments(chosen)
            return FilamentDatabase(fil), [f.id for f in fil], f"Colori scelti dall'immagine: {len(fil)}"
        ids = self.selected_filaments()
        return self.db, ids, f"I miei filamenti: {len(ids)}"

    def settings(self, fids: list[str] | None = None) -> JobSettings:
        s = JobSettings(printer_id=self.cb_printer.currentData(), changer_id=self.cb_changer.currentData(),
                        filament_ids=fids if fids is not None else self.selected_filaments(), slots=dict(self.slots),
                        width_mm=self.sp_width.value(), pixel_mm=self.sp_pixel.value(),
                        max_colors=self.sp_colors.value(), quality=self.cb_quality.currentData(),
                        mode=self.cb_mode.currentData(),
                        change_economy=self.sl_econ.value() / 100, use_importance=self.ck_imp.isChecked(),
                        relief_mm=self.sp_relief.value() if self.ck_relief.isChecked() else 0.0,
                        add_frame=self.ck_frame.isChecked(), add_hook=self.ck_hook.isChecked(),
                        add_stand=self.ck_stand.isChecked())
        s.print.layer_height_mm = self.sp_layer.value()
        s.print.first_layer_height_mm = max(self.sp_layer.value(), 0.16)
        s.print.base_thickness_mm = self.sp_base.value()
        s.print.max_thickness_mm = self.sp_max.value()
        s.prep.denoise, s.prep.sharpen = self.ck_denoise.isChecked(), self.ck_sharp.isChecked()
        s.prep.auto_contrast = self.ck_contrast.isChecked()
        return s

    # --- azioni --------------------------------------------------------------
    def open_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Carica fotografia", "",
                                           "Immagini (*.png *.jpg *.jpeg *.tif *.tiff *.webp *.bmp)")
        if p:
            self.load_image_path(p)

    def load_image_path(self, p: str):
        try:
            self.image = load_image(p)
        except ImageImportError as e:
            QMessageBox.warning(self, "Fotografia", str(e))
            return
        self.v_orig.set_image(self.image.srgb)
        self.v_sim.set_image(None)
        self.palette.set_image(self.image.srgb)
        self.tabs.setCurrentWidget(self.v_cmp)
        self._update_height()
        self.txt.setPlainText("Fotografia caricata.\n" + "\n".join(self.image.notes))

    def _nearest_owned(self, hx: str):
        from colorforge.color.spaces import delta_e2000, hex_to_srgb
        owned = [f for f in self.db if f.owned_spools > 0]
        if not owned:
            return None, 0.0
        lab = srgb_to_lab(hex_to_srgb(hx))
        de = [float(delta_e2000(lab, np.array(f.lab))) for f in owned]
        i = int(np.argmin(de))
        return owned[i].label, de[i]

    def suggest_purchases(self):
        if self.image is None:
            QMessageBox.information(self, "Suggerimenti", "Carica prima una fotografia.")
            return
        owned = [f.id for f in self.db if f.owned_spools > 0]
        if not owned:
            QMessageBox.information(self, "Suggerimenti", "Segna prima i filamenti che possiedi in 'I miei filamenti'.")
            return
        from colorforge.filaments.database import FilamentDatabase as FDB
        from colorforge.imaging.preprocess import resize_to_grid
        from colorforge.optics.km import OpticalModel
        from colorforge.optimize.regional import suggest_purchases
        from colorforge.optimize.search import LayerPlan
        s = self.settings(owned)
        pal = palette_filaments([hx for hx, _ in __import__("colorforge.pipeline", fromlist=["x"]).image_palette(self.image.srgb, 12)])
        cand_db = FDB(list(self.db) + [f for f in pal if f.id not in self.db])
        cands = [f.id for f in self.db if f.owned_spools == 0] + [f.id for f in pal]
        cap = get_printer(s.printer_id).changer(s.changer_id).capacity or s.max_colors
        img = self.image.srgb
        h, w = img.shape[:2]
        cols = 120
        target = srgb_to_lab(resize_to_grid(img, cols, max(8, int(cols * h / w))))
        plan = LayerPlan.from_settings(s.print.first_layer_height_mm, s.print.layer_height_mm,
                                       s.print.base_thickness_mm, s.print.max_thickness_mm)
        model = OpticalModel(list(cand_db))

        def work(prog):
            return suggest_purchases(model, plan, target, owned, cands, min(cap, s.max_colors), progress=prog)

        def show(r):
            base, cur, sug = r
            lines = [f"Con i tuoi filamenti attuali: ΔE medio previsto {base:.1f}.", ""]
            if not sug:
                lines.append("Nessun filamento in più migliorerebbe in modo apprezzabile questa immagine.")
            for c, rep, de, gain in sug:
                f = cand_db.get(c)
                what = f.label if f.name.startswith("#") else f"{f.label} {srgb_to_hex(f.srgb)}"
                how = f" al posto di {cand_db.get(rep).label}" if rep else ""
                lines.append(f"• {what}{how}: ΔE {de:.1f} (−{gain:.1f})")
            lines += ["", "I colori 'Colore immagine' sono indicativi: cerca un filamento reale simile.",
                      "Stima basata sui dati dei filamenti; più sono calibrati, più è affidabile."]
            QMessageBox.information(self, "Filamenti consigliati", "\n".join(lines))
        self._run_task(work, show, "Ricerca filamenti consigliati")

    def bambu_template(self):
        from colorforge import settings as appset
        from colorforge.export.bambu import load_template
        p = appset.bambu_template_path()
        if p is None:
            return None
        try:
            return load_template(p)
        except ValueError:
            return None

    def _cost_html(self, res) -> str:
        tpl = self.bambu_template()
        if tpl is None:
            return ("<p><small>Imposta il progetto Bambu di riferimento per stimare spurgo e costo con i dati "
                    "reali dei tuoi filamenti.</small></p>")
        try:
            from colorforge.export.bambu import cost_estimate, map_to_template_slots
            db = getattr(self, "result_db", self.db)
            mp = map_to_template_slots([(sw["id"], sw["colore"]) for sw in res.swaps], tpl)
            c = cost_estimate(res, tpl, {k: v[0] for k, v in mp.items()}, db)
        except Exception as e:  # la stima è informativa: mai bloccare l'interfaccia
            log.warning("Stima costi non disponibile: %s", e)
            return ""
        nz = f" · secondo ugello per lo slot {c['ugello_secondario_slot']}" if c["ugello_secondario_slot"] else ""
        return (f"<p><b>Stima con i dati del tuo progetto Bambu:</b> {c['cambi']} cambi, spurgo ~{c['spurgo_g']:.0f} g{nz}, "
                f"costo filamento ~{c['costo_eur']:.2f} €<br><small>Per i valori esatti: slicing in Bambu Studio, "
                "salva, poi File → Leggi tempi e costi reali dallo slicer.</small></p>")

    def read_slicer_estimate(self):
        from colorforge.export.bambu import read_slice_info
        p, _ = QFileDialog.getOpenFileName(self, "Progetto slicato da Bambu Studio", str(Path.home()),
                                           "Progetto Bambu (*.3mf)")
        if not p:
            return
        try:
            info = read_slice_info(p)
        except ValueError as e:
            QMessageBox.warning(self, "Slicer", str(e))
            return
        lines = []
        for pl in info["piatti"]:
            h, mnt = divmod(int(pl["tempo_s"] // 60), 60)
            lines.append(f"Piatto {pl['piatto']}: {h} h {mnt} min · {pl['peso_g']:.1f} g · ~{pl['costo_eur']:.2f} €")
            for f in pl["filamenti"]:
                cst = f" · {f['costo_eur']:.2f} €" if f["costo_eur"] is not None else ""
                lines.append(f"   slot {f['slot']} {f['tipo']} {f['colore']}: {f['grammi']:.1f} g ({f['metri']:.2f} m){cst}")
        QMessageBox.information(self, "Valori reali dallo slicer", "\n".join(lines) +
                                "\n\nCosti calcolati con i prezzi €/kg impostati nei profili filamento di Bambu Studio.")

    def set_bambu_template(self):
        import shutil
        from colorforge import settings as appset
        from colorforge.export.bambu import load_template, template_filaments
        from colorforge.filaments.database import default_user_db_path
        p, _ = QFileDialog.getOpenFileName(self, "Progetto Bambu Studio di riferimento", str(Path.home()),
                                           "Progetto Bambu (*.3mf)")
        if not p:
            return
        try:
            t = load_template(p)
        except ValueError as e:
            QMessageBox.warning(self, "Progetto Bambu", str(e))
            return
        dst = appset.app_dir() / "bambu_template.3mf"
        shutil.copy2(p, dst)
        cfg = appset.load(); cfg["bambu_template"] = str(dst); appset.save(cfg)
        only = QMessageBox.question(self, "Filamenti AMS",
                                    f"Trovati {len(t.filaments)} filamenti negli slot del progetto "
                                    f"({t.printer}, ugello {t.nozzle} mm).\n\nUsare solo questi come filamenti disponibili?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        for f in list(self.db):
            if f.id.startswith("ams-slot"):
                self.db.remove(f.id)
                self.slots.pop(f.id, None)
            elif only == QMessageBox.StandardButton.Yes:
                f.owned_spools = 0
        for fil, slot in template_filaments(t):
            self.db.add(fil, overwrite=True)
            self.slots[fil.id] = slot
        try:
            self.db.save(default_user_db_path())
        except OSError as e:
            QMessageBox.warning(self, "Filamenti", str(e))
        self.refresh_filaments()
        self.palette.extract()
        lay = t.settings.get("layer_height")
        msg = (f"Progetto di riferimento impostato. Da ora il 3MF esportato si apre in Bambu Studio con ogni parte "
               f"già assegnata al suo slot.\n\nNel tuo progetto l'altezza strato è {lay} mm: ")
        try:
            lay = float(lay)
            self.sp_layer.setValue(lay)
            self.sp_base.setValue(max(self.sp_base.value(), lay * 6))
            msg += "l'ho impostata anche qui."
        except (TypeError, ValueError):
            msg += "controllala qui a sinistra."
        QMessageBox.information(self, "Progetto Bambu", msg)

    def open_calibration(self):
        from colorforge.gui.calibration_dialog import CalibrationDialog
        dlg = CalibrationDialog(self.db, self.slots, self, ZoomView)
        if dlg.exec():
            self.refresh_filaments()

    def open_print_feedback(self):
        if not self.result:
            QMessageBox.information(self, "Confronto", "Apri o elabora prima il progetto che hai stampato "
                                    "(File → Apri progetto).")
            return
        from colorforge.gui.calibration_dialog import PrintFeedbackDialog
        db = getattr(self, "result_db", self.db)
        dlg = PrintFeedbackDialog(self.result, db, self, ZoomView, self._run_task)
        if dlg.exec():
            if db is not self.db:
                self.db.merge(db, overwrite=True)
            self.refresh_filaments()

    def edit_filaments(self):
        dlg = FilamentEditor(self.db, self.slots, self)
        if dlg.exec():
            self.db, self.slots = dlg.db, dlg.slots
            self.refresh_filaments()
            self.palette.extract()

    def _palette_changed(self):
        n = len(self.palette.selected())
        self.lb_source.setText(f"Colori scelti: {n}" if n else "Nessun colore scelto: uso i miei filamenti")
        if self.image is not None and n and self.ck_auto.isChecked():
            self.auto_timer.start()

    def palette_to_db(self):
        from colorforge.filaments.database import default_user_db_path
        added = 0
        for f in palette_filaments(self.palette.selected()):
            f.id = f"utente-{f.id}"
            f.brand, f.owned_spools = "", 1
            f.name = f"Filamento {f.name}"
            if f.id not in self.db:
                self.db.add(f)
                added += 1
        try:
            self.db.save(default_user_db_path())
        except OSError as e:
            QMessageBox.warning(self, "Filamenti", str(e))
        self.refresh_filaments()
        QMessageBox.information(self, "Filamenti", f"{added} colori aggiunti ai tuoi filamenti. "
                                "Puoi completarli (marca, TD, slot) in 'I miei filamenti'.")

    def start(self, auto: bool = False):
        if self.image is None:
            if not auto:
                QMessageBox.information(self, "ColorForge", "Carica prima una fotografia.")
            return
        if self.thread is not None and self.thread.isRunning():
            if auto:  # ricalcolo richiesto mentre ne gira un altro: annulla e rilancia
                self.pending_auto = True
                self.worker.cancel = True
            return
        db, fids, source = self.job_filaments()
        s = self.settings(fids)
        if not s.filament_ids:
            if not auto:
                QMessageBox.information(self, "ColorForge", "Seleziona almeno un colore o un filamento.")
            return
        s.max_colors = min(s.max_colors, len(fids))
        if auto:  # anteprima rapida: ricerca limitata e risoluzione ridotta
            s.quality = "veloce"
            s.pixel_mm = max(s.pixel_mm, s.width_mm / 160)
        self.lb_source.setText(source + (" · anteprima rapida" if auto else ""))
        self.auto_run = auto
        self.bt_run.setEnabled(False); self.bt_cancel.setEnabled(True)
        self.bt_export.setEnabled(False); self.bt_3mf.setEnabled(False)
        self.job_db = db
        self.thread = QThread(self)
        self.worker = Worker(self.image, db, s)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.done.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._after_thread)
        self.thread.start()

    def _after_thread(self):
        if self.pending_auto:
            self.pending_auto = False
            QTimer.singleShot(0, lambda: self.start(auto=True))

    def cancel(self):
        if self.worker:
            self.worker.cancel = True
            self.lb_phase.setText("Annullamento in corso…")

    def _on_progress(self, p: float, msg: str):
        self.progress.setValue(int(p * 1000))
        self.lb_phase.setText(msg)

    def _finish_ui(self):
        self.bt_run.setEnabled(True); self.bt_cancel.setEnabled(False)

    def _on_failed(self, msg: str):
        self._finish_ui()
        if self.pending_auto:
            return
        self.progress.setValue(0)
        self.lb_phase.setText("Interrotto")
        if getattr(self, "auto_run", False):
            self.lb_source.setText(msg)
        else:
            QMessageBox.warning(self, "Elaborazione", msg)

    def _on_done(self, res: JobResult):
        self._finish_ui()
        self.progress.setValue(1000)
        self.lb_phase.setText(f"Completato in {res.elapsed_s:.0f} s"
                              + (" (anteprima rapida: premi 'Elabora' per la qualità massima)" if self.auto_run else ""))
        self.show_result(res)

    def show_result(self, res: JobResult):
        self.result = res
        self.result_db = getattr(self, "job_db", self.db)
        self.bt_export.setEnabled(True); self.bt_3mf.setEnabled(True)
        first = self.v_orig.item.pixmap().isNull()
        if first:
            self.v_orig.set_image(res.original_srgb)
        self.v_prep.set_image(res.prepared_srgb)
        self.v_sim.set_image(res.simulated_srgb)
        self.v_err.set_image(res.error_map_rgb())
        if self.tabs.currentWidget() is not self.v_err:
            self.tabs.setCurrentWidget(self.v_cmp)
        r = build_report(res)
        m = res.metrics
        html = [f"<h3>Risultato previsto (simulazione, non garanzia del risultato fisico)</h3>",
                f"<p>ΔE2000 medio <b>{m['delta_e2000_medio']:.2f}</b> · 95° percentile {m['delta_e2000_p95']:.2f} · "
                f"SSIM {m['ssim_luminanza']:.3f} · affidabilità dati <b>{res.reliability:.2f}</b></p>",
                ("<p><b>Filamenti usati (multicolore per zona) — assegna ogni parte del 3MF allo slot indicato:</b></p><ol>"
                 if res.zone is not None else "<p><b>Sequenza filamenti (dal basso):</b></p><ol>")]
        for sw in res.swaps:
            where = ("base del pannello" if sw.get("ruolo") == "base" else "nelle zone") if res.zone is not None \
                else f"dallo strato {sw['strato_inizio']} (z={sw['z_inizio_mm']} mm)"
            html.append(f"<li><span style='background:{sw['colore']}'>&nbsp;&nbsp;&nbsp;&nbsp;</span> "
                        f"{sw['filamento']} — slot {sw['slot']} — {where}</li>")
        e = r["stima_stampa"]
        html.append(f"</ol><p>Altezza {e['altezza_totale_mm']} mm · {e['numero_strati']} strati · "
                    f"{e['cambi_colore']} cambi · ~{e['tempo_min_stimato']:.0f} min (stima)</p>")
        html.append(self._cost_html(res))
        if res.warnings:
            html.append("<p><b>Avvisi:</b></p><ul>" + "".join(f"<li>{w}</li>" for w in res.warnings) + "</ul>")
        html.append("<p><b>Operazioni:</b></p><ul>" + "".join(f"<li>{w}</li>" for w in res.log) + "</ul>")
        self.txt.setHtml("".join(html))

    def _run_task(self, fn, on_done, title):
        from PySide6.QtWidgets import QProgressDialog
        self.bt_export.setEnabled(False); self.bt_3mf.setEnabled(False); self.bt_run.setEnabled(False)
        dlg = QProgressDialog(f"{title} in corso…\nAttendi la fine prima di aprire il file.", None, 0, 1000, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.show()
        th = QThread(self)
        wk = TaskWorker(fn)
        wk.moveToThread(th)

        def finish():
            dlg.close()
            self.bt_export.setEnabled(True); self.bt_3mf.setEnabled(True); self.bt_run.setEnabled(True)

        def prog(p, m):
            self._on_progress(p, m)
            dlg.setValue(int(p * 1000))
            dlg.setLabelText(f"{m}\nAttendi la fine prima di aprire il file.")

        def ok(r):
            finish(); self.progress.setValue(1000); on_done(r)

        def ko(m):
            finish(); QMessageBox.warning(self, title, m)
        relay = Relay(prog, ok, ko, self)
        th.started.connect(wk.run)
        wk.progress.connect(relay.progress)
        wk.done.connect(relay.done)
        wk.failed.connect(relay.failed)
        wk.done.connect(th.quit)
        wk.failed.connect(th.quit)
        item = (th, wk, relay)
        self._bg.append(item)
        th.finished.connect(relay.deleteLater)
        th.finished.connect(self._cleanup_bg)
        self.lb_phase.setText(title + "…")
        th.start()

    @Slot()
    def _cleanup_bg(self):
        self._bg = [x for x in self._bg if x[0].isRunning()]

    def _done_msg(self, title: str, paths: list[str], extra: list[str] | None = None):
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText("File salvati:\n" + "\n".join(paths) + ("\n" + "\n".join(extra) if extra else ""))
        open_btn = box.addButton("Apri cartella", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(paths[0]).parent)))
        self.lb_phase.setText(title + " completata")

    def guided(self):
        if self.image is None:
            QMessageBox.information(self, "ColorForge", "Carica prima una fotografia.")
            return
        db, fids, _ = self.job_filaments()
        if not fids:
            QMessageBox.information(self, "ColorForge", "Seleziona almeno un colore o un filamento.")
            return
        import copy
        base = self.settings(fids)
        base.max_colors = min(base.max_colors, len(fids))
        image = self.image
        variants = [("Massima qualità", 0.0), ("Equilibrata", 0.4), ("Pochi cambi", 0.9)]

        def work(prog):
            out = []
            for i, (name, econ) in enumerate(variants):
                s = copy.deepcopy(base)
                s.change_economy = econ
                s.quality = "bilanciata" if s.quality == "massima" else s.quality
                r = run(image, db, s, progress=lambda p, m, i=i, name=name: prog((i + p) / len(variants), f"{name}: {m}"))
                out.append((name, r))
            return out

        def choose(results):
            from PySide6.QtWidgets import QDialog, QGridLayout
            dlg = QDialog(self)
            dlg.setWindowTitle("Scegli la soluzione")
            g = QGridLayout(dlg)
            for c, (name, r) in enumerate(results):
                v = ZoomView(); v.setMinimumSize(260, 320); v.set_image(r.simulated_srgb)
                e = r.estimate
                g.addWidget(QLabel(f"<b>{name}</b>"), 0, c)
                g.addWidget(v, 1, c)
                g.addWidget(QLabel(f"ΔE medio previsto: <b>{r.metrics['delta_e2000_medio']:.1f}</b><br>"
                                   f"Filamenti: {len(r.solution.filaments)} · cambi: {e['cambi_colore']}<br>"
                                   f"Tempo ~{e['tempo_min_stimato']:.0f} min · spurgo ~{e['spurgo_g_stimato']} g<br>"
                                   f"<small>stime, da verificare con lo slicer</small>"), 2, c)
                b = QPushButton("Scegli questa")
                b.clicked.connect(lambda _=False, r=r: (dlg.accept(), self._use_result(r, db)))
                g.addWidget(b, 3, c)
            dlg.resize(900, 600)
            dlg.exec()
        self._run_task(work, choose, "Calcolo delle 3 soluzioni")

    def _use_result(self, r, db):
        self.job_db = db
        self.show_result(r)
        self.lb_phase.setText("Soluzione scelta: puoi esportarla")

    def export_miniature(self):
        if not self.result:
            QMessageBox.information(self, "Miniatura", "Elabora prima l'immagine.")
            return
        import copy
        p, _ = QFileDialog.getSaveFileName(self, "Salva miniatura", str(Path.home() / "colorforge_prova_40mm.3mf"),
                                           "Progetto 3MF (*.3mf)")
        if not p:
            return
        if not p.lower().endswith(".3mf"):
            p += ".3mf"
        s = copy.deepcopy(self.result.settings)
        s.width_mm, s.height_mm = 40.0, 0.0
        s.pixel_mm = max(s.pixel_mm, 0.4)
        db, image = getattr(self, "result_db", self.db), self.image
        if image is None:
            QMessageBox.information(self, "Miniatura", "Carica la fotografia originale.")
            return

        def work(prog):
            r = run(image, db, s, progress=lambda f, m: prog(f * 0.7, m))
            export_3mf(r, p, progress=lambda f, m: prog(0.7 + 0.3 * f, m), template=self.bambu_template())
            return p
        self._run_task(work, lambda path: self._done_msg("Miniatura di prova", [path]), "Miniatura di prova")

    def export_3mf_file(self):
        if not self.result:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Esporta 3MF", str(Path.home() / "colorforge.3mf"),
                                           "Progetto 3MF (*.3mf)")
        if not p:
            return
        if not p.lower().endswith(".3mf"):
            p += ".3mf"
        res = self.result
        tpl = self.bambu_template()

        def work(prog):
            probs = export_3mf(res, p, progress=prog, template=tpl)
            return p, probs.get("_slot_mapping", [])

        def done(r):
            path, notes = r
            if tpl is None:
                extra = ["", "Suggerimento: imposta il tuo progetto Bambu (File → Imposta progetto Bambu di "
                         "riferimento) per avere gli slot già assegnati."]
            else:
                extra = ["", "Progetto Bambu Studio: le parti sono già assegnate agli slot AMS."]
                if notes:
                    extra += ["Colori senza filamento esatto, assegnati allo slot più simile:"] + notes
            self._done_msg("Esportazione 3MF", [path], extra)
        self._run_task(work, done, "Esportazione 3MF")

    def export(self):
        if not self.result:
            return
        d = QFileDialog.getExistingDirectory(self, "Cartella di esportazione", str(Path.home()))
        if not d:
            return
        res = self.result
        tpl = self.bambu_template()
        self._run_task(lambda prog: export_all(res, d, progress=prog, template=tpl),
                       lambda files: self._done_msg("Esportazione", list(files.values())), "Esportazione")

    def save_project(self):
        if not self.result:
            QMessageBox.information(self, "Progetto", "Nessun risultato da salvare.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Salva progetto", "progetto.cfproj", "Progetto ColorForge (*.cfproj)")
        if p:
            try:
                save_project(p, self.result, getattr(self, 'result_db', self.db))
            except OSError as e:
                QMessageBox.warning(self, "Progetto", str(e))

    def closeEvent(self, e):
        if self.worker:
            self.worker.cancel = True
        for th in [self.thread] + [x[0] for x in self._bg]:
            if th is not None and th.isRunning():
                th.quit()
                th.wait(15000)
        super().closeEvent(e)

    def open_project(self):
        p, _ = QFileDialog.getOpenFileName(self, "Apri progetto", "", "Progetto ColorForge (*.cfproj)")
        if not p:
            return
        try:
            res, fdb = load_project(p)
        except (ValueError, OSError, KeyError) as e:
            QMessageBox.warning(self, "Progetto", f"Impossibile aprire il progetto: {e}")
            return
        self.db.merge(fdb)
        self.slots.update(res.settings.slots)
        self.refresh_filaments()
        self.show_result(res)
