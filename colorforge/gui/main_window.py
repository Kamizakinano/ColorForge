"""Finestra principale di ColorForge AI."""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox,
                               QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from colorforge.color.spaces import srgb_to_hex
from colorforge.filaments.database import FilamentDatabase
from colorforge.gui.filament_editor import FilamentEditor
from colorforge.imaging.importer import ImageImportError, LoadedImage, load_image
from colorforge.pipeline import Cancelled, JobResult, JobSettings, build_report, export_all, run
from colorforge.printers.profiles import load_printers
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
        self.sp_pixel = QDoubleSpinBox(); self.sp_pixel.setRange(0.1, 2); self.sp_pixel.setSingleStep(0.05); self.sp_pixel.setValue(0.3); self.sp_pixel.setSuffix(" mm")
        self.sp_layer = QDoubleSpinBox(); self.sp_layer.setRange(0.04, 0.32); self.sp_layer.setSingleStep(0.02); self.sp_layer.setValue(0.08); self.sp_layer.setSuffix(" mm")
        self.sp_base = QDoubleSpinBox(); self.sp_base.setRange(0.2, 5); self.sp_base.setValue(0.64); self.sp_base.setSuffix(" mm")
        self.sp_max = QDoubleSpinBox(); self.sp_max.setRange(0.5, 10); self.sp_max.setValue(2.4); self.sp_max.setSuffix(" mm")
        self.sp_colors = QSpinBox(); self.sp_colors.setRange(1, 16); self.sp_colors.setValue(8)
        self.cb_quality = QComboBox()
        for lbl, k in QUALITY_LABELS:
            self.cb_quality.addItem(lbl, k)
        self.sp_width.valueChanged.connect(self._update_height)
        for lbl, w in (("Larghezza", self.sp_width), ("Altezza (auto)", self.lb_height),
                       ("Risoluzione", self.sp_pixel), ("Altezza strato", self.sp_layer),
                       ("Spessore base", self.sp_base), ("Spessore massimo", self.sp_max),
                       ("Colori massimi", self.sp_colors), ("Qualità", self.cb_quality)):
            f.addRow(lbl, w)
        sl.addWidget(g)

        g = QGroupBox("Filamenti disponibili")
        v = QVBoxLayout(g)
        self.lst_fil = QListWidget()
        v.addWidget(self.lst_fil)
        b = QPushButton("🎨  I miei filamenti…")
        b.clicked.connect(self.edit_filaments)
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

        self.bt_run = QPushButton("▶  Elabora automaticamente")
        self.bt_run.setStyleSheet("font-weight:bold; padding:8px;")
        self.bt_run.clicked.connect(self.start)
        self.bt_cancel = QPushButton("Annulla"); self.bt_cancel.setEnabled(False)
        self.bt_cancel.clicked.connect(self.cancel)
        self.bt_export = QPushButton("💾  Esporta progetto di stampa…"); self.bt_export.setEnabled(False)
        self.bt_export.clicked.connect(self.export)
        for w in (self.bt_run, self.bt_cancel, self.bt_export):
            sl.addWidget(w)

        self.tabs = QTabWidget()
        self.v_orig, self.v_prep, self.v_sim, self.v_err, self.v_cmp = (ZoomView() for _ in range(5))
        for v, t in ((self.v_orig, "Originale"), (self.v_prep, "Preparata"), (self.v_sim, "Risultato previsto"),
                     (self.v_cmp, "Confronto"), (self.v_err, "Mappa errori ΔE")):
            self.tabs.addTab(v, t)
        self.txt = QTextEdit(); self.txt.setReadOnly(True)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.tabs)
        right.addWidget(self.txt)
        right.setSizes([600, 240])
        split = QSplitter()
        split.addWidget(side)
        split.addWidget(right)
        split.setSizes([360, 1040])
        self.setCentralWidget(split)

        self.progress = QProgressBar(); self.progress.setRange(0, 1000)
        self.lb_phase = QLabel("Pronto")
        self.statusBar().addWidget(self.lb_phase, 1)
        self.statusBar().addPermanentWidget(self.progress)

        m = self.menuBar().addMenu("&File")
        for text, fn in (("Apri fotografia…", self.open_image), ("Apri progetto…", self.open_project),
                         ("Salva progetto…", self.save_project), ("Esporta…", self.export)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            m.addAction(a)
        self._printer_changed()

    # --- stato -------------------------------------------------------------
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

    def settings(self) -> JobSettings:
        s = JobSettings(printer_id=self.cb_printer.currentData(), changer_id=self.cb_changer.currentData(),
                        filament_ids=self.selected_filaments(), slots=dict(self.slots),
                        width_mm=self.sp_width.value(), pixel_mm=self.sp_pixel.value(),
                        max_colors=self.sp_colors.value(), quality=self.cb_quality.currentData())
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
        self.tabs.setCurrentWidget(self.v_orig)
        self._update_height()
        self.txt.setPlainText("Fotografia caricata.\n" + "\n".join(self.image.notes))

    def edit_filaments(self):
        dlg = FilamentEditor(self.db, self.slots, self)
        if dlg.exec():
            self.db, self.slots = dlg.db, dlg.slots
            self.refresh_filaments()

    def start(self):
        if self.image is None:
            QMessageBox.information(self, "ColorForge", "Carica prima una fotografia.")
            return
        s = self.settings()
        if not s.filament_ids:
            QMessageBox.information(self, "ColorForge", "Seleziona almeno un filamento.")
            return
        self.bt_run.setEnabled(False); self.bt_cancel.setEnabled(True); self.bt_export.setEnabled(False)
        self.thread = QThread(self)
        self.worker = Worker(self.image, self.db, s)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.done.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

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
        self.progress.setValue(0)
        self.lb_phase.setText("Interrotto")
        QMessageBox.warning(self, "Elaborazione", msg)

    def _on_done(self, res: JobResult):
        self._finish_ui()
        self.progress.setValue(1000)
        self.lb_phase.setText(f"Completato in {res.elapsed_s:.0f} s")
        self.show_result(res)

    def show_result(self, res: JobResult):
        self.result = res
        self.bt_export.setEnabled(True)
        self.v_orig.set_image(res.original_srgb)
        self.v_prep.set_image(res.prepared_srgb)
        self.v_sim.set_image(res.simulated_srgb)
        self.v_err.set_image(res.error_map_rgb())
        gap = np.ones((res.prepared_srgb.shape[0], 4, 3))
        self.v_cmp.set_image(np.concatenate([res.prepared_srgb, gap, res.simulated_srgb], 1))
        self.tabs.setCurrentWidget(self.v_cmp)
        r = build_report(res)
        m = res.metrics
        html = [f"<h3>Risultato previsto (simulazione, non garanzia del risultato fisico)</h3>",
                f"<p>ΔE2000 medio <b>{m['delta_e2000_medio']:.2f}</b> · 95° percentile {m['delta_e2000_p95']:.2f} · "
                f"SSIM {m['ssim_luminanza']:.3f} · affidabilità dati <b>{res.reliability:.2f}</b></p>",
                "<p><b>Sequenza filamenti (dal basso):</b></p><ol>"]
        for sw in res.swaps:
            html.append(f"<li><span style='background:{sw['colore']}'>&nbsp;&nbsp;&nbsp;&nbsp;</span> "
                        f"{sw['filamento']} — slot {sw['slot']} — dallo strato {sw['strato_inizio']} "
                        f"(z={sw['z_inizio_mm']} mm)</li>")
        e = r["stima_stampa"]
        html.append(f"</ol><p>Altezza {e['altezza_totale_mm']} mm · {e['numero_strati']} strati · "
                    f"{e['cambi_colore']} cambi · ~{e['tempo_min_stimato']:.0f} min (stima)</p>")
        if res.warnings:
            html.append("<p><b>Avvisi:</b></p><ul>" + "".join(f"<li>{w}</li>" for w in res.warnings) + "</ul>")
        html.append("<p><b>Operazioni:</b></p><ul>" + "".join(f"<li>{w}</li>" for w in res.log) + "</ul>")
        self.txt.setHtml("".join(html))

    def export(self):
        if not self.result:
            return
        d = QFileDialog.getExistingDirectory(self, "Cartella di esportazione")
        if not d:
            return
        try:
            files = export_all(self.result, d)
        except OSError as e:
            QMessageBox.warning(self, "Esportazione", f"Impossibile scrivere i file: {e}")
            return
        QMessageBox.information(self, "Esportazione", "File creati:\n" + "\n".join(files.values()))

    def save_project(self):
        if not self.result:
            QMessageBox.information(self, "Progetto", "Nessun risultato da salvare.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Salva progetto", "progetto.cfproj", "Progetto ColorForge (*.cfproj)")
        if p:
            try:
                save_project(p, self.result, self.db)
            except OSError as e:
                QMessageBox.warning(self, "Progetto", str(e))

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
