"""Finestra di calibrazione: crea il provino, leggilo da foto o colorimetro, salva."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGraphicsEllipseItem, QHBoxLayout,
                               QHeaderView, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

from colorforge.calibration.fit import apply_fits, fit_specimen, sample_photo, white_balance
from colorforge.calibration.specimen import SpecimenSpec, export_specimen
from colorforge.color.spaces import linear_to_lab, srgb_to_hex, srgb_to_lab, srgb_to_linear
from colorforge.filaments.database import default_user_db_path
from colorforge.imaging.importer import ImageImportError, load_image

CORNER_NAMES = ["alto a sinistra (vicino all'aletta)", "alto a destra", "basso a destra", "basso a sinistra"]


class ClickView(QWidget):
    """Vista con zoom che registra i clic sugli angoli del provino."""
    clicked = Signal(float, float)

    def __init__(self, zoom_view_cls):
        super().__init__()
        self.view = zoom_view_cls()
        self.view.setDragMode(self.view.DragMode.NoDrag)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self.view.viewport().installEventFilter(self)
        self.marks = []

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
            p = self.view.mapToScene(ev.position().toPoint())
            self.clicked.emit(p.x(), p.y())
            return True
        return False

    def mark(self, x, y):
        r = max(4.0, self.view.item.boundingRect().width() / 150)
        e = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        e.setPen(QPen(QColor("#FF00AA"), r / 2))
        e.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.view.scene().addItem(e)
        self.marks.append(e)

    def clear_marks(self):
        for m in self.marks:
            self.view.scene().removeItem(m)
        self.marks = []


class CalibrationDialog(QDialog):
    def __init__(self, db, slots, parent=None, zoom_view_cls=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrazione filamenti")
        self.resize(1100, 720)
        self.db, self.slots = db, slots
        self.spec: SpecimenSpec | None = None
        self.photo = None
        self.corners: list[tuple[float, float]] = []
        self.patch_lab = None
        self.fits = None
        self.method = "photo"
        tabs = QTabWidget()
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        tabs.addTab(self._tab_create(), "1 · Crea il provino")
        tabs.addTab(self._tab_read(zoom_view_cls), "2 · Leggi il provino")
        tabs.addTab(self._tab_results(), "3 · Risultati")
        self.tabs = tabs

    # --- 1: creazione ------------------------------------------------------
    def _tab_create(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel(
            "Il provino è una piastra del filamento chiaro di base con una scala di spessori (1–24 strati) "
            "per ogni filamento da calibrare.\nStampalo con le stesse impostazioni che userai per i quadri "
            "(altezza strato, temperatura)."))
        f = QFormLayout()
        self.cb_base = QComboBox()
        fl = sorted(self.db, key=lambda x: -x.lab[0])
        for x in fl:
            self.cb_base.addItem(f"{x.label}  {srgb_to_hex(x.srgb)}", x.id)
        f.addRow("Filamento di base (chiaro)", self.cb_base)
        self.sp_layer = QDoubleSpinBox(); self.sp_layer.setRange(0.04, 0.32); self.sp_layer.setValue(0.08); self.sp_layer.setSingleStep(0.02)
        f.addRow("Altezza strato (mm)", self.sp_layer)
        v.addLayout(f)
        v.addWidget(QLabel("Filamenti da calibrare (massimo 8 per provino):"))
        self.lst = QListWidget()
        self.lst.setSpacing(2)
        self.lst.setAlternatingRowColors(True)
        from PySide6.QtGui import QIcon, QPixmap
        for x in fl:
            it = QListWidgetItem(f"  {x.label}   {srgb_to_hex(x.srgb)}")
            pm = QPixmap(18, 18); pm.fill(QColor(srgb_to_hex(x.srgb))); it.setIcon(QIcon(pm))
            it.setData(Qt.ItemDataRole.UserRole, x.id)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if x.owned_spools > 0 else Qt.CheckState.Unchecked)
            self.lst.addItem(it)
        v.addWidget(self.lst, 1)
        b = QPushButton("Crea provino (3MF)…"); b.setObjectName("primary")
        b.clicked.connect(self.create_specimen)
        v.addWidget(b)
        return w

    def create_specimen(self):
        ids = [self.lst.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.lst.count())
               if self.lst.item(i).checkState() == Qt.CheckState.Checked]
        base = self.cb_base.currentData()
        ids = [i for i in ids if i != base]
        if not ids:
            QMessageBox.information(self, "Provino", "Seleziona almeno un filamento diverso dalla base.")
            return
        if len(ids) > 8:
            QMessageBox.information(self, "Provino", "Massimo 8 filamenti per provino: ne verranno usati i primi 8.")
            ids = ids[:8]
        spec = SpecimenSpec(base, ids, layer_mm=self.sp_layer.value())
        p, _ = QFileDialog.getSaveFileName(self, "Salva provino", str(Path.home() / f"provino_{spec.id}.3mf"),
                                           "3MF (*.3mf)")
        if not p:
            return
        try:
            slots = export_specimen(spec, p, self.db, self.slots)
        except OSError as e:
            QMessageBox.warning(self, "Provino", str(e))
            return
        self.spec = spec
        w, h = spec.size_mm
        QMessageBox.information(self, "Provino creato",
                                f"Provino {spec.id} ({w:.0f} × {h:.0f} mm) salvato in:\n{p}\n\n"
                                f"Il file {Path(p).with_suffix('.cfcal').name} accanto contiene la disposizione: "
                                "serve per leggere il provino.\n\nSlot: " +
                                ", ".join(f"{self.db.get(f).label} → {s}" for f, s in slots.items()))

    # --- 2: lettura -------------------------------------------------------
    def _tab_read(self, zoom_view_cls):
        w = QWidget()
        h = QHBoxLayout(w)
        left = QVBoxLayout()
        b = QPushButton("Apri file del provino (.cfcal)…"); b.clicked.connect(self.open_spec); left.addWidget(b)
        self.lb_spec = QLabel("Nessun provino caricato"); self.lb_spec.setWordWrap(True); left.addWidget(self.lb_spec)
        left.addWidget(QLabel("<b>Metodo A – Foto</b><br>Fotografa il provino dall'alto, con luce diurna "
                              "diffusa, senza ombre né riflessi."))
        b = QPushButton("Apri foto del provino…"); b.clicked.connect(self.open_photo); left.addWidget(b)
        self.lb_corner = QLabel(""); self.lb_corner.setWordWrap(True); left.addWidget(self.lb_corner)
        b = QPushButton("Ricomincia i clic"); b.clicked.connect(self.reset_corners); left.addWidget(b)
        left.addWidget(QLabel("<b>Metodo B – Colorimetro</b><br>Inserisci i valori L*a*b* misurati "
                              "di ogni quadratino nella tabella a destra."))
        b = QPushButton("Tabella per colorimetro"); b.clicked.connect(self.show_lab_table); left.addWidget(b)
        left.addStretch(1)
        b = QPushButton("▶ Calcola i parametri ottici"); b.setObjectName("primary"); b.clicked.connect(self.compute); left.addWidget(b)
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(320)
        h.addWidget(lw)
        from colorforge.gui.main_window import ZoomView
        self.click = ClickView(zoom_view_cls or ZoomView)
        self.click.clicked.connect(self.add_corner)
        self.lab_table = QTableWidget()
        self.lab_table.hide()
        h.addWidget(self.click, 1)
        h.addWidget(self.lab_table, 1)
        return w

    def open_spec(self):
        p, _ = QFileDialog.getOpenFileName(self, "File del provino", str(Path.home()), "Provino ColorForge (*.cfcal)")
        if p:
            try:
                self.load_spec(SpecimenSpec.load(p))
            except (ValueError, OSError, TypeError) as e:
                QMessageBox.warning(self, "Provino", f"File non valido: {e}")

    def load_spec(self, spec: SpecimenSpec):
        missing = [f for f in [spec.base_id] + spec.filament_ids if f not in self.db]
        if missing:
            QMessageBox.warning(self, "Provino", "Filamenti non presenti nel database: " + ", ".join(missing))
            return
        self.spec = spec
        self.lb_spec.setText(f"Provino {spec.id} del {spec.created}: base {self.db.get(spec.base_id).label}, "
                             f"{len(spec.filament_ids)} filamenti, {len(spec.columns)} quadratini per riga.")
        self.reset_corners()

    def open_photo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Foto del provino", str(Path.home()),
                                           "Immagini (*.png *.jpg *.jpeg *.tif *.tiff *.webp)")
        if not p:
            return
        try:
            self.photo = load_image(p).srgb
        except ImageImportError as e:
            QMessageBox.warning(self, "Foto", str(e))
            return
        self.lab_table.hide(); self.click.show()
        self.click.view.set_image(self.photo)
        self.method = "photo"
        self.reset_corners()

    def reset_corners(self):
        self.corners = []
        self.click.clear_marks()
        self._corner_hint()

    def _corner_hint(self):
        if self.photo is None:
            self.lb_corner.setText("")
        elif len(self.corners) < 4:
            self.lb_corner.setText(f"Clicca l'angolo della piastra: <b>{CORNER_NAMES[len(self.corners)]}</b> "
                                   "(esclusa l'aletta).")
        else:
            self.lb_corner.setText("Angoli acquisiti. Premi 'Calcola'.")

    def add_corner(self, x, y):
        if self.photo is None or len(self.corners) >= 4:
            return
        self.corners.append((x, y))
        self.click.mark(x, y)
        self._corner_hint()

    def show_lab_table(self):
        if not self.spec:
            QMessageBox.information(self, "Colorimetro", "Apri prima il file del provino.")
            return
        self.method = "colorimeter"
        s = self.spec
        self.lab_table.clear()
        self.lab_table.setRowCount(len(s.filament_ids) * len(s.columns))
        self.lab_table.setColumnCount(5)
        self.lab_table.setHorizontalHeaderLabels(["Filamento", "Strati", "L*", "a*", "b*"])
        r = 0
        for i, fid in enumerate(s.filament_ids):
            for n in s.columns:
                for c, v in enumerate([self.db.get(fid).label, "base" if n == 0 else str(n)]):
                    it = QTableWidgetItem(v); it.setFlags(Qt.ItemFlag.ItemIsEnabled); self.lab_table.setItem(r, c, it)
                for c in (2, 3, 4):
                    self.lab_table.setItem(r, c, QTableWidgetItem(""))
                r += 1
        self.lab_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.click.hide(); self.lab_table.show()

    def _lab_from_table(self) -> np.ndarray:
        s = self.spec
        out = np.zeros((len(s.filament_ids), len(s.columns), 3))
        r = 0
        for i in range(len(s.filament_ids)):
            for c in range(len(s.columns)):
                vals = [self.lab_table.item(r, k).text().replace(",", ".").strip() for k in (2, 3, 4)]
                try:
                    out[i, c] = [float(x) for x in vals]
                except ValueError:
                    raise ValueError(f"Valore mancante o non valido alla riga {r + 1} della tabella")
                r += 1
        return out

    def compute(self):
        if not self.spec:
            QMessageBox.information(self, "Calibrazione", "Apri prima il file del provino.")
            return
        try:
            if self.method == "colorimeter":
                lab = self._lab_from_table()
            else:
                if self.photo is None or len(self.corners) != 4:
                    QMessageBox.information(self, "Calibrazione", "Apri la foto e clicca i 4 angoli.")
                    return
                srgb = sample_photo(self.photo, self.corners, self.spec)
                lin = white_balance(srgb_to_linear(srgb), self.db.get(self.spec.base_id).linear_rgb)
                lab = linear_to_lab(lin)
            self.patch_lab = lab
            self.fits = fit_specimen(self.spec, lab)
        except ValueError as e:
            QMessageBox.warning(self, "Calibrazione", str(e))
            return
        self._show_results()
        self.tabs.setCurrentIndex(2)

    # --- 3: risultati -------------------------------------------------------
    def _tab_results(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.res_table = QTableWidget(0, 6)
        self.res_table.setHorizontalHeaderLabels(["Filamento", "Colore prima", "Colore misurato", "TD (mm)",
                                                  "Errore di adattamento ΔE", "Esito"])
        self.res_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.res_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.res_table.setAlternatingRowColors(True)
        self.res_table.setShowGrid(False)
        self.res_table.verticalHeader().setVisible(False)
        v.addWidget(self.res_table, 1)
        self.ck_force = QCheckBox("Salva anche le stime segnalate come poco affidabili")
        v.addWidget(self.ck_force)
        v.addWidget(QLabel("Le misure da foto valgono meno di quelle da colorimetro: il software ne tiene conto "
                           "nell'affidabilità delle previsioni."))
        b = QPushButton("💾 Salva nel database dei filamenti"); b.setObjectName("primary"); b.clicked.connect(self.save)
        v.addWidget(b)
        return w

    def _show_results(self):
        from colorforge.color.spaces import lab_to_srgb
        self.res_table.setRowCount(len(self.fits))
        for r, fit in enumerate(self.fits):
            f = self.db.get(fit.filament_id)
            vals = [f.label, srgb_to_hex(f.srgb), srgb_to_hex(lab_to_srgb(fit.r_inf_lab)),
                    f"{fit.td_mm:.2f}", f"{fit.rms_de:.2f}", fit.message]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c in (1, 2):
                    it.setBackground(QColor(v))
                    it.setForeground(QColor("#000000" if QColor(v).lightness() > 128 else "#FFFFFF"))
                self.res_table.setItem(r, c, it)

    def save(self):
        if not self.fits:
            return
        done = apply_fits(self.db, self.spec, self.patch_lab, self.fits, self.method, self.ck_force.isChecked())
        try:
            self.db.save(default_user_db_path())
        except OSError as e:
            QMessageBox.warning(self, "Calibrazione", str(e))
            return
        QMessageBox.information(self, "Calibrazione", f"Aggiornati {len(done)} filamenti su {len(self.fits)}.")
        self.accept()


class PrintFeedbackDialog(QDialog):
    """Confronto tra la stampa reale (foto) e la previsione, con correzione del modello."""

    def __init__(self, res, db, parent, zoom_view_cls, run_task):
        super().__init__(parent)
        self.setWindowTitle("Confronta con la stampa reale")
        self.resize(1200, 720)
        self.res, self.db, self.run_task = res, db, run_task
        self.photo, self.corners, self.fit = None, [], None
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(QLabel("1. Fotografa la stampa dall'alto, con luce diffusa.\n"
                              "2. Clicca i 4 angoli dell'immagine stampata.\n"
                              "3. Il programma confronta i colori reali con quelli previsti e corregge "
                              "trasparenza e colore dei filamenti usati."))
        b = QPushButton("Apri foto della stampa…"); b.clicked.connect(self.open_photo); left.addWidget(b)
        self.lb = QLabel(""); self.lb.setWordWrap(True); left.addWidget(self.lb)
        b = QPushButton("Ricomincia i clic"); b.clicked.connect(self.reset); left.addWidget(b)
        b = QPushButton("▶ Confronta e calcola la correzione"); b.clicked.connect(self.compute); left.addWidget(b)
        self.lb_res = QLabel(""); self.lb_res.setWordWrap(True); left.addWidget(self.lb_res)
        self.bt_apply = QPushButton("💾 Applica la correzione ai filamenti"); self.bt_apply.setEnabled(False)
        self.bt_apply.clicked.connect(self.apply); left.addWidget(self.bt_apply)
        left.addStretch(1)
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(320)
        lay.addWidget(lw)
        self.click = ClickView(zoom_view_cls)
        self.click.clicked.connect(self.add_corner)
        lay.addWidget(self.click, 1)
        self.cmp = zoom_view_cls()
        lay.addWidget(self.cmp, 1)
        self.reset()

    def open_photo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Foto della stampa", str(Path.home()),
                                           "Immagini (*.png *.jpg *.jpeg *.tif *.tiff *.webp)")
        if not p:
            return
        try:
            self.photo = load_image(p).srgb
        except ImageImportError as e:
            QMessageBox.warning(self, "Foto", str(e))
            return
        self.click.view.set_image(self.photo)
        self.reset()

    def reset(self):
        self.corners = []
        self.click.clear_marks()
        self._hint()

    def _hint(self):
        names = ["alto a sinistra", "alto a destra", "basso a destra", "basso a sinistra"]
        if self.photo is None:
            self.lb.setText("Nessuna foto.")
        elif len(self.corners) < 4:
            self.lb.setText(f"Clicca l'angolo <b>{names[len(self.corners)]}</b> dell'immagine stampata.")
        else:
            self.lb.setText("Angoli acquisiti.")

    def add_corner(self, x, y):
        if self.photo is None or len(self.corners) >= 4:
            return
        self.corners.append((x, y))
        self.click.mark(x, y)
        self._hint()

    def compute(self):
        if self.photo is None or len(self.corners) != 4:
            QMessageBox.information(self, "Confronto", "Apri la foto e clicca i 4 angoli.")
            return
        from colorforge.calibration.feedback import refit_from_print
        res, db, photo, corners = self.res, self.db, self.photo, list(self.corners)
        self.run_task(lambda prog: refit_from_print(res, db, photo, corners), self._done,
                      "Confronto con la stampa")

    def _done(self, fit):
        self.fit = fit
        gap = np.ones((fit["rectified"].shape[0], 4, 3))
        self.cmp.set_image(np.concatenate([fit["rectified"], gap, self.res.simulated_srgb], 1))
        better = fit["after"] < fit["before"] - 0.2
        self.lb_res.setText(f"Differenza media tra stampa reale e previsione su {fit['n']} punti:<br>"
                            f"<b>prima ΔE {fit['before']:.1f}</b> → <b>dopo la correzione ΔE {fit['after']:.1f}</b><br>"
                            + ("La correzione migliora il modello." if better else
                               "Nessun miglioramento significativo: il database non verrà modificato.")
                            + "<br><small>A destra: stampa raddrizzata | previsione.</small>")
        self.bt_apply.setEnabled(better)

    def apply(self):
        from colorforge.calibration.feedback import apply_refit
        done = apply_refit(self.db, self.fit)
        try:
            self.db.save(default_user_db_path())
        except OSError as e:
            QMessageBox.warning(self, "Correzione", str(e))
            return
        QMessageBox.information(self, "Correzione", f"Aggiornati {len(done)} filamenti. "
                                "Rielabora l'immagine per usare i nuovi valori.")
        self.accept()
