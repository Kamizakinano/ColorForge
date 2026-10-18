"""Finestra "I miei filamenti": inserimento manuale di colori, TD e slot AMS."""
from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from colorforge.color.spaces import hex_to_srgb, srgb_to_hex, srgb_to_lab
from colorforge.filaments.database import FilamentDatabase, default_user_db_path
from colorforge.filaments.model import Filament, Reliability

COLS = ["Usa", "Slot AMS", "Colore", "HEX", "Marca", "Nome", "Materiale", "TD (mm)", "Origine TD", "Note"]
REL_LABELS = {Reliability.UNKNOWN: "Sconosciuta", Reliability.ESTIMATED: "Stimata",
              Reliability.DECLARED: "Dichiarata/HueForge", Reliability.PHOTO: "Da foto",
              Reliability.MEASURED: "Misurata"}
HEX_RE = re.compile(r"^#?[0-9A-Fa-f]{6}$")


class FilamentEditor(QDialog):
    def __init__(self, db: FilamentDatabase, slots: dict[str, int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("I miei filamenti")
        self.resize(1180, 560)
        self.db, self.slots = db, dict(slots)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Inserisci i filamenti che possiedi. Colore: clicca sul riquadro o scrivi il codice HEX. "
            "TD: valore di trasparenza (es. da HueForge); se lo lasci a 0 viene considerato sconosciuto.\n"
            "Spunta 'Usa' per i filamenti disponibili e indica lo slot AMS (1-8, 0 = automatico)."))
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setDefaultSectionSize(32)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table)
        row = QHBoxLayout()
        for text, fn in (("Aggiungi", self.add_row), ("Duplica", self.duplicate), ("Elimina", self.delete),
                         ("Importa…", self.import_db), ("Esporta…", self.export_db),
                         ("Importa CSV…", self.import_csv), ("Esporta CSV…", self.export_csv)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Save).setText("Salva")
        bb.button(QDialogButtonBox.StandardButton.Save).setObjectName("primary")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("Annulla")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        for f in db:
            self._append(f)
        self.table.itemChanged.connect(self._hex_edited)

    # --- righe ---------------------------------------------------------
    def _append(self, f: Filament):
        r = self.table.rowCount()
        self.table.insertRow(r)
        use = QTableWidgetItem()
        use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        use.setCheckState(Qt.CheckState.Checked if f.owned_spools > 0 else Qt.CheckState.Unchecked)
        use.setData(Qt.ItemDataRole.UserRole, f.id)
        self.table.setItem(r, 0, use)
        slot = QSpinBox(); slot.setRange(0, 16); slot.setValue(self.slots.get(f.id, 0))
        self.table.setCellWidget(r, 1, slot)
        hx = srgb_to_hex(f.srgb)
        sw = QPushButton(); sw.clicked.connect(lambda _=False, b=sw: self._pick(b))
        self.table.setCellWidget(r, 2, sw)
        self.table.setItem(r, 3, QTableWidgetItem(hx))
        self._set_swatch(sw, hx)
        for c, v in ((4, f.brand), (5, f.name), (6, f.material), (9, f.calibration_notes or f.source)):
            self.table.setItem(r, c, QTableWidgetItem(v))
        td = QDoubleSpinBox(); td.setRange(0, 50); td.setDecimals(2); td.setSingleStep(0.1)
        td.setValue(f.td_mm or 0.0)
        self.table.setCellWidget(r, 7, td)
        rel = QComboBox()
        for k, lbl in REL_LABELS.items():
            rel.addItem(lbl, k.value)
        rel.setCurrentIndex(max(0, rel.findData(f.td_reliability.value)))
        self.table.setCellWidget(r, 8, rel)

    def _set_swatch(self, b: QPushButton, hx: str):
        b.setStyleSheet(f"background:{hx}; border:1px solid #666; min-width:40px;")
        b.setProperty("hex", hx)

    def _row_of(self, w: QWidget) -> int:
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 2) is w:
                return r
        return -1

    def _pick(self, b: QPushButton):
        c = QColorDialog.getColor(QColor(b.property("hex")), self, "Colore del filamento stampato")
        if c.isValid():
            r = self._row_of(b)
            self.table.item(r, 3).setText(c.name().upper())

    def _hex_edited(self, item: QTableWidgetItem):
        if item.column() != 3:
            return
        t = item.text().strip()
        if HEX_RE.match(t):
            t = "#" + t.lstrip("#").upper()
            self._set_swatch(self.table.cellWidget(item.row(), 2), t)

    def add_row(self):
        n = 1
        while f"utente-{n}" in {self._id(r) for r in range(self.table.rowCount())}:
            n += 1
        f = Filament.from_hex(f"utente-{n}", "", "Nuovo filamento", "#808080",
                              color_reliability=Reliability.DECLARED, owned_spools=1,
                              source="Inserito manualmente dall'utente")
        self._append(f)
        self.table.scrollToBottom()

    def duplicate(self):
        r = self.table.currentRow()
        if r < 0:
            return
        f = self._read_row(r)
        f.id = f"{f.id}-copia"
        self._append(f)

    def delete(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def _id(self, r):
        return self.table.item(r, 0).data(Qt.ItemDataRole.UserRole)

    def _read_row(self, r: int) -> Filament:
        fid = self._id(r)
        hx = self.table.item(r, 3).text().strip()
        if not HEX_RE.match(hx):
            raise ValueError(f"Riga {r + 1}: codice colore non valido '{hx}' (formato #RRGGBB)")
        base = self.db.get(fid) if fid in self.db else None
        td = self.table.cellWidget(r, 7).value()
        rel = Reliability(self.table.cellWidget(r, 8).currentData())
        f = Filament.from_dict(base.to_dict()) if base else Filament(id=fid, brand="", name="")
        new_lab = srgb_to_lab(hex_to_srgb("#" + hx.lstrip("#"))).tolist()
        if base is None or srgb_to_hex(base.srgb) != "#" + hx.lstrip("#").upper():
            f.lab = new_lab
            f.color_reliability = Reliability.DECLARED
            f.km_scattering = None if base is None else f.km_scattering
        f.brand = self.table.item(r, 4).text()
        f.name = self.table.item(r, 5).text()
        f.material = self.table.item(r, 6).text()
        f.calibration_notes = self.table.item(r, 9).text()
        f.td_mm = td if td > 0 else None
        f.td_reliability = rel if td > 0 else Reliability.UNKNOWN
        if td > 0 and rel == Reliability.UNKNOWN:
            f.td_reliability = Reliability.DECLARED
        f.owned_spools = 1 if self.table.item(r, 0).checkState() == Qt.CheckState.Checked else 0
        s = self.table.cellWidget(r, 1).value()
        if s:
            self.slots[fid] = s
        else:
            self.slots.pop(fid, None)
        return f

    def collect(self) -> FilamentDatabase:
        out = FilamentDatabase(calibration_history=self.db.calibration_history)
        for r in range(self.table.rowCount()):
            out.add(self._read_row(r), overwrite=True)
        return out

    def accept(self):
        try:
            self.db = self.collect()
            self.db.save(default_user_db_path())
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Filamenti", str(e))
            return
        super().accept()

    def import_db(self):
        p, _ = QFileDialog.getOpenFileName(self, "Importa filamenti", "", "JSON (*.json)")
        if not p:
            return
        try:
            other = FilamentDatabase.load(p)
        except (ValueError, OSError, KeyError) as e:
            QMessageBox.warning(self, "Importazione", f"File non valido: {e}")
            return
        existing = {self._id(r) for r in range(self.table.rowCount())}
        for f in other:
            if f.id not in existing:
                self._append(f)

    def export_db(self):
        p, _ = QFileDialog.getSaveFileName(self, "Esporta filamenti", "filamenti.json", "JSON (*.json)")
        if p:
            try:
                self.collect().save(p)
            except (ValueError, OSError) as e:
                QMessageBox.warning(self, "Esportazione", str(e))

    def import_csv(self):
        from colorforge.filaments.csv_io import read_csv
        p, _ = QFileDialog.getOpenFileName(self, "Importa libreria CSV", "", "CSV (*.csv *.txt)")
        if not p:
            return
        try:
            with open(p, encoding="utf-8-sig") as fh:
                items, errors = read_csv(fh.read())
        except (OSError, UnicodeDecodeError) as e:
            QMessageBox.warning(self, "CSV", f"Impossibile leggere il file: {e}")
            return
        existing = {self._id(r) for r in range(self.table.rowCount())}
        added = 0
        for f, slot in items:
            if f.id in existing:
                continue
            if slot:
                self.slots[f.id] = slot
            self._append(f)
            added += 1
        msg = f"Importati {added} filamenti (valori dichiarati, non misurati)."
        if errors:
            msg += "\n\nRighe scartate:\n" + "\n".join(errors[:15])
        QMessageBox.information(self, "CSV", msg)

    def export_csv(self):
        from colorforge.filaments.csv_io import write_csv
        p, _ = QFileDialog.getSaveFileName(self, "Esporta CSV", "filamenti.csv", "CSV (*.csv)")
        if p:
            try:
                db = self.collect()
                with open(p, "w", encoding="utf-8-sig", newline="") as fh:
                    fh.write(write_csv(list(db), self.slots))
            except (ValueError, OSError) as e:
                QMessageBox.warning(self, "CSV", str(e))
