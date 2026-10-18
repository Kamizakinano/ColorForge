"""Tavolozza dei colori della foto, in forma di tabella ordinata.

Ogni riga: spunta · campione · copertura (barra) · tuo filamento più simile · somiglianza.
Spuntare un colore significa "usalo come filamento"; nessuna spunta = si usano i
filamenti posseduti. Doppio clic sul campione per correggere il colore.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QColorDialog, QHBoxLayout, QHeaderView, QLabel,
                               QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from colorforge.pipeline import image_palette


def _quality(de: float) -> tuple[str, str]:
    if de < 6:
        return "Ottimo", "#2E9E44"
    if de < 15:
        return "Simile", "#D9A400"
    return "Lontano", "#C62828"


class PalettePanel(QWidget):
    changed = Signal()

    def __init__(self, to_pixmap):
        super().__init__()
        self._image = None
        self._rows: list[list] = []      # [hex, frazione]
        self.nearest = None              # hex -> (nome, ΔE, hex del filamento)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 4)
        lay.setSpacing(6)
        title = QLabel("<b>Colori dell'immagine</b>")
        hint = QLabel("<small>Spunta i colori da usare come filamenti. Nessuna spunta = uso i tuoi filamenti. "
                      "Doppio clic su un campione per correggerlo.</small>")
        hint.setWordWrap(True)
        top = QHBoxLayout()
        top.addWidget(title, 1)
        top.addWidget(QLabel("Colori:"))
        self.sp_k = QSpinBox(); self.sp_k.setRange(4, 32); self.sp_k.setValue(16)
        self.sp_k.setToolTip("Quanti colori principali estrarre dall'immagine")
        self.sp_k.valueChanged.connect(self.extract)
        top.addWidget(self.sp_k)
        lay.addLayout(top)
        lay.addWidget(hint)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Colore", "Copertura", "Tuo filamento più simile", "Somiglianza"])
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        hh = self.table.horizontalHeader()
        for c, mode in enumerate([QHeaderView.ResizeMode.ResizeToContents, QHeaderView.ResizeMode.ResizeToContents,
                                  QHeaderView.ResizeMode.Fixed, QHeaderView.ResizeMode.Stretch,
                                  QHeaderView.ResizeMode.ResizeToContents]):
            hh.setSectionResizeMode(c, mode)
        self.table.setColumnWidth(2, 80)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self._edit)
        self.table.itemChanged.connect(lambda it: self.changed.emit() if it.column() == 0 else None)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, fn in (("Seleziona tutti", lambda: self._set_all(True)), ("Nessuno", lambda: self._set_all(False))):
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        lay.addLayout(row)
        self.bt_add = QPushButton("Salva i colori spuntati tra i miei filamenti")
        lay.addWidget(self.bt_add)
        self.lb_sum = QLabel("")
        lay.addWidget(self.lb_sum)
        self.setMinimumWidth(420)

    # --- dati -----------------------------------------------------------------
    def set_image(self, srgb):
        self._image = srgb
        self.extract()

    def extract(self):
        self._rows = [] if self._image is None else [[hx, w] for hx, w in image_palette(self._image, self.sp_k.value())]
        self._fill()
        self.changed.emit()

    def _fill(self, checked: set | None = None):
        checked = checked or set()
        t = self.table
        t.blockSignals(True)
        t.setRowCount(len(self._rows))
        for r, (hx, w) in enumerate(self._rows):
            ck = QTableWidgetItem()
            ck.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            ck.setCheckState(Qt.CheckState.Checked if r in checked else Qt.CheckState.Unchecked)
            t.setItem(r, 0, ck)
            sw = QTableWidgetItem(hx)
            sw.setBackground(QBrush(QColor(hx)))
            sw.setForeground(QBrush(QColor("#000000" if QColor(hx).lightness() > 140 else "#FFFFFF")))
            sw.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            sw.setToolTip("Doppio clic per correggere il colore")
            t.setItem(r, 1, sw)
            bar = QProgressBar(); bar.setRange(0, 1000); bar.setValue(int(w * 1000))
            bar.setFormat(f"{w * 100:.1f}%"); bar.setTextVisible(True); bar.setMaximumHeight(18)
            t.setCellWidget(r, 2, bar)
            name, de, fhx = (self.nearest(hx) if self.nearest else (None, 0.0, None))
            if name:
                cell = QTableWidgetItem(f"  {name}")
                cell.setToolTip(f"{name}  {fhx}")
                if fhx:
                    cell.setIcon(self._icon(fhx))
                t.setItem(r, 3, cell)
                q, col = _quality(de)
                badge = QTableWidgetItem(f" {q} ")
                badge.setForeground(QBrush(QColor("#FFFFFF")))
                badge.setBackground(QBrush(QColor(col)))
                badge.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                badge.setToolTip(f"Differenza ΔE {de:.1f} (0 = identico; sotto 6 ottimo, sotto 15 simile)")
                t.setItem(r, 4, badge)
            else:
                t.setItem(r, 3, QTableWidgetItem("  —"))
                t.setItem(r, 4, QTableWidgetItem(""))
        t.blockSignals(False)
        self._summary()

    def _icon(self, hx):
        from PySide6.QtGui import QIcon, QPixmap
        pm = QPixmap(16, 16); pm.fill(QColor(hx)); return QIcon(pm)

    def _summary(self):
        n = len(self.selected())
        cov = sum(w for r, (_, w) in enumerate(self._rows) if self._checked(r)) * 100
        self.lb_sum.setText(f"<small>{n} colori spuntati · coprono il {cov:.0f}% dell'immagine</small>" if n
                            else "<small>Nessun colore spuntato: verranno usati i tuoi filamenti.</small>")

    def _checked(self, r):
        it = self.table.item(r, 0)
        return it is not None and it.checkState() == Qt.CheckState.Checked

    def _edit(self, r, c):
        if c != 1:
            return
        cdlg = QColorDialog.getColor(QColor(self._rows[r][0]), self, "Colore del filamento reale")
        if cdlg.isValid():
            keep = {i for i in range(len(self._rows)) if self._checked(i)} | {r}
            self._rows[r][0] = cdlg.name().upper()
            self._fill(keep)
            self.changed.emit()

    def _set_all(self, on: bool):
        self._fill(set(range(len(self._rows))) if on else set())
        self.changed.emit()

    # --- API usata dalla finestra e dai test -------------------------------------
    def count(self) -> int:
        return len(self._rows)

    def set_checked(self, r: int, on: bool = True):
        self.table.item(r, 0).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._summary()

    def selected(self) -> list[str]:
        out = []
        for r, (hx, _) in enumerate(self._rows):
            if self._checked(r) and hx not in out:
                out.append(hx)
        return out
