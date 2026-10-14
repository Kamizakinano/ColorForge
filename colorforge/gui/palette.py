"""Tavolozza dei colori della foto: l'utente spunta i colori che ha davvero."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QColorDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from colorforge.pipeline import image_palette


class PalettePanel(QWidget):
    """Elenco dei colori principali dell'immagine, ciascuno spuntabile e modificabile."""
    changed = Signal()

    def __init__(self, to_pixmap):
        super().__init__()
        self._to_pixmap = to_pixmap
        self._image = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        t = QLabel("<b>Colori dell'immagine</b><br><small>Spunta i colori dei filamenti che hai. "
                   "Doppio clic per correggere un colore.</small>")
        t.setWordWrap(True)
        lay.addWidget(t)
        row = QHBoxLayout()
        row.addWidget(QLabel("Colori estratti"))
        self.sp_k = QSpinBox(); self.sp_k.setRange(4, 32); self.sp_k.setValue(16)
        self.sp_k.valueChanged.connect(self.extract)
        row.addWidget(self.sp_k)
        lay.addLayout(row)
        self.list = QListWidget()
        self.list.itemChanged.connect(lambda _i: self.changed.emit())
        self.list.itemDoubleClicked.connect(self._edit)
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, fn in (("Tutti", lambda: self._set_all(True)), ("Nessuno", lambda: self._set_all(False))):
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        lay.addLayout(row)
        self.bt_add = QPushButton("Salva tra i miei filamenti")
        lay.addWidget(self.bt_add)
        self.lb_info = QLabel("")
        self.lb_info.setWordWrap(True)
        lay.addWidget(self.lb_info)
        self.setMinimumWidth(190)

    def set_image(self, srgb: np.ndarray | None):
        self._image = srgb
        self.extract()

    def extract(self):
        self.list.blockSignals(True)
        self.list.clear()
        if self._image is not None:
            for hx, w in image_palette(self._image, self.sp_k.value()):
                self._add(hx, w)
        self.list.blockSignals(False)
        self.changed.emit()

    def _add(self, hx: str, w: float):
        it = QListWidgetItem()
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(Qt.CheckState.Unchecked)
        self._style(it, hx, w)
        self.list.addItem(it)

    def _style(self, it: QListWidgetItem, hx: str, w: float):
        rgb = np.array([int(hx[i:i + 2], 16) for i in (1, 3, 5)]) / 255
        it.setIcon(self._to_pixmap(np.ones((20, 32, 3)) * rgb))
        it.setText(f"{hx}   {w * 100:4.1f}%")
        it.setData(Qt.ItemDataRole.UserRole, (hx, w))

    def _edit(self, it: QListWidgetItem):
        hx, w = it.data(Qt.ItemDataRole.UserRole)
        c = QColorDialog.getColor(QColor(hx), self, "Colore del filamento reale")
        if c.isValid():
            self._style(it, c.name().upper(), w)
            it.setCheckState(Qt.CheckState.Checked)
            self.changed.emit()

    def _set_all(self, on: bool):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.list.blockSignals(False)
        self.changed.emit()

    def selected(self) -> list[str]:
        out = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                hx = it.data(Qt.ItemDataRole.UserRole)[0]
                if hx not in out:
                    out.append(hx)
        return out
