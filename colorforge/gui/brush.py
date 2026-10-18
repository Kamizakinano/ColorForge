"""Pennello sull'immagine originale: proteggi (verde) / semplifica (rosso) / gomma."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QGraphicsPixmapItem, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QVBoxLayout, QWidget)

from colorforge.pipeline import PROTECT, SIMPLIFY

MAX_SIDE = 900  # risoluzione della maschera (lato lungo)


class BrushView(QWidget):
    """Avvolge una ZoomView aggiungendo una maschera dipingibile sopra l'immagine."""
    changed = Signal()

    def __init__(self, view):
        super().__init__()
        self.view = view
        self.mask: np.ndarray | None = None
        self.mode = 0          # 0 = sposta/zoom, 1 proteggi, 2 semplifica, 3 gomma
        self.radius = 18
        self.overlay = QGraphicsPixmapItem()
        self.overlay.setZValue(1)
        self.view.scene().addItem(self.overlay)
        self._painting = False
        bar = QHBoxLayout()
        self.group = QButtonGroup(self)
        for i, (txt, tip) in enumerate([("✋", "Sposta e zoom"), ("🟩 Proteggi", "Zone da riprodurre con la massima cura"),
                                        ("🟥 Semplifica", "Zone meno importanti: meno dettagli e meno cambi"),
                                        ("Gomma", "Cancella le pennellate")]):
            b = QPushButton(txt); b.setCheckable(True); b.setToolTip(tip)
            self.group.addButton(b, i); bar.addWidget(b)
        self.group.button(0).setChecked(True)
        self.group.idClicked.connect(self._set_mode)
        bar.addWidget(QLabel("Pennello"))
        self.sl = QSlider(Qt.Orientation.Horizontal); self.sl.setRange(3, 80); self.sl.setValue(18)
        self.sl.valueChanged.connect(lambda v: setattr(self, "radius", v))
        bar.addWidget(self.sl, 1)
        clr = QPushButton("Pulisci"); clr.clicked.connect(self.clear); bar.addWidget(clr)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar); lay.addWidget(view)
        view.viewport().installEventFilter(self)

    def _set_mode(self, i):
        self.mode = i
        self.view.setDragMode(self.view.DragMode.ScrollHandDrag if i == 0 else self.view.DragMode.NoDrag)

    def set_image(self, srgb):
        self.view.set_image(srgb)
        h, w = srgb.shape[:2]
        k = max(1.0, max(h, w) / MAX_SIDE)
        self.scale = k
        self.mask = np.zeros((int(round(h / k)), int(round(w / k))), np.uint8)
        self.img_size = (w, h)
        self._refresh()

    def set_mask(self, m):
        if m is not None and self.mask is not None:
            from PIL import Image
            self.mask = np.asarray(Image.fromarray(np.asarray(m, np.uint8)).resize(self.mask.shape[::-1], Image.Resampling.NEAREST)).copy()
            self._refresh()

    def clear(self):
        if self.mask is not None and self.mask.any():
            self.mask[:] = 0
            self._refresh()
            self.changed.emit()

    def _refresh(self):
        if self.mask is None:
            return
        rgba = np.zeros(self.mask.shape + (4,), np.uint8)
        rgba[self.mask == PROTECT] = (40, 200, 90, 110)
        rgba[self.mask == SIMPLIFY] = (230, 60, 60, 110)
        h, w = self.mask.shape
        qi = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        pm = QPixmap.fromImage(qi).scaled(self.img_size[0], self.img_size[1])
        self.overlay.setPixmap(pm)

    def _paint(self, pos):
        p = self.view.mapToScene(pos.toPoint())
        cx, cy = p.x() / self.scale, p.y() / self.scale
        r = self.radius / max(self.view.transform().m11(), 1e-6) / self.scale
        h, w = self.mask.shape
        yy, xx = np.ogrid[0:h, 0:w]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        self.mask[disk] = {1: PROTECT, 2: SIMPLIFY, 3: 0}[self.mode]
        self._refresh()

    def eventFilter(self, obj, ev):
        if self.mode == 0 or self.mask is None:
            return False
        t = ev.type()
        if t == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
            self._painting = True; self._paint(ev.position()); return True
        if t == QEvent.Type.MouseMove and self._painting:
            self._paint(ev.position()); return True
        if t == QEvent.Type.MouseButtonRelease and self._painting:
            self._painting = False; self.changed.emit(); return True
        return False

    def user_mask(self):
        return self.mask if self.mask is not None and self.mask.any() else None
