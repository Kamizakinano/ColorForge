"""Tema chiaro/scuro, guida rapida e controllo aggiornamenti."""
from __future__ import annotations

import json
import urllib.request

from PySide6.QtGui import QColor, QPalette

RELEASES_API = "https://api.github.com/repos/Kamizakinano/ColorForge/releases?per_page=5"

GUIDE_HTML = """<h3>ColorForge AI in 6 passi</h3><ol>
<li><b>File → Imposta progetto Bambu di riferimento</b> (una volta): importa i tuoi filamenti AMS.</li>
<li><b>Carica fotografia</b>.</li>
<li>Nella tavolozza al centro spunta i colori da usare, oppure lascia tutto vuoto per usare i tuoi filamenti.</li>
<li>Con il <b>pennello</b> dipingi in verde le zone importanti (volti) e in rosso quelle da semplificare.</li>
<li><b>✨ Proponi 3 soluzioni</b> oppure <b>▶ Elabora</b>; regola <i>Fedeltà ↔ Meno cambi</i>.</li>
<li><b>🔬 Miniatura di prova</b>, poi <b>Esporta 3MF</b> e aprilo in Bambu Studio: gli slot sono già assegnati.</li>
</ol><p>Per previsioni più precise: <b>Calibrazione → Calibra i filamenti</b> e, dopo una stampa,
<b>Confronta con la stampa reale</b>.</p>"""


ACCENT = "#FF7846"

_COMMON = """
QGroupBox { font-weight: 600; border: 1px solid {line}; border-radius: 8px; margin-top: 10px; padding-top: 2px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {accent}; }
QPushButton { border: 1px solid {line}; border-radius: 6px; padding: 5px 10px; background: {button}; }
QPushButton:hover { border-color: {accent}; }
QPushButton:pressed, QPushButton:checked { background: {pressed}; border-color: {accent}; }
QPushButton:disabled { color: {muted}; border-color: {line}; }
QPushButton#primary { background: {accent}; color: #1B1F2A; font-weight: 700; border: none; }
QPushButton#primary:hover { background: #FF8C62; }
QPushButton#primary:disabled { background: {line}; color: {muted}; }
QTabBar::tab { padding: 6px 14px; border: none; border-bottom: 2px solid transparent; background: transparent; }
QTabBar::tab:selected { border-bottom: 2px solid {accent}; color: {accent}; font-weight: 600; }
QHeaderView::section { background: {header}; border: none; border-bottom: 1px solid {line}; padding: 5px; font-weight: 600; }
QTableWidget, QListWidget, QTextEdit { border: 1px solid {line}; border-radius: 6px; gridline-color: {line}; }
QComboBox, QSpinBox, QDoubleSpinBox { border: 1px solid {line}; border-radius: 5px; padding: 1px 6px; background: {field}; min-height: 20px; }
QTableWidget QSpinBox, QTableWidget QDoubleSpinBox, QTableWidget QComboBox { border: none; border-radius: 0; background: transparent; }
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: {accent}; }
QProgressBar { border: 1px solid {line}; border-radius: 5px; text-align: center; background: {field}; }
QProgressBar::chunk { background: {accent}; border-radius: 4px; }
QSlider::groove:horizontal { height: 4px; background: {line}; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -6px 0; border-radius: 7px; background: {accent}; }
QSlider::sub-page:horizontal { background: {accent}; border-radius: 2px; }
QCheckBox::indicator, QListView::indicator, QTableView::indicator { width: 14px; height: 14px;
    border: 1px solid {ind}; border-radius: 3px; background: {field}; }
QCheckBox::indicator:checked, QListView::indicator:checked, QTableView::indicator:checked {
    background: {accent}; border-color: {accent}; }
QCheckBox::indicator:disabled { border-color: {line}; }
QToolTip { border: 1px solid {accent}; padding: 4px; }
QStatusBar { border-top: 1px solid {line}; }
"""

DARK_TOKENS = dict(accent=ACCENT, line="#3A3F4D", button="#2A2E3A", pressed="#3A2E2A", muted="#6D7280",
                   header="#262A35", field="#16181E", ind="#9097A6")
LIGHT_TOKENS = dict(accent="#E8602C", line="#D3D7E0", button="#FFFFFF", pressed="#FFE6DC", muted="#A0A4AE",
                    header="#EEF0F4", field="#FFFFFF", ind="#7A808C")


def stylesheet(dark: bool) -> str:
    t = DARK_TOKENS if dark else LIGHT_TOKENS
    out = _COMMON
    for k, v in t.items():
        out = out.replace("{" + k + "}", v)
    return out


def apply_theme(app, dark: bool) -> None:
    app.setStyle("Fusion")
    if not dark:
        app.setPalette(app.style().standardPalette())
        app.setStyleSheet(stylesheet(False))
        return
    p = QPalette()
    base, alt, text = QColor(30, 32, 40), QColor(40, 43, 54), QColor(225, 228, 235)
    for role, c in [(QPalette.ColorRole.Window, base), (QPalette.ColorRole.WindowText, text),
                    (QPalette.ColorRole.Base, QColor(22, 24, 30)), (QPalette.ColorRole.AlternateBase, alt),
                    (QPalette.ColorRole.ToolTipBase, alt), (QPalette.ColorRole.ToolTipText, text),
                    (QPalette.ColorRole.Text, text), (QPalette.ColorRole.Button, alt),
                    (QPalette.ColorRole.ButtonText, text), (QPalette.ColorRole.Highlight, QColor(255, 120, 70)),
                    (QPalette.ColorRole.HighlightedText, QColor(0, 0, 0)), (QPalette.ColorRole.Link, QColor(110, 170, 255))]:
        p.setColor(role, c)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(120, 120, 130))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(120, 120, 130))
    app.setPalette(p)
    app.setStyleSheet(stylesheet(True))


def quality_badge(de: float) -> tuple[str, str]:
    """(testo, colore) per un valore ΔE medio previsto."""
    if de < 3:
        return "Eccellente", "#2E9E44"
    if de < 6:
        return "Buona", "#5BAA3C"
    if de < 10:
        return "Discreta", "#D9A400"
    return "Bassa", "#C62828"


def fetch_latest_release(timeout: float = 6.0):
    """(numero build, url pagina) dell'ultima release 'build-N', o None."""
    try:
        req = urllib.request.Request(RELEASES_API, headers={"Accept": "application/vnd.github+json",
                                                            "User-Agent": "ColorForge-AI"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    best = None
    for rel in data:
        tag = str(rel.get("tag_name", ""))
        if tag.startswith("build-") and tag[6:].isdigit():
            n = int(tag[6:])
            if best is None or n > best[0]:
                best = (n, rel.get("html_url", ""))
    return best
