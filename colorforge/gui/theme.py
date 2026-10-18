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


def apply_theme(app, dark: bool) -> None:
    app.setStyle("Fusion")
    if not dark:
        app.setPalette(app.style().standardPalette())
        app.setStyleSheet("")
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
    app.setStyleSheet(
        "QCheckBox::indicator, QListView::indicator { width: 14px; height: 14px; border: 1px solid #9097A6;"
        " border-radius: 3px; background: #16181E; }"
        "QCheckBox::indicator:checked, QListView::indicator:checked { background: #FF7846; border-color: #FF7846; }"
        "QCheckBox::indicator:disabled { border-color: #50545E; }")


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
