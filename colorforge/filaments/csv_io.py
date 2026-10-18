"""Libreria filamenti da/verso CSV (es. valori TD presi da HueForge o dai produttori).

Colonne riconosciute (intestazione obbligatoria, separatore , o ;):
  marca, nome, materiale, colore (#RRGGBB), td (mm, facoltativo), fonte (facoltativo),
  prezzo_eur_kg (facoltativo), slot (facoltativo)
I valori importati sono marcati come "dichiarati", non misurati.
"""
from __future__ import annotations

import csv
import io
import re

from colorforge.filaments.model import Filament, Reliability

ALIASES = {"marca": "brand", "brand": "brand", "vendor": "brand", "nome": "name", "name": "name",
           "materiale": "material", "material": "material", "type": "material", "colore": "hex", "color": "hex",
           "colour": "hex", "hex": "hex", "td": "td", "transmission distance": "td", "fonte": "source",
           "source": "source", "prezzo_eur_kg": "price", "price": "price", "slot": "slot"}
HEX = re.compile(r"^#?[0-9A-Fa-f]{6}$")


def read_csv(text: str) -> tuple[list[tuple[Filament, int | None]], list[str]]:
    sample = text[:2000]
    delim = ";" if sample.count(";") > sample.count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    if not rows:
        return [], ["File vuoto"]
    head = [ALIASES.get(h.strip().lower(), h.strip().lower()) for h in rows[0]]
    if "hex" not in head or "name" not in head:
        return [], ["Intestazione non riconosciuta: servono almeno le colonne 'nome' e 'colore'"]
    out, errors = [], []
    for n, r in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in r):
            continue
        d = {head[i]: (r[i].strip() if i < len(r) else "") for i in range(len(head))}
        hx = d.get("hex", "")
        if not HEX.match(hx):
            errors.append(f"Riga {n}: colore non valido '{hx}'")
            continue
        hx = "#" + hx.lstrip("#").upper()
        td = None
        if d.get("td"):
            try:
                td = float(d["td"].replace(",", "."))
                if not 0 < td < 100:
                    raise ValueError
            except ValueError:
                errors.append(f"Riga {n}: TD non valida '{d['td']}' (ignorata)")
                td = None
        slug = re.sub(r"[^a-z0-9]+", "-", f"{d.get('brand', '')}-{d['name']}-{hx}".lower()).strip("-")
        f = Filament.from_hex(f"csv-{slug}", d.get("brand", ""), d["name"], hx,
                              material=(d.get("material") or "PLA").upper(),
                              color_reliability=Reliability.DECLARED, td_mm=td,
                              td_reliability=Reliability.DECLARED if td else Reliability.UNKNOWN,
                              source=d.get("source") or "Importato da CSV (valori dichiarati, non misurati)",
                              owned_spools=1)
        if d.get("price"):
            try:
                f.print_params["prezzo_eur_kg"] = float(d["price"].replace(",", "."))
            except ValueError:
                pass
        slot = int(d["slot"]) if d.get("slot", "").isdigit() else None
        out.append((f, slot))
    return out, errors


def write_csv(filaments, slots: dict[str, int] | None = None) -> str:
    from colorforge.color.spaces import srgb_to_hex
    b = io.StringIO()
    w = csv.writer(b, delimiter=";")
    w.writerow(["marca", "nome", "materiale", "colore", "td", "fonte", "prezzo_eur_kg", "slot"])
    for f in filaments:
        w.writerow([f.brand, f.name, f.material, srgb_to_hex(f.srgb), f.td_mm or "", f.source,
                    f.print_params.get("prezzo_eur_kg", ""), (slots or {}).get(f.id, "")])
    return b.getvalue()
