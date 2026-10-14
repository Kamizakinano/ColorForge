"""Esportazione 3MF (core spec 1.x + materiali base) con un oggetto per fascia di colore.

Ogni fascia (intervallo di quote stampato con un filamento) è un oggetto
separato, chiuso, con il proprio materiale `basematerials` e il colore di
visualizzazione. Lo slot AMS è nel nome dell'oggetto ("slot N").

Nota: Bambu Studio 2.8.2.60 rifiuta un `Metadata/model_settings.config`
scritto da terzi ("configurazione non valida"), quindi non viene più
generato: il file è un 3MF standard (solo geometria + materiali base). In alternativa resta sempre valido
il flusso STL singolo + cambi colore alle quote indicate nel report.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from colorforge.geometry.mesh import Mesh

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="config" ContentType="text/xml"/>
 <Default Extension="png" ContentType="image/png"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""


def _write_mesh(fh, m: Mesh) -> None:
    """Scrive la mesh in streaming (file grandi: centinaia di MB senza tenerli in memoria)."""
    fh.write(b"<mesh><vertices>\n")
    np.savetxt(fh, m.vertices, fmt='<vertex x="%.4f" y="%.4f" z="%.4f"/>')
    fh.write(b"</vertices><triangles>\n")
    np.savetxt(fh, m.faces, fmt='<triangle v1="%d" v2="%d" v3="%d"/>')
    fh.write(b"</triangles></mesh>")


def write_3mf(parts: list[tuple[str, str, int, Mesh]], path: str | Path,
              metadata: dict[str, str] | None = None, offset_xy=(0.0, 0.0),
              progress=None) -> None:
    """`parts`: lista di (nome, colore_hex, slot_estrusore_1based, mesh).
    Scrive prima su file temporaneo e poi rinomina: nessun file a metà."""
    metadata = metadata or {}
    path = Path(path)
    mats = "\n".join(f'<base name="{escape(n)}" displaycolor="{c}FF"/>' for n, c, _, _ in parts)
    meta = "\n".join(f'<metadata name="{escape(k)}">{escape(v)}</metadata>' for k, v in metadata.items())
    head = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="it-IT" '
            f'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n{meta}\n<resources>\n'
            f'<basematerials id="1">\n{mats}\n</basematerials>\n')
    tmp = path.with_suffix(path.suffix + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        with z.open("3D/3dmodel.model", "w", force_zip64=True) as fh:
            fh.write(head.encode("utf-8"))
            for i, (name, _c, _slot, mesh) in enumerate(parts):
                fh.write(f'<object id="{i + 2}" name="{escape(name)}" type="model" pid="1" pindex="{i}">'.encode())
                _write_mesh(fh, mesh)
                fh.write(b"</object>\n")
                if progress:
                    progress((i + 1) / len(parts), f"Scrittura 3MF: oggetto {i + 1}/{len(parts)}")
            items = "\n".join(
                f'<item objectid="{i + 2}" transform="1 0 0 0 1 0 0 0 1 {offset_xy[0]:.3f} {offset_xy[1]:.3f} 0" printable="1"/>'
                for i in range(len(parts)))
            fh.write(f"</resources>\n<build>\n{items}\n</build>\n</model>".encode())
    import os
    os.replace(tmp, path)


def read_3mf_meshes(path: str | Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Rilettura minimale per i test di integrità."""
    import xml.etree.ElementTree as ET
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("3D/3dmodel.model"))
    out = []
    for o in root.iterfind(".//m:object", ns):
        v = np.array([[float(e.get(k)) for k in "xyz"] for e in o.iterfind(".//m:vertex", ns)])
        t = np.array([[int(e.get(k)) for k in ("v1", "v2", "v3")] for e in o.iterfind(".//m:triangle", ns)])
        out.append((o.get("name"), v, t))
    return out
