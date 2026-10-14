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


def _mesh_xml(m: Mesh) -> str:
    vs = "\n".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in m.vertices)
    ts = "\n".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in m.faces)
    return f"<mesh><vertices>\n{vs}\n</vertices><triangles>\n{ts}\n</triangles></mesh>"


def write_3mf(parts: list[tuple[str, str, int, Mesh]], path: str | Path,
              metadata: dict[str, str] | None = None, offset_xy=(0.0, 0.0)) -> None:
    """`parts`: lista di (nome, colore_hex, slot_estrusore_1based, mesh)."""
    metadata = metadata or {}
    mats = "\n".join(f'<base name="{escape(n)}" displaycolor="{c}FF"/>' for n, c, _, _ in parts)
    objs, items, cfg = [], [], []
    for i, (name, color, slot, mesh) in enumerate(parts):
        oid = i + 2
        objs.append(f'<object id="{oid}" name="{escape(name)}" type="model" pid="1" pindex="{i}">'
                    f"{_mesh_xml(mesh)}</object>")
        items.append(f'<item objectid="{oid}" transform="1 0 0 0 1 0 0 0 1 {offset_xy[0]:.3f} {offset_xy[1]:.3f} 0" printable="1"/>')
        cfg.append(f'<object id="{oid}">\n <metadata key="name" value="{escape(name)}"/>\n'
                   f' <metadata key="extruder" value="{slot}"/>\n</object>')
    meta = "\n".join(f'<metadata name="{escape(k)}">{escape(v)}</metadata>' for k, v in metadata.items())
    model = f"""<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="it-IT" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
{meta}
<resources>
<basematerials id="1">
{mats}
</basematerials>
{chr(10).join(objs)}
</resources>
<build>
{chr(10).join(items)}
</build>
</model>"""
    settings = '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n' + "\n".join(cfg) + "\n</config>"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", model)


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
