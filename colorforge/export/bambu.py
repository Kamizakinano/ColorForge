"""Progetto Bambu Studio (.3mf) con filamenti già assegnati.

Struttura ricavata da un progetto reale salvato con Bambu Studio 2.8.2.60:
  3D/3dmodel.model                 oggetti mesh + oggetto "assieme" con <components>,
                                   metadata Application = BambuStudio-…
  Metadata/model_settings.config   per oggetto/parte: nome, extruder (= slot filamento),
                                   matrice; piatto con model_instance; assemble
  Metadata/project_settings.config impostazioni di stampante/filamenti, prese dal
                                   progetto modello dell'utente (i suoi slot AMS)
  Metadata/plate_1.png             miniatura

Tutte le parti appartengono a un unico oggetto: ogni parte ha il proprio
`extruder`, quindi Bambu Studio la stampa con lo slot corretto senza
assegnazioni manuali.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

from colorforge.export.threemf import _write_mesh
from colorforge.geometry.mesh import Mesh

NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_BBL = "http://schemas.bambulab.com/package/2021"


@dataclass
class BambuTemplate:
    path: str
    settings: dict
    version: str

    @property
    def filaments(self) -> list[dict]:
        s = self.settings
        n = len(s.get("filament_colour", []))
        get = lambda k, i, d="": (s.get(k) or [d] * n)[i] if i < len(s.get(k) or []) else d
        return [{"slot": i + 1, "colour": get("filament_colour", i, "#808080").upper()[:7],
                 "type": get("filament_type", i, "PLA"), "vendor": get("filament_vendor", i, ""),
                 "profile": get("filament_settings_id", i, "")} for i in range(n)]

    @property
    def printer(self) -> str:
        return str(self.settings.get("printer_model", ""))

    @property
    def nozzle(self) -> float:
        v = self.settings.get("nozzle_diameter", ["0.4"])
        return float(v[0] if isinstance(v, list) else v)

    @property
    def bed_center(self) -> tuple[float, float]:
        pts = self.settings.get("printable_area") or ["0x0", "256x0", "256x256", "0x256"]
        xy = np.array([[float(a) for a in p.split("x")] for p in pts])
        return float(xy[:, 0].mean()), float(xy[:, 1].mean())


def load_template(path: str | Path) -> BambuTemplate:
    try:
        with zipfile.ZipFile(path) as z:
            settings = json.loads(z.read("Metadata/project_settings.config"))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as e:
        raise ValueError("Il file non è un progetto Bambu Studio salvato (manca project_settings.config).") from e
    return BambuTemplate(str(path), settings, str(settings.get("version", "")))


def _thumb(srgb: np.ndarray, size: int) -> bytes:
    img = Image.fromarray((np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    img.thumbnail((size, size))
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    b = io.BytesIO()
    canvas.save(b, "PNG")
    return b.getvalue()


def write_bambu_project(parts: list[tuple[str, int, Mesh]], path: str | Path, template: BambuTemplate,
                        layer_height: float, first_layer_height: float, preview_srgb: np.ndarray | None = None,
                        title: str = "ColorForge", solid_infill: bool = True, progress=None) -> None:
    """`parts`: [(nome, slot 1-based, mesh in mm con origine nell'angolo)]."""
    path = Path(path)
    n_fil = len(template.filaments)
    for name, slot, _ in parts:
        if not 1 <= slot <= max(n_fil, 1):
            raise ValueError(f"La parte '{name}' usa lo slot {slot}, ma il progetto modello ha {n_fil} filamenti.")
    settings = dict(template.settings)
    # le quote dei cambi colore dipendono dall'altezza di strato: devono coincidere
    settings["layer_height"] = f"{layer_height:g}"
    settings["initial_layer_print_height"] = f"{first_layer_height:g}"
    if solid_infill:  # interno pieno: la trasparenza calcolata presuppone materiale compatto
        settings["sparse_infill_density"] = "100%"
    settings["from"] = "project"

    lo = np.min([m.vertices.min(0) for _, _, m in parts], axis=0)
    hi = np.max([m.vertices.max(0) for _, _, m in parts], axis=0)
    cx, cy = template.bed_center
    tx, ty = cx - (lo[0] + hi[0]) / 2, cy - (lo[1] + hi[1]) / 2
    asm_id = len(parts) + 1

    head = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
            f'xmlns="{NS_CORE}" xmlns:BambuStudio="{NS_BBL}">\n'
            f' <metadata name="Application">BambuStudio-{template.version or "02.08.02.60"}</metadata>\n'
            f' <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
            f' <metadata name="Title">{escape(title)}</metadata>\n'
            f' <metadata name="Designer">ColorForge AI</metadata>\n <resources>\n')

    fd, tmp = tempfile.mkstemp(prefix="colorforge_", suffix=".3mf.tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
            z.writestr("[Content_Types].xml",
                       '<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
                       ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
                       ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
                       ' <Default Extension="png" ContentType="image/png"/>\n'
                       ' <Default Extension="gcode" ContentType="text/x.gcode"/>\n</Types>')
            rels = ['<Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>']
            if preview_srgb is not None:
                z.writestr("Metadata/plate_1.png", _thumb(preview_srgb, 512))
                z.writestr("Metadata/plate_1_small.png", _thumb(preview_srgb, 128))
                rels += ['<Relationship Target="/Metadata/plate_1.png" Id="rel-2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>',
                         '<Relationship Target="/Metadata/plate_1.png" Id="rel-4" Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-middle"/>',
                         '<Relationship Target="/Metadata/plate_1_small.png" Id="rel-5" Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-small"/>']
            z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n '
                       + "\n ".join(rels) + "\n</Relationships>")
            with z.open("3D/3dmodel.model", "w", force_zip64=True) as fh:
                fh.write(head.encode())
                for i, (name, slot, mesh) in enumerate(parts):
                    fh.write(f'  <object id="{i + 1}" type="model">\n'.encode())
                    _write_mesh(fh, mesh)
                    fh.write(b"\n  </object>\n")
                    if progress:
                        progress((i + 1) / len(parts), f"Scrittura progetto Bambu: parte {i + 1}/{len(parts)}")
                comps = "\n".join(f'    <component objectid="{i + 1}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>'
                                  for i in range(len(parts)))
                fh.write(f'  <object id="{asm_id}" type="model">\n   <components>\n{comps}\n   </components>\n  </object>\n'
                         f' </resources>\n <build>\n  <item objectid="{asm_id}" transform="1 0 0 0 1 0 0 0 1 '
                         f'{tx:.4f} {ty:.4f} {-lo[2]:.4f}" printable="1"/>\n </build>\n</model>\n'.encode())
            part_cfg = []
            for i, (name, slot, mesh) in enumerate(parts):
                part_cfg.append(
                    f'    <part id="{i + 1}" subtype="normal_part" uuid="{uuid.uuid4()}">\n'
                    f'      <metadata key="name" value="{escape(name)}"/>\n'
                    f'      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
                    f'      <metadata key="extruder" value="{slot}"/>\n'
                    f'      <mesh_stat face_count="{len(mesh.faces)}" edges_fixed="0" degenerate_facets="0" '
                    f'facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n    </part>')
            model_settings = (
                '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n'
                f'  <object id="{asm_id}">\n    <metadata key="name" value="{escape(title)}"/>\n'
                f'    <metadata key="extruder" value="{parts[0][1]}"/>\n'
                f'    <metadata face_count="{sum(len(m.faces) for _, _, m in parts)}"/>\n'
                + "\n".join(part_cfg) + "\n  </object>\n"
                '  <plate>\n    <metadata key="plater_id" value="1"/>\n    <metadata key="plater_name" value=""/>\n'
                '    <metadata key="locked" value="false"/>\n    <metadata key="filament_map_mode" value="Auto For Flush"/>\n'
                + ('    <metadata key="thumbnail_file" value="Metadata/plate_1.png"/>\n' if preview_srgb is not None else "")
                + f'    <model_instance>\n      <metadata key="object_id" value="{asm_id}"/>\n'
                '      <metadata key="instance_id" value="0"/>\n      <metadata key="identify_id" value="100"/>\n'
                '    </model_instance>\n  </plate>\n  <assemble>\n'
                f'   <assemble_item object_id="{asm_id}" instance_id="0" transform="1 0 0 0 1 0 0 0 1 '
                f'{tx:.4f} {ty:.4f} {-lo[2]:.4f}" offset="0 0 0" />\n  </assemble>\n</config>\n')
            z.writestr("Metadata/model_settings.config", model_settings)
            z.writestr("Metadata/project_settings.config", json.dumps(settings, indent=4, ensure_ascii=False))
            z.writestr("Metadata/slice_info.config",
                       '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <header>\n'
                       '    <header_item key="X-BBL-Client-Type" value="slicer"/>\n'
                       f'    <header_item key="X-BBL-Client-Version" value="{template.version or "02.08.02.60"}"/>\n'
                       '  </header>\n</config>\n')
        shutil.move(tmp, path)
        try:
            os.chmod(path, 0o644)
        except OSError:
            pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def template_filaments(template: BambuTemplate):
    """Filamenti del progetto modello come schede ColorForge (colore dichiarato dallo slicer)."""
    from colorforge.filaments.model import Filament, Reliability
    out = []
    for f in template.filaments:
        name = f["profile"].split("@")[0].strip() or f["type"]
        vendor = f["vendor"] or "AMS"
        if name.lower().startswith(vendor.lower()):
            name = name[len(vendor):].strip() or f["type"]
        fil = Filament.from_hex(f"ams-slot{f['slot']}-{f['colour'].lstrip('#')}", vendor,
                                f"{name} (slot {f['slot']})", f["colour"], material=f["type"],
                                color_reliability=Reliability.DECLARED, owned_spools=1,
                                source=f"Importato dal progetto Bambu {Path(template.path).name}: colore impostato "
                                       "nello slicer, non misurato")
        out.append((fil, f["slot"]))
    return out


def map_to_template_slots(parts_info: list[tuple[str, str]], template: BambuTemplate) -> dict[str, tuple[int, float]]:
    """Per ogni (id filamento, colore hex) lo slot del progetto modello: quello
    dichiarato se l'id viene dal progetto, altrimenti il colore più vicino (ΔE00)."""
    from colorforge.color.spaces import delta_e2000, hex_to_srgb, srgb_to_lab
    tl = [(f["slot"], srgb_to_lab(hex_to_srgb(f["colour"]))) for f in template.filaments]
    out = {}
    for fid, hx in parts_info:
        if fid.startswith("ams-slot"):
            try:
                out[fid] = (int(fid.split("-")[1][4:]), 0.0)
                continue
            except ValueError:
                pass
        lab = srgb_to_lab(hex_to_srgb(hx))
        de = [(float(delta_e2000(lab, l)), sl) for sl, l in tl]
        d, sl = min(de)
        out[fid] = (sl, d)
    return out


def _floats(v, n, default):
    try:
        out = [float(x) for x in (v or [])][:n]
    except (TypeError, ValueError):
        out = []
    return out + [default] * (n - len(out))


def template_tables(template: BambuTemplate):
    """(matrice di spurgo mm³ [da][a], costo €/kg, densità g/cm³, n. ugelli, moltiplicatore)."""
    s = template.settings
    n = len(template.filaments)
    m = _floats(s.get("flush_volumes_matrix"), n * n, 300.0)
    mat = np.array(m[:n * n]).reshape(n, n)
    cost = _floats(s.get("filament_cost"), n, 20.0)
    dens = _floats(s.get("filament_density"), n, 1.24)
    nozzles = len(s.get("nozzle_diameter") or [1])
    mult = _floats(s.get("flush_multiplier"), 1, 1.0)[0]
    return mat, np.array(cost), np.array(dens), nozzles, mult


def layer_slot_sets(res, slot_of: dict[str, int]) -> list[list[int]]:
    """Per ogni strato, gli slot (1-based) presenti, dal basso."""
    if res.zone is None:
        seq = res.solution.sequence
        top = int(res.levels.max())
        out = []
        for l in range(top):
            f = [fid for i, (fid, st) in enumerate(seq)
                 if st <= l < (seq[i + 1][1] if i + 1 < len(seq) else 10 ** 9)]
            out.append([slot_of[f[0]]] if f else [])
        return out
    rs, idx = res.zone["recipes"], res.zone["idx"]
    lift = res.zone.get("lift")
    B = res.plan.base_layers + (0 if lift is None else lift)
    fa, a, fb, b = rs.fa[idx], rs.a[idx], rs.fb[idx], rs.b[idx]
    top = int(res.levels.max())
    base_slot = slot_of[rs.fids[rs.base]]
    out = []
    for l in range(top):
        rel = l - B
        here = np.where(rel < 0, -3, np.where(rel < a, fa, np.where(rel < a + b, fb, -1)))
        vals = set(np.unique(here[res.levels > l]).tolist())
        sl = set()
        if -3 in vals:
            sl.add(base_slot)
        sl |= {slot_of[rs.fids[v]] for v in vals if v >= 0}
        out.append(sorted(sl))
    return out


def purge_estimate(res, template: BambuTemplate, slot_of: dict[str, int]) -> dict:
    """Spurgo stimato con la matrice reale del progetto. Ordine per strato: si parte
    dall'ultimo filamento usato e si sceglie ogni volta il cambio meno costoso.
    Con due ugelli (X2D) il secondo filamento più usato resta sul secondo ugello:
    i passaggi da/verso di lui non richiedono spurgo."""
    mat, cost, dens, nozzles, mult = template_tables(template)
    layers = layer_slot_sets(res, slot_of)
    usage = {}
    for ls in layers:
        for s_ in ls:
            usage[s_] = usage.get(s_, 0) + 1
    parked = None
    if nozzles >= 2 and len(usage) >= 2:
        parked = sorted(usage, key=lambda k: -usage[k])[1]
    cur, changes, vol = None, 0, np.zeros(len(mat))
    for ls in layers:
        todo = list(ls)
        while todo:
            if cur is None:
                nxt = todo[0]
            else:
                nxt = min(todo, key=lambda t: 0 if t == cur else
                          (0 if parked in (t, cur) else mat[cur - 1, t - 1]))
            if cur is not None and nxt != cur:
                changes += 1
                if parked not in (nxt, cur):
                    vol[nxt - 1] += mat[cur - 1, nxt - 1] * mult
            cur = nxt
            todo.remove(nxt)
    grams = vol * dens / 1000.0
    return {"cambi": changes, "spurgo_g": float(grams.sum()),
            "spurgo_g_per_slot": {i + 1: round(float(g), 2) for i, g in enumerate(grams) if g > 0},
            "ugello_secondario_slot": parked, "costo_eur_kg": cost.tolist(), "densita": dens.tolist()}


def cost_estimate(res, template: BambuTemplate, slot_of: dict[str, int], db) -> dict:
    """Costo stimato: filamento del modello + spurgo, con i prezzi €/kg del progetto."""
    pe = purge_estimate(res, template, slot_of)
    model_g = {}
    for label, g in res.estimate.get("filamento_g", {}).items():
        model_g[label] = g
    per_slot = {}
    for f in db:
        if f.label in model_g and f.id in slot_of:
            per_slot[slot_of[f.id]] = per_slot.get(slot_of[f.id], 0.0) + model_g[f.label]
    costs = pe["costo_eur_kg"]
    total_g = {s: per_slot.get(s, 0.0) + pe["spurgo_g_per_slot"].get(s, 0.0) for s in set(per_slot) | set(pe["spurgo_g_per_slot"])}
    eur = {s: round(g * costs[s - 1] / 1000.0, 2) for s, g in total_g.items() if 1 <= s <= len(costs)}
    return {**pe, "modello_g_per_slot": {k: round(v, 2) for k, v in per_slot.items()},
            "totale_g_per_slot": {k: round(v, 2) for k, v in total_g.items()},
            "costo_eur_per_slot": eur, "costo_eur": round(sum(eur.values()), 2)}


def read_slice_info(path: str | Path) -> dict:
    """Valori reali dallo slicer: apri in Bambu Studio, fai lo slicing e salva il
    progetto (o esporta il file .gcode.3mf). Ritorna tempo, peso, filamenti."""
    import xml.etree.ElementTree as ET
    try:
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("Metadata/slice_info.config"))
            try:
                settings = json.loads(z.read("Metadata/project_settings.config"))
            except KeyError:
                settings = {}
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as e:
        raise ValueError("Nel file non ci sono dati di slicing: fai lo slicing in Bambu Studio e salva il progetto.") from e
    plates = []
    costs = _floats(settings.get("filament_cost"), 32, 20.0)
    for p in root.iterfind("plate"):
        md = {m.get("key"): m.get("value") for m in p.iterfind("metadata")}
        fil = []
        for f in p.iterfind("filament"):
            fid = int(f.get("id", "0") or 0)
            g = float(f.get("used_g", "0") or 0)
            fil.append({"slot": fid, "tipo": f.get("type", ""), "colore": f.get("color", ""),
                        "metri": float(f.get("used_m", "0") or 0), "grammi": g,
                        "costo_eur": round(g * costs[fid - 1] / 1000.0, 2) if 1 <= fid <= len(costs) else None})
        if md.get("prediction") is None and not fil:
            continue
        plates.append({"piatto": md.get("index", "1"), "tempo_s": float(md.get("prediction", 0) or 0),
                       "peso_g": float(md.get("weight", 0) or 0), "filamenti": fil,
                       "costo_eur": round(sum(x["costo_eur"] or 0 for x in fil), 2)})
    if not plates:
        raise ValueError("Il progetto non è ancora stato slicato: premi 'Slice' in Bambu Studio e salva di nuovo.")
    return {"piatti": plates}
