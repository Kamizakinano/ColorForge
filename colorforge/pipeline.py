"""Flusso completo: foto -> preparazione -> ottimizzazione -> quote -> mesh -> export."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from colorforge import __version__
from colorforge.color.spaces import delta_e2000, linear_to_lab, linear_to_srgb, srgb_to_hex, srgb_to_lab
from colorforge.export.stl import write_stl
from colorforge.export.threemf import write_3mf
from colorforge.filaments.database import FilamentDatabase
from colorforge.geometry.mesh import Mesh, column_mesh, fix_saddles, validate
from colorforge.imaging.analysis import analyze
from colorforge.imaging.importer import LoadedImage
from colorforge.imaging.preprocess import PrepOptions, prepare, resize_to_grid
from colorforge.optics.km import OpticalModel
from colorforge.optimize.heightmap import assign_levels, smooth_levels
from colorforge.optimize.search import LayerPlan, QualityProfile, Solution, optimize
from colorforge.printers.profiles import PrintSettings, get_printer


class Cancelled(Exception):
    pass


@dataclass
class JobSettings:
    printer_id: str = "bambu-x2d"
    changer_id: str = "ams2pro-x2"
    filament_ids: list[str] = field(default_factory=list)
    slots: dict[str, int] = field(default_factory=dict)  # filamento -> slot AMS (1..8)
    width_mm: float = 100.0
    height_mm: float = 0.0          # 0 = da rapporto d'aspetto
    pixel_mm: float = 0.3
    max_colors: int = 8
    quality: str = "massima"
    print: PrintSettings = field(default_factory=PrintSettings)
    prep: PrepOptions = field(default_factory=PrepOptions)
    background_reflectance: float = 0.0
    seed: int = 0
    # "zone": filamenti diversi nello stesso strato (AMS); "layers": cambi per strato (HueForge)
    mode: str = "zone"
    # 0 = massima fedeltà; 1 = meno cambi filamento possibile (accetta più ΔE)
    change_economy: float = 0.3
    # pesi maggiori a contorni, dettagli e soggetto nella scelta dei filamenti
    use_importance: bool = True
    # usa un solo materiale (quello più presente tra i selezionati): PLA e PETG non aderiscono bene
    single_material: bool = True
    # materiale della stampa: "auto" (il più presente tra i selezionati), "PLA" o "PETG"
    material: str = "auto"
    # rilievo 3D (solo modalità per zona): altezza in mm del soggetto sopra lo sfondo, 0 = piatto
    relief_mm: float = 0.0
    # accessori opzionali (generati solo se richiesti)
    add_frame: bool = False
    add_hook: bool = False
    add_stand: bool = False
    frame_mm: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JobSettings":
        d = dict(d)
        d["print"] = PrintSettings(**d.get("print", {}))
        p = d.get("prep", {})
        if "background_fill" in p:
            p["background_fill"] = tuple(p["background_fill"])
        d["prep"] = PrepOptions(**p)
        return cls(**d)


@dataclass
class JobResult:
    settings: JobSettings
    plan: LayerPlan
    solution: Solution
    candidates: list[Solution]
    original_srgb: np.ndarray
    prepared_srgb: np.ndarray
    levels: np.ndarray
    simulated_srgb: np.ndarray
    error_de: np.ndarray
    reliability: float
    warnings: list[str]
    log: list[str]
    metrics: dict
    estimate: dict
    swaps: list[dict]
    mesh_problems: list[str]
    elapsed_s: float
    zone: dict | None = None   # modalità per zona: {"recipes": RecipeSet, "idx": array}
    objects: dict | None = field(default=None, repr=False)  # mesh per oggetto (cache export)
    user_mask: np.ndarray | None = field(default=None, repr=False)  # pennellate dell'utente

    def error_map_rgb(self, vmax: float = 20.0) -> np.ndarray:
        t = np.clip(self.error_de / vmax, 0, 1)[..., None]
        green, yellow, red = np.array([0.1, 0.7, 0.2]), np.array([1, 0.85, 0.1]), np.array([0.85, 0.1, 0.1])
        return np.where(t < 0.5, green + (yellow - green) * t * 2, yellow + (red - yellow) * (t - 0.5) * 2)


def _grid_size(img: np.ndarray, s: JobSettings) -> tuple[int, int, float, float]:
    h, w = img.shape[:2]
    width = s.width_mm
    height = s.height_mm or width * h / w
    cols = max(8, int(round(width / s.pixel_mm)))
    rows = max(8, int(round(height / s.pixel_mm)))
    return cols, rows, cols * s.pixel_mm, rows * s.pixel_mm


PROTECT, SIMPLIFY = 1, 2


def run(image: LoadedImage, db: FilamentDatabase, s: JobSettings,
        progress: Callable[[float, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None, user_mask: np.ndarray | None = None) -> JobResult:
    """`user_mask` (stessa forma della foto o qualsiasi, 0/1/2): zone dipinte col pennello,
    1 = proteggi (massima fedeltà, soggetto del rilievo), 2 = semplifica."""
    t0 = time.time()
    progress = progress or (lambda p, m: None)
    cancelled = cancelled or (lambda: False)

    def check():
        if cancelled():
            raise Cancelled()

    warnings, log = list(image.notes), []
    printer = get_printer(s.printer_id)
    changer = printer.changer(s.changer_id)
    if not s.filament_ids:
        raise ValueError("Nessun filamento selezionato.")
    from collections import Counter
    mats = Counter(db.get(f).material.upper() for f in s.filament_ids)
    if s.material != "auto":
        main = s.material.upper()
        if main not in mats:
            raise ValueError(f"Nessun filamento {main} selezionato: cambia 'Materiale' o i filamenti.")
    else:
        main = mats.most_common(1)[0][0] if s.single_material and len(mats) > 1 else None
    if main is not None and len(mats) > 1:
        dropped = [db.get(f).label for f in s.filament_ids if db.get(f).material.upper() != main]
        s = __import__("copy").deepcopy(s)
        s.filament_ids = [f for f in s.filament_ids if db.get(f).material.upper() == main]
        warnings.append(f"Usato solo {main} (materiali diversi aderiscono male): esclusi " + ", ".join(dropped) + ".")
    filaments = [db.get(f) for f in s.filament_ids]
    max_colors = s.max_colors
    if changer.capacity and max_colors > changer.capacity:
        warnings.append(f"{changer.name}: massimo {changer.capacity} filamenti, limite applicato.")
        max_colors = changer.capacity

    progress(0.02, "Analisi della fotografia")
    report = analyze(image.srgb)
    warnings += report.warnings
    cols, rows, w_mm, h_mm = _grid_size(image.srgb, s)
    ps = s.print
    if not printer.fits(w_mm, h_mm, ps.max_thickness_mm):
        raise ValueError(f"Dimensioni {w_mm:.0f}×{h_mm:.0f} mm fuori dal volume di stampa di {printer.name} "
                         f"({printer.build_volume_mm[0]}×{printer.build_volume_mm[1]} mm).")
    if s.pixel_mm < printer.nozzle_mm * 0.5:
        warnings.append("Risoluzione più fine di metà ugello: i dettagli minori non saranno stampabili.")
    check()
    progress(0.05, "Preparazione dell'immagine")
    alpha = None
    if image.alpha is not None:
        alpha = resize_to_grid(np.repeat(image.alpha[..., None], 3, -1), cols, rows)[..., 0]
    small = resize_to_grid(image.srgb, cols, rows)
    prepared, plog = prepare(small, s.prep, alpha=alpha)
    log += plog
    target_lab = srgb_to_lab(prepared)
    brush = None
    if user_mask is not None and np.any(user_mask):
        from PIL import Image as _I
        brush = np.asarray(_I.fromarray(np.asarray(user_mask, np.uint8)).resize((cols, rows), _I.Resampling.NEAREST))
        simp = brush == SIMPLIFY
        if simp.any():
            from scipy import ndimage as _nd
            sm = np.stack([_nd.gaussian_filter(target_lab[..., c], 2.0) for c in range(3)], -1)
            target_lab = np.where(simp[..., None], sm, target_lab)
        log.append(f"Pennello: {np.mean(brush == PROTECT) * 100:.0f}% protetto, {np.mean(simp) * 100:.0f}% semplificato.")

    model = OpticalModel(filaments, background=s.background_reflectance)
    warnings += model.warnings
    plan = LayerPlan.from_settings(ps.first_layer_height_mm, ps.layer_height_mm,
                                   ps.base_thickness_mm, ps.max_thickness_mm)
    quality = QualityProfile.get(s.quality)
    check()
    if s.mode == "zone":
        res = _run_zone(image, db, s, model, plan, target_lab, prepared, max_colors, changer,
                        warnings, log, progress, check, t0, brush)
        res.user_mask = user_mask
        return res
    if s.relief_mm > 0:
        warnings.append("Il rilievo 3D è disponibile solo in modalità 'Multicolore per zona': ignorato.")
    best, cands, ev = optimize(model, plan, target_lab, s.filament_ids, max_colors, quality,
                               progress=progress, cancelled=cancelled, seed=s.seed)
    check()
    log.append(f"Ottimizzazione '{quality.name}': {ev.evaluations} combinazioni valutate, "
               f"ΔE00 medio sui colori rappresentativi {best.mean_de:.2f}.")
    if len(best.sequence) - 1 > ps.max_color_changes:
        warnings.append(f"La soluzione richiede {best.changes} cambi colore (limite {ps.max_color_changes}).")

    progress(0.85, "Calcolo della mappa delle altezze")
    levels, _ = assign_levels(target_lab, best.lut_lab, plan.base_layers)
    levels, _, msg = smooth_levels(levels, target_lab, best.lut_lab)
    log.append(msg)
    levels, nfix = fix_saddles(levels)
    levels = np.minimum(levels, plan.max_layers)
    if nfix:
        log.append(f"Corrette {nfix} celle per garantire una geometria manifold (+1 strato ciascuna).")
    sim_lab = best.lut_lab[levels]
    err = delta_e2000(target_lab, sim_lab)
    simulated = linear_to_srgb(best.lut_lin[levels])
    check()

    progress(0.92, "Verifica della geometria")
    mesh = column_mesh(levels, s.pixel_mm, plan.z_of_level)
    problems = validate(mesh)

    ssim = float(structural_similarity(target_lab[..., 0], sim_lab[..., 0], data_range=100))
    metrics = {
        "delta_e2000_medio": float(err.mean()),
        "delta_e2000_mediana": float(np.median(err)),
        "delta_e2000_p95": float(np.percentile(err, 95)),
        "percentuale_pixel_dE_sopra_10": float((err > 10).mean() * 100),
        "ssim_luminanza": ssim,
        "tipo": "previsione del modello ottico, NON misura su stampa reale",
    }
    reliability = model.reliability(best.filaments)
    if reliability < 0.5:
        warnings.append(f"Affidabilità della previsione bassa ({reliability:.2f}): i dati ottici dei "
                        "filamenti sono stimati. Calibrare i filamenti per previsioni attendibili.")
    swaps, estimate = _swaps_and_estimate(best, plan, levels, s, db, mesh)
    res = JobResult(s, plan, best, cands, image.srgb, prepared, levels, simulated, err, reliability,
                    warnings, log, metrics, estimate, swaps, problems, time.time() - t0)
    res.user_mask = user_mask
    return res


def _run_zone(image, db, s, model, plan, target_lab, prepared, max_colors, changer,
              warnings, log, progress, check, t0, brush=None) -> JobResult:
    from colorforge.imaging.analysis import importance_map
    from colorforge.optimize.regional import (build_objects, build_recipes, choose_filaments,
                                              clean_recipes, layer_changes, nearest_recipes,
                                              reduce_changes)
    progress(0.15, "Scelta dei filamenti")
    weights = importance_map(prepared) if s.use_importance else None
    if brush is not None:
        weights = np.ones(brush.shape) if weights is None else weights
        weights = weights * np.where(brush == PROTECT, 4.0, np.where(brush == SIMPLIFY, 0.3, 1.0))
        weights = weights / weights.mean()
    chosen, _ = choose_filaments(model, plan, target_lab, s.filament_ids, max_colors,
                                 progress=progress, seed=s.seed, weights=weights,
                                 economy=s.change_economy)
    check()
    progress(0.72, "Calcolo delle combinazioni di strati")
    rs = build_recipes(model, plan, chosen, 0, step=1 if s.quality == "massima" else 2)
    progress(0.78, "Assegnazione dei colori per zona")
    idx, _ = nearest_recipes(target_lab, rs)
    idx, nclean, _ = clean_recipes(idx, rs, plan.base_layers, target_lab)
    check()
    ch0 = layer_changes(rs, idx, plan.base_layers)
    if s.change_economy > 0:
        progress(0.8, "Riduzione dei cambi filamento")
        idx, ch1, rlog = reduce_changes(rs, idx, target_lab, plan.base_layers, s.change_economy, weights)
        log += rlog
        if ch1 < ch0:
            log.append(f"Cambi filamento ridotti da {ch0} a {ch1}.")
    check()
    log.append(f"Multicolore per zona: {len(rs)} combinazioni di strati con {len(chosen)} filamenti; "
               f"{nclean} pixel isolati uniti ai vicini.")
    levels = plan.base_layers + rs.a[idx] + rs.b[idx]
    sim_lab = rs.lab[idx]
    err = delta_e2000(target_lab, sim_lab)
    simulated = linear_to_srgb(rs.lin[idx])
    lift = None
    if s.relief_mm > 0:
        from colorforge.imaging.analysis import relief_map, subject_mask_robust
        subj = (brush == PROTECT) if brush is not None and np.any(brush == PROTECT) else subject_mask_robust(prepared)
        rel = relief_map(prepared, subj, s.relief_mm, s.pixel_mm)
        lift = np.rint(rel / plan.layer_mm).astype(np.int64)
        levels = levels + lift
        log.append(f"Rilievo 3D: soggetto rialzato fino a {s.relief_mm:.1f} mm ({subj.mean() * 100:.0f}% dell'immagine).")
    progress(0.85, "Costruzione e verifica della geometria")
    objects = build_objects(rs, idx, plan, s.pixel_mm, lift)
    problems = [f"{k[1]} ({k[0]}): {p}" for k, m in objects.items() for p in validate(m)]
    ssim = float(structural_similarity(target_lab[..., 0], sim_lab[..., 0], data_range=100))
    metrics = {
        "delta_e2000_medio": float(err.mean()), "delta_e2000_mediana": float(np.median(err)),
        "delta_e2000_p95": float(np.percentile(err, 95)),
        "percentuale_pixel_dE_sopra_10": float((err > 10).mean() * 100), "ssim_luminanza": ssim,
        "tipo": "previsione del modello ottico, NON misura su stampa reale",
    }
    reliability = model.reliability(chosen)
    if reliability < 0.5:
        warnings.append(f"Affidabilità della previsione bassa ({reliability:.2f}): i dati ottici dei "
                        "filamenti sono stimati. Calibrare i filamenti per previsioni attendibili.")
    mats = {db.get(f).material.upper() for f in chosen}
    if len(mats) > 1:
        warnings.append("Materiali diversi nella stessa stampa (" + ", ".join(sorted(mats)) + "): PLA e PETG "
                        "aderiscono male tra loro. Usa filamenti dello stesso materiale.")
    sol = Solution([(f, 0) for f in chosen], float(err.mean()), float(err.mean()))
    slots = {}
    for f in chosen:
        slots[f] = s.slots.get(f) or len(slots) + 1
    swaps = []
    px2 = s.pixel_mm ** 2
    zl = plan.z_of_level
    vols = {}
    for (role, f), m in objects.items():
        vols[f] = vols.get(f, 0.0) + m.volume
    for i, f in enumerate(chosen):
        fl = db.get(f)
        swaps.append({"ordine": i + 1, "filamento": fl.label, "id": f, "colore": srgb_to_hex(fl.srgb),
                      "materiale": fl.material.upper(),
                      "slot": slots[f], "ruolo": "base" if i == 0 else "zona",
                      "strato_inizio": 1 if i == 0 else plan.base_layers + 1,
                      "z_inizio_mm": 0.0 if i == 0 else round(float(zl(plan.base_layers)), 3),
                      "z_fine_mm": round(float(zl(plan.max_layers)), 3)})
    top = int(levels.max())
    changes = layer_changes(rs, idx, plan.base_layers)
    volume = sum(vols.values())
    estimate = {
        "nota": "Stime approssimative: verificare con lo slicer.",
        "altezza_totale_mm": round(float(zl(top)), 3), "numero_strati": top,
        "cambi_colore": int(changes), "volume_mm3": round(volume, 1),
        "filamento_g": {db.get(f).label: round(v * db.get(f).density_g_cm3 / 1000, 2) for f, v in vols.items()},
        "spurgo_g_stimato": round(0.3 * changes, 1),
        "ipotesi_spurgo": "0.3 g per cambio (ipotesi; dipende da spurgo, torre e doppio ugello)",
        "tempo_min_stimato": round(volume / 8.0 / 60 + top * 0.25 + 1.5 * changes, 0),
        "ipotesi_tempo": "flusso medio 8 mm³/s, 15 s per strato, 90 s per cambio filamento",
    }
    if changes > 60:
        warnings.append(f"Circa {changes} cambi filamento: stampa lunga e molto spurgo. "
                        "Riduci 'Colori massimi' o usa la modalità 'Cambi per strato' per stampe più rapide.")
    res = JobResult(s, plan, sol, [sol], image.srgb, prepared, levels, simulated, err, reliability,
                    warnings, log, metrics, estimate, swaps, problems, time.time() - t0,
                    zone={"recipes": rs, "idx": idx, "lift": lift}, objects=objects)
    return res


def _swaps_and_estimate(sol: Solution, plan: LayerPlan, levels, s: JobSettings, db, mesh: Mesh):
    px2 = s.pixel_mm ** 2
    swaps, est_mass = [], {}
    seq = sol.sequence
    auto_slot: dict[str, int] = {}
    for fid, _ in seq:  # stesso filamento = stesso slot, in ordine di primo utilizzo
        auto_slot.setdefault(fid, s.slots.get(fid) or len(auto_slot) + 1)
    for i, (fid, start) in enumerate(seq):
        f = db.get(fid)
        end = seq[i + 1][1] if i + 1 < len(seq) else plan.max_layers
        z0, z1 = plan.z_of_level(start), plan.z_of_level(end)
        thick = np.clip(plan.z_of_level(levels) - z0, 0, z1 - z0)
        vol = float(thick.sum() * px2)
        est_mass[f.label] = round(vol * f.density_g_cm3 / 1000, 2)
        swaps.append({"ordine": i + 1, "filamento": f.label, "id": fid, "colore": srgb_to_hex(f.srgb),
                      "materiale": f.material.upper(), "slot": auto_slot[fid], "strato_inizio": start + 1,
                      "z_inizio_mm": round(float(z0), 3), "z_fine_mm": round(float(z1), 3)})
    top = int(levels.max())
    used = [w for w in swaps if w["strato_inizio"] <= top]
    volume = mesh.volume
    estimate = {
        "nota": "Stime approssimative: verificare con lo slicer.",
        "altezza_totale_mm": round(float(plan.z_of_level(top)), 3),
        "numero_strati": top,
        "cambi_colore": max(0, len(used) - 1),
        "volume_mm3": round(volume, 1),
        "filamento_g": est_mass,
        "spurgo_g_stimato": round(0.3 * max(0, len(used) - 1), 1),
        "ipotesi_spurgo": "0.3 g per cambio (ipotesi, dipende dalle impostazioni di spurgo dello slicer)",
        "tempo_min_stimato": round(volume / 8.0 / 60 + top * 0.25 + 1.5 * max(0, len(used) - 1), 0),
        "ipotesi_tempo": "flusso medio 8 mm³/s, 15 s per strato, 90 s per cambio colore AMS",
    }
    return swaps, estimate


def band_meshes(res: JobResult) -> list[tuple[str, str, int, Mesh, dict]]:
    """Una mesh chiusa per ogni fascia (cambi per strato) o per ogni
    (ruolo, filamento) (multicolore per zona), per il 3MF multi-materiale."""
    out = []
    if res.zone is not None:
        from colorforge.optimize.regional import build_objects
        objs = res.objects or build_objects(res.zone["recipes"], res.zone["idx"], res.plan, res.settings.pixel_mm,
                                            res.zone.get("lift"))
        by_id = {sw["id"]: sw for sw in res.swaps}
        order = {"base": 0, "medio": 1, "sopra": 2}
        for (role, f), m in sorted(objs.items(), key=lambda kv: (order[kv[0][0]], by_id[kv[0][1]]["slot"])):
            sw = by_id[f]
            out.append((f"slot {sw['slot']} - {sw['filamento']} ({role})", sw["colore"], int(sw["slot"]), m, sw))
        return out
    seq = res.solution.sequence
    for i, sw in enumerate(res.swaps):
        a = seq[i][1]
        b = seq[i + 1][1] if i + 1 < len(seq) else res.plan.max_layers
        rel = np.clip(res.levels, a, b) - a
        if rel.max() <= 0:
            continue
        m = column_mesh(rel, res.settings.pixel_mm, res.plan.z_of_level, level_offset=a)
        out.append((f"{i + 1:02d} slot {sw['slot']} - {sw['filamento']}", sw["colore"], int(sw["slot"]), m, sw))
    return out


def _with_accessories(res: JobResult, bands: list) -> list:
    s = res.settings
    if not (s.add_frame or s.add_hook or s.add_stand) or not bands:
        return bands
    from colorforge.color.spaces import srgb_to_lab as _lab
    from colorforge.geometry.accessories import accessories_for
    H, W = res.levels.shape
    top = max(float(m.vertices[:, 2].max()) for _, _, _, m, _ in bands)
    # filamento più scuro tra quelli usati: cornice e accessori stanno bene scuri
    sws = {sw["id"]: sw for _, _, _, _, sw in bands}
    dark = min(sws.values(), key=lambda sw: _lab(__import__("colorforge.color.spaces", fromlist=["x"]).hex_to_srgb(sw["colore"]))[0])
    for name, m in accessories_for(W * s.pixel_mm, H * s.pixel_mm, top, s.add_frame, s.add_hook, s.add_stand, s.frame_mm):
        bands.append((f"slot {dark['slot']} - {name}", dark["colore"], int(dark["slot"]), m, dark))
    return bands


def export_3mf(res: JobResult, path: str | Path, progress=None, template=None) -> dict:
    """Esporta il 3MF. Con `template` (progetto Bambu modello) scrive un progetto
    Bambu Studio con le parti già assegnate agli slot; altrimenti un 3MF standard.
    Ritorna i problemi geometrici per parte."""
    bands = _with_accessories(res, band_meshes(res))
    problems = {name: validate(m) for name, _, _, m, _ in bands}
    if template is not None:
        from colorforge.export.bambu import map_to_template_slots, write_bambu_project
        mats = {db_mat for db_mat in (sw.get("materiale") for _, _, _, _, sw in bands) if db_mat}
        job_mat = res.settings.material.upper() if res.settings.material != "auto" else (mats.pop() if len(mats) == 1 else None)
        mapping = map_to_template_slots([(sw["id"], c) for _, c, _, _, sw in bands], template, material=job_mat)
        parts = []
        notes = []
        for name, c, _sl, m, sw in bands:
            slot, de = mapping[sw["id"]]
            role = name.rsplit("(", 1)[-1].rstrip(")") if "(" in name else name.split(" - ", 1)[-1]
            parts.append((f"slot {slot} - {sw['filamento']}" + (f" ({role})" if role else ""), slot, m))
            if de > 5 and f"{sw['filamento']}" not in " ".join(notes):
                notes.append(f"{sw['filamento']} → slot {slot} (colore diverso, ΔE {de:.0f})")
        ps = res.settings.print
        write_bambu_project(parts, path, template, ps.layer_height_mm, ps.first_layer_height_mm,
                            res.simulated_srgb, Path(path).stem, progress=progress)
        problems["_slot_mapping"] = notes
        return problems
    write_3mf([(n, c, sl, m) for n, c, sl, m, _ in bands], path,
              {"Title": Path(path).stem, "Application": f"ColorForge AI {__version__}",
               "Designer": "ColorForge AI"}, progress=progress)
    return problems


def export_all(res: JobResult, out_dir: str | Path, stem: str = "colorforge", progress=None,
               template=None) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda p, m: None)
    files = {}
    progress(0.02, "Creazione STL")
    mesh = column_mesh(res.levels, res.settings.pixel_mm, res.plan.z_of_level)
    p = out / f"{stem}.stl"
    write_stl(mesh, p)
    files["stl"] = str(p)
    p = out / f"{stem}.3mf"
    band_problems = export_3mf(res, p, progress=lambda f, m: progress(0.1 + 0.8 * f, m), template=template)
    files["3mf"] = str(p)
    for key, arr in (("anteprima", res.simulated_srgb), ("preparata", res.prepared_srgb),
                     ("errori", res.error_map_rgb())):
        p = out / f"{stem}_{key}.png"
        Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(p)
        files[key] = str(p)
    report = build_report(res, band_problems)
    p = out / f"{stem}_report.json"
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    files["report_json"] = str(p)
    p = out / f"{stem}_istruzioni.txt"
    p.write_text(report_text(report), "utf-8")
    files["istruzioni"] = str(p)
    return files


def image_palette(srgb: np.ndarray, k: int = 16, seed: int = 0) -> list[tuple[str, float]]:
    """Colori principali della foto: [(hex, frazione)], dal più presente."""
    from colorforge.color.spaces import lab_to_srgb
    from colorforge.optimize.search import representative_colors
    h, w = srgb.shape[:2]
    step = max(1, int(np.sqrt(h * w / 60000)))
    lab = srgb_to_lab(srgb[::step, ::step])
    cent, wts = representative_colors(lab, k, seed)
    order = np.argsort(-wts)
    return [(srgb_to_hex(lab_to_srgb(cent[i])), float(wts[i])) for i in order]


def palette_filaments(hex_colors: list[str], material: str = "PLA"):
    """Filamenti temporanei dai colori scelti nella tavolozza (dati ottici stimati)."""
    from colorforge.filaments.model import Filament, Reliability
    return [Filament.from_hex(f"img-{h.lstrip('#').upper()}", "Colore immagine", h.upper(), h,
                              material=(material if material != "auto" else "PLA").upper(),
                              color_reliability=Reliability.ESTIMATED,
                              source="Colore estratto dalla foto, non un filamento misurato")
            for h in hex_colors]


def build_report(res: JobResult, band_problems: dict | None = None) -> dict:
    s = res.settings
    return {
        "applicazione": f"ColorForge AI {__version__}",
        "stampante": s.printer_id, "sistema_cambio": s.changer_id, "modalita": s.mode,
        "dimensioni_mm": [round(res.levels.shape[1] * s.pixel_mm, 2), round(res.levels.shape[0] * s.pixel_mm, 2)],
        "risoluzione_mm": s.pixel_mm, "qualita": s.quality,
        "altezza_strato_mm": s.print.layer_height_mm, "primo_strato_mm": s.print.first_layer_height_mm,
        "sequenza_filamenti": res.swaps,
        "metriche_previste": res.metrics,
        "affidabilita_previsione": res.reliability,
        "stima_stampa": res.estimate,
        "controllo_geometria": res.mesh_problems or "OK (chiusa, orientata, nessun triangolo degenere)",
        "controllo_fasce_3mf": band_problems,
        "avvisi": res.warnings, "operazioni": res.log,
        "candidate": [{"filamenti": c.filaments, "cambi": c.changes, "dE_medio": round(c.mean_de, 2)}
                      for c in res.candidates],
        "tempo_calcolo_s": round(res.elapsed_s, 1),
    }


def report_text(r: dict) -> str:
    lines = ["COLORFORGE AI - ISTRUZIONI DI STAMPA", "",
             "Modalità: " + ("multicolore per zona (AMS)" if r.get("modalita") == "zone" else "cambi per strato"),
             f"Dimensioni: {r['dimensioni_mm'][0]} x {r['dimensioni_mm'][1]} mm, "
             f"altezza strato {r['altezza_strato_mm']} mm (primo strato {r['primo_strato_mm']} mm)",
             f"Altezza totale: {r['stima_stampa']['altezza_totale_mm']} mm, {r['stima_stampa']['numero_strati']} strati",
             "", "FILAMENTI:" if r.get("modalita") == "zone" else "SEQUENZA FILAMENTI (dal basso verso l'alto):"]
    for sw in r["sequenza_filamenti"]:
        lines.append(f"  {sw['ordine']}. {sw['filamento']} ({sw['colore']}) slot {sw['slot']}: "
                     f"dallo strato {sw['strato_inizio']} (z = {sw['z_inizio_mm']} mm)")
    if r.get("modalita") == "zone":
        lines += ["", "MODALITÀ MULTICOLORE PER ZONA (AMS):",
                  "  1. Apri il file .3mf in Bambu Studio e rispondi 'Sì' a 'caricare come singolo oggetto",
                  "     con più parti'.",
                  "  2. Nell'elenco oggetti assegna a ogni parte lo slot scritto nel suo nome",
                  "     (es. 'slot 3 - ...'). Più parti possono avere lo stesso slot.",
                  "  3. Il file .stl contiene solo la forma: in questa modalità serve il 3MF.",
                  "", "QUALITÀ PREVISTA (simulazione, non misura reale):"]
        m = r["metriche_previste"]
        lines.append(f"  ΔE2000 medio {m['delta_e2000_medio']:.2f}, 95° percentile {m['delta_e2000_p95']:.2f}, "
                     f"SSIM luminanza {m['ssim_luminanza']:.3f}")
        lines.append(f"  Cambi filamento stimati: {r['stima_stampa']['cambi_colore']}, "
                     f"tempo ~{r['stima_stampa']['tempo_min_stimato']:.0f} min (stima)")
        if r["avvisi"]:
            lines += ["", "AVVISI:"] + [f"  - {w}" for w in r["avvisi"]]
        return "\n".join(lines) + "\n"
    lines += ["", "METODO A - STL singolo (consigliato, sempre valido):",
              "  Importa il file .stl in Bambu Studio, assegna il primo filamento e aggiungi un",
              "  cambio filamento (tasto destro sulla barra degli strati) all'inizio di ogni strato",
              "  indicato sopra, scegliendo lo slot corrispondente.",
              "", "METODO B - 3MF multi-oggetto:",
              "  Il file .3mf contiene un oggetto per ogni fascia di colore. All'apertura rispondi 'Sì'",
              "  alla domanda 'caricare come singolo oggetto con più parti', poi assegna a ogni parte",
              "  lo slot indicato nel suo nome (es. '02 slot 2 - ...').",
              "", "QUALITÀ PREVISTA (simulazione, non misura reale):"]
    m = r["metriche_previste"]
    lines.append(f"  ΔE2000 medio {m['delta_e2000_medio']:.2f}, 95° percentile {m['delta_e2000_p95']:.2f}, "
                 f"SSIM luminanza {m['ssim_luminanza']:.3f}")
    lines.append(f"  Affidabilità della previsione: {r['affidabilita_previsione']:.2f} (0 = dati stimati, 1 = calibrati)")
    if r["avvisi"]:
        lines += ["", "AVVISI:"] + [f"  - {w}" for w in r["avvisi"]]
    return "\n".join(lines) + "\n"
