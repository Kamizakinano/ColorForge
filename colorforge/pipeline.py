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


def run(image: LoadedImage, db: FilamentDatabase, s: JobSettings,
        progress: Callable[[float, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None) -> JobResult:
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

    model = OpticalModel(filaments, background=s.background_reflectance)
    warnings += model.warnings
    plan = LayerPlan.from_settings(ps.first_layer_height_mm, ps.layer_height_mm,
                                   ps.base_thickness_mm, ps.max_thickness_mm)
    quality = QualityProfile.get(s.quality)
    check()
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
    return JobResult(s, plan, best, cands, image.srgb, prepared, levels, simulated, err, reliability,
                     warnings, log, metrics, estimate, swaps, problems, time.time() - t0)


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
                      "slot": auto_slot[fid], "strato_inizio": start + 1,
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
    """Una mesh chiusa per ogni fascia di filamento (per 3MF multi-materiale)."""
    out = []
    seq = res.solution.sequence
    for i, sw in enumerate(res.swaps):
        a = seq[i][1]
        b = seq[i + 1][1] if i + 1 < len(seq) else res.plan.max_layers
        rel = np.clip(res.levels, a, b) - a
        if rel.max() <= 0:
            continue
        m = column_mesh(rel, res.settings.pixel_mm, res.plan.z_of_level, level_offset=a)
        out.append((f"{i + 1:02d} {sw['filamento']}", sw["colore"], int(sw["slot"]), m, sw))
    return out


def export_all(res: JobResult, out_dir: str | Path, stem: str = "colorforge") -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    mesh = column_mesh(res.levels, res.settings.pixel_mm, res.plan.z_of_level)
    p = out / f"{stem}.stl"
    write_stl(mesh, p)
    files["stl"] = str(p)
    bands = band_meshes(res)
    band_problems = {name: validate(m) for name, _, _, m, _ in bands}
    p = out / f"{stem}.3mf"
    write_3mf([(n, c, sl, m) for n, c, sl, m, _ in bands], p,
              {"Title": stem, "Application": f"ColorForge AI {__version__}",
               "Designer": "ColorForge AI"})
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


def build_report(res: JobResult, band_problems: dict | None = None) -> dict:
    s = res.settings
    return {
        "applicazione": f"ColorForge AI {__version__}",
        "stampante": s.printer_id, "sistema_cambio": s.changer_id,
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
             f"Dimensioni: {r['dimensioni_mm'][0]} x {r['dimensioni_mm'][1]} mm, "
             f"altezza strato {r['altezza_strato_mm']} mm (primo strato {r['primo_strato_mm']} mm)",
             f"Altezza totale: {r['stima_stampa']['altezza_totale_mm']} mm, {r['stima_stampa']['numero_strati']} strati",
             "", "SEQUENZA FILAMENTI (dal basso verso l'alto):"]
    for sw in r["sequenza_filamenti"]:
        lines.append(f"  {sw['ordine']}. {sw['filamento']} ({sw['colore']}) slot {sw['slot']}: "
                     f"dallo strato {sw['strato_inizio']} (z = {sw['z_inizio_mm']} mm)")
    lines += ["", "METODO A - STL singolo (consigliato, sempre valido):",
              "  Importa il file .stl in Bambu Studio, assegna il primo filamento e aggiungi un",
              "  cambio filamento (tasto destro sulla barra degli strati) all'inizio di ogni strato",
              "  indicato sopra, scegliendo lo slot corrispondente.",
              "", "METODO B - 3MF multi-oggetto (da verificare con la tua versione di Bambu Studio):",
              "  Il file .3mf contiene un oggetto per ogni fascia di colore con lo slot già assegnato.",
              "", "QUALITÀ PREVISTA (simulazione, non misura reale):"]
    m = r["metriche_previste"]
    lines.append(f"  ΔE2000 medio {m['delta_e2000_medio']:.2f}, 95° percentile {m['delta_e2000_p95']:.2f}, "
                 f"SSIM luminanza {m['ssim_luminanza']:.3f}")
    lines.append(f"  Affidabilità della previsione: {r['affidabilita_previsione']:.2f} (0 = dati stimati, 1 = calibrati)")
    if r["avvisi"]:
        lines += ["", "AVVISI:"] + [f"  - {w}" for w in r["avvisi"]]
    return "\n".join(lines) + "\n"
