"""Formato progetto interno versionato (.cfproj = zip documentato).

Contenuto:
  manifest.json   {"format": "colorforge-project", "version": N, "app": ...}
  settings.json   JobSettings
  filaments.json  schede dei filamenti usati (con i dati di calibrazione)
  result.json     soluzione, metriche, stime, avvisi, cronologia
  original.png, prepared.png, simulated.png
  levels.npy, lut_lin.npy
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from colorforge import __version__
from colorforge.filaments.database import FilamentDatabase
from colorforge.optimize.search import LayerPlan, Solution
from colorforge.pipeline import JobResult, JobSettings

FORMAT_VERSION = 1


def _png(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "PNG")
    return b.getvalue()


def _npy(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.save(b, arr, allow_pickle=False)
    return b.getvalue()


def save_project(path: str | Path, res: JobResult, db: FilamentDatabase, history: list[str] | None = None) -> None:
    path = Path(path)
    fil = FilamentDatabase([db.get(f) for f in res.settings.filament_ids])
    result = {
        "sequence": res.solution.sequence, "score": res.solution.score, "mean_de": res.solution.mean_de,
        "plan": vars(res.plan), "metrics": res.metrics, "estimate": res.estimate, "swaps": res.swaps,
        "warnings": res.warnings, "log": res.log, "reliability": res.reliability,
        "mesh_problems": res.mesh_problems, "elapsed_s": res.elapsed_s,
        "candidates": [{"sequence": c.sequence, "score": c.score, "mean_de": c.mean_de} for c in res.candidates],
        "history": history or [],
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps({"format": "colorforge-project", "version": FORMAT_VERSION,
                                                "app": __version__}))
        z.writestr("settings.json", json.dumps(res.settings.to_dict(), ensure_ascii=False, indent=1))
        z.writestr("filaments.json", json.dumps(fil.to_dict(), ensure_ascii=False, indent=1))
        z.writestr("result.json", json.dumps(result, ensure_ascii=False, indent=1))
        z.writestr("original.png", _png(res.original_srgb))
        z.writestr("prepared.png", _png(res.prepared_srgb))
        z.writestr("simulated.png", _png(res.simulated_srgb))
        z.writestr("levels.npy", _npy(res.levels))
        z.writestr("lut_lin.npy", _npy(res.solution.lut_lin))
        z.writestr("lut_lab.npy", _npy(res.solution.lut_lab))
        z.writestr("error_de.npy", _npy(res.error_de.astype(np.float32)))
    if path.exists():  # mai sovrascrivere senza copia
        os.replace(path, path.with_suffix(path.suffix + ".bak"))
    os.replace(tmp, path)


def _migrate(manifest: dict) -> None:
    v = int(manifest.get("version", 0))
    if v > FORMAT_VERSION:
        raise ValueError("Progetto creato con una versione più recente di ColorForge.")
    # v < FORMAT_VERSION: qui andranno le migrazioni future.


def load_project(path: str | Path) -> tuple[JobResult, FilamentDatabase]:
    """Riapre un progetto senza ricalcolare nulla."""
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise ValueError(f"File progetto danneggiato: {e}") from e
    with z:
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("format") != "colorforge-project":
            raise ValueError("Il file non è un progetto ColorForge.")
        _migrate(manifest)
        settings = JobSettings.from_dict(json.loads(z.read("settings.json")))
        fdb = FilamentDatabase.from_dict(json.loads(z.read("filaments.json")))
        r = json.loads(z.read("result.json"))
        img = lambda n: np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB"), dtype=float) / 255
        npy = lambda n: np.load(io.BytesIO(z.read(n)), allow_pickle=False)
        seq = [tuple(x) for x in r["sequence"]]
        sol = Solution(seq, r["score"], r["mean_de"], npy("lut_lin.npy"), npy("lut_lab.npy"))
        cands = [Solution([tuple(x) for x in c["sequence"]], c["score"], c["mean_de"]) for c in r["candidates"]]
        res = JobResult(settings, LayerPlan(**r["plan"]), sol, cands, img("original.png"), img("prepared.png"),
                        npy("levels.npy"), img("simulated.png"), npy("error_de.npy").astype(float),
                        r["reliability"], r["warnings"], r["log"], r["metrics"], r["estimate"], r["swaps"],
                        r["mesh_problems"], r["elapsed_s"])
    return res, fdb
