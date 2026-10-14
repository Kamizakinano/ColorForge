"""Importazione fotografie: validazione, EXIF, profilo ICC, trasparenza."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
MAX_PIXELS = 80_000_000


class ImageImportError(Exception):
    """Errore comprensibile per l'utente durante l'importazione."""


@dataclass
class LoadedImage:
    srgb: np.ndarray            # (H, W, 3) float in [0,1], sRGB
    alpha: np.ndarray | None    # (H, W) float in [0,1] o None
    source_path: str
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return self.srgb.shape[1], self.srgb.shape[0]


def load_image(path: str | Path) -> LoadedImage:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ImageImportError(f"Formato non supportato: {path.suffix}. Usare PNG, JPEG, TIFF o WebP.")
    if not path.is_file():
        raise ImageImportError(f"File non trovato: {path}")
    try:
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        img = Image.open(path)
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ImageImportError(f"Impossibile leggere l'immagine (file danneggiato o troppo grande): {e}") from e
    return from_pil(img, str(path))


def from_pil(img: Image.Image, source: str = "") -> LoadedImage:
    notes: list[str] = []
    exif_orient = img.getexif().get(0x0112, 1)
    img = ImageOps.exif_transpose(img)
    if exif_orient not in (1, None):
        notes.append(f"Orientamento corretto da EXIF (valore {exif_orient}).")

    if img.mode in ("I;16", "I;16B", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        arr = arr / (65535.0 if arr.max() > 255 else 255.0)
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "L")
        notes.append("Immagine ad alta profondità convertita a 8 bit.")

    alpha = None
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        alpha = np.asarray(rgba, dtype=np.float64)[..., 3] / 255.0
        if alpha.min() >= 0.999:
            alpha = None
        else:
            notes.append("Trasparenza rilevata: le aree trasparenti saranno escluse o riempite.")
        rgb = rgba.convert("RGB") if alpha is None else Image.merge("RGB", rgba.split()[:3])
        icc = img.info.get("icc_profile")
    else:
        icc = img.info.get("icc_profile")
        if img.mode == "CMYK" and not icc:
            notes.append("Immagine CMYK senza profilo: conversione approssimata.")
        rgb = img
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            desc = ImageCms.getProfileDescription(src).strip()
            rgb = ImageCms.profileToProfile(rgb, src, ImageCms.createProfile("sRGB"),
                                            renderingIntent=ImageCms.Intent.PERCEPTUAL,
                                            outputMode="RGB")
            notes.append(f"Profilo colore incorporato '{desc}' convertito in sRGB.")
        except (ImageCms.PyCMSError, OSError) as e:
            notes.append(f"Profilo ICC non leggibile ({e}); si assume sRGB.")
            rgb = rgb.convert("RGB")
    else:
        rgb = rgb.convert("RGB")
        notes.append("Nessun profilo colore: si assume sRGB.")
    srgb = np.asarray(rgb, dtype=np.float64) / 255.0
    return LoadedImage(srgb=srgb, alpha=alpha, source_path=source, notes=notes)
