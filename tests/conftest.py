import numpy as np
import pytest
from PIL import Image

from colorforge.filaments.database import FilamentDatabase
from colorforge.imaging.importer import from_pil


@pytest.fixture
def db():
    return FilamentDatabase.builtin()


def make_image(kind: str, w=60, h=40):
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    if kind == "gradient":
        a = np.stack([x, x, x], -1)
    elif kind == "bw":
        a = np.where((x > 0.5)[..., None], 1.0, 0.0) * np.ones(3)
    elif kind == "saturated":
        a = np.zeros((h, w, 3)); a[:, : w // 3, 0] = 1; a[:, w // 3: 2 * w // 3, 1] = 1; a[:, 2 * w // 3:, 2] = 1
    elif kind == "many":
        a = np.stack([x, y, 1 - x * y], -1)
    else:
        raise ValueError(kind)
    return from_pil(Image.fromarray((a * 255).astype(np.uint8)))
