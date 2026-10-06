from __future__ import annotations

import pathlib

import numpy as np
import pytest
from PIL import Image

_DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def frames() -> list[np.ndarray]:
    paths = sorted(_DATA.glob("teaser_*.png"))
    return [np.asarray(Image.open(path).convert("RGB").resize((640, 480), Image.BILINEAR)) for path in paths]
