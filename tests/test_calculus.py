import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calculus import gradiente  # noqa: E402
from functions import paraboloide  # noqa: E402


def test_gradiente():
    # ∇(x² + 3y²) = (2x, 6y)  ->  em (1, 2): (2, 12)
    assert np.allclose(gradiente(paraboloide, 1.0, 2.0), [2, 12], atol=1e-6)


if __name__ == "__main__":
    for nome, teste in list(globals().items()):
        if nome.startswith("test_"):
            teste()
    print("ok")
