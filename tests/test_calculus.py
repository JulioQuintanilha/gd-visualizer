import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calculus import descida, gradiente  # noqa: E402
from functions import himmelblau, paraboloide  # noqa: E402


def test_gradiente():
    # ∇(x² + 3y²) = (2x, 6y)  ->  em (1, 2): (2, 12)
    assert np.allclose(gradiente(paraboloide, 1.0, 2.0), [2, 12], atol=1e-6)


def test_descida():
    assert np.allclose(descida(paraboloide, (3, -2), lr=0.1)[-1], [0, 0], atol=1e-4)  # GD
    assert himmelblau(*descida(himmelblau, (1, 1), lr=0.005, beta=0.9)[-1]) < 1e-6  # Momentum


if __name__ == "__main__":
    for nome, teste in list(globals().items()):
        if nome.startswith("test_"):
            teste()
    print("ok")
