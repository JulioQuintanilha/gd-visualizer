import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calculus import descida, gradiente  # noqa: E402
from functions import himmelblau, paraboloide, rosenbrock, sela  # noqa: E402


def test_gradiente():
    # ∇(x² + 3y²) = (2x, 6y)  ->  em (1, 2): (2, 12)
    assert np.allclose(gradiente(paraboloide, 1.0, 2.0), [2, 12], atol=1e-6)


def test_descida():
    c, motivo = descida(paraboloide, (3, -2), lr=0.1)  # GD
    assert motivo == "convergiu" and np.allclose(c[-1], [0, 0], atol=1e-4)
    c, motivo = descida(himmelblau, (1, 1), lr=0.005, beta=0.9)  # Momentum
    assert motivo == "convergiu" and himmelblau(*c[-1]) < 1e-6


def test_motivo_da_parada():
    # falsa convergência: antes, o ponto ia a ~1e20, o gradiente numérico dava 0 e o GD "convergia"
    c, motivo = descida(rosenbrock, (-1, 2), lr=0.1)
    assert motivo == "divergiu" and np.all(np.abs(c) < 1e6)
    assert descida(paraboloide, (3, -2), lr=0.0001, max_iter=10)[1] == "limite de iterações"
    # na sela, partindo de y = 0, o GD para exatamente no ponto de sela (gradiente nulo, não é mínimo)
    c, motivo = descida(sela, (-1.5, 0), lr=0.05)
    assert motivo == "convergiu" and np.allclose(c[-1], [0, 0], atol=1e-6)


if __name__ == "__main__":
    for nome, teste in list(globals().items()):
        if nome.startswith("test_"):
            teste()
    print("ok")
