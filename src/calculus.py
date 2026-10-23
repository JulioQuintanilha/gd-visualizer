"""Núcleo de cálculo: tudo implementado pela equipe, sem bibliotecas de otimização.
NumPy aparece só como vetor (somar, multiplicar por escalar)."""

import numpy as np


def gradiente(f, x, y, h=1e-5):
    """∇f = (∂f/∂x, ∂f/∂y) por diferença central: [f(x+h) - f(x-h)] / 2h"""
    dfdx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    dfdy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([dfdx, dfdy])


# ---------------------------------------------------------------------------
# Otimizadores: recebem f e o ponto inicial, devolvem a trajetória (array N×2)
# ---------------------------------------------------------------------------

def descida(f, p0, lr=0.01, beta=0.0, max_iter=500, tol=1e-6):
    """Gradient Descent com Momentum (beta=0 → Gradient Descent puro).
    v ← β·v + ∇f(p)
    p ← p − η·v
    """
    p, v = np.array(p0, dtype=float), np.zeros(2)
    caminho = [p]
    for _ in range(max_iter):
        g = gradiente(f, *p)
        if np.hypot(*g) < tol or not np.all(np.isfinite(p)):  # convergiu ou divergiu
            break
        v = beta * v + g
        p = p - lr * v
        caminho.append(p)
    return np.array(caminho)


# nome -> função (f, p0, lr, max_iter) usada pela interface
OTIMIZADORES = {
    "Gradient Descent": lambda f, p0, lr, n: descida(f, p0, lr, beta=0.0, max_iter=n),
    "Momentum": lambda f, p0, lr, n: descida(f, p0, lr, beta=0.9, max_iter=n),
}
