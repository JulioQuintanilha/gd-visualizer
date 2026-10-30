"""Núcleo de cálculo: tudo implementado pela equipe, sem bibliotecas de otimização.
NumPy aparece só como vetor (somar, multiplicar por escalar)."""

import numpy as np


def gradiente(f, x, y, h=1e-5):
    """∇f = (∂f/∂x, ∂f/∂y) por diferença central: [f(x+h) - f(x-h)] / 2h"""
    dfdx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    dfdy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([dfdx, dfdy])


# ---------------------------------------------------------------------------
# Otimizadores: recebem f e o ponto inicial, devolvem (trajetória N×2, motivo da parada)
# ---------------------------------------------------------------------------

def descida(f, p0, lr=0.01, beta=0.0, max_iter=500, tol=1e-6, limite=1e6):
    """Gradient Descent com Momentum (beta=0 → Gradient Descent puro).
    v ← β·v + ∇f(p)
    p ← p − η·v
    Devolve (trajetória, motivo da parada).
    """
    p, v = np.array(p0, dtype=float), np.zeros(2)
    caminho = [p]
    for _ in range(max_iter):
        g = gradiente(f, *p)
        if np.hypot(*g) < tol:
            return np.array(caminho), "convergiu"
        v = beta * v + g
        p = p - lr * v
        # divergência: sem este teste, em |x| enorme x + h == x, o gradiente numérico dá 0
        # e o algoritmo "converge" falsamente. A comparação também pega inf e NaN.
        # limite fixo: os domínios vão só até ±5, então |x| > 1e6 já indica divergência
        if not np.all(np.abs(p) < limite):
            return np.array(caminho), "divergiu"
        caminho.append(p)
    return np.array(caminho), "limite de iterações"


# nome -> função (f, p0, lr, max_iter) usada pela interface
OTIMIZADORES = {
    "Gradient Descent": lambda f, p0, lr, n: descida(f, p0, lr, beta=0.0, max_iter=n),
    "Momentum": lambda f, p0, lr, n: descida(f, p0, lr, beta=0.9, max_iter=n),
}
