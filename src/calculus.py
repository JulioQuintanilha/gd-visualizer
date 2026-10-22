"""Núcleo de cálculo: tudo implementado pela equipe, sem bibliotecas de otimização.
NumPy aparece só como vetor (somar, multiplicar por escalar)."""

import numpy as np


def gradiente(f, x, y, h=1e-5):
    """∇f = (∂f/∂x, ∂f/∂y) por diferença central: [f(x+h) - f(x-h)] / 2h"""
    dfdx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    dfdy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([dfdx, dfdy])
