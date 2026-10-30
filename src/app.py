"""Interface interativa. Rodar com:  streamlit run src/app.py"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from calculus import OTIMIZADORES, gradiente
from functions import FUNCOES

# cor fixa por otimizador: a cor segue o método, nunca muda de posição
CORES = {"Gradient Descent": "#2a78d6", "Momentum": "#eb6834"}

st.set_page_config(page_title="Visualizador de Gradient Descent", layout="wide")
st.title("Visualizador de Gradient Descent")
st.caption("Matemática em Código · Cálculo III · UNIR")

with st.sidebar:
    fn = FUNCOES[st.selectbox("Função", list(FUNCOES))]
    st.latex(fn["formula"])
    st.caption(fn["descricao"])
    nome = st.selectbox("Otimizador", list(OTIMIZADORES))
    xmin, xmax, ymin, ymax = fn["dominio"]
    x0 = st.slider("x inicial", float(xmin), float(xmax), float(fn["p0"][0]), 0.1)
    y0 = st.slider("y inicial", float(ymin), float(ymax), float(fn["p0"][1]), 0.1)
    lr = st.select_slider("Learning rate (η)", [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], value=fn["lr"])
    n = st.slider("Máx. iterações", 10, 5000, 3000, 10)

f = fn["f"]
c, motivo = OTIMIZADORES[nome](f, (x0, y0), lr, n)

col1, col2 = st.columns([3, 2])

with col1:
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))
    Z = f(X, Y)
    Z = np.log1p(Z - Z.min())  # escala log: sem isso o vale da Rosenbrock some
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(X, Y, Z, levels=30, cmap="Greys", alpha=0.35)
    ax.contour(X, Y, Z, levels=30, colors="#9a9994", linewidths=0.5)
    for mx, my in fn["minimos"]:
        ax.plot(mx, my, "*", ms=14, color="black", zorder=5)
    ax.plot(c[:, 0], c[:, 1], color=CORES[nome], lw=2, zorder=4)
    ax.plot(*c[-1], "X", ms=10, color=CORES[nome], mec="white", zorder=6)
    ax.plot(x0, y0, "o", ms=8, color="black", mec="white", zorder=6)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x", ylabel="y")
    ax.set_title("Curvas de nível e trajetória  (● início · ✕ fim · ★ mínimo)", fontsize=10)
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 3.6))
    ax2.plot(f(c[:, 0], c[:, 1]), color=CORES[nome], lw=2)
    if fn["minimos"]:
        ax2.set_yscale("symlog", linthresh=1e-8)  # log que aceita f = 0
    ax2.set(xlabel="iteração", ylabel="f(x, y)", title="Convergência")
    ax2.grid(alpha=0.2)
    st.pyplot(fig2)

    (st.success if motivo == "convergiu" else st.warning)(f"Parada: {motivo}")
    st.metric("Iterações", len(c) - 1)
    st.metric("Ponto final", f"({c[-1, 0]:.4f}, {c[-1, 1]:.4f})")
    st.metric("f final", f"{f(*c[-1]):.2e}")
    st.metric("‖∇f‖ final", f"{np.hypot(*gradiente(f, *c[-1])):.2e}")

with st.expander("📐 A matemática por trás"):
    st.latex(r"\frac{\partial f}{\partial x} \approx \frac{f(x+h,y)-f(x-h,y)}{2h}")
    st.latex(r"\text{GD: } \mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\nabla f(\mathbf{x}_k)")
    st.latex(r"\text{Momentum: } \mathbf{v}_{k+1} = \beta\mathbf{v}_k + \nabla f(\mathbf{x}_k),\ \mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\mathbf{v}_{k+1}")
