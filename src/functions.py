"""Funções de teste clássicas de otimização em 2D."""


def paraboloide(x, y):
    return x**2 + 3 * y**2


def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def himmelblau(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def sela(x, y):
    return x**2 - y**2


# nome -> f, fórmula (LaTeX), domínio (xmin, xmax, ymin, ymax), mínimos conhecidos, descrição
FUNCOES = {
    "Paraboloide": dict(
        f=paraboloide, formula=r"f(x,y) = x^2 + 3y^2", dominio=(-4, 4, -4, 4),
        minimos=[(0, 0)],
        descricao="Convexa, um único mínimo. Caso ideal: todo método converge.",
    ),
    "Rosenbrock": dict(
        f=rosenbrock, formula=r"f(x,y) = (1-x)^2 + 100(y-x^2)^2", dominio=(-2, 2, -1, 3),
        minimos=[(1, 1)],
        descricao="Vale estreito e curvo. O Gradient Descent puro anda em zigue-zague.",
    ),
    "Himmelblau": dict(
        f=himmelblau, formula=r"f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2", dominio=(-5, 5, -5, 5),
        minimos=[(3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)],
        descricao="Quatro mínimos. O ponto inicial decide em qual o algoritmo cai.",
    ),
    "Sela": dict(
        f=sela, formula=r"f(x,y) = x^2 - y^2", dominio=(-3, 3, -3, 3),
        minimos=[],
        descricao="Ponto de sela na origem: gradiente nulo, mas não é mínimo.",
    ),
}


if __name__ == "__main__":
    # autoteste: f vale ~0 em todo mínimo conhecido
    for nome, fn in FUNCOES.items():
        for x, y in fn["minimos"]:
            assert abs(fn["f"](x, y)) < 1e-9, (nome, x, y)
    print("ok")
