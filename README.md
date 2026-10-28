# 📉 Visualizador de Gradient Descent e Otimizadores

## 🏫 Informações Institucionais
* **Instituição:** Fundação Universidade Federal de Rondônia (UNIR)
* **Curso:** Bacharelado em Ciência da Computação
* **Disciplina:** Cálculo III (DIN00024)
* **Professor:** Prof. Esp. Rodrigo Duarte de Oliveira Tolêdo
* **Período:** 2º Semestre / 2026

### 👥 Integrantes do Grupo
* Júlio César Maldonado Quintanilha Neto - Matrícula: 20251000252
* Leonardo Setti - Matrícula: 20231008834

---

## 🔍 Visão Geral do Projeto
Redes neurais aprendem ajustando seus parâmetros para diminuir uma função de erro, e o algoritmo mais usado para isso é o **Gradient Descent**. A cada passo, ele calcula as derivadas parciais do erro, monta o vetor gradiente e anda no sentido contrário, que é a direção em que o erro cai mais rápido. Derivadas parciais e vetor gradiente são conteúdo de Cálculo III.

Este projeto é uma ferramenta interativa e de código aberto que mostra esse processo em funções de duas variáveis, $f(x, y)$, nas quais é possível enxergar a superfície inteira. O usuário escolhe a função, o otimizador, o ponto inicial e o *learning rate*, e acompanha a trajetória do algoritmo sobre as curvas de nível, a curva de convergência e o ponto onde ele parou.

O público-alvo são estudantes de exatas e de computação que querem ver, na prática, por que o treinamento de redes neurais depende do Cálculo.

---

## 📐 Conceitos de Cálculo Aplicados

### 1. Funções de várias variáveis e curvas de nível
O software trabalha com quatro funções $f:\mathbb{R}^2 \to \mathbb{R}$ clássicas de otimização, desenhadas como curvas de nível:

| Função | Fórmula | Por que foi escolhida |
|---|---|---|
| Paraboloide | $x^2 + 3y^2$ | Convexa, com um único mínimo |
| Rosenbrock | $(1-x)^2 + 100(y-x^2)^2$ | Vale estreito e curvo, difícil para o Gradient Descent |
| Himmelblau | $(x^2+y-11)^2 + (x+y^2-7)^2$ | Quatro mínimos: o ponto inicial decide onde o algoritmo para |
| Sela | $x^2 - y^2$ | Gradiente nulo na origem sem que ela seja um mínimo |

### 2. Derivadas parciais e vetor gradiente
O gradiente aponta para a direção em que $f$ cresce mais rápido. O código calcula cada derivada parcial por **diferença central**:

$$\frac{\partial f}{\partial x} \approx \frac{f(x+h,\,y) - f(x-h,\,y)}{2h}, \qquad \nabla f = \left(\frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y}\right)$$

com $h = 10^{-5}$. A diferença central tem erro proporcional a $h^2$, menor que o da diferença simples $[f(x+h) - f(x)]/h$, cujo erro é proporcional a $h$.

### 3. Otimização: Gradient Descent e Momentum
O **Gradient Descent** dá um passo de tamanho $\eta$ (*learning rate*) no sentido oposto ao gradiente:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\nabla f(\mathbf{x}_k)$$

O **Momentum** acumula uma velocidade $\mathbf{v}$, que reduz o zigue-zague em vales estreitos:

$$\mathbf{v}_{k+1} = \beta\,\mathbf{v}_k + \nabla f(\mathbf{x}_k), \qquad \mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\mathbf{v}_{k+1}$$

Com $\beta = 0$ o Momentum vira o Gradient Descent puro, por isso os dois usam a mesma função no código (`descida`). O algoritmo para quando $\|\nabla f\| < 10^{-6}$ ou quando atinge o número máximo de iterações.

### 🔒 O que é implementado pela equipe
Derivadas parciais, gradiente e otimizadores estão em `src/calculus.py`, escritos pela equipe. O projeto não usa bibliotecas de otimização (SciPy, PyTorch, TensorFlow). NumPy serve só como estrutura de vetores, e Matplotlib e Streamlit cuidam da visualização.

---

## 🛠️ Tecnologias e Dependências

* **Linguagem Principal:** Python 3.10+
* **Interface Interativa:** Streamlit (1.30+)
* **Gráficos:** Matplotlib (3.7+)
* **Processamento Vetorial:** NumPy (1.24+)
* **Testes:** pytest (7.0+)

---

## 📂 Estrutura do Repositório

```text
├── src/
│   ├── app.py            # Interface interativa (Streamlit) e gráficos
│   ├── calculus.py       # Núcleo de cálculo: gradiente e otimizadores
│   └── functions.py      # Funções de teste (Paraboloide, Rosenbrock, Himmelblau, Sela)
├── tests/
│   └── test_calculus.py  # Testes automáticos do núcleo matemático
├── docs/                 # Relatório extensionista e slides do pitch
├── README.md
├── LICENSE               # MIT (código aberto)
└── requirements.txt
```

---

## 🚀 Como Executar o Projeto

### 1. Clonar o Repositório
```bash
git clone https://github.com/JulioQuintanilha/gd-visualizer.git
cd gd-visualizer
```

### 2. Instalar as Dependências
Recomenda-se o uso de um ambiente virtual (venv):
```bash
python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Rodar a Aplicação
```bash
streamlit run src/app.py
```
O navegador abre automaticamente em `http://localhost:8501`.

### 4. Rodar os Testes
```bash
pytest
```

---

## 📈 Exemplos de Uso e Resultados

*(Em desenvolvimento.)*

---

## 💡 Conclusão e Aprendizados

*(Em desenvolvimento.)*

---

## 🗺️ Roadmap

- [x] Estrutura do repositório
- [x] Funções de teste (`src/functions.py`)
- [x] Núcleo de cálculo: gradiente por diferença central (`src/calculus.py`)
- [x] Otimizadores: Gradient Descent e Momentum
- [x] Interface interativa (`src/app.py`)
- [ ] Matriz Hessiana e classificação de pontos críticos
- [ ] Método de Newton
- [ ] Otimizador Adam
- [ ] Modo de comparação entre otimizadores
- [ ] Backpropagation: mini rede neural com a Regra da Cadeia

---

## 📄 Licença
Distribuído sob a licença MIT. Veja `LICENSE`.
