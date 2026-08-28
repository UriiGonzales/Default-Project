"""Unidad 2 · Límite y Continuidad (Programa 2026: UN 2).

2.1 Límite finito (ε-δ), límites laterales, infinitésimos e indeterminaciones 0/0 y ∞/∞.
2.2 Asíntotas horizontales, verticales y oblicuas.
2.3 Continuidad, tipos de discontinuidad y teoremas fundamentales.
"""

from __future__ import annotations

import random

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from ..graficos import figura_funciones
from ..matematicas import (
    x,
    calcular_limite,
    asintotas_de,
    clasificar_discontinuidad,
)
from .. import ui


# ===========================================================================
# TEORÍA
# ===========================================================================

def teoria() -> None:
    st.markdown(
        "El **límite** cuantifica el comportamiento de una función cerca de un punto aunque "
        "allí no esté definida; es la puerta de entrada a la derivada y a la integral."
    )

    with st.expander("2.1 Límite finito y límites laterales", expanded=True):
        st.markdown(
            r"""
            Escribimos $\lim_{x\to a} f(x) = L$ cuando los valores de $f(x)$ se acercan a $L$
            todo lo que se quiera al tomar $x$ suficientemente cerca de $a$ (pero distinto de $a$).
            "
            """
        )
        st.latex(r"0<|x-a|<\delta \;\Rightarrow\; |f(x)-L|<\varepsilon")
        st.markdown(
            r"""
            **Álgebra de límites:** si ambos existen, el límite de la suma, el producto y el
            cociente (con denominador no nulo) se comporta "linealmente". El concepto principal
            es el de **infinitésimo**: $f$ es un infinitésimo en $a$ si $\lim_{x\to a}f(x)=0$.

            **Indeterminaciones** habitiles: $0/0$, $\infty/\infty$, $0\cdot\infty$,
            $\infty-\infty$, $1^\infty$, $0^0$, $\infty^0$. Las dos primeras se resuelven:
            """
        )
        st.latex(
            r"\begin{aligned}"
            r"&\bullet \dfrac{x^2-a^2}{x-a}\to 2a & "
            r"&\bullet \dfrac{\sin u}{u}\to 1 \\"
            r"&\bullet \dfrac{1-\cos u}{u}\to 0 & "
            r"&\bullet \dfrac{\tan u}{u}\to 1 \\"
            r"&\bullet \dfrac{e^u-1}{u}\to 1 & "
            r"&\bullet \dfrac{\ln(1+u)}{u}\to 1 "
            r"\end{aligned}"
        )
        st.markdown(
            r"**Límites laterales:** $\lim_{x\to a^+}f(x)$ y $\lim_{x\to a^-}f(x)$. "
            "El límite existe si y solo si ambos laterales existen y coinciden."
        )
        st.latex(r"\lim_{x\to a}f(x)=L \iff \lim_{x\to a^+}f(x)=\lim_{x\to a^-}f(x)=L")
        st.markdown(
            r"Si $x\to \infty$, el límite da las **asíntotas horizontales** y el comportamiento "
            "a largo plazo de la función."
        )

    with st.expander("2.2 Asíntotas", expanded=False):
        st.markdown(
            r"""
            - **Vertical** en $x=a$: cuando algún límite lateral tiende a infinito
              ($\pm\infty$). Suelen aparecer en polos de funciones racionales.
            - **Horizontal** en $y=L$: cuando $\lim_{x\to\pm\infty} f(x)=L$.
            - **Oblicua** $y=mx+b$: cuando existen
            """
        )
        st.latex(r"m=\lim_{x\to\pm\infty}\frac{f(x)}{x},\qquad b=\lim_{x\to\pm\infty}\big(f(x)-mx\big)")
        st.markdown(
            "Una función racional con grado del numerador = grado del denominador + 1 tiene "
            "asíntota oblicua (y no horizontal)."
        )

    with st.expander("2.3 Continuidad y teoremas", expanded=False):
        st.markdown(
            r"$f$ es **continua en $a$** si $f(a)$ está definida y $\lim_{x\to a}f(x)=f(a)$."
            " Si no hay continuidad, la discontinuidad puede clasificarse:"
        )
        st.latex(
            r"\text{evitable (el límite existe)},\qquad \text{de salto finito},\qquad "
            r"\text{esencial (infinita u oscilante)}"
        )
        st.markdown(
            "**Propiedades clave de las funciones continuas en intervalos cerrados $[a,b]$:**"
        )
        st.markdown(
            r"""
            - **Teorema de Bolzano (TVI):** si $f$ es continua y $f(a)$ y $f(b)$ tienen signos
              opuestos, existe $c\in(a,b)$ con $f(c)=0$.
            - **Teorema de los valores intermedios:** $f([a,b])$ es un intervalo.
            - **Teorema de Weierstrass:** $f$ alcanza máximo y mínimo absolutos en $[a,b]$.
            """
        )
        st.success(
            "El TVI es la justificación teórica del **método de bisección** para resolver "
            "ecuaciones numéricamente."
        )

    with st.expander("2.4 Demostraciones: Bolzano, TVI y Weierstrass", expanded=False):
        st.markdown(
            r"**Teorema de Bolzano (enunciado).** Si $f$ es continua en $[a,b]$ y "
            r"$f(a)\,f(b)<0$, existe $c\in(a,b)$ tal que $f(c)=0$."
        )
        st.markdown(
            r"**Demostración (método de bisección).** Construimos intervalos encajados "
            r"$[a_n,b_n]$ que conservan el cambio de signo:"
        )
        st.latex(
            r"f(a_n)\le 0 \le f(b_n),\qquad b_n-a_n=\frac{b-a}{2^n},\qquad "
            r"[a_{n+1},b_{n+1}]\subset[a_n,b_n]"
        )
        st.markdown(
            r"En cada paso se divide el intervalo por la mitad y se conserva la mitad que "
            r"mantiene el signo. Como $a_n$ es creciente y acotada (por $b$) y $b_n$ es "
            r"decreciente y acotada (por $a$), ambas convergen; y por $b_n-a_n\to 0$ lo hacen "
            r"al mismo límite $c$. Por continuidad,"
        )
        st.latex(r"f(c)=\lim f(a_n)\le 0 \quad\text{y}\quad f(c)=\lim f(b_n)\ge 0 \;\Rightarrow\; f(c)=0")
        st.markdown(
            r"Como $f(a)$ y $f(b)$ son no nulos, el cero cae en el interior: $c\in(a,b)$. "
            r"$\blacksquare$"
        )

        st.markdown(
            r"**Teorema del valor intermedio.** Si $f$ es continua en $[a,b]$ y $y_0$ es un "
            r"valor entre $f(a)$ y $f(b)$, existe $c\in(a,b)$ con $f(c)=y_0$."
        )
        st.markdown(
            r"**Demostración.** Se aplica Bolzano a $g(x)=f(x)-y_0$: $g(a)$ y $g(b)$ tienen "
            r"signos opuestos, luego existe $c$ con $g(c)=0$, es decir $f(c)=y_0$. "
            r"$\blacksquare$"
        )

        st.markdown(
            r"**Teorema de Weierstrass.** $f$ continua en $[a,b]$ está acotada y alcanza máximo "
            r"y mínimo absolutos."
        )
        st.markdown(
            r"**Demostración (acotación).** Si $f$ no estuviera acotada, existiría "
            r"$\{x_n\}\subset[a,b]$ con $f(x_n)>n$. Por el teorema de Bolzano–Weierstrass, "
            r"$\{x_n\}$ tiene una subsucesión convergente $x_{n_k}\to c\in[a,b]$; por "
            r"continuidad $f(x_{n_k})\to f(c)$, en contradicción con $f(x_{n_k})> n_k\to\infty$."
        )
        st.markdown(
            r"**Demostración (alcanza extremos).** Sea $M=\sup f([a,b])$ (existe por la "
            r"acotación). Por la propiedad aproximante del supremo, hay $x_n$ con "
            r"$f(x_n)> M-\tfrac{1}{n}$; una subsucesión converge a $c$ y $f(c)=\lim f(x_{n_k})=M$, "
            r"el máximo. Para el mínimo se procede análogamente con $\inf$. $\blacksquare$"
        )


# ===========================================================================
# INTUICIÓN VISUAL
# ===========================================================================

_FUNCIONES_LIMITE = {
    "sin(x) / x": sp.sin(x) / x,
    "(1 − cos(x)) / x²": (1 - sp.cos(x)) / x**2,
    "(x² − 1) / (x − 1)": (x**2 - 1) / (x - 1),
    "1 / x": 1 / x,
    "sen(2x) / (3x)": sp.sin(2 * x) / (3 * x),
    "x² / (x² + 1)": x**2 / (x**2 + 1),
}

_CONTINUIDADES_VISUALES = [
    {"a": 2, "nom": r"$f(x)=\begin{cases} kx+1 & x<2 \\ x^2-1 & x\ge 2\end{cases}$",
     "iz": lambda k: k * x + 1, "de": x**2 - 1, "k_teorica": 1},
    {"a": 1, "nom": r"$f(x)=\begin{cases} x^2+k & x<1 \\ 2x+1 & x\ge 1\end{cases}$",
     "iz": lambda k: x**2 + k, "de": 2 * x + 1, "k_teorica": 2},
    {"a": 0, "nom": r"$f(x)=\begin{cases} \sin x + k & x<0 \\ 2x+1 & x\ge 0\end{cases}$",
     "iz": lambda k: sp.sin(x) + k, "de": 2 * x + 1, "k_teorica": 1},
]


def _limite_numerico() -> None:
    st.markdown("##### Aproximación numérica de un límite")
    nombre = st.selectbox("Función", list(_FUNCIONES_LIMITE), key="u2_flim")
    expr = _FUNCIONES_LIMITE[nombre]
    a = st.slider("Punto a que tiende x", -4.0, 4.0, 0.0, 0.1, key="u2_a_lim")

    h1 = st.slider("h (paso)", 0.01, 0.5, 0.1, 0.01, key="u2_h_lim")

    def _fmt(valor) -> str:
        v = sp.nsimplify(valor)
        if v.is_finite:
            return f"{float(valor):.5f}"
        return "±∞"

    tabla = {
        f"x = a − h  (={a - h1:.3f})": _fmt(sp.limit(expr, x, a - h1)),
        "x = a − h/10": _fmt(sp.limit(expr, x, a - h1 / 10)),
        "x = a + h/10": _fmt(sp.limit(expr, x, a + h1 / 10)),
        f"x = a + h  (={a + h1:.3f})": _fmt(sp.limit(expr, x, a + h1)),
    }
    st.table(tabla)

    L = sp.limit(expr, x, a)
    fig = figura_funciones([(nombre, expr)], a - 5, a + 5, titulo=f"Límite en x = {float(a):g}")
    fig.add_vline(x=a, line_dash="dash", line_color="red", annotation_text="x = a")
    if L.is_finite:
        fig.add_hline(y=float(L), line_dash="dot", line_color="green",
                      annotation_text=f"L = {float(L):.3f}")
        fig.add_trace(go.Scatter(x=[a], y=[float(L)], mode="markers", name="(a, L)",
                                 marker=dict(size=10, color="green")))
        st.latex(rf"\lim_{{x\to {float(a):g}}} {sp.latex(expr)} = {sp.latex(L)}")
    else:
        st.warning("El límite NO es finito: f tiende a ±∞ (asíntota vertical).")
        st.latex(rf"\lim_{{x\to {float(a):g}}} {sp.latex(expr)} = \pm\infty")
    st.plotly_chart(fig, width="stretch")


def _continuidad_visual() -> None:
    st.markdown("##### Continuidad de una función a trozos")
    indice = st.selectbox("Función definida por tramos", range(len(_CONTINUIDADES_VISUALES)),
                          format_func=lambda i: _CONTINUIDADES_VISUALES[i]["nom"], key="u2_cont_elegir")
    caso = _CONTINUIDADES_VISUALES[indice]
    k = st.slider("Parámetro k", -8.0, 8.0, float(caso["k_teorica"]), 0.1, key="u2_cont_k")

    a = caso["a"]
    xs = np.linspace(a - 6, a + 6, 2000)
    fiz = sp.lambdify(x, caso["iz"](k), modules=["numpy"])
    fde = sp.lambdify(x, caso["de"], modules=["numpy"])
    with np.errstate(all="ignore"):
        ys_iz = np.where(xs < a, fiz(xs), np.nan)
        ys_de = np.where(xs >= a, fde(xs), np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys_iz, mode="lines", name="rama izquierda", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=xs, y=ys_de, mode="lines", name="rama derecha", line=dict(color="#ff7f0e", width=3)))
    li_iz = float(sp.limit(caso["iz"](k), x, a, dir="-"))
    val_der = float(caso["de"].subs(x, a))
    fig.add_trace(go.Scatter(x=[a], y=[li_iz], mode="markers", name="límite izq (∅)",
                             marker=dict(size=11, color="#1f77b4", symbol="circle-open")))
    fig.add_trace(go.Scatter(x=[a], y=[val_der], mode="markers", name="f(a)",
                             marker=dict(size=9, color="#ff7f0e")))
    fig.add_vline(x=a, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"Continuidad en x = {a}", height=420,
                      xaxis_title="x", yaxis_title="y", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    continua = abs(li_iz - val_der) < 1e-6
    k_ok = caso["k_teorica"]
    if continua:
        st.success(f"Con k = {k:.1f} la función es CONTÍNUA en x={a} (el salto se cerró).")
    else:
        st.error(f"Con k = {k:.1f} hay un SALTO de {val_der - li_iz:.2f} en x={a}.")
        st.markdown(
            rf"**k óptimo:** {k_ok} — entonces límite izquierdo = f(a) = {val_der:.0f}."
        )


def _asintotas_visual() -> None:
    st.markdown("##### Asíntotas de una función racional")
    f1 = (2 * x + 3) / (x - 1)
    f2 = x**2 / (x - 1)
    f3 = (x**3) / (x**2 - 1)
    f4 = x / (x**2 + 1)
    opciones = [("f(x)= (2x+3)/(x−1)", f1), ("f(x)= x²/(x−1)", f2),
                ("f(x)= x³/(x²−1)", f3), ("f(x)= x/(x²+1)", f4)]
    nom = st.selectbox("Función", [o[0] for o in opciones], key="u2_asint_nom")
    expr = dict(opciones)[nom]

    datos = asintotas_de(expr)
    verticales = datos["verticales"]
    horizontales = [datos["horizontal"]] if datos["horizontal"] is not None else []
    oblicua = datos["oblicua"]
    fig = figura_funciones([(nom, expr)], -6, 6, titulo=nom)
    for v in verticales:
        fig.add_vline(x=float(v), line_dash="dash", line_color="red", annotation_text=f"x={sp.latex(v)}")
    for h in horizontales:
        val = float(h)
        if abs(val) < 25:
            fig.add_hline(y=val, line_dash="dot", line_color="green", annotation_text=f"y={sp.latex(h)}")
    if oblicua is not None:
        xslinea = np.linspace(-6, 6, 2)
        fig.add_trace(go.Scatter(x=xslinea, y=[float(oblicua.subs(x, v)) for v in xslinea],
                                 mode="lines", name="asíntota oblicua",
                                 line=dict(color="orange", dash="dot")))
    st.plotly_chart(fig, width="stretch")

    partes = []
    if verticales:
        partes.append("**Verticales:** " + ", ".join(f"$x={sp.latex(v)}$" for v in verticales))
    if horizontales:
        partes.append("**Horizontales:** " + ", ".join(f"$y={sp.latex(h)}$" for h in horizontales))
    if oblicua is not None:
        partes.append(rf"**Oblicua:** $y={sp.latex(oblicua)}$")
    st.info("  \n".join(partes) if partes else "No se detectaron asíntotas en el rango elegido.")


def intuicion() -> None:
    st.markdown(
        "Visualizá la convergencia de los límites numéricamente, el cierre del salto de una "
        "función a trozos y las asíntotas."
    )
    t1, t2, t3 = st.tabs(["Límite numérico", "Continuidad con k", "Asíntotas"])
    with t1:
        _limite_numerico()
    with t2:
        _continuidad_visual()
    with t3:
        _asintotas_visual()


# ===========================================================================
# EJERCICIOS
# ===========================================================================

_LIMITES_ALG = [
    (r"\frac{x^2-1}{x-1}", (x**2 - 1) / (x - 1), 1, 2),
    (r"\frac{\sin x}{x}", sp.sin(x) / x, 0, 1),
    (r"\frac{1-\cos x}{x}", (1 - sp.cos(x)) / x, 0, 0),
    (r"\frac{x^2-5x+6}{x-2}", (x**2 - 5 * x + 6) / (x - 2), 2, -1),
    (r"\frac{\sqrt{x}-2}{x-4}", (sp.sqrt(x) - 2) / (x - 4), 4, sp.Rational(1, 4)),
    (r"\frac{x^3-8}{x-2}", (x**3 - 8) / (x - 2), 2, 12),
    (r"\frac{\tan x}{x}", sp.tan(x) / x, 0, 1),
    (r"\frac{e^x-1}{x}", (sp.exp(x) - 1) / x, 0, 1),
    (r"\frac{\ln(1+x)}{x}", sp.ln(1 + x) / x, 0, 1),
    (r"\frac{x^2-4}{x+2}", (x**2 - 4) / (x + 2), -2, -4),
]

_LIMITES_INF = [
    (r"\frac{3x^2+x+1}{2x^2-1}", (3 * x**2 + x + 1) / (2 * x**2 - 1), sp.Rational(3, 2)),
    (r"\frac{2x-1}{x+3}", (2 * x - 1) / (x + 3), 2),
    (r"\frac{x^2+3x}{x^2-1}", (x**2 + 3 * x) / (x**2 - 1), 1),
    (r"\frac{x+\sin x}{x}", (x + sp.sin(x)) / x, 1),
    (r"\frac{5x+2}{2x}", (5 * x + 2) / (2 * x), sp.Rational(5, 2)),
    (r"\frac{1}{x}", 1 / x, 0),
    (r"\frac{x^2+5}{1-3x^2}", (x**2 + 5) / (1 - 3 * x**2), sp.Rational(-1, 3)),
]

_LIMITES_LATERALES = [
    (r"\frac{|x|}{x}", sp.Abs(x) / x, 0, "+", 1),
    (r"\frac{|x|}{x}", sp.Abs(x) / x, 0, "-", -1),
    (r"\frac{1}{x-1}", 1 / (x - 1), 1, "+", sp.oo),
    (r"\frac{1}{x-1}", 1 / (x - 1), 1, "-", -sp.oo),
    (r"\frac{x-1}{|x-1|}", (x - 1) / sp.Abs(x - 1), 1, "+", 1),
    (r"e^{1/x}", sp.exp(1 / x), 0, "-", 0),
]

_DISCONTINUIDAD_BANCO = [
    ("f(x)= (x²−1)/(x−1)", (x**2 - 1) / (x - 1), 1),
    ("f(x)= sin(x)/x", sp.sin(x) / x, 0),
    ("f(x)= 1/x", 1 / x, 0),
    ("f(x)= |x|/x", sp.Abs(x) / x, 0),
    ("f(x)= (x²−4)/(x+2)", (x**2 - 4) / (x + 2), -2),
    ("f(x)= x/(x−1)", x / (x - 1), 1),
    ("f(x)= 1/(x²−1)", 1 / (x**2 - 1), 1),
]


def _lima_evaluar(clave: str) -> None:
    st.session_state.pop(f"in_{clave}", None)
    st.session_state.pop(f"w_{clave}", None)


def _tab_limites_alg() -> None:
    st.markdown("#### Límites con indeterminación 0/0")
    if "u2_alg" not in st.session_state:
        st.session_state["u2_alg"] = random.randint(0, len(_LIMITES_ALG) - 1)
    fr, expr, a, correcto = _LIMITES_ALG[st.session_state["u2_alg"]]

    if st.button("🎲 Otro ejercicio (0/0)", key="u2_alg_nuevo"):
        st.session_state["u2_alg"] = random.randint(0, len(_LIMITES_ALG) - 1)
        _lima_evaluar("u2_alg_resp")
        st.rerun()

    st.latex(rf"\lim_{{x\to {a}}} {fr}")
    ui.resolver_valor("u2_alg_resp", correcto, tema="U2-limite-00", enunciado=fr)


def _tab_limites_inf() -> None:
    st.markdown("#### Límites en el infinito")
    if "u2_inf" not in st.session_state:
        st.session_state["u2_inf"] = random.randint(0, len(_LIMITES_INF) - 1)
    fr, expr, correcto = _LIMITES_INF[st.session_state["u2_inf"]]

    if st.button("🎲 Otro ejercicio (∞/∞)", key="u2_inf_nuevo"):
        st.session_state["u2_inf"] = random.randint(0, len(_LIMITES_INF) - 1)
        _lima_evaluar("u2_inf_resp")
        st.rerun()

    st.latex(rf"\lim_{{x\to \infty}} {fr}")
    ui.resolver_valor("u2_inf_resp", correcto,
                      tema="U2-limite-inf", enunciado=fr)


def _tab_laterales() -> None:
    st.markdown("#### Límites laterales")
    if "u2_lat" not in st.session_state:
        st.session_state["u2_lat"] = random.randint(0, len(_LIMITES_LATERALES) - 1)
    fr, expr, a, lado, correcto = _LIMITES_LATERALES[st.session_state["u2_lat"]]

    if st.button("🎲 Otro ejercicio (laterales)", key="u2_lat_nuevo"):
        st.session_state["u2_lat"] = random.randint(0, len(_LIMITES_LATERALES) - 1)
        _lima_evaluar("u2_lat_resp")
        st.rerun()

    dir_sym = "+" if lado == "+" else "-"
    st.latex(rf"\lim_{{x\to {a}^{dir_sym}}} {fr}")
    ui.resolver_valor("u2_lat_resp", correcto,
                      tema="U2-limite-lateral", enunciado=rf"x\to {a}^{dir_sym} de {fr}")


def _tab_continuidad() -> None:
    st.markdown("#### Continuidad de una función a trozos (hallar k)")
    k = sp.Symbol("k")
    casos = [
        (r"f(x)=\begin{cases}kx+1 & x<2\\ x^2-1 & x\ge 2\end{cases}", 2, k * x + 1, x**2 - 1),
        (r"f(x)=\begin{cases}x^2+k & x<1\\ 2x+1 & x\ge 1\end{cases}", 1, x**2 + k, 2 * x + 1),
        (r"f(x)=\begin{cases}\sin x + k & x<0\\ 2x+1 & x\ge 0\end{cases}", 0, sp.sin(x) + k, 2 * x + 1),
        (r"f(x)=\begin{cases}e^x & x<1\\ k & x\ge 1\end{cases}", 1, sp.exp(x), k),
    ]
    if "u2_cont" not in st.session_state:
        st.session_state["u2_cont"] = random.randint(0, len(casos) - 1)
    nom, a, iz, de = casos[st.session_state["u2_cont"]]

    if st.button("🎲 Otro ejercicio (continuidad)", key="u2_cont_nuevo"):
        st.session_state["u2_cont"] = random.randint(0, len(casos) - 1)
        _lima_evaluar("u2_cont_resp")
        st.rerun()

    st.latex("Hallá k para que f sea continua en " + nom)
    correcto = sp.solve(sp.limit(iz, x, a, dir="-") - de.subs(x, a), k)
    if not correcto:
        st.warning("No hay k que haga continua a la función en el punto indicado.")
        return
    kk = sp.simplify(correcto[0])
    st.latex(rf"\lim_{{x\to {a}^-}} f(x) = f({a}) \;\Rightarrow\; k = {sp.latex(kk)}")
    ui.resolver_valor("u2_cont_resp", kk,
                      tema="U2-continuidad-k", enunciado=nom)


def _tab_discontinuidad() -> None:
    st.markdown("#### Clasificación de discontinuidades")
    if "u2_disc" not in st.session_state:
        st.session_state["u2_disc"] = random.randint(0, len(_DISCONTINUIDAD_BANCO) - 1)
    nombre, expr, a = _DISCONTINUIDAD_BANCO[st.session_state["u2_disc"]]

    if st.button("🎲 Otro ejercicio (discontinuidad)", key="u2_disc_nuevo"):
        st.session_state["u2_disc"] = random.randint(0, len(_DISCONTINUIDAD_BANCO) - 1)
        st.session_state.pop("w_u2_disc_resp", None)
        st.session_state.pop("mc_u2_disc_resp", None)
        st.rerun()

    st.write(f"**{nombre}**  — analizá la discontinuidad en **x = {a}**.")
    tip = clasificar_discontinuidad(expr, a)
    etiquetas = {
        "evitable": "Evitable (el límite existe y la función se puede redefinir)",
        "salto finito": "Salto finito (laterales finitos y distintos)",
        "esencial (infinita)": "Esencial infinita (algún lateral tiende a ±∞)",
    }
    if tip not in etiquetas:
        st.info(f"En x = {a} la función resulta **{tip}**: no hay discontinuidad que clasificar.")
        return
    orden = list(etiquetas)
    random.Random(st.session_state["u2_disc"]).shuffle(orden)
    ui.elegir_opcion("u2_disc_resp", "El punto x = a es una discontinuidad...",
                     [f"$\\Rightarrow$ {etiquetas[o]}" for o in orden],
                     orden.index(tip),
                     explicacion=etiquetas[tip],
                     tema="U2-discontinuidad", enunciado=nombre)


def ejercicios(repo=None) -> None:
    st.markdown(
        "Ejercitá límites algebraicos, límites en el infinito, laterales, continuidad y "
        "clasificación de discontinuidades."
    )
    t1, t2, t3, t4, t5 = st.tabs(
        ["Límites 0/0", "∞/∞ y asíntotas", "Laterales", "Continuidad (k)", "Discontinuidades"]
    )
    with t1:
        _tab_limites_alg()
    with t2:
        _tab_limites_inf()
    with t3:
        _tab_laterales()
    with t4:
        _tab_continuidad()
    with t5:
        _tab_discontinuidad()