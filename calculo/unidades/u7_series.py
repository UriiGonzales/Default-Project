"""Unidad 7 · Sucesiones y Series (Programa 2026: UN 7).

7.1 Sucesiones: convergencia, monotonía y acotación.
7.2 Series numéricas y criterios de convergencia (comparación, D'Alembert, Raabe, Leibniz).
7.3 Series de potencias y desarrollo de Taylor/Maclaurin.
"""

from __future__ import annotations

import random

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from ..matematicas import (
    x,
    n,
    limite_sucesion,
    suma_serie_geometrica,
    radio_convergencia,
)
from .. import ui


# ===========================================================================
# TEORÍA
# ===========================================================================

def teoria() -> None:
    st.markdown(
        "Las **sucesiones** ordenan infinitos números; las **series** los suman. Determinar "
        "cuándo esa suma infinita da un número finito es el objetivo de esta unidad."
    )

    with st.expander("7.1 Sucesiones numéricas", expanded=True):
        st.markdown(
            "Una sucesión $a_n$ converge a $L$ si los términos se acercan a $L$ todo lo que "
            "se quiera. **Toda sucesión convergente es acotada**; si es monótona y acotada, "
            "converge (teorema de convergencia monótona)."
        )
        st.latex(r"\lim_{n\to\infty} a_n = L")

    with st.expander("7.2 Series numéricas y criterios", expanded=False):
        st.markdown(
            "La serie $\\sum a_n$ converge si sus **sumas parciales** $S_N$ tienen límite finito. "
            "Caso base indispensable: la serie **geométrica**."
        )
        st.latex(r"\sum_{n=0}^\infty r^n = \frac{1}{1-r}\iff |r|<1;\qquad \text{serie armónica }\sum\frac1n\ \text{diverge}")
        st.markdown("**Criterios:**")
        st.markdown(
            r"""
            - **D'Alembert (razón):** si $\lim |a_{n+1}/a_n|=L$, converge si $L<1$, diverge si $L>1$.
            - **Comparación y límite de comparación:** con series de referencia (geométricas, $p$-series).
            - **Raabe:** cuando la razón da $L=1$.
            - **Leibniz:** si $a_n\downarrow 0$, $\sum(-1)^n a_n$ converge.
            """
        )
        st.latex(r"p\text{-serie }\sum\frac1{n^p}:\ \text{converge}\iff p>1")

    with st.expander("7.3 Series de potencias y Taylor", expanded=False):
        st.markdown(
            "Una serie de potencias $\\sum a_n (x-c)^n$ converge en un **intervalo** centrado "
            "en $c$ con **radio de convergencia** $R$."
        )
        st.latex(r"R=\lim_{n\to\infty}\left|\frac{a_n}{a_{n+1}}\right|")
        st.markdown("**Desarrollos de Maclaurin** (válidos para todo $x$ salvo indicación):")
        st.latex(
            r"\begin{aligned}"
            r"& e^x=\sum\frac{x^n}{n!},\qquad \sin x=\sum(-1)^n\frac{x^{2n+1}}{(2n+1)!},\\"
            r"&\cos x=\sum(-1)^n\frac{x^{2n}}{(2n)!},\qquad \frac1{1-x}=\sum x^n\quad(|x|<1),\\"
            r"&\ln(1+x)=\sum(-1)^{n+1}\frac{x^n}{n}\quad(|x|<1)"
            r"\end{aligned}"
        )

    with st.expander("7.4 Demostraciones: serie geométrica y sucesiones monótonas", expanded=False):
        st.markdown(
            r"**Teorema (criterio de la serie geométrica).** "
            r"$\sum_{n=0}^{\infty} r^n$ converge si y solo si $|r|<1$, y en ese caso su suma "
            r"es $\dfrac{1}{1-r}$."
        )
        st.markdown(r"**Demostración.** Las sumas parciales satisfacen $S_N=1+r+\cdots+r^N$. Multiplicando por $r$ y restando,")
        st.latex(
            r"S_N - r\,S_N = (1+r+\cdots+r^N)-(r+r^2+\cdots+r^{N+1}) = 1 - r^{N+1}"
        )
        st.markdown(
            r"de donde $S_N=\dfrac{1-r^{N+1}}{1-r}$ (para $r\neq1$). Entonces:"
        )
        st.markdown(
            r"- Si $|r|<1$, $r^{N+1}\to 0$ y $S_N\to \dfrac{1}{1-r}$. "
            r"- Si $|r|>1$, $|r|^{N+1}\to\infty$: la serie diverge. "
            r"- Si $r=1$: $S_N=N+1\to\infty$. Si $r=-1$: $S_N$ alterna entre $1$ y $0$, no "
            r"converge. "
            r"$\blacksquare$"
        )

        st.markdown(
            r"**Teorema de las sucesiones monótonas.** Toda sucesión creciente y acotada "
            r"superiormente converge (a su supremo); toda sucesión decreciente y acotada "
            r"inferiormente converge (a su ínfimo)."
        )
        st.markdown(
            r"**Demostración (caso creciente).** Sea $S=\sup\{a_n\}$, que existe por la "
            r"acotación. Dado $\varepsilon>0$, el número $S-\varepsilon$ no es cota superior, "
            r"luego existe $N$ con $a_N>S-\varepsilon$. Como la sucesión es creciente, para "
            r"todo $n\ge N$:"
        )
        st.latex(r"S-\varepsilon < a_N \le a_n \le S < S+\varepsilon\;\Rightarrow\;|a_n-S|<\varepsilon")
        st.markdown(
            r"Esto es exactamente $a_n\to S$. El caso decreciente se reduce al anterior "
            r"considerando $-a_n$. $\blacksquare$"
        )


# ===========================================================================
# INTUICIÓN VISUAL
# ===========================================================================

_SECUENCIAS = {
    "a_n = 1/n   → 0": 1 / n,
    "a_n = n/(n+1)   → 1": n / (n + 1),
    "a_n = (1+1/n)^n   → e": (1 + 1 / n) ** n,
    "a_n = n²/(n²+3)   → 1": n**2 / (n**2 + 3),
    "a_n = n/eⁿ   → 0": n / sp.exp(n),
}


def _sucesiones_visual() -> None:
    st.markdown("##### Convergencia de sucesiones")
    nom = st.selectbox("Sucesión", list(_SECUENCIAS), key="u7_suc_f")
    termino = _SECUENCIAS[nom]
    Nmax = st.slider("Mostrar hasta n =", 10, 80, 30, 1, key="u7_suc_n")

    fn = sp.lambdify(n, termino, "numpy")
    ns = np.arange(1, Nmax + 1, dtype=float)
    valores = fn(ns)
    L = limite_sucesion(termino)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=valores, mode="markers", name="a_n",
                             marker=dict(size=6, color="#1f77b4")))
    if L.is_finite:
        fig.add_hline(y=float(L.evalf()), line_dash="dot", line_color="green",
                      annotation_text=f"L = {float(L.evalf()):.4f}")
    fig.update_layout(title=f"{nom}  →  límite {L}",
                      height=430, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="n", yaxis_title="a_n")
    st.plotly_chart(fig, width="stretch")


def _serie_geometrica_visual() -> None:
    st.markdown("##### Sumas parciales de la serie geométrica")
    r = st.slider("Razón r", -0.95, 0.95, 0.5, 0.01, key="u7_geo_r")
    Nmax = st.slider("Sumar hasta N =", 1, 60, 12, 1, key="u7_geo_n")

    ns = np.arange(1, Nmax + 1)
    parciales = np.cumsum(r ** ns)
    L = 1 / (1 - r)
    if r == 0:
        L = 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=parciales, mode="lines+markers", name="S_N",
                             line=dict(color="#ff7f0e", width=2)))
    fig.add_hline(y=L, line_dash="dot", line_color="green", annotation_text=f"L = {L:.3f}")
    fig.update_layout(title=f"s = Σ r^n = 1/(1−r) = {L:.3f}",
                      height=430, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="N", yaxis_title="S_N")
    st.plotly_chart(fig, width="stretch")
    if abs(r) >= 1:
        st.error("|r| ≥ 1: la serie no converge (las parciales se disparan o no se estabilizan).")


def _serie_potencia_visual() -> None:
    st.markdown("##### Series de potencias: Maclaurin aproximando a la función")
    opciones = {
        "1/(1−x)  (|x|<1)": 1 / (1 - sp.Symbol("t")),
        "e^x": sp.exp(sp.Symbol("t")),
        "sin(x)": sp.sin(sp.Symbol("t")),
        "cos(x)": sp.cos(sp.Symbol("t")),
        "ln(1+x)  (|x|<1)": sp.ln(1 + sp.Symbol("t")),
    }
    nom = st.selectbox("Función", list(opciones), key="u7_pot_f")
    orden = st.slider("Orden del polinomio de Taylor", 1, 12, 4, 1, key="u7_pot_k")
    t = sp.Symbol("t")

    lo, hi = (-0.99, 0.99) if "|x|<1" in nom else (-3.5, 3.5)
    xs = np.linspace(lo, hi, 900)
    f = opciones[nom]
    fn = sp.lambdify(t, f, "numpy")
    serie = sp.series(f, t, 0, orden + 1).removeO()
    fs = sp.lambdify(t, serie, "numpy")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=fn(xs), mode="lines", name="f(x)",
                             line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=xs, y=fs(xs), mode="lines", name=f"P_{orden}(x)",
                             line=dict(color="#ff7f0e", width=2, dash="dash")))
    fig.update_layout(title=f"{nom} · Maclaurin de orden {orden}",
                      height=430, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig, width="stretch")
    st.markdown("Con más términos, el polinomio se 'pega' a la función dentro del intervalo de convergencia.")


def intuicion() -> None:
    st.markdown(
        "Observá cómo las sucesiones tienden a su límite, cómo las parciales de la geométrica "
        "se estabilizan y cómo Maclaurin aproxima a la función."
    )
    t1, t2, t3 = st.tabs(["Sucesiones", "Serie geométrica", "Series de potencias"])
    with t1:
        _sucesiones_visual()
    with t2:
        _serie_geometrica_visual()
    with t3:
        _serie_potencia_visual()


# ===========================================================================
# EJERCICIOS
# ===========================================================================

_LIMITES_SUCESION = [
    ("a_n = 1/n", 1 / n),
    ("a_n = (n+1)/n", (n + 1) / n),
    ("a_n = (n²+1)/(n²−n)", (n**2 + 1) / (n**2 - n)),
    ("a_n = 3n/(2n+1)", 3 * n / (2 * n + 1)),
    ("a_n = (1+1/n)^n", (1 + 1 / n) ** n),
    ("a_n = n²/eⁿ", n**2 / sp.exp(n)),
    ("a_n = (2n²−1)/(3n²+1)", (2 * n**2 - 1) / (3 * n**2 + 1)),
]

_GEOMETRICAS_BANCO = [
    (r"\sum_{n=0}^\infty \left(\frac{1}{2}\right)^n", 0.5),
    (r"\sum_{n=0}^\infty \left(\frac{1}{3}\right)^n", 1 / 3),
    (r"\sum_{n=0}^\infty \left(-\frac{1}{2}\right)^n", -0.5),
    (r"\sum_{n=1}^\infty \left(\frac{1}{2}\right)^n", 0.5, 1),
    (r"\sum_{n=0}^\infty \left(\frac{3}{4}\right)^n", 0.75),
]

_CONVERGENCIA_BANCO = [
    ("Σ 1/n² converge (p=2>1)", True),
    ("Σ 1/n (armónica) converge", False),
    ("Σ (1/2)ⁿ converge", True),
    ("Σ (5/4)ⁿ converge", False),
    ("Σ 1/n! converge", True),
    ("Σ (−1)ⁿ/n converge (Leibniz)", True),
]

_CRITERIOS_BANCO = [
    ("La p-serie Σ 1/nᵖ converge si p > 1.", True),
    ("D'Alembert: si el límite de |a_{n+1}/a_n| es L > 1, la serie converge.", False),
    ("El criterio de Leibniz aplica a series alternadas con término a_n decreciente a 0.", True),
    ("La serie armónica Σ 1/n es convergente.", False),
    ("Si los términos a_n no tienden a 0, la serie no puede converger.", True),
]

_RADIO_BANCO = [
    ("Σ xⁿ  (a_n = 1)", 1),
    ("Σ xⁿ/n  (a_n = 1/n)", 1),
    ("Σ xⁿ/n!  (a_n = 1/n!)", sp.oo),
    ("Σ (2x)ⁿ  (a_n = 2ⁿ)", sp.Rational(1, 2)),
    ("Σ xⁿ/n²  (a_n = 1/n²)", 1),
    ("Σ (5x)ⁿ  (a_n = 5ⁿ)", sp.Rational(1, 5)),
]

_TAYLOR_BANCO = [
    ("e^x = Σ xⁿ/n!", True),
    ("sin x = Σ (−1)ⁿ x^{2n+1}/(2n+1)!", True),
    ("cos x = Σ x^{2n}/(2n)!  (falta el (−1)ⁿ → es un desarrollo incorrecto)", False),
    ("1/(1−x) = Σ xⁿ vale para todo x real", False),
    ("ln(1+x) = Σ (−1)^{n+1} xⁿ/n vale para |x|<1", True),
]


def _reset(clave: str) -> None:
    for k in (f"w_{clave}", f"in_{clave}", f"mc_{clave}"):
        st.session_state.pop(k, None)


def _tab_limites() -> None:
    st.markdown("#### Límite de sucesiones")
    if "u7_l" not in st.session_state:
        st.session_state["u7_l"] = random.randint(0, len(_LIMITES_SUCESION) - 1)
    nombre, termino = _LIMITES_SUCESION[st.session_state["u7_l"]]

    if st.button("🎲 Otra sucesión", key="u7_l_nuevo"):
        st.session_state["u7_l"] = random.randint(0, len(_LIMITES_SUCESION) - 1)
        _reset("u7_l_resp")
        st.rerun()

    st.markdown(f"Hallá el límite si existe.  **{nombre}**")
    correcto = sp.simplify(limite_sucesion(termino))
    if correcto.has(sp.factorial, sp.gamma):
        st.info("Este límite requiere cuidado; pensalo numéricamente.")
    ui.resolver_valor("u7_l_resp", correcto, placeholder="ej.: 1  o  0",
                      tema="U7-limite-sucesion", enunciado=nombre)


def _tab_geometrica() -> None:
    st.markdown("#### Suma de series geométricas")
    if "u7_g" not in st.session_state:
        st.session_state["u7_g"] = random.randint(0, len(_GEOMETRICAS_BANCO) - 1)
    enunciado, r = _GEOMETRICAS_BANCO[st.session_state["u7_g"]][:2]
    n0 = _GEOMETRICAS_BANCO[st.session_state["u7_g"]][2] if len(_GEOMETRICAS_BANCO[st.session_state["u7_g"]]) > 2 else 0

    if st.button("🎲 Otra serie geométrica", key="u7_g_nuevo"):
        st.session_state["u7_g"] = random.randint(0, len(_GEOMETRICAS_BANCO) - 1)
        _reset("u7_g_resp")
        st.rerun()

    st.latex(enunciado + r"\quad =\quad ?")
    correcto = suma_serie_geometrica(r, n0)
    ui.resolver_valor("u7_g_resp", correcto, placeholder="ej.: 2  (o oo si diverge)",
                      tema="U7-serie-geometrica", enunciado=enunciado)


def _tab_convergencia() -> None:
    st.markdown("#### ¿Converge? (serie por término general)")
    if "u7_c" not in st.session_state:
        st.session_state["u7_c"] = random.randint(0, len(_CONVERGENCIA_BANCO) - 1)
    enunciado, verdadero = _CONVERGENCIA_BANCO[st.session_state["u7_c"]]

    if st.button("🎲 Otra serie", key="u7_c_nuevo"):
        st.session_state["u7_c"] = random.randint(0, len(_CONVERGENCIA_BANCO) - 1)
        _reset("u7_c_resp")
        st.rerun()

    st.markdown(enunciado)
    ui.elegir_opcion("u7_c_resp", "La afirmación es:",
                     ["Verdadera", "Falsa"], 0 if verdadero else 1,
                     tema="U7-convergencia", enunciado=enunciado)


def _tab_criterios() -> None:
    st.markdown("#### Criterios de convergencia (conceptos, V/F)")
    if "u7_cr" not in st.session_state:
        st.session_state["u7_cr"] = random.randint(0, len(_CRITERIOS_BANCO) - 1)
    enunciado, verdadero = _CRITERIOS_BANCO[st.session_state["u7_cr"]]

    if st.button("🎲 Otro enunciado (criterios)", key="u7_cr_nuevo"):
        st.session_state["u7_cr"] = random.randint(0, len(_CRITERIOS_BANCO) - 1)
        _reset("u7_cr_resp")
        st.rerun()

    st.markdown(enunciado)
    ui.elegir_opcion("u7_cr_resp", "La afirmación es:",
                     ["Verdadera", "Falsa"], 0 if verdadero else 1,
                     tema="U7-criterios", enunciado=enunciado)


def _tab_radio() -> None:
    st.markdown("#### Radio de convergencia (R)")
    if "u7_r" not in st.session_state:
        st.session_state["u7_r"] = random.randint(0, len(_RADIO_BANCO) - 1)
    enunciado, a_n = _RADIO_BANCO[st.session_state["u7_r"]]

    if st.button("🎲 Otra serie de potencias", key="u7_r_nuevo"):
        st.session_state["u7_r"] = random.randint(0, len(_RADIO_BANCO) - 1)
        _reset("u7_r_resp")
        st.rerun()

    st.markdown(f"Hallá el radio de convergencia de **{enunciado}** (a_n = {sp.latex(a_n)}).")
    correcto = radio_convergencia(a_n)
    if correcto is None:
        st.warning("No se pudo calcular el radio con el criterio de la razón.")
        return
    ui.resolver_valor("u7_r_resp", sp.oo if correcto == float("inf") else sp.nsimplify(correcto),
                      placeholder="ej.: 1  o  1/2  (oo si infinito)",
                      tema="U7-radio-convergencia", enunciado=enunciado)


def _tab_taylor() -> None:
    st.markdown("#### Desarrollos de Taylor/Maclaurin (reconocimiento, V/F)")
    if "u7_t" not in st.session_state:
        st.session_state["u7_t"] = random.randint(0, len(_TAYLOR_BANCO) - 1)
    enunciado, verdadero = _TAYLOR_BANCO[st.session_state["u7_t"]]

    if st.button("🎲 Otro desarrollo", key="u7_t_nuevo"):
        st.session_state["u7_t"] = random.randint(0, len(_TAYLOR_BANCO) - 1)
        _reset("u7_t_resp")
        st.rerun()

    st.markdown(enunciado)
    ui.elegir_opcion("u7_t_resp", "La afirmación es:",
                     ["Verdadera", "Falsa"], 0 if verdadero else 1,
                     tema="U7-taylor", enunciado=enunciado)


def ejercicios(repo=None) -> None:
    st.markdown("Límites de sucesiones, series geométricas, convergencia, criterios, radio y Taylor.")
    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Límites de sucesiones", "Series geométricas", "¿Converge?", "Criterios", "Radio R", "Taylor"]
    )
    with t1:
        _tab_limites()
    with t2:
        _tab_geometrica()
    with t3:
        _tab_convergencia()
    with t4:
        _tab_criterios()
    with t5:
        _tab_radio()
    with t6:
        _tab_taylor()