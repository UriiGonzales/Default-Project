"""Unidad 5 · Integral (Programa 2026: UN 5).

5.1 Primitivas, integral indefinida y teorema fundamental del cálculo.
5.2 Métodos de integración: sustitución y por partes.
5.3 Integral definida, regla de Barrow, propiedades y área bajo la curva.
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
    es_primitiva,
    integral_definida,
    area_entre,
    sumas_riemann,
)
from .. import ui


# ===========================================================================
# TEORÍA
# ===========================================================================

def teoria() -> None:
    st.markdown(
        "La **integral** es el 'cálculo de áreas' y, a la vez, la operación inversa de derivar: "
        "esos dos sentidos se conectan en el **Teorema Fundamental del Cálculo**."
    )

    with st.expander("5.1 Primitivas y Teorema Fundamental del Cálculo", expanded=True):
        st.markdown(
            "$F$ es una **primitiva** de $f$ en un intervalo si $F'(x)=f(x)$. Toda primitiva "
            "difiere de las demás en una constante."
        )
        st.latex(r"\int f(x)\,dx = F(x) + C,\qquad F'(x)=f(x)")
        st.markdown(
            "**Teorema Fundamental del Cálculo (2ª parte):** si $F$ es una primitiva de $f$,"
        )
        st.latex(r"\int_a^b f(x)\,dx = F(b)-F(a)\qquad\text{(regla de Barrow)}")
        st.markdown(
            r"""
            **Primitivas inmediatas:** $\int x^n dx=\frac{x^{n+1}}{n+1}+C$ ($n\ne-1$),
            $\int e^x dx=e^x+C$, $\int \frac{dx}{x}=\ln|x|+C$, $\int\sin x\,dx=-\cos x+C$,
            $\int\cos x\,dx=\sin x+C$, $\int\frac{dx}{1+x^2}=\arctan x+C$.
            """
        )
        st.success(
            "**1ª parte del TFC.** Si $F(x)=\\int_a^x f(t)\\,dt$, entonces $F'(x)=f(x)$: la "
            "derivada 'deshace' la integración. Más adelante permite derivar bajo el signo."
        )

    with st.expander("5.2 Métodos de integración", expanded=False):
        st.markdown("**Sustitución (regla de la cadena al revés):**")
        st.latex(r"\int f(g(x))\,g'(x)\,dx = \int f(u)\,du,\qquad u=g(x)")
        st.markdown("**Partes (derivada del producto al revés):**")
        st.latex(r"\int u\,dv = u\,v-\int v\,du")
        st.markdown(
            "Regla práctica para elegir $u$: **ILATE** (Inversas, Logarítmicas, Algebraicas, "
            "Trigonométricas, Exponenciales) — cuanto más 'arriba', mejor candidato para $u$."
        )

    with st.expander("5.3 Integral definida, propiedades y áreas", expanded=False):
        st.markdown("**Propiedades de linealidad y aditividad:**")
        st.latex(r"\int_a^b cf = c\int_a^b f,\qquad \int_a^b (f+g)=\int_a^b f+\int_a^b g,\qquad \int_a^c f=\int_a^b f+\int_b^c f")
        st.markdown(
            "Si $f(x)\\ge 0$ en $[a,b]$, la integral es el **área bajo la curva**. Si $f$ cambia "
            "de signo, la integral suma las áreas con signo."
        )

    with st.expander("5.4 Demostraciones: TVM del cálculo integral, TFC y Barrow", expanded=False):
        st.markdown(
            r"**Teorema del valor medio del cálculo integral.** Si $f$ es continua en $[a,b]$, "
            r"existe $c\in[a,b]$ tal que"
        )
        st.latex(r"\int_a^b f(x)\,dx = f(c)\,(b-a)")
        st.markdown(
            r"**Demostración.** Por Weierstrass, $m\le f(x)\le M$ en $[a,b]$, de donde "
            r"$m(b-a)\le \int_a^b f \le M(b-a)$. Sea "
            r"$y=\frac{1}{b-a}\int_a^b f$; entonces $y\in[m,M]$ y, por el teorema del valor "
            r"intermedio aplicado a $f$, existe $c$ con $f(c)=y$. Multiplicando por $b-a$ "
            r"queda la fórmula. $\blacksquare$"
        )

        st.markdown(
            r"**Teorema Fundamental del Cálculo (1ª parte).** Si $f$ es continua en $[a,b]$ y "
            r"definimos $F(x)=\int_a^x f(t)\,dt$, entonces $F'(x)=f(x)$ para todo $x\in(a,b)$."
        )
        st.markdown(
            r"**Demostración.** Para $h$ pequeño,"
        )
        st.latex(r"\frac{F(x+h)-F(x)}{h}=\frac{1}{h}\int_x^{x+h} f(t)\,dt")
        st.markdown(
            r"Sean $m_h$ y $M_h$ el mínimo y el máximo de $f$ en $[x,x+h]$. Acotando el "
            r"integrando,"
        )
        st.latex(r"m_h \le \frac{F(x+h)-F(x)}{h}\le M_h")
        st.markdown(
            r"Por la continuidad de $f$, $m_h\to f(x)$ y $M_h\to f(x)$ cuando $h\to 0$, y por "
            r"el teorema del encaje el cociente tiende a $f(x)$. El caso $h<0$ es idéntico "
            r"(con los límites de integración invertidos). $\blacksquare$"
        )

        st.markdown(
            r"**Regla de Barrow (TFC, 2ª parte).** Si $F$ es una primitiva de $f$ continua en "
            r"$[a,b]$,"
        )
        st.latex(r"\int_a^b f(x)\,dx = F(b)-F(a)")
        st.markdown(
            r"**Demostración.** Por la 1ª parte, $G(x)=\int_a^x f$ es una primitiva. Como "
            r"$(G-F)'=0$ en $(a,b)$, el corolario del TVM diferencial asegura que "
            r"$G-F$ es constante. Evaluando en $a$ (donde $G(a)=0$): $G(x)=F(x)-F(a)$ para "
            r"todo $x$; tomando $x=b$ queda el resultado. $\blacksquare$"
        )

        st.markdown(
            r"**Integración por partes.** Para $u,v$ derivables con $uv'$ integrable en $[a,b]$:"
        )
        st.latex(r"\int_a^b u\,v'\,dx=\Big[u\,v\Big]_a^b-\int_a^b v\,u'\,dx")
        st.markdown(
            r"**Demostración.** De la regla del producto $(uv)'=u'v+uv'$ se despeja "
            r"$uv'=(uv)'-u'v$; integrando entre $a$ y $b$ y aplicando Barrow queda la "
            r"identidad. $\blacksquare$"
        )


# ===========================================================================
# INTUICIÓN VISUAL
# ===========================================================================

_FUNCIONES_INT = {
    "x² en [0,2]": (x**2, 0, 2),
    "x³ en [0,2]": (x**3, 0, 2),
    "sin(x) en [0,π]": (sp.sin(x), 0, sp.pi),
    "x² en [−1,2]": (x**2, -1, 2),
}


def _riemann_visual() -> None:
    st.markdown("##### Sumas de Riemann: la integral como límite")
    nom = st.selectbox("Función e intervalo", list(_FUNCIONES_INT), key="u5_ri_f")
    f, a, b = _FUNCIONES_INT[nom]
    N = st.slider("N (número de subintervalos)", 2, 80, 8, 1, key="u5_ri_n")
    modo = st.radio("Punto de muestra", ["izquierdo", "derecho", "central"], horizontal=True, key="u5_ri_modo")

    valores = {"izquierdo": "izquierda", "derecho": "derecha", "central": "media"}[modo]
    suma = sumas_riemann(f, a, b, N, modo=valores)
    dx = (b - a) / N
    fnum = sp.lambdify(x, f, "numpy")

    fig = go.Figure()
    base = np.linspace(float(a), float(b), 600)
    fig.add_trace(go.Scatter(x=base, y=fnum(base), mode="lines", name="f(x)",
                             line=dict(color="#1f77b4", width=3)))
    if valores == "izquierda":
        xs = np.linspace(float(a), float(b) - dx, N)
    elif valores == "derecha":
        xs = np.linspace(float(a) + dx, float(b), N)
    else:
        xs = np.linspace(float(a) + dx / 2, float(b) - dx / 2, N)
    ys = fnum(xs)
    xs = np.clip(xs, float(a), float(b))
    fig.add_trace(go.Bar(x=xs + dx / 2, y=ys, width=dx * 0.95, name=f"rectángulos ({modo})",
                         marker_color="rgba(255,127,14,0.55)", marker_line_color="orange"))
    fig.update_layout(title=f"Suma de Riemann ({modo}) = {suma:.4f}",
                      height=430, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig, width="stretch")
    exacta = float(sp.integrate(f, (x, a, b)))
    st.markdown(
        rf"**Al crecer $N$, la suma se acerca a la integral exacta** $= {exacta:.4f}$ "
        "(error actual: {:.4f}).".format(abs(float(suma) - exacta))
    )


def _integral_definida_visual() -> None:
    st.markdown("##### Área bajo la curva")
    nom = st.selectbox("Función e intervalo", list(_FUNCIONES_INT), key="u5_id_f",
                       index=0)
    f, a, b = _FUNCIONES_INT[nom]
    fnum = sp.lambdify(x, f, "numpy")
    lo = float(a) - 0.6
    hi = float(b) + 0.6
    base = np.linspace(lo, hi, 800)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base, y=fnum(base), mode="lines", name="f(x)",
                             line=dict(color="#1f77b4", width=3)))
    xs = np.linspace(float(a), float(b), 400)
    fig.add_trace(go.Scatter(x=xs, y=fnum(xs), fill="tozeroy", mode="lines",
                             name="área = ∫f",
                             fillcolor="rgba(76,175,80,0.35)", line=dict(width=0)))
    fig.update_layout(title=f"∫ = {sp.integrate(f, (x, a, b))}",
                      height=430, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig, width="stretch")


def _tfc_visual() -> None:
    st.markdown("##### TFC en acción: F(x) = ∫₀^x f(t) dt")
    f = sp.sin(x) + 1
    fnum = sp.lambdify(x, f, "numpy")
    xs = np.linspace(0, 6, 700)
    dxv = xs[1] - xs[0]
    F_num = np.concatenate([[0.0], np.cumsum((fnum(xs[:-1]) + fnum(xs[1:])) / 2 * dxv)])

    x0 = st.slider("Punto x₀", 0.0, 6.0, 2.0, 0.1, key="u5_tfc_x")
    idx0 = int(np.argmin(np.abs(xs - x0)))
    Fx0 = float(F_num[idx0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=fnum(xs), mode="lines", name="f(t)",
                             line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=xs, y=F_num, mode="lines", name="F(x)=∫₀ˣf", yaxis="y2",
                             line=dict(color="purple", width=3)))
    fig.add_trace(go.Scatter(x=[x0], y=[fnum(x0)], mode="markers", name="(x₀, f(x₀))",
                             marker=dict(size=9, color="red")))
    fig.add_trace(go.Scatter(x=[x0], y=[Fx0], mode="markers", name="F(x₀)", yaxis="y2",
                             marker=dict(size=9, color="darkorange")))
    fig.update_layout(title="La pendiente de F en x₀ es f(x₀)",
                      height=430, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="x", yaxis_title="f",
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="F"))
    st.plotly_chart(fig, width="stretch")

    idx2 = min(idx0 + 3, len(xs) - 4)
    idx1 = max(idx0 - 3, 3)
    pendiente = (F_num[idx2] - F_num[idx1]) / (xs[idx2] - xs[idx1])
    st.markdown(
        rf"F(${x0:.1f}$) = {Fx0:.3f} y la pendiente de F ahí es **{pendiente:.3f}** ≈ "
        rf"**f({x0:.1f}) = {float(fnum(x0)):.3f}**."
    )


def intuicion() -> None:
    st.markdown(
        "Las sumas de Riemann convergen a la integral, el área se visualiza directamente y el "
        "TFC muestra que $F$ es una primitiva."
    )
    t1, t2, t3 = st.tabs(["Sumas de Riemann", "Área bajo la curva", "Teorema Fundamental"])
    with t1:
        _riemann_visual()
    with t2:
        _integral_definida_visual()
    with t3:
        _tfc_visual()


# ===========================================================================
# EJERCICIOS
# ===========================================================================

_PRIMITIVAS_BANCO = [
    ("f(x) = 3x²", 3 * x**2),
    ("f(x) = cos(2x)", sp.cos(2 * x)),
    ("f(x) = e^{3x}", sp.exp(3 * x)),
    ("f(x) = 1/x", 1 / x),
    ("f(x) = x/(x²+1)", x / (x**2 + 1)),
    ("f(x) = (2x+1)³", (2 * x + 1) ** 3),
    ("f(x) = √x", sp.sqrt(x)),
    ("f(x) = x·e^{x²}", x * sp.exp(x**2)),
]

_SUSTITUCION_BANCO = [
    (r"\int \frac{x}{x^2+1}\,dx", "u = x²+1", ["u = x", "u = x²+1", "u = 1", "u = x²−1"]),
    (r"\int x\,e^{x^2}\,dx", "u = x²", ["u = e^{x²}", "u = x²", "u = x", "u = 3x"]),
    (r"\int \sin x \cos x\,dx", "u = sin(x)", ["u = sin(x)", "u = x", "u = cos(2x)", "u = tan(x)"]),
    (r"\int \frac{e^x}{1+e^x}\,dx", "u = 1+e^{x}", ["u = e^x", "u = 1+e^x", "u = x", "u = e^{2x}"]),
    (r"\int 2x (x^2+1)^4\,dx", "u = x²+1", ["u = x", "u = 2x", "u = x²+1", "u = (x²+1)^4"]),
]

_PARTES_BANCO = [
    (r"\int x\,e^x\,dx", "u = x, dv = e^x dx", ["u = e^x, dv = x dx", "u = x, dv = e^x dx", "u = x·e^x, dv = dx"]),
    (r"\int x\cos x\,dx", "u = x, dv = cos(x) dx", ["u = cos(x), dv = x dx", "u = x, dv = cos(x) dx"]),
    (r"\int \ln x\,dx", "u = ln(x), dv = dx", ["u = x, dv = ln(x) dx", "u = ln(x), dv = dx"]),
    (r"\int x\ln x\,dx", "u = ln(x), dv = x dx", ["u = ln(x), dv = x dx", "u = x, dv = ln(x) dx"]),
]

_DEFINIDAS_BANCO = [
    ("∫₀¹ 3x² dx", 3 * x**2, 0, 1),
    ("∫₀^π sin(x) dx", sp.sin(x), 0, sp.pi),
    ("∫₀¹ e^x dx", sp.exp(x), 0, 1),
    ("∫₁² 1/x dx", 1 / x, 1, 2),
    ("∫₀^{π/2} cos(x) dx", sp.cos(x), 0, sp.pi / 2),
    ("∫₋₁¹ (x²+1) dx", x**2 + 1, -1, 1),
]

_AREAS_BANCO = [
    ("Entre f=x² y g=x", x**2, x, 0, 1),
    ("Entre f=x² y g=4x", x**2, 4 * x, 0, 4),
    ("Entre f=sin(x) y g=cos(x)", sp.sin(x), sp.cos(x), sp.pi / 4, 5 * sp.pi / 4),
    ("Entre f=x³−x y g=0", x**3 - x, 0, -1, 1),
]


def _reset(clave: str) -> None:
    for k in (f"w_{clave}", f"in_{clave}", f"mc_{clave}"):
        st.session_state.pop(k, None)


def _tab_primitivas() -> None:
    st.markdown("#### Integral indefinida (hallar una primitiva)")
    if "u5_p" not in st.session_state:
        st.session_state["u5_p"] = random.randint(0, len(_PRIMITIVAS_BANCO) - 1)
    i = st.session_state["u5_p"]
    nombre, f_expr = _PRIMITIVAS_BANCO[i]

    if st.button("🎲 Otro ejercicio (primitiva)", key="u5_p_nuevo"):
        st.session_state["u5_p"] = random.randint(0, len(_PRIMITIVAS_BANCO) - 1)
        _reset("u5_p_resp")
        st.rerun()

    st.latex(rf"\int {sp.latex(f_expr)}\,dx  \quad=\quad ?")
    ui.resolver_expresion(
        "u5_p_resp", sp.integrate(f_expr, x),
        placeholder="ej.: x**3  (sin escribir +C)",
        ayuda="Se compara algebraicamente: tu primitiva debe derivar en f(x).",
        tema="U5-primitiva", enunciado=nombre,
    )


def _tab_sustitucion() -> None:
    st.markdown("#### Método de sustitución (elegir el cambio correcto)")
    if "u5_s" not in st.session_state:
        st.session_state["u5_s"] = random.randint(0, len(_SUSTITUCION_BANCO) - 1)
    integral_l, correcta, distractores = _SUSTITUCION_BANCO[st.session_state["u5_s"]]

    if st.button("🎲 Otro ejercicio (sustitución)", key="u5_s_nuevo"):
        st.session_state["u5_s"] = random.randint(0, len(_SUSTITUCION_BANCO) - 1)
        _reset("u5_s_resp")
        st.rerun()

    st.markdown(f"Para calcular {integral_l}, ¿cuál es el cambio de variable $u$ más adecuado?")
    opciones = [correcta] + list(distractores)
    random.Random(st.session_state["u5_s"]).shuffle(opciones)
    ui.elegir_opcion("u5_s_resp", "Elijo:",
                     [f"$ {o} $" for o in opciones], opciones.index(correcta),
                     explicacion=f"Se elige u = {correcta.split('=')[1]}.",
                     tema="U5-sustitucion", enunciado=integral_l)


def _tab_partes() -> None:
    st.markdown("#### Método de integración por partes (u y dv)")
    if "u5_pt" not in st.session_state:
        st.session_state["u5_pt"] = random.randint(0, len(_PARTES_BANCO) - 1)
    integral_l, correcta, distractores = _PARTES_BANCO[st.session_state["u5_pt"]]

    if st.button("🎲 Otro ejercicio (por partes)", key="u5_pt_nuevo"):
        st.session_state["u5_pt"] = random.randint(0, len(_PARTES_BANCO) - 1)
        _reset("u5_pt_resp")
        st.rerun()

    st.markdown(f"Para calcular {integral_l}, la elección correcta de $u$ y $dv$ es:")
    opciones = [correcta] + list(distractores)
    random.Random(st.session_state["u5_pt"]).shuffle(opciones)
    ui.elegir_opcion("u5_pt_resp", "Aplico la fórmula",
                     [f"$ {o} $" for o in opciones], opciones.index(correcta),
                     explicacion=f"Elegí {correcta} y luego ∫u dv = uv − ∫v du.",
                     tema="U5-por-partes", enunciado=integral_l)


def _tab_definidas() -> None:
    st.markdown("#### Integrales definidas (regla de Barrow)")
    if "u5_d" not in st.session_state:
        st.session_state["u5_d"] = random.randint(0, len(_DEFINIDAS_BANCO) - 1)
    nombre, f_expr, a, b = _DEFINIDAS_BANCO[st.session_state["u5_d"]]

    if st.button("🎲 Otro ejercicio (definida)", key="u5_d_nuevo"):
        st.session_state["u5_d"] = random.randint(0, len(_DEFINIDAS_BANCO) - 1)
        _reset("u5_d_resp")
        st.rerun()

    st.latex(rf"\int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} {sp.latex(f_expr)}\,dx")
    correcto = sp.simplify(integral_definida(f_expr, a, b))
    ui.resolver_valor("u5_d_resp", correcto, placeholder="ej.: 2",
                      tema="U5-definida", enunciado=nombre)


def _tab_areas() -> None:
    st.markdown("#### Área entre curvas")
    if "u5_a" not in st.session_state:
        st.session_state["u5_a"] = random.randint(0, len(_AREAS_BANCO) - 1)
    nombre, f, g, a, b = _AREAS_BANCO[st.session_state["u5_a"]]

    if st.button("🎲 Otro ejercicio (área entre curvas)", key="u5_a_nuevo"):
        st.session_state["u5_a"] = random.randint(0, len(_AREAS_BANCO) - 1)
        _reset("u5_a_resp")
        st.rerun()

    fn = sp.lambdify(x, f, "numpy")
    gn = sp.lambdify(x, g, "numpy")
    xs = np.linspace(float(a), float(b), 400)
    with np.errstate(all="ignore"):
        ys1, ys2 = np.asarray(fn(xs), dtype=float), np.asarray(gn(xs), dtype=float)
    if ys1.ndim == 0:
        ys1 = np.full_like(xs, float(ys1))
    if ys2.ndim == 0:
        ys2 = np.full_like(xs, float(ys2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys1, mode="lines", name=sp.latex(f), line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=xs, y=ys2, mode="lines", name=sp.latex(g), line=dict(color="#ff7f0e")))
    fig.add_trace(go.Scatter(x=np.concatenate([xs, xs[::-1]]),
                             y=np.concatenate([ys1, ys2[::-1]]), fill="toself",
                             fillcolor="rgba(76,175,80,0.25)", line=dict(width=0),
                             name="área entre curvas"))
    fig.update_layout(title=nombre, height=420, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig, width="stretch")

    correcto = sp.simplify(area_entre(f, g, a, b))
    ui.resolver_valor("u5_a_resp", correcto, placeholder="ej.: 32/3",
                      tema="U5-area-entre-curvas", enunciado=nombre)


def ejercicios(repo=None) -> None:
    st.markdown("Primitivas, métodos de integración, definidas y áreas entre curvas.")
    t1, t2, t3, t4, t5 = st.tabs(
        ["Primitivas", "Sustitución", "Por partes", "Definidas", "Área entre curvas"]
    )
    with t1:
        _tab_primitivas()
    with t2:
        _tab_sustitucion()
    with t3:
        _tab_partes()
    with t4:
        _tab_definidas()
    with t5:
        _tab_areas()