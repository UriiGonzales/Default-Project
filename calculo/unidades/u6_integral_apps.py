"""Unidad 6 · Aplicaciones del Cálculo Integral (Programa 2026: UN 6).

6.1 Área entre curvas. Sólidos de revolución (discos y arandelas).
6.2 Longitudinal de arco y métodos de capas (opcional).
6.3 Integrales impropias de primera y segunda especie.
6.4 Aplicaciones físicas: trabajo y presión.
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
    volumen_revolucion,
    longitud_arco,
    integral_impropia,
)
from .. import ui


# ===========================================================================
# TEORÍA
# ===========================================================================

def teoria() -> None:
    st.markdown(
        "La integral se aplica a **geometría** (áreas, volúmenes, longitudes) y a **física** "
        "(trabajo, presión). Todas siguen el mismo esquema: *partir, aproximar, sumar, pasar "
        "al límite*."
    )

    with st.expander("6.1 Área entre curvas y sólidos de revolución", expanded=True):
        st.markdown("**Área entre dos curvas** en $[a,b]$ donde $f\\ge g$:")
        st.latex(r"A = \int_a^b \big(f(x)-g(x)\big)\,dx")
        st.markdown("**Sólido de revolución** al girar $y=f(x)\\ge0$ alrededor del eje $x$:")
        st.latex(r"V = \pi \int_a^b f(x)^2\,dx\qquad\text{(método de discos)}")
        st.markdown(
            "Si la región queda entre $f$ y $g$ ($f\\ge g\\ge0$), se usa el **método de "
            "arandelas**:"
        )
        st.latex(r"V = \pi \int_a^b \big(f(x)^2-g(x)^2\big)\,dx")

    with st.expander("6.2 Longitud de arco (y área de superficie)", expanded=False):
        st.markdown(
            "La **longitud de arco** de la gráfica de $f$ en $[a,b]$ se obtiene integrando "
            "el factor de escala $\\sqrt{1+(f')^2}$:"
        )
        st.latex(r"L = \int_a^b \sqrt{1+\big(f'(x)\big)^2}\,dx")

    with st.expander("6.3 Integrales impropias", expanded=False):
        st.markdown(
            "Cuando un límite de integración es infinito (**1ª especie**) o la función no es "
            "acotada en el intervalo (**2ª especie**):"
        )
        st.latex(r"\int_a^\infty f =\lim_{T\to\infty}\int_a^T f,\qquad \int_a^b \frac{1}{(x-a)^p}\,dx\ \text{converge si } p<1")
        st.markdown(
            "Convergen cuando el **límite es finito**. Criterios: comparación, límite de "
            "comparación y $p$-integrales."
        )

    with st.expander("6.4 Aplicaciones físicas", expanded=False):
        st.markdown(
            r"""
            **Trabajo** de una fuerza variable $F(x)$ a lo largo de un desplazamiento:
            $W=\int_a^b F(x)\,dx$.
            **Presión** sobre una placa vertical:
            $F=\rho g \int (a-y)\,w(y)\,dy$.
            """
        )
        st.info(
            "Estos problemas se resuelven siempre *localizando* un elemento diferencial "
            "(rebanada) y luego integrando."
        )

    with st.expander("6.5 Demostración: volumen de un sólido de revolución", expanded=False):
        st.markdown(
            r"**Teorema.** Si $f\ge 0$ es continua en $[a,b]$, el volumen del sólido que se "
            r"obtiene al girar la región bajo $y=f(x)$ alrededor del eje $x$ es"
        )
        st.latex(r"V=\pi\int_a^b f(x)^2\,dx")
        st.markdown(
            r"**Demostración (por rebanadas).** Tomemos una partición "
            r"$a=x_0<x_1<\cdots<x_n=b$ de norma $\delta$. En cada subintervalo $I_i$, la "
            r"sección transversal del sólido es un disco de radio $f(x)$; los radios mínimo y "
            r"máximo de $f$ en $I_i$ son $m_i$ y $M_i$. El trozo de sólido queda así entre dos "
            r"cilindros:"
        )
        st.latex(r"\pi\,m_i^2\,\Delta x_i\ \le\ V_i\ \le\ \pi\,M_i^2\,\Delta x_i")
        st.markdown(
            r"Sumando sobre $i$, $V$ queda encuadrado entre las sumas de Darboux (inferior y "
            r"superior) de $\pi f(x)^2$:"
        )
        st.latex(r"\sum_i \pi\,m_i^2\,\Delta x_i\ \le\ V\ \le\ \sum_i \pi\,M_i^2\,\Delta x_i")
        st.markdown(
            r"Como $f^2$ es continua (y por lo tanto integrable), ambas sumas convergen a la "
            r"misma integral cuando $\delta\to0$; por el teorema del encaje,"
        )
        st.latex(r"V=\pi\int_a^b f(x)^2\,dx\qquad\blacksquare")
        st.info(
            r"El argumento es el del **principio de Cavalieri**: el volumen es la integral de "
            r"las áreas de las secciones transversales $A(x)=\pi f(x)^2$."
        )


# ===========================================================================
# INTUICIÓN VISUAL
# ===========================================================================

_FUNCIONES_REV = {
    "f(x)=√x en [0,4]": (sp.sqrt(x), 0, 4),
    "f(x)=x² en [0,1]": (x**2, 0, 1),
    "f(x)=sin(x) en [0,π]": (sp.sin(x), 0, sp.pi),
    "f(x)=x³ en [0,1]": (x**3, 0, 1),
}


def _revolucion_3d() -> None:
    st.markdown("##### Sólido de revolución alrededor del eje x")
    nom = st.selectbox("Función", list(_FUNCIONES_REV), key="u6_rev_f")
    f, a0, b0 = _FUNCIONES_REV[nom]

    V = sp.simplify(volumen_revolucion(f, a0, b0))
    fn = sp.lambdify(x, f, "numpy")
    t = np.linspace(float(a0), float(b0), 60)
    tt, ang = np.meshgrid(t, np.linspace(0, 2 * np.pi, 100))
    r = fn(tt)
    X = tt
    Y = r * np.cos(ang)
    Z = r * np.sin(ang)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis", opacity=0.85,
                                     showscale=False)])
    fig.update_layout(title=f"Sólido: V = π ∫ f² = {V} ≈ {float(V.evalf()):.3f}",
                      height=520, margin=dict(l=10, r=10, t=60, b=10),
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    st.plotly_chart(fig, width="stretch")


def _impropia_visual() -> None:
    st.markdown("##### Impropia de 1ª especie: área hasta el infinito")
    elegida = st.selectbox("Función (desde x = 1)", ["1/x²  (converge)", "1/x  (diverge)"],
                           key="u6_imp_f")
    f = 1 / x**2 if "1/x²" in elegida else 1 / x
    T = st.slider("Límite superior T (hacia ∞)", 1.5, 30.0, 5.0, 0.5, key="u6_imp_t")

    fn = sp.lambdify(x, f, "numpy")
    xs = np.linspace(1, T, 900)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=fn(xs), mode="lines", name=f"{elegida}",
                             line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=xs, y=fn(xs), fill="tozeroy", mode="lines",
                             fillcolor="rgba(255,127,14,0.4)", line=dict(width=0),
                             name=f"área hasta T"))
    fig.update_layout(title=f"Área [1,{T:.1f}] = {float(sp.integrate(f, (x, 1, T))):.3f}",
                      height=430, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig, width="stretch")

    estudio = integral_impropia(f, 1, sp.oo)
    if estudio["converge"]:
        st.success(f"Cuando T→∞ el área tiende a un valor finito: **{estudio['valor']}**.")
    else:
        st.error("Cuando T→∞ el área crece sin límite: la integral **diverge**.")
    st.caption("Interpretación: convergen = 'el área acumulada se estabiliza'.")


def _longitud_visual() -> None:
    st.markdown("##### Longitud de arco por segmentos")
    f = sp.sin(x)
    a, b = 0.0, 4.0
    fn = sp.lambdify(x, f, "numpy")
    N = st.slider("N segmentos de la poligonal", 2, 60, 8, 1, key="u6_lon_n")

    xs = np.linspace(0, 4, 600)
    pts = np.linspace(a, b, N + 1)
    long = float(np.sum(np.hypot(np.diff(pts), np.diff(fn(pts)))))
    exacta = float(longitud_arco(f, a, b))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=fn(xs), mode="lines", name="f(x)=sin(x)",
                             line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=pts, y=fn(pts), mode="lines+markers", name=f"poligonal ({N})",
                             line=dict(color="red", width=2, dash="dot")))
    fig.update_layout(title=f"Longitud poligonal = {long:.4f}  (exacta = {exacta:.4f})",
                      height=430, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig, width="stretch")
    st.markdown("Al aumentar $N$, la suma de segmentos se aproxima a $\\sqrt{1+(f')^2}$ integrado.")


def intuicion() -> None:
    st.markdown(
        "Visualizá el sólido de revolución, el área 'infinita' de las impropias y la "
        "aproximación poligonal de la longitud de arco."
    )
    t1, t2, t3 = st.tabs(["Revolución 3D", "Impropias", "Longitud de arco"])
    with t1:
        _revolucion_3d()
    with t2:
        _impropia_visual()
    with t3:
        _longitud_visual()


# ===========================================================================
# EJERCICIOS
# ===========================================================================

_DISCOS_BANCO = [
    ("y = √x, girada sobre x, [0,4]", sp.sqrt(x), 0, 4),
    ("y = x², girada sobre x, [0,1]", x**2, 0, 1),
    ("y = sin(x), girada sobre x, [0,π]", sp.sin(x), 0, sp.pi),
    ("y = 4x, girada sobre x, [0,1]", 4 * x, 0, 1),
    ("y = 3, girada sobre x, [0,2]", sp.Integer(3), 0, 2),
]

_ARANDELAS_BANCO = [
    ("entre y=√x (superior) e y=x², [0,1]", sp.sqrt(x), x**2, 0, 1),
    ("entre y=1 (superior) e y=x², [0,1]", sp.Integer(1), x**2, 0, 1),
    ("entre y=2 (superior) e y=x, [0,2]", sp.Integer(2), x, 0, 2),
]

_LONGITUD_BANCO = [
    ("f(x)=2x en [0,2]", 2 * x, 0, 2),
    ("f(x)=x^{3/2}·(2/3) en [0,1]", sp.Rational(2, 3) * x ** sp.Rational(3, 2), 0, 1),
    ("f(x)=3x+1 en [0,1]", 3 * x + 1, 0, 1),
    ("f(x)=x² en [0,1]", x**2, 0, 1),
]

_IMPROPIAS_BANCO = [
    ("∫₁^∞ 1/x² dx", 1 / x**2, 1, sp.oo),
    ("∫₀^∞ e^{-x} dx", sp.exp(-x), 0, sp.oo),
    ("∫₁^∞ 1/x dx", 1 / x, 1, sp.oo),
    ("∫₀¹ 1/√x dx  (2ª especie)", 1 / sp.sqrt(x), 0, 1),
    ("∫₁^∞ 1/x^{3/2} dx", 1 / x ** sp.Rational(3, 2), 1, sp.oo),
]

_FISICA_BANCO = [
    ("Una fuerza F(x)=x² N empuja un cuerpo desde x=1 hasta x=3 m. El trabajo W es:", 26 / 3, ["8 J", "26/3 J", "10 J", "9 J"]),
    ("Una fuerza F(x)=3 N constante actúa de x=0 a x=5 m. El trabajo W es:", 15, ["15 J", "5 J", "8 J", "3 J"]),
    ("F(x)=2x+1 N de x=0 a x=4 m. El trabajo W es:", 20, ["20 J", "9 J", "16 J", "24 J"]),
    ("La presión sobre una represa de ancho 2 y profundidad 3 (ρg=1, w(y)=2): F = ∫₀³ 2·(3−y) dy = ?", 9, ["9", "18", "27", "6"]),
]


def _reset(clave: str) -> None:
    for k in (f"w_{clave}", f"in_{clave}", f"mc_{clave}"):
        st.session_state.pop(k, None)


def _tab_discos() -> None:
    st.markdown("#### Volumen de revolución: método de discos")
    if "u6_d" not in st.session_state:
        st.session_state["u6_d"] = random.randint(0, len(_DISCOS_BANCO) - 1)
    nombre, f, a, b = _DISCOS_BANCO[st.session_state["u6_d"]]

    if st.button("🎲 Otro ejercicio (discos)", key="u6_d_nuevo"):
        st.session_state["u6_d"] = random.randint(0, len(_DISCOS_BANCO) - 1)
        _reset("u6_d_resp")
        st.rerun()

    st.write(f"Calcular el volumen del sólido generado por **{nombre}**.")
    st.latex(rf"V = \pi\int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} f(x)^2\,dx")
    correcto = sp.simplify(volumen_revolucion(f, a, b))
    ui.resolver_valor("u6_d_resp", correcto, placeholder="ej.: 8*pi",
                      tema="U6-volumen-discos", enunciado=nombre)


def _tab_arandelas() -> None:
    st.markdown("#### Volumen de revolución: método de arandelas")
    if "u6_a" not in st.session_state:
        st.session_state["u6_a"] = random.randint(0, len(_ARANDELAS_BANCO) - 1)
    nombre, f, g, a, b = _ARANDELAS_BANCO[st.session_state["u6_a"]]

    if st.button("🎲 Otro ejercicio (arandelas)", key="u6_a_nuevo"):
        st.session_state["u6_a"] = random.randint(0, len(_ARANDELAS_BANCO) - 1)
        _reset("u6_a_resp")
        st.rerun()

    st.write(f"Hallar el volumen girando la región **{nombre}** alrededor del eje x.")
    correcto = sp.simplify(sp.pi * sp.integrate(f**2 - g**2, (x, a, b)))
    ui.resolver_valor("u6_a_resp", correcto, placeholder="ej.: 3*pi/10",
                      tema="U6-volumen-arandelas", enunciado=nombre)


def _tab_longitud() -> None:
    st.markdown("#### Longitud de arco")
    if "u6_l" not in st.session_state:
        st.session_state["u6_l"] = random.randint(0, len(_LONGITUD_BANCO) - 1)
    nombre, f, a, b = _LONGITUD_BANCO[st.session_state["u6_l"]]

    if st.button("🎲 Otro ejercicio (longitud)", key="u6_l_nuevo"):
        st.session_state["u6_l"] = random.randint(0, len(_LONGITUD_BANCO) - 1)
        _reset("u6_l_resp")
        st.rerun()

    st.write(f"Calcular la longitud de arco de **{nombre}**.")
    st.latex(rf"L = \int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} \sqrt{{1+(f')^2}}\,dx")
    correcto = sp.nsimplify(longitud_arco(f, a, b))
    ui.resolver_valor("u6_l_resp", correcto, placeholder="ej.: sqrt(20)",
                      tema="U6-longitud-arco", enunciado=nombre)


def _tab_impropias() -> None:
    st.markdown("#### Integrales impropias (¿convergen? ¿cuánto?)")
    if "u6_i" not in st.session_state:
        st.session_state["u6_i"] = random.randint(0, len(_IMPROPIAS_BANCO) - 1)
    nombre, f, a, tope = _IMPROPIAS_BANCO[st.session_state["u6_i"]]

    if st.button("🎲 Otro ejercicio (impropias)", key="u6_i_nuevo"):
        st.session_state["u6_i"] = random.randint(0, len(_IMPROPIAS_BANCO) - 1)
        _reset("u6_i_resp")
        st.rerun()

    st.markdown(f"**{nombre}**" + "  (escribí `oo` si diverge a +∞, `-oo` si diverge a −∞)")
    correcto = sp.simplify(sp.integrate(f, (x, a, tope)))
    ui.resolver_valor("u6_i_resp", correcto, placeholder="ej.: 1  o  oo",
                      tema="U6-integral-impropia", enunciado=nombre)


def _tab_fisica() -> None:
    st.markdown("#### Aplicaciones físicas: trabajo y presión")
    if "u6_f" not in st.session_state:
        st.session_state["u6_f"] = random.randint(0, len(_FISICA_BANCO) - 1)
    enunciado, correcto, distractores = _FISICA_BANCO[st.session_state["u6_f"]]

    if st.button("🎲 Otra aplicación física", key="u6_f_nuevo"):
        st.session_state["u6_f"] = random.randint(0, len(_FISICA_BANCO) - 1)
        _reset("u6_f_resp")
        st.rerun()

    st.markdown(enunciado)
    opciones = [str(correcto)] + list(distractores)
    random.Random(st.session_state["u6_f"]).shuffle(opciones)
    ui.elegir_opcion("u6_f_resp", "El valor correcto es:",
                     [f"$ {o} $" for o in opciones], opciones.index(str(correcto)),
                     explicacion=f"Integrá la fuerza entre los límites: {correcto}.",
                     tema="U6-aplicaciones-fisicas", enunciado=enunciado)


def ejercicios(repo=None) -> None:
    st.markdown("Volúmenes, longitudes, impropias y aplicaciones físicas.")
    t1, t2, t3, t4, t5 = st.tabs(
        ["Volumen (discos)", "Volumen (arandelas)", "Longitud de arco", "Impropias", "Física"]
    )
    with t1:
        _tab_discos()
    with t2:
        _tab_arandelas()
    with t3:
        _tab_longitud()
    with t4:
        _tab_impropias()
    with t5:
        _tab_fisica()