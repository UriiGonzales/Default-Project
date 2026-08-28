"""Unidad 4 · Aplicaciones del Cálculo Diferencial (Programa 2026: UN 4).

4.1 Teoremas de Rolle y del valor medio. Monotonía, extremos y concavidad.
4.2 Regla de L'Hôpital.
4.3 Estudio completo de funciones y problemas de optimización.
4.4 Polinomios de Taylor.
"""

from __future__ import annotations

import random

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from ..graficos import figura_funciones
from ..matematicas import x, taylor_de
from .. import ui


# ===========================================================================
# TEORÍA
# ===========================================================================

def teoria() -> None:
    st.markdown(
        "Se usa la derivada para **analizar funciones**: crecimiento, extremos, curvatura, "
        "cálculo de límites (L'Hôpital) y aproximaciones (Taylor)."
    )

    with st.expander("4.1 Teoremas de Rolle, Lagrange y Cauchy (con demostración)", expanded=True):
        st.markdown(
            r"**Lema de Fermat.** Si $f$ tiene un extremo local en un punto interior $c$ y "
            r"existe $f'(c)$, entonces $f'(c)=0$."
        )
        st.markdown(
            r"**Demostración.** Si $c$ es un máximo local, $f(c+h)\le f(c)$ para $|h|$ "
            r"pequeño. Entonces"
        )
        st.latex(
            r"h>0:\ \frac{f(c+h)-f(c)}{h}\le 0 \;\Rightarrow\; f'(c)\le 0;"
            r"\qquad h<0:\ \frac{f(c+h)-f(c)}{h}\ge 0 \;\Rightarrow\; f'(c)\ge 0"
        )
        st.markdown(
            r"Luego $f'(c)=0$. El caso de mínimo es análogo. $\blacksquare$"
        )

        st.markdown(
            r"**Teorema de Rolle.** Si $f$ es continua en $[a,b]$, derivable en $(a,b)$ y "
            r"$f(a)=f(b)$, existe $c\in(a,b)$ con $f'(c)=0$."
        )
        st.markdown(
            r"**Demostración.** Por Weierstrass, $f$ alcanza máximo $M$ y mínimo $m$ en $[a,b]$."
        )
        st.markdown(
            r"- Si $M=m$, $f$ es constante y $f'(c)=0$ para todo $c$. "
            r"- Si $M>m$, como $f(a)=f(b)$, al menos uno de los dos extremos se alcanza en un "
            r"punto interior $c\in(a,b)$ (si no fuera así, $M$ y $m$ se alcanzarían en $a$ y "
            r"en $b$, contradictoriamente con $M>m$ y $f(a)=f(b)$). "
            r"Por el lema de Fermat, $f'(c)=0$. $\blacksquare$"
        )

        st.markdown(
            r"**Teorema de Lagrange (del valor medio).** $f$ continua en $[a,b]$ y derivable en "
            r"$(a,b)$ $\Rightarrow$ existe $c\in(a,b)$ con"
        )
        st.latex(r"f'(c)=\frac{f(b)-f(a)}{b-a}")
        st.markdown(
            r"**Demostración.** Se define la función auxiliar (la 'distancia' de $f$ a la recta "
            r"que une los extremos):"
        )
        st.latex(r"h(x)=f(x)-f(a)-\frac{f(b)-f(a)}{b-a}\,(x-a)")
        st.markdown(
            r"que es continua en $[a,b]$, derivable en $(a,b)$ y cumple $h(a)=h(b)=0$. Por "
            r"Rolle existe $c\in(a,b)$ con $h'(c)=0$, y como "
            r"$h'(x)=f'(x)-\dfrac{f(b)-f(a)}{b-a}$, se despeja la fórmula. $\blacksquare$"
        )
        st.success(
            "**Consecuencia clave:** si $f'(x)=0$ en todo un intervalo, $f$ es constante ahí. "
            "Es el puente hacia las primitivas (Unidad 5)."
        )

        st.markdown(
            r"**Teorema de Cauchy.** $f,g$ continuas en $[a,b]$ y derivables en $(a,b)$ con "
            r"$g'(x)\neq 0$ $\Rightarrow$ existe $c\in(a,b)$ con"
        )
        st.latex(r"\frac{f(b)-f(a)}{g(b)-g(a)}=\frac{f'(c)}{g'(c)}")
        st.markdown(
            r"**Demostración.** Notemos que $g(b)\neq g(a)$: si fueran iguales, Rolle daría un "
            r"punto con $g'=0$, prohibido. Definimos"
        )
        st.latex(r"H(x)=f(x)-f(a)-\frac{f(b)-f(a)}{g(b)-g(a)}\,\big(g(x)-g(a)\big)")
        st.markdown(
            r"Entonces $H(a)=H(b)=f(a)$ y $H$ es continua y derivable; por Rolle existe $c$ "
            r"con $H'(c)=0$, es decir "
            r"$f'(c)-\dfrac{f(b)-f(a)}{g(b)-g(a)}\,g'(c)=0$, de donde la fórmula. "
            r"$\blacksquare$"
        )

    with st.expander("4.2 Monotonía, extremos y concavidad", expanded=False):
        st.markdown("**Criterio de la primera derivada:**")
        st.latex(
            r"f'>0 \Rightarrow \text{creciente},\qquad f'<0 \Rightarrow \text{decreciente}"
        )
        st.markdown(
            r"""
            Si $f'$ cambia de signo en un punto crítico ($f'(c)=0$ o no existe), hay un
            **extremo local**. La concavidad la da $f''$:
            """
        )
        st.latex(r"f''>0\Rightarrow\text{cóncava hacia arriba},\qquad f''<0\Rightarrow\text{cóncava hacia abajo}")
        st.markdown("**Criterio de la segunda derivada.** Si $f'(c)=0$:")
        st.latex(r"f''(c)>0 \Rightarrow\text{mínimo local},\qquad f''(c)<0\Rightarrow\text{máximo local}")
        st.markdown(
            "Los **puntos de inflexión** ($f''$ cambia de signo) separan un tipo de curvatura "
            "del otro."
        )

    with st.expander("4.3 Regla de L'Hôpital (con demostración)", expanded=False):
        st.markdown(
            r"**Regla (caso $0/0$).** Sean $f,g$ derivables en un entorno de $a$ (salvo quizá "
            r"en $a$), con $f(a)=g(a)=0$, $g'(x)\neq0$ y tal que existe "
            r"$\lim_{x\to a}\frac{f'(x)}{g'(x)}=L$. Entonces"
        )
        st.latex(r"\lim_{x\to a}\frac{f(x)}{g(x)}=\lim_{x\to a}\frac{f'(x)}{g'(x)}=L")
        st.markdown(r"**Demostración.** Extendiendo $f(a)=g(a)=0$, ambas funciones son continuas en $a$; para $x$ cercano a $a$ el teorema de Cauchy en $[a,x]$ da")
        st.latex(r"\frac{f(x)}{g(x)}=\frac{f(x)-f(a)}{g(x)-g(a)}=\frac{f'(c_x)}{g'(c_x)},\qquad c_x\in(a,x)")
        st.markdown(
            r"Cuando $x\to a$ se tiene $c_x\to a$ y el cociente tiende a $L$. La misma idea "
            r"cubre $\infty/\infty$ y los límites en el infinito (con la sustitución "
            r"$t=1/x$). $\blacksquare$"
        )
        st.warning(
            "Solo aplica cuando hay indeterminación del cociente; si sigue indeterminado, se "
            "puede repetir. NO es la derivada de un cociente."
        )

    with st.expander("4.4 Polinomios de Taylor y optimización", expanded=False):
        st.markdown(
            "El polinomio de **Taylor de orden $n$** alrededor de $a$ aproxima a $f$ "
            "con error de orden $(x-a)^{n+1}$:"
        )
        st.latex(r"P_n(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}{2!}(x-a)^2+\cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n")
        st.markdown(
            "Es la base de las **linealizaciones** ($n=1$), de los desarrollos de potencias para "
            "funciones trascendentes y del estudio de extremos vía signo."
        )


# ===========================================================================
# INTUICIÓN VISUAL
# ===========================================================================

_FUNCS = {
    "x² − 4x + 3": x**2 - 4 * x + 3,
    "x³ − 3x": x**3 - 3 * x,
    "x⁴ − 8x²": x**4 - 8 * x**2,
    "−x³ + 6x² − 9x + 5": -x**3 + 6 * x**2 - 9 * x + 5,
}


def _sombreado_signo(fig: go.Figure, g: sp.Expr, lo: float, hi: float, color: str) -> None:
    """Sombrea en donde g(x)>0 dentro de [lo,hi] con vspan."""
    fg = sp.lambdify(x, g, "numpy")
    punto = np.linspace(lo, hi, 900)
    with np.errstate(all="ignore"):
        vals = fg(punto)
    vals = np.asarray(vals, dtype=float)
    if vals.ndim == 0:
        vals = np.full_like(punto, float(vals))
    regiones = []
    activa = vals[0] > 0
    inicio = lo
    for i in range(1, len(punto)):
        ahora = vals[i] > 0
        if ahora != activa:
            if activa:
                regiones.append((inicio, punto[i]))
            inicio = punto[i]
            activa = ahora
    if activa:
        regiones.append((inicio, hi))
    for a0, a1 in regiones:
        fig.add_vrect(x0=a0, x1=a1, fillcolor=color, opacity=0.18, line_width=0)


def _monotonia_visual() -> None:
    st.markdown("##### Monotonía: f, f' y extremos")
    nom = st.selectbox("Función", list(_FUNCS), key="u4_mon_f")
    f = _FUNCS[nom]
    lo, hi = -4.0, 4.0
    fp = sp.diff(f, x)
    fig = figura_funciones([(nom, f)], lo, hi, titulo="Zona sombreada: f creciente (f'>0)")
    _sombreado_signo(fig, fp, lo, hi, "green")
    criticos = [float(c) for c in sp.solve(sp.Eq(fp, 0), x) if c.is_real]
    for c in criticos:
        fpp = sp.diff(f, x, 2).subs(x, c)
        tipo = "mín" if float(sp.re(fpp)) > 0 else "máx"
        fig.add_trace(go.Scatter(x=[c], y=[float(f.subs(x, c))], mode="markers", name=f"{tipo}. local",
                                 marker=dict(size=9, color="red" if tipo == "máx" else "orange")))
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Puntos críticos:** " + (", ".join(f"${c:.2f}$" for c in criticos) if criticos else "ninguno"))


def _concavidad_visual() -> None:
    st.markdown("##### Concavidad y puntos de inflexión")
    nom = st.selectbox("Función", list(_FUNCS), key="u4_cc_f")
    f = _FUNCS[nom]
    lo, hi = -4.0, 4.0
    fpp = sp.diff(f, x, 2)
    fig = figura_funciones([(nom, f)], lo, hi, titulo="Sombreo: cóncava hacia arriba (f''>0)")
    _sombreado_signo(fig, fpp, lo, hi, "purple")
    pi = [float(c) for c in sp.solve(sp.Eq(fpp, 0), x) if c.is_real]
    for c in pi:
        fig.add_trace(go.Scatter(x=[c], y=[float(f.subs(x, c))], mode="markers", name="P.I.",
                                 marker=dict(size=9, color="black", symbol="x")))
    st.plotly_chart(fig, width="stretch")
    if pi:
        st.markdown("**Posibles puntos de inflexión:** " + ", ".join(f"${c:.2f}$" for c in pi))


def _optimizacion_visual() -> None:
    st.markdown("##### Optimización geométrica: rectángulo bajo la parábola")
    f = -x**2 + 4 * x
    w = st.slider("Base w del rectángulo (0 ≤ w ≤ 4)", 0.5, 4.0, 2.0, 0.05, key="u4_opt_w")
    area = w * (-w**2 + 4 * w)
    Ws = sp.Symbol("Ws")
    w_opt = max((float(c) for c in sp.solve(sp.diff(Ws * (-Ws**2 + 4 * Ws), Ws), Ws)
                 if 0 < float(c) < 4), default=8 / 3)
    A_opt = float(w_opt * (-w_opt**2 + 4 * w_opt))

    fig = go.Figure()
    xs = np.linspace(0, 4, 500)
    fig.add_trace(go.Scatter(x=xs, y=[-v * v + 4 * v for v in xs], mode="lines",
                             name="f(x) = −x² + 4x", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=[0, w, w, 0, 0], y=[0, 0, -w*w + 4*w, -w*w + 4*w, 0], fill="toself",
                             fillcolor="orange", opacity=0.5, name="rectángulo",
                             line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=[w_opt], y=[A_opt], mode="markers", name="óptimo",
                             marker=dict(size=10, color="green")))
    fig.update_layout(title=f"Área = w·f(w) = {area:.3f}   (máximo {A_opt:.3f} en w = {w_opt:.3f})",
                      height=420, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig, width="stretch")
    if abs(w - w_opt) < 0.15:
        st.success("Estás en el óptimo: el área se maximiza alrededor de w = 8/3.")


def intuicion() -> None:
    st.markdown(
        "Visualizá crecimiento, concavidad y un problema clásico de optimización: todo se "
        "decide con los signos de $f'$ y $f''$."
    )
    t1, t2, t3 = st.tabs(["Monotonía y extremos", "Concavidad y P.I.", "Optimización"])
    with t1:
        _monotonia_visual()
    with t2:
        _concavidad_visual()
    with t3:
        _optimizacion_visual()


# ===========================================================================
# EJERCICIOS
# ===========================================================================

_FUNCIONES_CRITICAS = [
    (r"f(x)=x^2-4x+3", x**2 - 4 * x + 3),
    (r"f(x)=x^3-3x", x**3 - 3 * x),
    (r"f(x)=x^4-8x^2", x**4 - 8 * x**2),
    (r"f(x)=-x^3+6x^2-9x+5", -x**3 + 6 * x**2 - 9 * x + 5),
    (r"f(x)=x^3-3x^2", x**3 - 3 * x**2),
]


def _intervalos_signo(fp: sp.Expr) -> tuple[list[str], list[str]]:
    """Intervalos (latex) donde fp>0 (crec) y fp<0 (decr)."""
    raices = [float(c) for c in sp.solve(sp.Eq(fp, 0), x) if c.is_real]
    raices = list(dict.fromkeys(sorted(raices)))
    if not raices:
        f = sp.lambdify(x, fp, "numpy")
        if f(0.0) > 0:
            return [r"(-\infty,\infty)"], []
        return [], [r"(-\infty,\infty)"]

    bordes = [float("-inf")] + raices + [float("inf")]
    crec, decr = [], []
    f = sp.lambdify(x, fp, "numpy")
    for i in range(len(bordes) - 1):
        a0, a1 = bordes[i], bordes[i + 1]
        medio = a0 + 1 if a0 != float("-inf") and abs(a0) < 1e9 else a0 + 0.5
        signo = f(medio) > 0
        ia = r"(-\infty," if a0 == float("-inf") else f"({a0:g}"
        ib = r"\infty)" if a1 == float("inf") else f"{a1:g})"
        etiqueta = ia + "," + ib
        (crec if signo else decr).append(etiqueta)
    return crec, decr


def _tab_monotonia() -> None:
    st.markdown("#### Intervalos de crecimiento y decrecimiento")
    if "u4_mon" not in st.session_state:
        st.session_state["u4_mon"] = random.randint(0, len(_FUNCIONES_CRITICAS) - 1)
    i = st.session_state["u4_mon"]
    nombre, expr_f = _FUNCIONES_CRITICAS[i]

    if st.button("🎲 Otro ejercicio (monotonía)", key="u4_mon_nuevo"):
        st.session_state["u4_mon"] = random.randint(0, len(_FUNCIONES_CRITICAS) - 1)
        st.session_state.pop("w_u4_mon_resp", None)
        st.session_state.pop("mc_u4_mon_resp", None)
        if not hasattr(st.session_state, "_seed_u4"):
            st.session_state["_seed_u4"] = i
        st.rerun()

    fp = sp.diff(expr_f, x)
    crec, decr = _intervalos_signo(fp)
    texto_crec = r" \cup ".join(crec) if crec else r"\varnothing"
    texto_decr = r" \cup ".join(decr) if decr else r"\varnothing"

    discriminantes = [texto_crec, texto_decr]
    raices = sorted({float(c) for c in sp.solve(sp.Eq(fp, 0), x) if c.is_real})
    if raices:
        discriminantes.append(rf"({raices[0]:g},\infty)")
        discriminantes.append(rf"(-\infty,{raices[-1]:g})")
    discriminantes.append(r"\varnothing")
    vistos = []
    for s in discriminantes:
        if s not in vistos:
            vistos.append(s)
    opciones = vistos
    while len(opciones) < 4:
        opciones.append(r"(0,1)")
    random.Random(i).shuffle(opciones)
    idx = opciones.index(texto_crec)
    st.latex("¿Dónde es CRECIENTE f?    " + nombre)
    ui.elegir_opcion("u4_mon_resp", "Conjunto de crecimiento:",
                     [f"$ {s} $" for s in opciones], idx,
                     explicacion=f"Resolvés f'(x)>0 → crece en {texto_crec}.",
                     tema="U4-monotonia", enunciado=nombre)


def _tab_extremos() -> None:
    st.markdown("#### Clasificación de extremos locales")
    if "u4_ext" not in st.session_state:
        st.session_state["u4_ext"] = random.randint(0, len(_FUNCIONES_CRITICAS) - 1)
    nombre, expr_f = _FUNCIONES_CRITICAS[st.session_state["u4_ext"]]

    if st.button("🎲 Otro ejercicio (extremos)", key="u4_ext_nuevo"):
        st.session_state["u4_ext"] = random.randint(0, len(_FUNCIONES_CRITICAS) - 1)
        st.session_state.pop("w_u4_ext_resp", None)
        st.session_state.pop("mc_u4_ext_resp", None)
        st.rerun()

    fp = sp.diff(expr_f, x)
    criticos = [c for c in sp.solve(sp.Eq(fp, 0), x) if c.is_real]
    if not criticos:
        st.info("Esta función no tiene puntos críticos reales.")
        return
    c = random.Random(st.session_state["u4_ext"]).choice(criticos)
    fpp = sp.diff(expr_f, x, 2).subs(x, c)
    fpp_val = float(sp.re(fpp)) if fpp.is_real or sp.im(fpp) == 0 else 0.0
    if abs(fpp_val) < 1e-9:
        idx = 2
    elif fpp_val < 0:
        idx = 0
    else:
        idx = 1
    st.latex(rf"f({sp.latex(c)}) \text{{ con }}  f''({sp.latex(c)}) = {sp.latex(sp.simplify(fpp))}")
    ui.elegir_opcion("u4_ext_resp",
                     f"En $x={sp.latex(c)}$ la función presenta:",
                     ["un MÁXIMO local", "un MÍNIMO local", "un punto de inflexión horizontal"],
                     idx,
                     explicacion="Usá el criterio de la segunda derivada.",
                     tema="U4-extremos", enunciado=f"{nombre} en x={c}")


_TVM_BANCO = [
    ("La función f(x)=x² en [1,3] cumple el TVM y el punto c = 2 satisface f'(c) = (f(3)−f(1))/2.", True),
    ("f(x)=x³−4x en [−2,2] cumple el Teorema de Rolle (f(−2)=f(2)).", True),
    ("f(x)=|x| en [−1,1] NO permite aplicar Rolle porque no es derivable en 0.", True),
    ("Si f'(x)=0 en todo (a,b) y f es continua en [a,b], entonces f es constante.", True),
    ("f(x)=1/x en [−1,1] cumple el TVM porque es continua en todo [−1,1].", False),
    ("Rolle requiere que f sea derivable en todo el intervalo cerrado [a,b].", False),
]


def _tab_tvm() -> None:
    st.markdown("#### Teoremas de Rolle y del valor medio (V/F)")
    if "u4_tvm" not in st.session_state:
        st.session_state["u4_tvm"] = random.randint(0, len(_TVM_BANCO) - 1)
    enunciado, es_verdadero = _TVM_BANCO[st.session_state["u4_tvm"]]

    if st.button("🎲 Otro enunciado (Rolle/TVM)", key="u4_tvm_nuevo"):
        st.session_state["u4_tvm"] = random.randint(0, len(_TVM_BANCO) - 1)
        st.session_state.pop("w_u4_tvm_resp", None)
        st.session_state.pop("mc_u4_tvm_resp", None)
        st.rerun()

    st.markdown(enunciado)
    ui.elegir_opcion("u4_tvm_resp", "La afirmación es:",
                     ["Verdadera", "Falsa"],
                     0 if es_verdadero else 1,
                     tema="U4-tvm", enunciado=enunciado)


_LHOSPITAL_BANCO = [
    (r"\frac{\sin 2x}{3x}", sp.sin(2 * x) / (3 * x), 0),
    (r"\frac{1-\cos x}{x^2}", (1 - sp.cos(x)) / x**2, 0),
    (r"\frac{e^x-1-x}{x^2}", (sp.exp(x) - 1 - x) / x**2, 0),
    (r"\frac{\ln x}{x-1}", sp.ln(x) / (x - 1), 1),
    (r"\frac{x^2}{e^x}", x**2 / sp.exp(x), sp.oo),
    (r"\frac{\tan x - \sin x}{x^3}", (sp.tan(x) - sp.sin(x)) / x**3, 0),
]


def _tab_lhospital() -> None:
    st.markdown("#### Límites con la regla de L'Hôpital")
    if "u4_lh" not in st.session_state:
        st.session_state["u4_lh"] = random.randint(0, len(_LHOSPITAL_BANCO) - 1)
    fr, expr, x0 = _LHOSPITAL_BANCO[st.session_state["u4_lh"]]

    if st.button("🎲 Otro ejercicio (L'Hôpital)", key="u4_lh_nuevo"):
        st.session_state["u4_lh"] = random.randint(0, len(_LHOSPITAL_BANCO) - 1)
        st.session_state.pop("w_u4_lh_resp", None)
        st.session_state.pop("in_u4_lh_resp", None)
        st.rerun()

    cond = rf"x\to {sp.latex(x0)}"
    st.latex(rf"\lim_{{{cond}}} {fr}")
    correcto = sp.limit(expr, x, x0)
    ui.resolver_valor("u4_lh_resp", correcto,
                      tema="U4-lhopital", enunciado=rf"\lim {cond} {fr}")


_OPTIMIZACION_BANCO = [
    ("Con 100 m de alambre se quiere cercar un rectángulo de área máxima. ¿Cuánto mide cada lado (x) del rectángulo óptimo?", 25),
    ("Dos números positivos suman 12 y su producto debe ser máximo. ¿Qué valor único deben tener (x = y)?", 6),
    ("Dos números positivos tienen producto 100 y su suma debe ser mínima. ¿Cuánto vale cada uno (x = y)?", 10),
    ("La función de ingresos es R(x)=40x−x². ¿Para qué nivel x el ingreso es máximo?", 20),
]


def _tab_optimizacion() -> None:
    st.markdown("#### Problemas de optimización")
    if "u4_opt" not in st.session_state:
        st.session_state["u4_opt"] = random.randint(0, len(_OPTIMIZACION_BANCO) - 1)
    enunciado, correcto = _OPTIMIZACION_BANCO[st.session_state["u4_opt"]]

    if st.button("🎲 Otro problema (optimización)", key="u4_opt_nuevo"):
        st.session_state["u4_opt"] = random.randint(0, len(_OPTIMIZACION_BANCO) - 1)
        st.session_state.pop("w_u4_opt_resp", None)
        st.session_state.pop("in_u4_opt_resp", None)
        st.rerun()

    st.markdown(enunciado)
    ui.resolver_valor("u4_opt_resp", correcto,
                      tema="U4-optimizacion", enunciado=enunciado)


_TAYLOR_BANCO = [
    ("e^x", sp.exp(x), 0, 3),
    ("sin(x)", sp.sin(x), 0, 5),
    ("ln(1+x)", sp.ln(1 + x), 0, 4),
    ("cos(x)", sp.cos(x), 0, 6),
    ("sqrt(1+x)", sp.sqrt(1 + x), 0, 3),
]


def _tab_taylor() -> None:
    st.markdown("#### Polinomio de Taylor en x = 0")
    if "u4_tay" not in st.session_state:
        st.session_state["u4_tay"] = random.randint(0, len(_TAYLOR_BANCO) - 1)
    nom, expr, x0, orden = _TAYLOR_BANCO[st.session_state["u4_tay"]]

    if st.button("🎲 Otro ejercicio (Taylor)", key="u4_tay_nuevo"):
        st.session_state["u4_tay"] = random.randint(0, len(_TAYLOR_BANCO) - 1)
        st.session_state.pop("w_u4_tay_resp", None)
        st.session_state.pop("in_u4_tay_resp", None)
        st.rerun()

    poly = sp.simplify(taylor_de(expr, x0, orden))
    st.latex(rf"P_{{\text{{{orden}}}}}(x) \text{{ de }} {nom} \text{{ en }} a={x0}:")
    ui.resolver_expresion(
        "u4_tay_resp", poly,
        placeholder="ej.: 1 + x + x**2/2 + x**3/6",
        ayuda="El polinomio de Taylor de orden n se escribe expandido; cualquier forma equivalente vale.",
        tema="U4-taylor", enunciado=f"{nom} orden {orden}",
    )


def ejercicios(repo=None) -> None:
    st.markdown(
        "Aplicaciones: monotonía, extremos, Rolle/TVM, L'Hôpital, optimización y Taylor."
    )
    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Monotonía", "Extremos", "Rolle y TVM", "L'Hôpital", "Optimización", "Taylor"]
    )
    with t1:
        _tab_monotonia()
    with t2:
        _tab_extremos()
    with t3:
        _tab_tvm()
    with t4:
        _tab_lhospital()
    with t5:
        _tab_optimizacion()
    with t6:
        _tab_taylor()