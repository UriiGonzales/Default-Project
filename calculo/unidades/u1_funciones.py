"""Unidad 1 · Funciones (Programa 2026: UN 1).

1.1 Números reales, inecuaciones y valor absoluto.
1.2 Funciones: dominio, imagen, composición, inversa, paridad, monoticía.
1.3 Funciones algebraicas y trascendentes; especiales (signo, parte entera, mantisa).
"""

from __future__ import annotations

import random

import numpy as np
import streamlit as st
import sympy as sp

from ..graficos import figura_funciones
from ..matematicas import x
from .. import ui


# ===========================================================================
# TEORÍA
# ===========================================================================

def teoria() -> None:
    st.markdown(
        "En esta unidad se consolidan los **números reales** como herramienta y se estudia "
        "el objeto central de todo el cálculo: la **función de variable real**."
    )

    with st.expander("1.1 Números reales, inecuaciones y valor absoluto", expanded=True):
        st.markdown(
            r"""
            Los números reales $\mathbb{R}$ se organizan mediante **inecuaciones**,
            que se resuelven con análisis de signos:

            - **Cuadráticas** ($ax^2+bx+c \ge 0$): se factorizan (o se usa la fórmula de
              Baskara), se marcan las raíces en la recta y se estudia el signo de cada
              factor en los intervalos resultantes.
            - **Fraccionarias** ($\frac{P(x)}{Q(x)}<0$): se analiza el signo del numerador y
              del denominador por separado (nunca se "multiplica cruzado" sin considerar el signo).
            - **Modulares**: $|x-a|<d$ representa el intervalo $a-d < x < a+d$
              ("dentro de la distancia $d$ a $a$"), mientras que
              $|x-a|>d$ resulta la unión de dos semirrectas.
            """
        )
        st.latex(r"|x| = \begin{cases} x & x \ge 0 \\ -x & x < 0 \end{cases}")
        st.latex(r"|a\cdot b| = |a|\,|b|,\qquad \left|\tfrac{a}{b}\right|=\tfrac{|a|}{|b|},\qquad |a+b|\le |a|+|b|")
        st.markdown(
            "Esto permite hallar **entornos** (intervalos simétricos) y **entornos reducidos** "
            "(sin el punto central), esenciales para definir límites en la Unidad 2."
        )

    with st.expander("1.2 Funciones: definición, dominio, operaciones y características", expanded=False):
        st.markdown(
            r"""
            Una **función** $f: A \to B$ asigna a cada $x$ de su dominio exactamente un valor
            $f(x)$.

            - **Dominio** $\mathcal{D}$: conjunto de $x$ para los que la regla tiene sentido.
              Para denominador $\ne 0$, radicando $\ge 0$ (índice par), argumento de
              logaritmos $>0$, etc.
            - **Imagen** $\mathcal{I}$: conjunto de valores que efectivamente toma $f$.
            - **Operaciones**: suma, producto y cociente (salvo ceros del denominador);
              **composición** $(f\circ g)(x)=f(g(x))$.
            - **Inversa**: $f^{-1}$ existe si $f$ es biyectiva; sus gráficas son simétricas
              respecto de $y=x$.
            """
        )
        st.latex(r"f\ \text{par} \iff f(-x)=f(x), \qquad f\ \text{impar} \iff f(-x)=-f(x)")
        st.latex(r"\text{monótona creciente} \iff x_1<x_2 \Rightarrow f(x_1)\le f(x_2)")
        st.markdown(
            "Las **traslaciones** desplazan la gráfica ($f(x)+k$ vertical, $f(x+c)$ horizontal "
            "en sentido opuesto) y las multiplicaciones por $-1$ la **reflejan** sobre los ejes."
        )

    with st.expander("1.3 Funciones elementales y especiales", expanded=False):
        st.markdown(
            r"""
            **Algebraicas:** potenciales $x^n$, racionales $P/Q$, irracionales $\sqrt[n]{x}$.

            **Trascendentes:** exponencial $e^x$ (o $a^x$), logarítmica $\log_a x$,
            trigonométricas ($\sin$, $\cos$, $\tan$ y sus inversas) e **hiperbólicas**:
            """
        )
        st.latex(r"\sinh x=\frac{e^x-e^{-x}}{2},\quad \cosh x=\frac{e^x+e^{-x}}{2},\quad \tanh x=\frac{\sinh x}{\cosh x}")
        st.markdown(
            r"""
            **Especiales** (construidas por tramos):
            """
        )
        st.latex(r"\operatorname{sgn}(x)=\begin{cases}1& x>0\\0& x=0\\-1& x<0\end{cases},\qquad |x|=\begin{cases}x&x\ge0\\-x&x<0\end{cases}")
        st.latex(r"\lfloor x \rfloor = \max\{k\in\mathbb{Z}: k\le x\}\quad(\text{parte entera}),\qquad \{x\}=x-\lfloor x\rfloor\quad(\text{mantisa})")
        st.info(
            "La **parte entera** y la **mantisa** tienen discontinuidades de tipo salto en cada "
            "entero; el **signo** salta en $x=0$. Es el primer contacto con los tipos de "
            "discontinuidad que se clasificarán en la Unidad 2."
        )

    with st.expander("Funciones hiperbólicas: identidades útiles", expanded=False):
        st.latex(r"\cosh^2 x - \sinh^2 x = 1,\qquad 1-\tanh^2 x = \operatorname{sech}^2 x")
        st.latex(r"\sinh(a+b)=\sinh a\cosh b+\cosh a\sinh b")
        st.markdown(
            r"Su derivada ($\sinh' = \cosh$, $\cosh' = \sinh$, $\tanh' = \operatorname{sech}^2$) "
            "se usará en la Unidad 3."
        )


# ===========================================================================
# INTUICIÓN VISUAL
# ===========================================================================

_FUNCIONES_BASE = {
    "x²": x**2,
    "√x": sp.sqrt(x),
    "1/x": 1 / x,
    "e^x": sp.exp(x),
    "ln(x)": sp.ln(x),
    "sin(x)": sp.sin(x),
    "cos(x)": sp.cos(x),
    "tan(x)": sp.tan(x),
    "|x|": sp.Abs(x),
    "x³ − 3x": x**3 - 3 * x,
}


def _explorador_transformaciones() -> None:
    st.markdown(r"##### Transformaciones: $y = a\,f(b\,(x+c)) + d$")
    base = st.selectbox("Función base f(x)", list(_FUNCIONES_BASE), key="u1_base")
    c1, c2, c3, c4 = st.columns(4)
    a = c1.slider("a (vertical)", -3.0, 3.0, 1.0, 0.1, key="u1_a")
    b = c2.slider("b (horizontal)", -3.0, 3.0, 1.0, 0.1, key="u1_b")
    c = c3.slider("c (desplaz. horiz.)", -4.0, 4.0, 0.0, 0.1, key="u1_c")
    d = c4.slider("d (vertical)", -4.0, 4.0, 0.0, 0.1, key="u1_d")

    f = _FUNCIONES_BASE[base]
    if b == 0:
        st.warning("El parámetro b no puede ser 0.")
        return
    transformada = a * f.subs(x, b * (x + c)) + d
    fig = figura_funciones(
        [("f(x) original", f), ("transformada", transformada)],
        -6, 6, titulo="Efecto de los parámetros",
    )
    st.plotly_chart(fig, width="stretch")
    signos = []
    if a != 1:
        signos.append("reflexión vertical " + ("(estira)" if abs(a) > 1 else "(comprime)"))
    if b < 0:
        signos.append("reflexión horizontal")
    if c:
        signos.append(f"corrimiento horizontal {'izquierda' if c > 0 else 'derecha'} (c={-c:.1f})")
    if d:
        signos.append(f"corrimiento vertical {'arriba' if d > 0 else 'abajo'} ({d:.1f})")
    st.markdown("**Efectos activos:** " + (", ".join(signos) if signos else "ninguno extra sobre la curva base."))


def _especiales() -> None:
    import plotly.graph_objects as go

    st.markdown("##### Funciones especiales (construidas por tramos)")
    elegida = st.selectbox("Función", ["signo sgn(x)", "valor absoluto |x|", "parte entera ⌊x⌋", "mantisa {x}"],
                           key="u1_especial")
    xs = np.linspace(-5, 5, 2001)
    if "signo" in elegida:
        ys = np.sign(xs)
        label = r"\operatorname{sgn}(x)"
    elif "valor" in elegida:
        ys = np.abs(xs)
        label = r"|x|"
    elif "parte entera" in elegida:
        ys = np.floor(xs)
        label = r"\lfloor x\rfloor"
    else:
        ys = xs - np.floor(xs)
        label = r"\{x\}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)",
                             line=dict(color="#1f77b4", width=2)))
    fig.update_layout(
        title=f"${label}$",
        xaxis_title="x", yaxis_title="y",
        height=420, margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, width="stretch")
    explicaciones = [
        "Salta de -1 a +1 en x=0, pasando por 0: función con discontinuidad de salto en el origen.",
        "Gráfica en 'V': coincide con x para x≥0 y con -x para x<0; esquina no suave en x=0.",
        "Escalera: constante en cada [k, k+1), con saltos de altura 1 en cada entero.",
        "Sierra: repite el patrón [0,1) en cada intervalo; es periódica de período 1.",
    ]
    if "signo" in elegida:
        idx = 0
    elif "valor" in elegida:
        idx = 1
    elif "parte entera" in elegida:
        idx = 2
    else:
        idx = 3
    st.caption(explicaciones[idx])


def _composicion_visual() -> None:
    st.markdown("##### Composición de funciones")
    lista = ["x²", "√x", "e^x", "ln(x)", "sin(x)", "1/x"]
    f_nom = st.selectbox("f(x)", lista, key="u1_f")
    g_nom = st.selectbox("g(x)", lista, key="u1_g")
    f = _FUNCIONES_BASE[f_nom]
    g = _FUNCIONES_BASE[g_nom]
    try:
        h = sp.simplify(f.subs(x, g))
    except Exception:
        h = x
    fig = figura_funciones([
        ("g(x)", g),
        ("f(g(x))", h),
    ], -4, 4, titulo=f"f = {sp.latex(f)},  g = {sp.latex(g)}")
    st.plotly_chart(fig, width="stretch")
    st.latex(r"(f\circ g)(x) = f\big(g(x)\big) = " + sp.latex(h))


def intuicion() -> None:
    st.markdown(
        "Explorá el efecto de los parámetros sobre una curva, las funciones por tramos y la "
        "composición antes de definirlas formalmente."
    )
    t1, t2, t3 = st.tabs(["Transformaciones", "Funciones especiales", "Composición"])
    with t1:
        _explorador_transformaciones()
    with t2:
        _especiales()
    with t3:
        _composicion_visual()


# ===========================================================================
# EJERCICIOS
# ===========================================================================

_DOMINIO_BANCO = [
    {"f": r"f(x)=\dfrac{1}{x-3}", "expr": 1 / (x - 3), "sol": r"(-\infty,3)\cup(3,\infty)",
     "distractores": [r"(-\infty,3]", r"[3,\infty)", r"(-\infty,\infty)"]},
    {"f": r"f(x)=\sqrt{x-2}", "expr": sp.sqrt(x - 2), "sol": r"[2,\infty)",
     "distractores": [r"(-\infty,2]", r"(-2,2)", r"(-\infty,2)\cup(2,\infty)"]},
    {"f": r"f(x)=\ln(2-x)", "expr": sp.ln(2 - x), "sol": r"(-\infty,2)",
     "distractores": [r"(2,\infty)", r"[2,\infty)", r"(-\infty,2]"]},
    {"f": r"f(x)=\dfrac{x}{x^2-4}", "expr": x / (x**2 - 4), "sol": r"(-\infty,-2)\cup(-2,2)\cup(2,\infty)",
     "distractores": [r"(-\infty,\infty)", r"(-2,2)", r"(-\infty,2)\cup(2,\infty)"]},
    {"f": r"f(x)=\sqrt[4]{x^2}", "expr": (x**2) ** sp.Rational(1, 4), "sol": r"(-\infty,\infty)",
     "distractores": [r"(-\infty,0)", r"[0,\infty)", r"(-\infty,0)\cup(0,\infty)"]},
    {"f": r"f(x)=\dfrac{1}{x+1}+\sqrt{x}", "expr": 1 / (x + 1) + sp.sqrt(x), "sol": r"[0,\infty)",
     "distractores": [r"(-\infty,\infty)", r"(-\infty,-1)\cup(-1,\infty)", r"(-1,\infty)"]},
]

_COMPOSICION_BANCO = [
    ("x²", "2x+1", x**2, 2 * x + 1),
    ("√x", "x-1", sp.sqrt(x), x - 1),
    ("sin(x)", "x²", sp.sin(x), x**2),
    ("e^x", "-x", sp.exp(x), -x),
    ("1/x", "x+3", 1 / x, x + 3),
    ("ln(x)", "x²", sp.ln(x), x**2),
    ("x³", "x−1", x**3, x - 1),
    ("cos(x)", "2x", sp.cos(x), 2 * x),
    ("e^x", "x/2", sp.exp(x), x / 2),
    ("tan(x)", "x²", sp.tan(x), x**2),
]

_PARIDAD_BANCO = [
    ("f(x)=x²", x**2), ("f(x)=x³", x**3), ("f(x)=cos(x)", sp.cos(x)),
    ("f(x)=sin(x)", sp.sin(x)), ("f(x)=x²+x", x**2 + x), ("f(x)=e^x", sp.exp(x)),
    ("f(x)=|x|", sp.Abs(x)), ("f(x)=x³−x", x**3 - x), ("f(x)=cos(2x)", sp.cos(2 * x)),
    ("f(x)=x²−4", x**2 - 4), ("f(x)=1/x", 1 / x), ("f(x)=tan(x)", sp.tan(x)),
]

_INECUACIONES_BANCO = [
    ("x² − 5x + 6 ≥ 0", r"(-\infty,2]\cup[3,\infty)", [r"(-\infty,2)\cup(3,\infty)", r"[2,3]", r"(-\infty,3]"]),
    ("|x| > 2", r"(-\infty,-2)\cup(2,\infty)", [r"(-2,2)", r"[-2,2]", r"(-\infty,2)"]),
    ("|x−3| < 1", r"(2,4)", [r"(-\infty,2)\cup(4,\infty)", r"[2,4]", r"(3,4)"]),
    ("(x−1)(x+2) ≤ 0", r"[-2,1]", [r"(-\infty,-2]\cup[1,\infty)", r"(-2,1)", r"(-\infty,1]"]),
    ("|x+1| ≥ 3", r"(-\infty,-4]\cup[2,\infty)", [r"(-4,2)", r"[-4,2]", r"(-\infty,2)"]),
    ("(x+3)/(x−1) < 0", r"(-3,1)", [r"(-\infty,-3)\cup(1,\infty)", r"(-\infty,1)", r"(-3,\infty)"]),
]


def _resp_estado(clave: str) -> None:
    st.session_state.pop(clave, None)


def _nuevo_ejercicio(clave: str) -> None:
    _resp_estado(clave)


def _tab_dominio() -> None:
    st.markdown("#### Dominio natural de una función")
    if "u1_dom" not in st.session_state:
        st.session_state["u1_dom"] = random.randint(0, len(_DOMINIO_BANCO) - 1)
    dato = _DOMINIO_BANCO[st.session_state["u1_dom"]]

    if st.button("🎲 Otro ejercicio (dominio)", key="u1_dom_nuevo"):
        st.session_state["u1_dom"] = random.randint(0, len(_DOMINIO_BANCO) - 1)
        _nuevo_ejercicio("u1_dom_resp")
        st.rerun()

    st.latex("Hallá el dominio natural de  " + dato["f"])
    opciones = [dato["sol"]] + list(dato["distractores"])
    op = opciones.copy()
    random.Random(st.session_state["u1_dom"]).shuffle(op)
    idx_correcto = op.index(dato["sol"])
    ui.elegir_opcion("u1_dom_resp", "El dominio D_f es:",
                     [f"$ {s} $" for s in op], idx_correcto,
                     explicacion=f"La función tiene dominio {dato['sol']}.",
                     tema="U1-dominio", enunciado=dato["f"])


def _tab_composicion() -> None:
    st.markdown("#### Composición de funciones")
    if "u1_comp" not in st.session_state:
        st.session_state["u1_comp"] = random.randint(0, len(_COMPOSICION_BANCO) - 1)
    f_nom, g_nom, f_expr, g_expr = _COMPOSICION_BANCO[st.session_state["u1_comp"]]

    if st.button("🎲 Otro ejercicio (composición)", key="u1_comp_nuevo"):
        st.session_state["u1_comp"] = random.randint(0, len(_COMPOSICION_BANCO) - 1)
        _nuevo_ejercicio("u1_comp_resp")
        st.rerun()

    correcto = sp.simplify(f_expr.subs(x, g_expr))
    st.markdown(f"Si $f(x) = {sp.latex(f_expr)}$ y $g(x) = {sp.latex(g_expr)}$, hallá $(f\\circ g)(x)$.")
    ui.resolver_expresion("u1_comp_resp", correcto, placeholder="ej.: (2x+1)^2",
                          ayuda="Simplificá el resultado tanto como puedas.",
                          tema="U1-composicion",
                          enunciado=f"(f∘g) con f={f_nom}, g={g_nom}")


def _tab_paridad() -> None:
    st.markdown("#### Paridad de funciones")
    if "u1_par" not in st.session_state:
        st.session_state["u1_par"] = random.randint(0, len(_PARIDAD_BANCO) - 1)
    nombre, expr = _PARIDAD_BANCO[st.session_state["u1_par"]]

    if st.button("🎲 Otro ejercicio (paridad)", key="u1_par_nuevo"):
        st.session_state["u1_par"] = random.randint(0, len(_PARIDAD_BANCO) - 1)
        _nuevo_ejercicio("u1_par_resp")
        st.rerun()

    # Clasificación simbólica.
    es_par = sp.simplify(expr.subs(x, -x) - expr) == 0
    es_impar = sp.simplify(expr.subs(x, -x) + expr) == 0
    if es_par and es_impar:
        idx = None  # f ≡ 0: caso borde
    elif es_par:
        idx = 0
    elif es_impar:
        idx = 1
    else:
        idx = 2
    st.latex(nombre)
    if idx is None:
        st.info("Esta función es idénticamente nula: es par e impar a la vez.")
        return
    ui.elegir_opcion("u1_par_resp",
                     f"¿Cómo es la simetría de la gráfica de {nombre.split('=')[0]}=...?",
                     ["Es par (simetría respecto del eje y)",
                      "Es impar (simetría respecto del origen)",
                      "No es par ni impar"],
                     idx,
                     explicacion="Recordá: par → f(−x)=f(x); impar → f(−x)=−f(x).",
                     tema="U1-paridad", enunciado=nombre)


def _tab_inecuaciones() -> None:
    st.markdown("#### Inecuaciones (cuadráticas y modulares)")
    if "u1_ine" not in st.session_state:
        st.session_state["u1_ine"] = random.randint(0, len(_INECUACIONES_BANCO) - 1)
    pregunta, solucion, distractores = _INECUACIONES_BANCO[st.session_state["u1_ine"]]

    if st.button("🎲 Otro ejercicio (inecuación)", key="u1_ine_nuevo"):
        st.session_state["u1_ine"] = random.randint(0, len(_INECUACIONES_BANCO) - 1)
        _nuevo_ejercicio("u1_ine_resp")
        st.rerun()

    st.write(f"Resolver en ℝ:  **{pregunta}**")
    opciones = [solucion] + list(distractores)
    op = opciones.copy()
    random.Random(st.session_state["u1_ine"]).shuffle(op)
    idx = op.index(solucion)
    ui.elegir_opcion("u1_ine_resp", "El conjunto solución es:",
                     [f"$ S = {s} $" for s in op], idx,
                     explicacion="Usá el estudio del signo de los factores.",
                     tema="U1-inecuacion", enunciado=pregunta)


def ejercicios(repo=None) -> None:
    st.markdown(
        "Practicá dominio, composición, paridad e inecuaciones. Las respuestas de composición "
        "se comparan algebraicamente."
    )
    t1, t2, t3, t4 = st.tabs(["Dominio", "Composición", "Paridad", "Inecuaciones"])
    with t1:
        _tab_dominio()
    with t2:
        _tab_composicion()
    with t3:
        _tab_paridad()
    with t4:
        _tab_inecuaciones()