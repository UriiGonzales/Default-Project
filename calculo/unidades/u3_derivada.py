"""Unidad 3 · Derivada y Diferencial (Programa 2026: UN 3).

3.1 Cociente incremental, derivada, reglas de derivación y regla de la cadena.
3.2 Derivadas de orden superior. Derivadas de funciones trascendentes.
3.3 Derivada en un punto, recta tangente y normal. Diferencial.
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
    generar_ejercicio,
    derivar,
    Ejercicio,
)
from .. import ui


# ===========================================================================
# TEORÍA
# ===========================================================================

def teoria() -> None:
    st.markdown(
        "La **derivada** es la pendiente instantánea: generaliza el concepto de velocidad y "
        "permite estudiar localmente cualquier función."
    )

    with st.expander("3.1 Definición y reglas de derivación", expanded=True):
        st.markdown(
            "Definida como el límite del **cociente incremental** cuando $h\\to 0$:"
        )
        st.latex(r"f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}")
        st.markdown("**Reglas fundamentales:**")
        st.latex(
            r"\begin{aligned}"
            r"&(u+v)'=u'+v' && (cu)'=c\,u' \\"
            r"&(uv)'=u'v+uv' && \left(\frac{u}{v}\right)'=\frac{u'v-uv'}{v^2} \\"
            r"&(u^n)'=n\,u^{n-1}u' && \text{(regla de la cadena)}\; (f\circ g)'=f'(g)\,g'"
            r"\end{aligned}"
        )
        st.markdown("**Derivadas elementales:**")
        st.latex(
            r"\begin{aligned}"
            r"&(\sin u)'=\cos u\,u' &&(\cos u)'=-\sin u\,u' \\"
            r"&(\tan u)'=\sec^2 u\,u' &&(e^u)'=e^u\,u' \\"
            r"&(\ln u)'=\frac{u'}{u} &&(x^a)'=a\,x^{a-1}"
            r"\end{aligned}"
        )
        st.success(
            "**Interpretación geométrica:** $f'(x_0)$ es la pendiente de la recta tangente a la "
            "gráfica en $(x_0, f(x_0))$: $y = f(x_0) + f'(x_0)(x-x_0)$."
        )

    with st.expander("3.2 Derivadas de orden superior y trascendentes", expanded=False):
        st.markdown(
            "$f''$ es la derivada de $f'$ (aceleración si $f$ es posición); en general $f^{(n)}$ "
            "se obtiene derivando $n$ veces. Las funciones hiperbólicas cierran el repertorio:"
        )
        st.latex(r"(\sinh u)'=\cosh u\,u',\qquad (\cosh u)'=\sinh u\,u',\qquad (\tanh u)'=\operatorname{sech}^2u\,u'")
        st.markdown(
            r"""
            **Derivación logarítmica:** para $f(x)^{g(x)}$ se toma logaritmo y se deriva
            implícitamente.
            """
        )

    with st.expander("3.3 Derivada en un punto, tangente y diferencial", expanded=False):
        st.markdown(
            r"""
            - **Recta tangente:** $y=f(x_0)+f'(x_0)(x-x_0)$; **normal:** pendiente $-1/f'(x_0)$.
            - **Diferencial:** $dy = f'(x)\,dx$ aproxima el cambio de $f$ ante un cambio $dx$:
              $f(x+dx) \approx f(x)+f'(x)\,dx$ (linealización de primer orden).
            - **Derivabilidad y continuidad:** si existe $f'$, $f$ es continua; el recíproco es
              **falso** (ej.: $|x|$ es continua en $0$ pero no derivable).
            """
        )
        st.info(
            "Una función derivable tiene la gráfica 'suave' (tangente única); los picos "
            "como el de $|x|$ en el origen no admiten tangente."
        )

    with st.expander("3.4 Demostraciones: suma, producto, inversa y logarítmica", expanded=False):
        st.markdown(
            r"**Derivada de una suma.** Si existen $f'(x)$ y $g'(x)$, entonces "
            r"$(f+g)'(x)=f'(x)+g'(x)$."
        )
        st.markdown(r"**Demostración.** Por definición y por el álgebra de límites,")
        st.latex(
            r"(f+g)'(x)=\lim_{h\to0}\left[\frac{f(x+h)-f(x)}{h}+\frac{g(x+h)-g(x)}{h}\right]"
            r"=f'(x)+g'(x)\qquad\blacksquare"
        )

        st.markdown(
            r"**Derivada de un producto.** Si existen $f'(x)$ y $g'(x)$, entonces "
            r"$(fg)'(x)=f'(x)\,g(x)+f(x)\,g'(x)$."
        )
        st.markdown(r"**Demostración.** Sumando y restando el término mixto $f(x+h)\,g(x)$,")
        st.latex(
            r"\frac{f(x+h)g(x+h)-f(x)g(x)}{h}=f(x+h)\,\frac{g(x+h)-g(x)}{h}"
            r"+\frac{f(x+h)-f(x)}{h}\,g(x)"
        )
        st.markdown(
            r"Cuando $h\to0$: el primer término da $f(x)\,g'(x)$ y el segundo "
            r"$f'(x)\,g(x)$ (usamos que $g$ es continua en $x$ por ser derivable). "
            r"$\blacksquare$"
        )

        st.markdown(
            r"**Derivada de la función inversa.** Si $f$ es continua y biyectiva en un entorno "
            r"de $x_0$ con $f'(x_0)\neq 0$, entonces $f^{-1}$ es derivable en "
            r"$y_0=f(x_0)$ y"
        )
        st.latex(r"(f^{-1})'(y_0)=\frac{1}{f'(x_0)},\qquad x_0=f^{-1}(y_0)")
        st.markdown(
            r"**Demostración.** Para $y\to y_0$, poniendo $x=f^{-1}(y)$ (que tiende a $x_0$ "
            r"por la continuidad de $f^{-1}$):"
        )
        st.latex(
            r"\frac{f^{-1}(y)-f^{-1}(y_0)}{y-y_0}=\frac{x-x_0}{f(x)-f(x_0)}"
            r"=\left(\frac{f(x)-f(x_0)}{x-x_0}\right)^{-1}\;\xrightarrow[y\to y_0]\;"
            r"\frac{1}{f'(x_0)}"
        )
        st.markdown(
            r"La condición $f'(x_0)\neq 0$ permite tomar el límite del cociente. $\blacksquare$"
        )

        st.markdown(
            r"**Método logarítmico de derivación.** Para $u(x)>0$:"
        )
        st.latex(r"\left(u^v\right)'=u^v\left(v'\,\ln u + v\,\frac{u'}{u}\right)")
        st.markdown(r"**Demostración.** Sea $y=u^v$; tomando logaritmo en ambos miembros,")
        st.latex(r"\ln y = v\,\ln u\;\Rightarrow\;\frac{y'}{y}=v'\,\ln u + v\,\frac{u'}{u}")
        st.markdown(
            r"Despejando $y'$ se obtiene la fórmula. Es útil con $f(x)^{g(x)}$ y con "
            r"expresiones de muchos factores (se transforman en sumas de logaritmos). "
            r"$\blacksquare$"
        )


# ===========================================================================
# INTUICIÓN VISUAL
# ===========================================================================

def _cociente_incremental() -> None:
    st.markdown("##### Cociente incremental: cómo nace la derivada")
    f = x**2
    a = st.slider("Punto x₀ (fijo)", -3.0, 3.0, 1.0, 0.1, key="u3_a_sec")
    h = st.slider("h (longitud del intervalo)", 0.05, 3.0, 1.0, 0.05, key="u3_h_sec")

    pendiente = sp.simplify((f.subs(x, a + h) - f.subs(x, a)) / h)
    xs = np.linspace(-4, 4, 600)
    x1, x2 = a, a + h
    y1, y2 = float(f.subs(x, x1)), float(f.subs(x, x2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[float(f.subs(x, v)) for v in xs], mode="lines",
                             name="f(x) = x²", line=dict(color="#1f77b4", width=3)))
    seg_x = np.linspace(x1 - 0.4, x2 + 0.4, 20)
    seg_y = [y1 + float(pendiente) * (v - x1) for v in seg_x]
    fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines", name="recta secante",
                             line=dict(color="red", dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode="markers", name="puntos",
                             marker=dict(size=8, color="red")))
    fig.update_layout(title=f"Pendiente de la secante = {sp.latex(sp.sympify(pendiente))}",
                      height=420, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        rf"Cuando $h \to 0$, la pendiente $\frac{{f(x_0+h)-f(x_0)}}{{h}} = {sp.latex(pendiente)}$ "
        "tiende a la pendiente de la **tangente**, $f'(x_0) = " 
        f"{float(sp.diff(f, x).subs(x, a)):.2f}$."
    )


def _tangente_visual() -> None:
    st.markdown("##### Recta tangente a una función con slider")
    banco = {
        "f(x) = x² − 4x + 3": x**2 - 4 * x + 3,
        "f(x) = sin(x)": sp.sin(x),
        "f(x) = e^x": sp.exp(x),
        "f(x) = x³ − 3x": x**3 - 3 * x,
        "f(x) = ln(x)": sp.ln(x),
    }
    nom = st.selectbox("Función", list(banco), key="u3_tg_f")
    f = banco[nom]
    low = 1.0 if "ln" in nom else -4.0
    a = st.slider("Punto de tangencia x₀", float(low), 4.0, 1.0, 0.1, key="u3_tg_a")

    fp = sp.diff(f, x)
    m = float(fp.subs(x, a))
    b0 = float(f.subs(x, a))
    tangente = m * (x - a) + b0

    fig = figura_funciones([(nom, f)], min(low, a - 2), a + 3, titulo="Recta tangente")
    xs = np.linspace(min(low, a - 2), a + 3, 300)
    fig.add_trace(go.Scatter(x=xs, y=[m * (v - a) + b0 for v in xs], mode="lines",
                             name=f"tangente en x₀={a:.1f}", line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=[a], y=[b0], mode="markers", name="punto (x₀, f(x₀))",
                             marker=dict(size=10, color="red")))
    st.plotly_chart(fig, width="stretch")
    st.latex(rf"y = f({sp.latex(sp.sympify(a))}) + f'({sp.latex(sp.sympify(a))})(x-{sp.latex(sp.sympify(a))})"
             rf" = {sp.latex(sp.sympify(tangente))}")


def intuicion() -> None:
    st.markdown(
        "Observá cómo la secante se convierte en tangente, y cómo la tangente depende del "
        "punto elegido."
    )
    t1, t2 = st.tabs(["Cociente incremental", "Recta tangente"])
    with t1:
        _cociente_incremental()
    with t2:
        _tangente_visual()


# ===========================================================================
# EJERCICIOS
# ===========================================================================

_CADENA_BANCO = [
    ("f(x) = sin(3x)", sp.sin(3 * x)),
    ("f(x) = cos(2x)", sp.cos(2 * x)),
    ("f(x) = e^(2x)", sp.exp(2 * x)),
    ("f(x) = ln(2x+1)", sp.ln(2 * x + 1)),
    ("f(x) = (2x+1)^3", (2 * x + 1) ** 3),
    ("f(x) = sqrt(x²+1)", sp.sqrt(x**2 + 1)),
    ("f(x) = tan(3x)", sp.tan(3 * x)),
    ("f(x) = e^(x²)", sp.exp(x**2)),
    ("f(x) = ln(sin(x))", sp.ln(sp.sin(x))),
    ("f(x) = sin(x²)", sp.sin(x**2)),
]

_PUNTO_BANCO = [
    ("f(x) = x²−4x+3", x**2 - 4 * x + 3, 3),
    ("f(x) = sin(x)", sp.sin(x), sp.pi / 4),
    ("f(x) = e^x", sp.exp(x), 0),
    ("f(x) = x³", x**3, 2),
    ("f(x) = ln(x)", sp.ln(x), 1),
    ("f(x) = 1/x", 1 / x, 2),
    ("f(x) = √x", sp.sqrt(x), 4),
]

_TANGENTE_BANCO = [
    ("f(x) = x²−4x+3", x**2 - 4 * x + 3, 3),
    ("f(x) = x³", x**3, 1),
    ("f(x) = sin(x)", sp.sin(x), 0),
    ("f(x) = e^x", sp.exp(x), 0),
    ("f(x) = 1/x", 1 / x, 1),
    ("f(x) = √x", sp.sqrt(x), 1),
]

_SEGUNDA_BANCO = [
    ("f(x) = x³−2x²+x−5", x**3 - 2 * x**2 + x - 5),
    ("f(x) = sin(x)", sp.sin(x)),
    ("f(x) = 1/x", 1 / x),
    ("f(x) = ln(x)", sp.ln(x)),
    ("f(x) = x⁴/4 − x²", x**4 / 4 - x**2),
    ("f(x) = cos(2x)", sp.cos(2 * x)),
]


def _reset_widget(clave: str) -> None:
    st.session_state.pop(f"w_{clave}", None)
    st.session_state.pop(f"in_{clave}", None)
    st.session_state.pop(f"mc_{clave}", None)


def _tab_derivacion() -> None:
    st.markdown("#### Derivación directa (reglas básicas + cociente)")
    if "u3_d" not in st.session_state:
        st.session_state["u3_d"] = generar_ejercicio()

    c1, _ = st.columns([1, 3])
    if c1.button("🎲 Nuevo ejercicio", key="u3_d_nuevo"):
        nuevo = generar_ejercicio(random.Random())
        while nuevo.enunciado == st.session_state["u3_d"].enunciado:
            nuevo = generar_ejercicio(random.Random())
        st.session_state["u3_d"] = nuevo
        _reset_widget("u3_d_resp")
        st.rerun()

    ejercicio: Ejercicio = st.session_state["u3_d"]
    nombre_f = ejercicio.enunciado.split("=", 1)[1].strip()
    st.latex(rf"f(x) = {nombre_f}")
    resultado = ui.resolver_expresion(
        "u3_d_resp", ejercicio.derivada,
        placeholder="ej.: 3*x**2 - 4  o  cos(3x)*3",
        ayuda="Usá regla del producto o cociente si hace falta; la cadena para compuestas.",
        tema="U3-derivada-directa", enunciado=ejercicio.enunciado,
    )
    if resultado is False:
        a = ejercicio.enunciado
        st.markdown("**Pistas:**")
        if any(o in a for o in ("sin", "cos", "tan")):
            st.markdown("- **Regla de la cadena:** $(\\sin u)'=\\cos u\\, u'$.")
        elif "sqrt" in a:
            st.markdown("- $u^{1/2}$ con la regla de la potencia compuesta: $\\tfrac12 u^{-1/2} u'$.")
        elif "ln" in a:
            st.markdown("- $(\\ln u)'=\\dfrac{u'}{u}$.")
        elif "/" in a:
            st.markdown("- **Regla del cociente:** $(u/v)'=\\dfrac{u'v-uv'}{v^2}$.")
        elif "*" in a:
            st.markdown("- **Regla del producto:** $(uv)'=u'v+uv'$.")


def _tab_cadena() -> None:
    st.markdown("#### Regla de la cadena")
    if "u3_c" not in st.session_state:
        st.session_state["u3_c"] = random.randint(0, len(_CADENA_BANCO) - 1)
    nombre, expr = _CADENA_BANCO[st.session_state["u3_c"]]

    c1, _ = st.columns([1, 3])
    if c1.button("🎲 Nuevo ejercicio", key="u3_c_nuevo"):
        st.session_state["u3_c"] = random.randint(0, len(_CADENA_BANCO) - 1)
        _reset_widget("u3_c_resp")
        st.rerun()

    st.latex(nombre)
    ui.resolver_expresion(
        "u3_c_resp", derivar(expr),
        placeholder="ej.: cos(3x)*3",
        ayuda="Deriva 'afuera' y multiplicá por la derivada 'adentro'.",
        tema="U3-regla-cadena", enunciado=nombre,
    )


def _tab_en_punto() -> None:
    st.markdown("#### Valor de la derivada en un punto")
    if "u3_p" not in st.session_state:
        st.session_state["u3_p"] = random.randint(0, len(_PUNTO_BANCO) - 1)
    nombre, expr, a = _PUNTO_BANCO[st.session_state["u3_p"]]

    c1, _ = st.columns([1, 3])
    if c1.button("🎲 Nuevo ejercicio", key="u3_p_nuevo"):
        st.session_state["u3_p"] = random.randint(0, len(_PUNTO_BANCO) - 1)
        _reset_widget("u3_p_resp")
        st.rerun()

    st.latex(nombre)
    correcto = sp.simplify(sp.diff(expr, x).subs(x, a))
    st.latex(rf"f'({sp.latex(sp.sympify(a))}) =")
    ui.resolver_valor("u3_p_resp", correcto, placeholder="ej.: 2  o  sqrt(2)/2",
                      tema="U3-derivada-en-punto", enunciado=f"{nombre} en x={a}")


def _tab_tangente() -> None:
    st.markdown("#### Recta tangente en un punto")
    if "u3_t" not in st.session_state:
        st.session_state["u3_t"] = random.randint(0, len(_TANGENTE_BANCO) - 1)
    nombre, expr, a = _TANGENTE_BANCO[st.session_state["u3_t"]]

    c1, _ = st.columns([1, 3])
    if c1.button("🎲 Nuevo ejercicio", key="u3_t_nuevo"):
        st.session_state["u3_t"] = random.randint(0, len(_TANGENTE_BANCO) - 1)
        _reset_widget("u3_t_resp")
        st.rerun()

    m = sp.simplify(sp.diff(expr, x).subs(x, a))
    b0 = sp.simplify(expr.subs(x, a))
    tangente = sp.simplify(m * (x - a) + b0)
    st.latex(nombre)
    st.latex(rf"Hallá la recta tangente a la gráfica en $x_0 = {sp.latex(sp.sympify(a))}$.")
    fig = figura_funciones([("f", expr)], -2, 5, titulo="Recta tangente")
    xs = np.linspace(-2, 5, 300)
    fig.add_trace(go.Scatter(x=xs, y=[float(m * (v - a) + b0) for v in xs], mode="lines",
                             name="tangente", line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=[float(a)], y=[float(b0)], mode="markers",
                             marker=dict(size=10, color="red")))
    st.plotly_chart(fig, width="stretch")
    ui.resolver_expresion("u3_t_resp", tangente, placeholder="ej.: 2*x - 6",
                          tema="U3-recta-tangente",
                          enunciado=f"{nombre} en x0={a}")


def _tab_orden_superior() -> None:
    st.markdown("#### Derivadas de orden superior (f'')")
    if "u3_s" not in st.session_state:
        st.session_state["u3_s"] = random.randint(0, len(_SEGUNDA_BANCO) - 1)
    nombre, expr = _SEGUNDA_BANCO[st.session_state["u3_s"]]

    c1, _ = st.columns([1, 3])
    if c1.button("🎲 Nuevo ejercicio", key="u3_s_nuevo"):
        st.session_state["u3_s"] = random.randint(0, len(_SEGUNDA_BANCO) - 1)
        _reset_widget("u3_s_resp")
        st.rerun()

    f1 = sp.simplify(sp.diff(expr, x))
    f2 = sp.simplify(sp.diff(expr, x, 2))
    f3 = sp.simplify(sp.diff(expr, x, 3))

    candidatos = [f2, f1, -f2, f3]
    opciones = []
    for e in candidatos:
        if all(sp.simplify(e - v) != 0 for v in opciones):
            opciones.append(e)
    pad = 1
    while len(opciones) < 4:
        opciones.append(sp.simplify(f2 + pad))
        pad += 1

    random.Random(st.session_state["u3_s"]).shuffle(opciones)
    idx_correcto = opciones.index(f2)
    st.latex(nombre)
    ui.elegir_opcion(
        "u3_s_resp", "La segunda derivada $f''(x)$ es:",
        [rf"$ f''(x) = {sp.latex(v)}$" for v in opciones],
        idx_correcto,
        explicacion=f"Se obtiene derivando dos veces: $f''(x)={sp.latex(f2)}$.",
        tema="U3-derivada-segunda", enunciado=nombre,
    )


def ejercicios(repo=None) -> None:
    st.markdown(
        "Derivá con las reglas básicas, la cadena, en un punto, encontrá rectas tangentes y "
        "derivadas de orden superior."
    )
    t1, t2, t3, t4, t5 = st.tabs(
        ["Derivación directa", "Regla de la cadena", "Derivada en un punto",
         "Recta tangente", "f''"]
    )
    with t1:
        _tab_derivacion()
    with t2:
        _tab_cadena()
    with t3:
        _tab_en_punto()
    with t4:
        _tab_tangente()
    with t5:
        _tab_orden_superior()