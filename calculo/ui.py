"""Widgets reutilizables para los ejercicios de las distintas unidades.

Estandarizan el flujo "respuesta -> evaluación -> feedback" en toda la app:
    * `resolver_expresion`  : texto libre que se compara algebraicamente.
    * `resolver_valor`      : texto libre que representa un número (o ∞).
    * `elegir_opcion`       : opción múltiple (radio + evaluar).
Cada widget persiste su estado en ``st.session_state`` bajo una clave única.
"""

from __future__ import annotations

import streamlit as st

from .matematicas import (
    ErrorMatematico,
    parsear_expresion,
    parsear_evaluacion,
    son_equivalentes,
    comparar_resultado,
)


def _estado(clave: str):
    return st.session_state.setdefault(f"w_{clave}", {})


def _resultado_nuevo(clave: str) -> None:
    st.session_state[f"w_{clave}"] = {}


def boton_reset(clave: str):
    """Limpia el estado de un widget (para 'nuevo ejercicio')."""
    _resultado_nuevo(clave)


def _auto_registrar(tema, enunciado, respuesta, correcto, es_correcta) -> None:
    """Guarda el intento si la app configuró el repositorio y el estudiante."""
    import streamlit as st

    repo = st.session_state.get("_repo")
    sid = st.session_state.get("estudiante_id")
    if repo is None or not sid or tema is None:
        return
    try:
        repo.guardar_registro(sid, tema, enunciado or "(ejercicio)",
                              respuesta or "(sin respuesta)", correcto, bool(es_correcta))
    except Exception:
        pass


def resolver_expresion(
    clave: str,
    correcto,
    placeholder: str = "escribe tu respuesta",
    ayuda: str = "",
    tolerancia: float = 1e-3,
    tema: str | None = None,
    enunciado: str | None = None,
) -> bool | None:
    """Input de texto con comparación algebraica (con tolerancia numérica).

    Devuelve True/False si ya fue evaluado, y None si todavía no.
    """
    estado = _estado(clave)
    respuesta = st.text_input(
        "Tu respuesta:",
        placeholder=placeholder,
        help=ayuda,
        key=f"in_{clave}",
    )

    c1, c2 = st.columns([1, 1])
    evaluar = c1.button("Evaluar", key=f"ev_{clave}", disabled=not respuesta.strip())
    ver_sol = c2.button("Ver solución", key=f"sol_{clave}")

    if evaluar:
        try:
            expr_usuario = parsear_expresion(respuesta)
            es_correcta = son_equivalentes(expr_usuario, correcto) or comparar_resultado(
                expr_usuario, correcto, tolerancia
            )
        except ErrorMatematico as exc:
            st.error(f"⚠️ {exc}")
            return None
        estado["resultado"] = bool(es_correcta)
        estado["respuesta"] = respuesta.strip()
        estado["mostrar_sol"] = False
        _auto_registrar(tema, enunciado, respuesta.strip(), correcto, es_correcta)

    resultado = estado.get("resultado")
    if resultado is True:
        st.success("¡Correcto! Tu respuesta es algebraicamente equivalente. 🎉")
    elif resultado is False:
        st.error("Todavía no es correcta. Revisa el cálculo o mira la solución.")

    if ver_sol or estado.get("mostrar_sol"):
        estado["mostrar_sol"] = True
        st.info(f"**Solución:**  $S = {sp_latex(correcto)}$")

    return resultado


def resolver_valor(
    clave: str,
    correcto,
    placeholder: str = "ej.: 2, -3, oo",
    tolerancia: float = 1e-3,
    tema: str | None = None,
    enunciado: str | None = None,
) -> bool | None:
    """Input de texto para un valor numérico (o ±∞) con tolerancia."""
    estado = _estado(clave)
    respuesta = st.text_input("Tu respuesta:", placeholder=placeholder, key=f"in_{clave}")

    c1, c2 = st.columns([1, 1])
    evaluar = c1.button("Evaluar", key=f"ev_{clave}", disabled=not respuesta.strip())
    ver_sol = c2.button("Ver solución", key=f"sol_{clave}")

    if evaluar:
        try:
            expr_usuario = parsear_evaluacion(respuesta)
            es_correcta = comparar_resultado(expr_usuario, correcto, tolerancia)
        except ErrorMatematico as exc:
            st.error(f"⚠️ {exc}")
            return None
        estado["resultado"] = bool(es_correcta)
        estado["respuesta"] = respuesta.strip()
        estado["mostrar_sol"] = False
        _auto_registrar(tema, enunciado, respuesta.strip(), correcto, es_correcta)

    resultado = estado.get("resultado")
    if resultado is True:
        st.success("¡Correcto! 🎉")
    elif resultado is False:
        st.error("No es el valor esperado. Revisa el procedimiento o mira la solución.")

    if ver_sol or estado.get("mostrar_sol"):
        estado["mostrar_sol"] = True
        st.info(rf"**Solución:**  $= {sp_latex(correcto)}$")

    return resultado


def elegir_opcion(
    clave: str,
    pregunta: str,
    opciones: list[str],
    idx_correcto: int,
    explicacion: str = "",
    tema: str | None = None,
    enunciado: str | None = None,
) -> bool | None:
    """Opción múltiple con evaluación.
    `opciones` son cadenas con sintaxis LaTeX (se muestran con `st.latex`).
    """
    estado = _estado(clave)
    seleccion = st.radio(pregunta, list(range(len(opciones))),
                         format_func=lambda i: opciones[i], key=f"mc_{clave}")

    if st.button("Evaluar", key=f"ev_{clave}", disabled=seleccion is None):
        es_correcta = (seleccion == idx_correcto)
        estado["resultado"] = bool(es_correcta)
        estado["respuesta"] = opciones[seleccion]
        _auto_registrar(tema, enunciado, opciones[seleccion], opciones[idx_correcto], es_correcta)

    resultado = estado.get("resultado")
    if resultado is True:
        st.success("¡Correcto! 🎉 " + explicacion if explicacion else "¡Correcto! 🎉")
    elif resultado is False:
        st.error("No es la opción correcta. " + (explicacion if explicacion else ""))
    return resultado


def sp_latex(expr) -> str:
    """LaTeX de una expresión SymPy (o cadena, si es string)."""
    import sympy as sp

    if isinstance(expr, str):
        return expr
    if expr is sp.oo:
        return r"\infty"
    if expr == -sp.oo:
        return r"-\infty"
    try:
        return sp.latex(expr)
    except Exception:
        return str(expr)


# ---------------------------------------------------------------------------
# Identidad del estudiante y registro de progreso (compartido por todas las
# unidades)
# ---------------------------------------------------------------------------

def panel_identidad(repo, contenedor=st.sidebar) -> None:
    """Permite registrar (o revivir) la identidad del estudiante en una sesión."""
    id_actual = st.session_state.get("estudiante_id")
    nombre = st.session_state.get("estudiante_nombre", "")

    if id_actual:
        contenedor.markdown(f"👤 **{nombre}** · progreso activo")
        if contenedor.button("Cambiar usuario", key="cambiar_usuario"):
            st.session_state.pop("estudiante_id", None)
            st.session_state.pop("estudiante_nombre", None)
            del st.session_state["cambiar_usuario"]
            st.rerun()
        return

    with contenedor.expander("🔐 Identifícate para guardar progreso", expanded=False):
        nombre_inp = st.text_input("Nombre", key="id_nombre", value="")
        email_inp = st.text_input("Email", key="id_email", value="")
        if st.button("Registrar", key="id_registrar", disabled=not (nombre_inp.strip() and email_inp.strip())):
            try:
                sid = repo.crear_estudiante(nombre_inp, email_inp)
                st.session_state["estudiante_id"] = sid
                st.session_state["estudiante_nombre"] = nombre_inp.strip()
                st.rerun()
            except Exception as exc:
                st.error(f"No se pudo registrar: {exc}")


def registrar_practica(repo, estudiante_id, tema: str, enunciado: str,
                       respuesta: str, correcta, es_correcta: bool) -> None:
    """Registra un intento si el estudiante está identificado y el repo existe."""
    if not estudiante_id or repo is None:
        return
    try:
        repo.guardar_registro(
            estudiante_id, tema, enunciado,
            respuesta or "(sin respuesta)", correcta, bool(es_correcta),
        )
    except Exception as exc:
        st.warning(f"No se pudo registrar el intento: {exc}")


def resumen_progreso(repo, estudiante_id: str) -> None:
    """Métricas del progreso global del estudiante (todas las unidades)."""
    try:
        intentos = repo.progreso_por_estudiante(estudiante_id)
    except Exception:
        intentos = []
    if not intentos:
        st.caption("Aún no hay intentos registrados.")
        return
    correctos = sum(1 for i in intentos if i["es_correcta"])
    total = len(intentos)
    c1, c2, c3 = st.columns(3)
    c1.metric("Intentos", total)
    c2.metric("Correctos", correctos)
    c3.metric("Acierto (%)", f"{100 * correctos / total:.0f}")

    por_tema: dict[str, list] = {}
    for i in intentos:
        clase = i["enunciado"].split("]")[0][1:] if i["enunciado"].startswith("[") else "general"
        por_tema.setdefault(clase, []).append(i)
    st.caption("Intentos por tema: " + ", ".join(f"{k} ({len(v)})" for k, v in por_tema.items()))