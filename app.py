"""CalculusLab — Aplicación interactiva de Cálculo 1 (Análisis Matemático I).

Punto de entrada. Ejecutar con:

    python -m streamlit run app.py

Estructura modular:
    calculo/matematicas     Motor simbólico (SymPy): límites, derivadas, integrales, series.
    calculo/graficos        Figuras y estilo de Plotly.
    calculo/ui              Widgets de ejercicios con evaluación algebraica y progreso.
    calculo/persistencia    Repositorio de progreso (SQLite por defecto, PostgreSQL lista).
    calculo/unidades        Las 7 unidades temáticas, cada una con Teoría/Intuición/Ejercicios.
"""

from __future__ import annotations

import streamlit as st

from calculo.persistencia import crear_repositorio
from calculo.unidades import UNIDADES, etiqueta, mostrar

_HOME = "Inicio"

try:
    from calculo.unidades import armar_navegacion
    _NAV = armar_navegacion()
except Exception:  # pragma: no cover
    _NAV = [_HOME] + [etiqueta(u) for u in UNIDADES]


# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CalculusLab · Cálculo 1",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Conectando con el almacenamiento…")
def _obtener_repositorio():
    """Instancia única (por sesión de servidor) del repositorio de progreso."""
    return crear_repositorio()


# ---------------------------------------------------------------------------
# Barra lateral: navegación
# ---------------------------------------------------------------------------

def _sidebar() -> tuple[str, object]:
    with st.sidebar:
        st.title("📐 CalculusLab")
        st.caption("Análisis Matemático I · Cálculo 1 · UTN FRM 2026")
        st.divider()

        pagina = st.radio(
            "Navegación",
            _NAV,
            key="nav",
            label_visibility="collapsed",
        )

        st.divider()
        repo = _obtener_repositorio()
        st.markdown("**Progreso personal**")
        from calculo import ui

        ui.panel_identidad(repo)

        st.divider()
        st.caption(
            "Teoría · Intuición visual · Ejercicios con corrección algebraica. "
            "Almacenamiento local SQLite; lista para PostgreSQL via `POSTGRES_URL`."
        )
    return pagina, repo


# ---------------------------------------------------------------------------
# Portada
# ---------------------------------------------------------------------------

def _portada() -> None:
    st.title("CalculusLab · Preparación de Cálculo 1")
    st.markdown(
        """
        Aplicación interactiva para **estudiantes universitarios** que preparan
        **Análisis Matemático I** siguiendo el programa oficial 2026 de la UTN
        FRM: teoría rigurosa, visualización dinámica y ejercicios con
        corrección matemática real, **clasificados por unidad**.
        """
    )

    cols = st.columns([2, 1])
    with cols[0]:
        st.markdown("### Las 7 unidades del programa")
        with st.expander("Ver unidades", expanded=True):
            for u in UNIDADES:
                st.markdown(f"**{etiqueta(u)}**  — {u.subtitulo}  (~{u.horas} h)")

    with cols[1]:
        st.info("**Sugerencia:** empezá por **Funciones** y seguí el orden del menú lateral.")

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.info("**Rigor:** demostraciones completas, no solo fórmulas sueltas.")
    c2.info("**Interactividad:** deslizadores y gráficos actualizados en tiempo real.")
    c3.info("**Evaluación inteligente:** $2x$ y $x+x$ cuentan como lo mismo.")


# ---------------------------------------------------------------------------
# ``main``
# ---------------------------------------------------------------------------

def main() -> None:
    repo = _obtener_repositorio()
    st.session_state["_repo"] = repo

    pagina, _repo_menu = _sidebar()

    if pagina == _HOME:
        _portada()
    else:
        indice = next(
            (i for i, u in enumerate(UNIDADES) if etiqueta(u) == pagina),
            None,
        )
        if indice is None:
            _portada()
        else:
            mostrar(UNIDADES[indice], repo)


if __name__ == "__main__":
    main()