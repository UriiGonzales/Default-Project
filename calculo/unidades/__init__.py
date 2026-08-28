"""Paquete `unidades`: una página por unidad temática del programa de AM I.

Cada módulo expone exactamente tres funciones:
    * ``teoria()``      — Teoría y Demostraciones (rigor).
    * ``intuicion()``   — Intuición Visual (interactiva).
    * ``ejercicios()``  — Ejercicios Prácticos (con evaluación algebraica).

`UNIDADES` es el registro que permite a la UI recorrer el programa completo
ordenadamente y renderizar cada unidad con sus tres pestañas.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from . import (
    u1_funciones,
    u2_limite,
    u3_derivada,
    u4_diferencial,
    u5_integral,
    u6_integral_apps,
    u7_series,
)

_TAB_TEORIA = "📖 Teoría y Demostraciones"
_TAB_INTUICION = "👀 Intuición Visual"
_TAB_EJERCICIOS = "✏️ Ejercicios Prácticos"


@dataclass(frozen=True)
class Unidad:
    numero: int
    titulo: str       # "Funciones"
    subtitulo: str    # tema segundo, ejemplo: "Límite y Continuidad"
    horas: int        # horas cátedra según el programa
    modulo: ModuleType


UNIDADES: tuple[Unidad, ...] = (
    Unidad(1, "Funciones", "Números reales, inecuaciones, dominio, funciones especiales", 10, u1_funciones),
    Unidad(2, "Límite y Continuidad", "ε-δ, asíntotas, discontinuidades, teoremas de continuidad", 20, u2_limite),
    Unidad(3, "Derivada y Diferencial", "Recta tangente, reglas de derivación, diferencial", 20, u3_derivada),
    Unidad(4, "Aplicaciones del Cálculo Diferencial", "Crecimiento, extremos, Taylor, L'Hôpital, optimización", 20, u4_diferencial),
    Unidad(5, "Integral", "Primitivas, sumas de Riemann, TFC, métodos de integración", 20, u5_integral),
    Unidad(6, "Aplicaciones del Cálculo Integral", "Área, longitud de arco, volumen, integrales impropias", 10, u6_integral_apps),
    Unidad(7, "Sucesiones y Series", "Convergencia, criterios, series de potencias, Taylor/MacLaurin", 20, u7_series),
)


def etiqueta(unidad: Unidad) -> str:
    return f"UN {unidad.numero} · {unidad.titulo}"


def mostrar(unidad: Unidad, repo=None) -> None:
    """Renderiza una unidad completa con sus tres pestañas.

    `repo` (opcional) habilita el registro automático de intentos de los
    ejercicios, junto con la identidad guardada en `st.session_state`.
    """
    import streamlit as st

    st.title(f"**Unidad {unidad.numero}:** {unidad.titulo}")
    st.caption(f"{unidad.subtitulo} · ~{unidad.horas} h cátedra")

    tab_teoria, tab_intui, tab_ejer = st.tabs(
        [_TAB_TEORIA, _TAB_INTUICION, _TAB_EJERCICIOS]
    )
    with tab_teoria:
        unidad.modulo.teoria()
    with tab_intui:
        unidad.modulo.intuicion()
    with tab_ejer:
        unidad.modulo.ejercicios(repo)


def armar_navegacion() -> list[str]:
    """Etiquetas para el menú de páginas (Stepper / radio de la sidebar)."""
    base = ["Inicio"]
    return base + [etiqueta(u) for u in UNIDADES]


__all__ = ["Unidad", "UNIDADES", "etiqueta", "mostrar", "armar_navegacion"]