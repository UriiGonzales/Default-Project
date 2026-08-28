"""Graficador genérico sobre Plotly.

Evita repetir lógica en cada unidad: construye una figura con varias curvas
simbólicas, máscara de valores no finitos (asíntotas, dominios), punto de
marcado opcional y leyenda unificada.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .matematicas import x, funcion_numerica


def figura_funciones(
    funciones: list[tuple[str, "Expr"]],
    xmin: float,
    xmax: float,
    titulo: str = "",
    marcadores: list[tuple[float, float] | tuple[float, float, str]] | None = None,
    vlineas: list[float] | None = None,
    puntos: int = 600,
    altura: int | None = None,
) -> go.Figure:
    """Construye una figura con las curvas dadas.

    `funciones` es una lista de (nombre, expresión SymPy). Los valores no
    finitos de cada curva se enmascaran.
    """
    xs = np.linspace(xmin, xmax, puntos)
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#17becf"]

    fig = go.Figure()
    for i, (nombre, expr) in enumerate(funciones):
        try:
            f_num = funcion_numerica(expr)
            ys = np.array([float(f_num(t)) for t in xs], dtype=float)
        except Exception:
            continue
        ys = np.where(np.isfinite(ys), ys, np.nan)
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", name=nombre,
                       line=dict(color=colores[i % len(colores)], width=2.5),
                       connectgaps=False)
        )

    if vlineas:
        for x0 in vlineas:
            fig.add_vline(x=x0, line_dash="dash", line_color="gray",
                          annotation_text=f"x={x0:.2f}")

    if marcadores:
        for marcador in marcadores:
            mx, my = marcador[0], marcador[1]
            color = marcador[2] if len(marcador) > 2 else "red"
            fig.add_trace(
                go.Scatter(x=[mx], y=[my], mode="markers",
                           marker=dict(size=12, color=color, symbol="circle-open"),
                           showlegend=False)
            )

    fig.update_layout(
        title=titulo or None,
        xaxis_title="x", yaxis_title="y",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50 if titulo else 30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=altura,
    )
    return fig


def estilo_numerico(valor: float, decimales: int = 4) -> str:
    """Formatea un número para mostrarlo en un data frame o caption."""
    return f"{valor:.{decimales}f}"