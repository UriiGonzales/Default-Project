"""Paquete `calculo`: aplicación modular de Cálculo 1 construida con Streamlit + SymPy.

Módulos:
    matematicas   - Motor simbólico: parseo, límites, derivadas, integrales, series.
    graficos      - Figuras y estilo de Plotly.
    ui            - Widgets de ejercicios con evaluación algebraica y registro de progreso.
    persistencia  - Repositorio de progreso (SQLite por defecto, lista para PostgreSQL).
    unidades      - Las 7 unidades temáticas (teoria/intuicion/ejercicios por unidad).
"""

__version__ = "1.0.0"