# 📐 CalculusLab · Cálculo 1

Aplicación interactiva para estudiantes universitarios que preparan
**Análisis Matemático I (Cálculo 1)**, construida con **Python + Streamlit**,
motor simbólico **SymPy** y gráficos interactivos **Plotly**. Sigue el
programa oficial de la UTN FRM (2026): 7 unidades, cada una con **Teoría**,
**Intuición visual** y **Ejercicios con corrección algebraica**.

## Instalación y ejecución

1. Crea un entorno virtual (recomendado):

   ```bash
   python -m venv .venv        # Windows
   .venv\Scripts\activate      # PowerShell
   ```

2. Instala dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Ejecuta la aplicación:

   ```bash
   python -m streamlit run app.py
   ```

   Se abrirá la app en `http://localhost:8501`.

## Las 7 unidades

| Unidad | Contenido central |
|---|---|
| **UN 1 · Funciones** | Dominio e imagen, paridad, inversas, composición, funciones especiales, inecuaciones. |
| **UN 2 · Límite y Continuidad** | Definición formal, laterales, infinitos, asíntotas, teoremas, clasificación de discontinuidades. |
| **UN 3 · Derivada y Diferencial** | Cociente incremental, reglas y cadena, derivada en un punto, rectas tangentes, derivadas de orden superior. |
| **UN 4 · Aplicaciones del Cálculo Diferencial** | Rolle y TVM, monotonía y extremos, concavidad y puntos de inflexión, L'Hôpital, Taylor, optimización. |
| **UN 5 · Integral** | Primitivas e inmediatas, sustitución, por partes, integral definida, TFC, áreas entre curvas. |
| **UN 6 · Aplicaciones del Cálculo Integral** | Volúmenes (discos y arandelas), longitudes de arco, trabajo, integrales impropias. |
| **UN 7 · Sucesiones y Series** | Convergencia de sucesiones, series y criterios, series de potencias, radio de convergencia, Taylor/Maclaurin. |

Cada unidad organiza su contenido en **Teoría y Demostraciones**, **Intuición
Visual** (gráficos interactivos con Plotly) y **Ejercicios Prácticos**
(evaluación algebraica con SymPy).

## Uso

- Identifícate con **nombre** en la barra lateral para que tu progreso quede
  guardado (SQLite local).
- Cada ejercicio registra los intentos automáticamente con su **unidad** y
  **tema**; el resumen se ve en la barra lateral.

## Arquitectura

```
app.py                  # Punto de entrada + navegación (radio lateral)
calculo/
├── matematicas.py      # Motor simbólico SymPy: parseo, límites, derivadas, integrales, series
├── graficos.py         # Figuras y estilo de Plotly
├── ui.py               # Widgets de ejercicios con evaluación algebraica y registro de progreso
├── persistencia.py     # Repositorio de progreso: en memoria / SQLite / PostgreSQL
└── unidades/
    ├── u1_funciones.py      ... u7_series.py   # Las 7 unidades (teoria/intuicion/ejercicios)
    └── __init__.py          # Registro de unidades, etiquetas y página común
```

El código es **modular y orientado a interfaz**: la UI depende de la clase
abstracta `RepositorioProgreso`, nunca de un motor concreto.

## Persistencia y PostgreSQL (a futuro)

Por defecto la app guarda el progreso en **SQLite** (`progreso.db`, junto al
proyecto, no requiere configuración). Para conectar una base de datos
**PostgreSQL** (por ejemplo en un contenedor Docker):

```bash
# 1) Levanta PostgreSQL
docker run --name calc-postgres -e POSTGRES_PASSWORD=admin \
           -e POSTGRES_DB=calculus -p 5432:5432 -d postgres:16

# 2) Instala el driver opcional
pip install psycopg2-binary

# 3) Indica la URL de conexión y arranca
$env:POSTGRES_URL = "postgresql://postgres:admin@localhost:5432/calculus"
python -m streamlit run app.py
```

Las tablas (`estudiantes`, `intentos`) se crean automáticamente. Cambiar de
motor no altera la interfaz: basta definir `POSTGRES_URL`.

## Notas para estudiantes

- El **parseo** acepta notación natural: `sin(3x)`, `e^(2x)`, `sqrt(x)`, `ln(x)`,
  `x^2`, `cos(2x)*x`, etc. (el `*` implícito es opcional).
- La **solución** se muestra como *equivalentes*, no como cadenas idénticas:
  `2*x`, `2x` y `x+x` cuentan como lo mismo.
- Identifícate con nombre en la barra lateral para guardar tu historial y el
  progreso del 100% del programa.# Default-Project
