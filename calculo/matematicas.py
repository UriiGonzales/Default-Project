"""Motor matemático basado en SymPy.

Responsabilidades:
    * Definir la variable simbólica principal.
    * Parsear expresiones escritas por el usuario (texto libre -> SymPy).
    * Calcular derivadas simbólicas.
    * Comparar algebraicamente dos expresiones (equivalencia matemática,
      no igualdad de cadenas).
    * Construir la ecuación de la recta tangente y funciones numéricas
      (lambdify) para visualización.

Cualquier parte de la aplicación que necesite matemática simbólica debe
importar de este módulo; así el resto del código se mantiene independiente
de los detalles de SymPy (separación de responsabilidades).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import sympy as sp
from sympy.calculus.util import continuous_domain, singularities

# ---------------------------------------------------------------------------
# Símbolos y configuraciones compartidas
# ---------------------------------------------------------------------------

#: Variable independiente usada por toda la aplicación.
x = sp.Symbol("x", real=True)

#: Índice natural usado por sucesiones y series (UNIDAD 7).
n = sp.Symbol("n", integer=True, positive=True)

#: Funciones admitidas al parsear texto del usuario.
_FUNCIONES = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sqrt": sp.sqrt,
    "ln": sp.ln,
    "log": sp.ln,
    "exp": sp.exp,
    "abs": sp.Abs,
}

#: Constantes simbólicas disponibles al parsear.
_CONSTANTES = {
    "e": sp.E,
    "pi": sp.pi,
}


class ErrorMatematico(ValueError):
    """Excepción lanzada cuando una expresión no puede interpretarse."""


# ---------------------------------------------------------------------------
# Parseo de entrada de usuario
# ---------------------------------------------------------------------------

def parsear_expresion(texto: str) -> sp.Expr:
    """Convierte texto libre del usuario en una expresión SymPy.

    El parseo es *tolerante*: acepta ``2x+1``, ``2*x + 1``, ``x^2``, ``e^x``,
    ``sin(2x)``, ``sqrt(x)``, ``ln(x)``, etc. Lanza `ErrorMatematico` si la
    entrada no es interpretable, de modo que la UI pueda mostrar un mensaje
    amigable en lugar de quebrar.
    """
    if not isinstance(texto, str) or not texto.strip():
        raise ErrorMatematico("La expresión está vacía.")

    limpio = texto.strip().lower().replace(" ", "")
    limpio = limpio.replace("^", "**")

    # Convierte "2x" en "2*x" (regla de producto implícito). Recorremos la
    # cadena insertando '*' entre un token alfanumérico y una variable/función.
    limpio = _insertar_productos(limpio)

    # IMPORTANTE: incluimos la variable global `x` (con sus supuestos) en el
    # contexto de parseo para que el símbolo parseado sea el MISMO objeto que
    # usa `derivar`, `subs`, etc. De lo contrario `diff` devolvería 0.
    contexto = {**_CONSTANTES, **{k: v for k, v in _FUNCIONES.items()}, "x": x}

    try:
        expr = sp.sympify(limpio, locals=contexto)
    except (sp.SympifyError, SyntaxError, TypeError, AttributeError) as exc:
        raise ErrorMatematico("No pude interpretar la expresión. Usa notación como '2*x + 1'.") from exc

    if not isinstance(expr, sp.Expr):
        raise ErrorMatematico("La entrada no es una expresión matemática válida.")

    # Validación: la expresión debe depender solo de la variable x (o ser una
    # constante). Detecta entradas sin sentido como "estonoes" que SymPy
    # interpretaría como otro símbolo.
    variables_extra = expr.free_symbols - {x}
    if variables_extra:
        nombres = ", ".join(sorted(str(s) for s in variables_extra))
        raise ErrorMatematico(f"La expresión usa símbolos desconocidos ({nombres}); responde como función de 'x'.")

    return expr


def _insertar_productos(texto: str) -> str:
    """Inserta '*' implícito respetando los nombres de funciones conocidas.

    Transforma ``2x`` -> ``2*x``, ``sin(2x)`` -> ``sin(2*x)``,
    ``(x+1)(x-1)`` -> ``(x+1)*(x-1)`` — pero **no** ``sin(x)`` -> ``sin*(x)``.

    Un escáner de caracteres es más robusto que las regex para distinguir
    una llamada de función (``sin(``) de una multiplicación implícita
    (``2(``, ``)(``, ``x(``).
    """
    nombres_funcion = dict.fromkeys(_FUNCIONES.keys(), True)

    def es_parte_ident(ch: str) -> bool:
        return ch.isalpha() or ch == "_"

    def token_precedente(out: str) -> str:
        """Devuelve la palabra alfanumérica que termina en el último char de `out`."""
        j = len(out) - 1
        while j >= 0 and (out[j].isalpha() or out[j].isdigit() or out[j] == "_"):
            j -= 1
        return out[j + 1 :].lower()

    out: list[str] = []
    i, n = 0, len(texto)

    while i < n:
        ch = texto[i]

        if ch == "(":
            if out:
                ultimo = out[-1]
                if ultimo in ")]" or (ultimo.isdigit()):
                    out.append("*")
                elif ultimo.isalpha():
                    if token_precedente("".join(out)) not in nombres_funcion:
                        out.append("*")
            out.append("(")

        elif ch.isalpha():
            j = i
            while j < n and es_parte_ident(texto[j]):
                j += 1
            ident = texto[i:j].lower()
            if out and (out[-1].isdigit() or out[-1] == ")" or out[-1].isalpha()):
                out.append("*")
            out.append(ident)
            i = j
            continue

        elif ch.isdigit():
            if out and (out[-1].isalpha() or out[-1] == ")"):
                out.append("*")
            out.append(ch)

        elif ch == ")":
            out.append(ch)

        else:  # operadores, espacios ya eliminados
            out.append(ch)

        i += 1

    return "".join(out)


# ---------------------------------------------------------------------------
# Álgebra simbólica
# ---------------------------------------------------------------------------

def derivar(expr: sp.Expr, orden: int = 1) -> sp.Expr:
    """Deriva `expr` respecto de la variable global `x`."""
    return sp.diff(expr, x, orden)


def simplificar(expr: sp.Expr) -> sp.Expr:
    """Simplificación estándar (algebraica y trigonométrica)."""
    expr = sp.simplify(expr)
    return sp.trigsimp(sp.expand(expr))


def son_equivalentes(respuesta: sp.Expr, correcta: sp.Expr) -> bool:
    """Devuelve True si `respuesta` y `correcta` son algebraicamente iguales.

    La comparación es matemática: restamos ambas expresiones y verificamos
    que su forma canónica simplificada sea idénticamente cero. Así,
    ``2*x`` y ``x + x`` (o ``2x`` y ``2/x**(-1)``) cuentan como
    equivalentes, mientras que ``x + 1`` no equivale a ``x + 2``.
    """
    if respuesta is None or correcta is None:
        return False
    try:
        diferencia = simplificar(respuesta - correcta)
        return sp.S(diferencia) == 0
    except Exception:
        return False


def recta_tangente(expr: sp.Expr, x0: float) -> sp.Expr:
    """Ecuación de la recta tangente a `expr` en ``x = x0``.

    ``y = f(x0) + f'(x0) * (x - x0)``
    """
    m = sp.diff(expr, x).subs(x, x0)
    return sp.expand(expr.subs(x, x0) + m * (x - x0))


def funcion_numerica(expr: sp.Expr) -> Callable[[float], float]:
    """Compila `expr` a una función numérica (numpy/scipy) para graficar."""
    return sp.lambdify(x, expr, modules="numpy")


# ---------------------------------------------------------------------------
# Banco de ejercicios
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ejercicio:
    """Un ejercicio de derivación generado aleatoriamente."""

    enunciado: str          # Función mostrada al usuario, ej. "f(x) = sin(3x)"
    funcion: sp.Expr        # Expresión SymPy asociada
    derivada: sp.Expr       # Respuesta correcta (derivada simbólica)

    # --- Compatibilidad con persistencia --------------------------------
    def a_diccionario(self) -> dict:
        return {
            "enunciado": self.enunciado,
            "funcion_str": sp.srepr(self.funcion),
            "derivada_str": sp.srepr(self.derivada),
        }

    @staticmethod
    def desde_diccionario(datos: dict) -> "Ejercicio":
        return Ejercicio(
            enunciado=datos["enunciado"],
            funcion=sp.sympify(datos["funcion_str"]),
            derivada=sp.sympify(datos["derivada_str"]),
        )


def _banco_de_funciones() -> dict[str, sp.Expr]:
    """Catálogo de funciones diferenciables de dificultad progresiva."""
    return {
        "f(x) = x^2 - 4x + 3":        x ** 2 - 4 * x + 3,
        "f(x) = x^3 - 2x^2 + x - 5":  x ** 3 - 2 * x ** 2 + x - 5,
        "f(x) = 3x^5 - x + 2":        3 * x ** 5 - x + 2,
        "f(x) = x^4/4 - x^2":         x ** 4 / 4 - x ** 2,
        "f(x) = sqrt(x)":             sp.sqrt(x),
        "f(x) = 1/x":                 1 / x,
        "f(x) = 1/x^2":               1 / x ** 2,
        "f(x) = x^(1/3) + x":         x ** (sp.Rational(1, 3)) + x,
        "f(x) = sin(x)":              sp.sin(x),
        "f(x) = cos(x)":              sp.cos(x),
        "f(x) = sin(3x)":             sp.sin(3 * x),
        "f(x) = cos(2x)":             sp.cos(2 * x),
        "f(x) = tan(x)":              sp.tan(x),
        "f(x) = e^x":                 sp.exp(x),
        "f(x) = e^(2x)":              sp.exp(2 * x),
        "f(x) = ln(x)":               sp.ln(x),
        "f(x) = ln(2x + 1)":          sp.ln(2 * x + 1),
        "f(x) = x * sin(x)":          x * sp.sin(x),
        "f(x) = x^2 * e^x":           x ** 2 * sp.exp(x),
        "f(x) = sin(x) * cos(x)":     sp.sin(x) * sp.cos(x),
        "f(x) = x / (x + 1)":         x / (x + 1),
        "f(x) = x^2 / (x - 2)":       x ** 2 / (x - 2),
    }


def generar_ejercicio(aleatorio: random.Random | None = None) -> Ejercicio:
    """Genera un ejercicio aleatorio de derivación y su solución."""
    if aleatorio is None:
        aleatorio = random.Random()
    nombre, funcion = aleatorio.choice(list(_banco_de_funciones().items()))
    return Ejercicio(enunciado=nombre, funcion=funcion, derivada=derivar(funcion))


# ---------------------------------------------------------------------------
# Métricas de entrenamiento (opcional, para el dashboard de progreso)
# ---------------------------------------------------------------------------

def nivel_dificultad(expr: sp.Expr) -> str:
    """Clasificación heurística de dificultad de un ejercicio."""
    if expr.has(sp.sin, sp.cos, sp.tan, sp.exp, sp.ln, sp.sqrt, sp.Abs):
        if expr.has(sp.Mul) and len(expr.atoms(sp.Mul)) >= 4:
            return "alta"
        return "media"
    return "baja"


# ===========================================================================
# UNIDAD 2 · LÍMITE Y CONTINUIDAD
# ===========================================================================

#: Infinitésimos equivalentes (x → 0) usados en teoría y ejercicios.
INFINITESIMOS_EQUIVALENTES = {
    r"\sin x": r"x",
    r"\tan x": r"x",
    r"\arcsin x": r"x",
    r"\arctan x": r"x",
    r"\ln(1+x)": r"x",
    r"e^x - 1": r"x",
    r"(1+x)^k - 1": r"k\,x",
    r"1-\cos x": r"\frac{x^2}{2}",
    r"\sqrt{1+x}-1": r"\frac{x}{2}",
}


def calcular_limite(expr: sp.Expr, x0: sp.Expr | float, direccion: str | None = None) -> sp.Expr:
    """Límite de `expr` cuando x → x0. `direccion` puede ser '+' (derecha) o '-' (izquierda)."""
    if direccion is None:
        return sp.limit(expr, x, x0)
    return sp.limit(expr, x, x0, dir=direccion)


def asintotas_de(expr: sp.Expr) -> dict:
    """Determina asíntotas horizontales, verticales y oblicuas de `expr`.

    Devuelve: ``{"verticales": [x0, ...], "horizontal": valor|None, "oblicua": mx+b|None}``
    """
    out: dict = {"verticales": [], "horizontal": None, "oblicua": None}

    # Horizontales en ±∞ (la primera finita encontrada).
    for signo in (sp.oo, -sp.oo):
        try:
            lim = sp.limit(expr, x, signo)
            if lim.is_finite:
                out["horizontal"] = sp.simplify(lim)
                break
        except Exception:
            continue

    # Oblicua (misma rama derecha: y = mx + b).
    try:
        m = sp.limit(expr / x, x, sp.oo)
        if m.is_finite and m != 0:
            b = sp.limit(expr - m * x, x, sp.oo)
            if b.is_finite:
                out["oblicua"] = sp.simplify(m * x + b)
    except Exception:
        pass

    # Verticales: singularidades donde la función diverge.
    try:
        for p in singularities(expr, x):
            try:
                lim = sp.limit(expr, x, p)
                if lim in (sp.oo, -sp.oo) or lim.has(sp.oo, -sp.oo):
                    out["verticales"].append(sp.simplify(p))
            except Exception:
                continue
    except Exception:
        pass
    return out


def dominio_de(expr: sp.Expr) -> sp.Set:
    """Dominio natural de f: mayor subconjunto real donde f está definida y es continua."""
    return continuous_domain(expr, x, sp.S.Reals)


def puntos_de_discontinuidad(expr: sp.Expr) -> list:
    """Puntos del dominio real donde la función deja de ser continua."""
    try:
        return list(singularities(expr, x))
    except Exception:
        return []


def clasificar_discontinuidad(expr: sp.Expr, p: sp.Expr | float) -> str:
    """Clasifica la discontinuidad de `expr` en `p`.

    Devuelve una de: 'continua', 'evitable', 'salto finito' o 'esencial (infinita)'.
    """
    L_iz = L_de = None
    try:
        L_iz = sp.limit(expr, x, p, dir="-")
    except Exception:
        pass
    try:
        L_de = sp.limit(expr, x, p, dir="+")
    except Exception:
        pass

    diverge = lambda L: L is None or (isinstance(L, sp.Expr) and L.has(sp.oo, -sp.oo, sp.zoo))

    # Caso límite general (existe y es finito).
    if L_iz is not None and L_de is not None and L_iz.equals(L_de):
        try:
            f_p = sp.simplify(expr.subs(x, p))
            if f_p.is_finite and f_p.equals(L_iz):
                return "continua"
            return "evitable"
        except Exception:
            return "evitable"

    if diverge(L_iz) or diverge(L_de):
        return "esencial (infinita)"
    if L_iz is not None and L_de is not None:
        return "salto finito"
    return "esencial"


def formato_conjunto(S: sp.Set) -> str:
    """Serializa un conjunto de reales (Interval/Union/FiniteSet/...) a intervalos LaTeX."""

    def finito(v) -> str:
        if v is sp.oo:
            return r"\infty"
        if v == -sp.oo:
            return r"-\infty"
        return sp.latex(v)

    def trozo(i) -> str:
        iz, de = i.start, i.end
        ciz = "[" if (iz is not sp.oo and not i.left_open) else "("
        cde = "]" if (de is not -sp.oo and not i.right_open) else ")"
        return f"{ciz}{finito(iz)},{finito(de)}{cde}"

    if S is sp.S.EmptySet:
        return r"\varnothing"
    if S is sp.S.Reals or S is sp.S.UniversalSet:
        return r"(-\infty,\infty)"
    if isinstance(S, sp.FiniteSet):
        return "{" + ", ".join(sp.latex(v) for v in S) + "}"
    if isinstance(S, sp.Interval):
        return trozo(S)
    if isinstance(S, sp.Union):
        return r"\;\cup\;".join(trozo(i) for i in S.args)
    return sp.latex(S)


# ===========================================================================
# UNIDAD 4 · POLINOMIO DE TAYLOR
# ===========================================================================

def taylor_de(expr: sp.Expr, x0: sp.Expr | float, orden: int) -> sp.Expr:
    """Polinomio de Taylor de `expr` de grado `orden` alrededor de x = x0."""
    return sp.series(expr, x, x0, orden + 1).removeO()


# ===========================================================================
# UNIDAD 5 · INTEGRAL
# ===========================================================================

def primitiva(f: sp.Expr) -> sp.Expr:
    """Una primitiva (integral indefinida) de f."""
    return sp.integrate(f, x)


def es_primitiva(respuesta: sp.Expr, f: sp.Expr) -> bool:
    """Indica si `respuesta` es una primitiva de `f` (es decir, F' = f)."""
    try:
        return son_equivalentes(derivar(respuesta), f)
    except Exception:
        return False


def integral_definida(f: sp.Expr, a: float, b: float) -> sp.Expr | float:
    """Integral definida ∫_a^b f, simbólica con caída numérica."""
    try:
        valor = sp.integrate(f, (x, a, b))
        return sp.nsimplify(valor) if valor.has(sp.pi, sp.exp, sp.log, sp.sqrt) else sp.simplify(valor)
    except Exception:
        import numpy as np

        f_num = funcion_numerica(f)
        xs = np.linspace(a, b, 2001)
        ys = np.array([float(f_num(t)) for t in xs], dtype=float)
        return float(np.trapezoid(ys, xs))


def area_entre(f: sp.Expr, g: sp.Expr, a: float, b: float) -> sp.Expr | float:
    """Área entre las curvas f y g en [a, b]."""
    return integral_definida(sp.Abs(f - g), a, b)


def volumen_revolucion(f: sp.Expr, a: float, b: float) -> sp.Expr | float:
    """Volumen del sólido de revolución (método de discos) de f alrededor del eje x en [a,b]."""
    return integral_definida(sp.pi * f**2, a, b)


def longitud_arco(f: sp.Expr, a: float, b: float) -> sp.Expr | float:
    """Longitud de arco de la curva y=f(x) en [a,b]."""
    return integral_definida(sp.sqrt(1 + derivar(f) ** 2), a, b)


def integral_impropia(f: sp.Expr, a: float, tope: sp.Expr | float = sp.oo) -> dict:
    """Estudia ∫_a^tope f. Devuelve {'converge': bool, 'valor': ..., 'exacto': bool}."""
    try:
        valor = sp.integrate(f, (x, a, tope))
    except Exception:
        valor = None

    if isinstance(valor, sp.Expr):
        if valor.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
            return {"converge": False, "valor": sp.oo, "exacto": True}
        try:
            numero = float(valor.evalf())
            if not (numero == numero):  # NaN
                return {"converge": False, "valor": sp.oo, "exacto": True}
            return {"converge": True, "valor": sp.nsimplify(valor), "exacto": True}
        except Exception:
            return {"converge": False, "valor": sp.oo, "exacto": True}

    # Caída numérica (aproximada): usamos colas en escala geométrica.
    import numpy as np

    try:
        f_num = funcion_numerica(f)
        if tope is sp.oo:
            xs = np.geomspace(max(1.0, float(a) if float(a) > 0 else 1.0), 1e5, 100000)
        else:
            xs = np.linspace(float(a), float(tope), 100000)
        ys = np.array([float(f_num(t)) for t in xs], dtype=float)
        ys = np.nan_to_num(ys, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.trapezoid(ys, xs))
        return {"converge": bool(np.isfinite(total) and abs(total) < 1e10),
                "valor": total, "exacto": False}
    except Exception:
        return {"converge": None, "valor": None, "exacto": False}


def sumas_riemann(f: sp.Expr, a: float, b: float, N: int, modo: str = "media") -> float:
    """Suma de Riemann de f en [a,b] con N subintervalos.

    `modo` puede ser 'izquierda', 'derecha', 'media', 'sup' o 'inf'.
    """
    import numpy as np

    f_num = funcion_numerica(f)
    dx = (b - a) / N
    puntos = [a + dx * i for i in range(N + 1)]
    if modo == "izquierda":
        xs = puntos[:-1]
    elif modo == "derecha":
        xs = puntos[1:]
    elif modo == "media":
        xs = [(puntos[i] + puntos[i + 1]) / 2 for i in range(N)]
    elif modo in ("sup", "inf"):
        muestras = np.linspace(0, 1, 21)
        vals = []
        for i in range(N):
            fk = [float(f_num(puntos[i] + dx * s)) for s in muestras]
            vals.append(max(fk) if modo == "sup" else min(fk))
        return dx * sum(vals)
    else:
        raise ValueError(f"Modo desconocido: {modo}")
    return dx * sum(float(f_num(t)) for t in xs)


# ===========================================================================
# UNIDAD 7 · SUCESIONES Y SERIES
# ===========================================================================

def limite_sucesion(termino: sp.Expr) -> sp.Expr:
    """Límite de la sucesión a(n) cuando n → ∞."""
    return sp.limit(termino, n, sp.oo)


def suma_serie_geometrica(r: float, n0: int = 0) -> float | None:
    """Suma de Σ_{n=n0}^∞ r^n si |r|<1; None si diverge."""
    if abs(r) >= 1:
        return None
    return r**n0 / (1 - r)


def radio_convergencia(termo_potencia: sp.Expr) -> float | None:
    """Radio de convergencia de Σ a_n x^n a partir del término a_n.

    R = lim |a_n / a_{n+1}|  (criterio de la razón inverso).
    """
    try:
        a = sp.sympify(termo_potencia)
        razon = sp.simplify(sp.Abs(a / a.subs(n, n + 1)))
        R = sp.limit(razon, n, sp.oo)
        if R is sp.oo:
            return float("inf")
        if R == 0:
            return 0.0
        return float(R.evalf())
    except Exception:
        return None


def es_convergente_serie(termino_serie: sp.Expr) -> bool | None:
    """Clasifica Σ a_n por criterio de la razón (heurística simbólica)."""
    try:
        a = sp.sympify(termino_serie)
        razon = sp.simplify(sp.Abs((a.subs(n, n + 1)) / a))
        L = sp.limit(razon, n, sp.oo)
        if L.is_finite and L < 1:
            return True
        if L.is_finite and L > 1:
            return False
        return None  # criterio no concluyente
    except Exception:
        return None


def es_absolutamente_convergente(term_absoluto: sp.Expr) -> bool | None:
    """Convergencia absoluta de Σ |a_n| (para series alternadas) por la razón."""
    return es_convergente_serie(term_absoluto)


# ===========================================================================
# COMPARADORES DE RESPUESTAS (usan ∞ y tolerancia numérica)
# ===========================================================================

def parsear_evaluacion(texto: str) -> sp.Expr:
    """Como `parsear_expresion` pero además acepta infinitos ('oo', 'inf', '∞')."""
    t = texto.strip().lower().replace(" ", "").replace("∞", "oo")
    if t in ("oo", "+oo", "inf", "+inf", "infinito", "infty"):
        return sp.oo
    if t in ("-oo", "-inf", "-infinito", "-infty"):
        return -sp.oo
    return parsear_expresion(texto)


def son_equivalentes_numerico(a: sp.Expr, b: sp.Expr, tolerancia: float = 1e-3) -> bool:
    """Devuelve True si a y b representan el mismo número real (con tolerancia)."""
    try:
        va, vb = float(a.evalf()), float(b.evalf())
        return abs(va - vb) <= tolerancia
    except Exception:
        return False


def comparar_resultado(usuario: sp.Expr, correcto: sp.Expr, tolerancia: float = 1e-3) -> bool:
    """Compara la respuesta del usuario con la correcta.

    Primero equivalencia algebraica exacta; si el resultado es numérico y no
    coincide exactamente, permite una tolerancia (acepta p. ej. 0.3333 ≈ 1/3).
    """
    if usuario is None or correcto is None:
        return False
    try:
        if simplificar(usuario - correcto) == 0:
            return True
    except Exception:
        pass
    if correcto.is_finite is False or usuario.is_finite is False:
        return usuario == correcto
    return son_equivalentes_numerico(usuario, correcto, tolerancia)