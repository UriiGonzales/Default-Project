"""Capa de persistencia del progreso de los estudiantes.

Diseño orientado a futuro:
    * `RepositorioProgreso` es la interfaz abstracta. Toda la UI depende de
      esta interfaz, nunca de un motor concreto.
    * `RepositorioSQLite` (por defecto) guarda datos en un archivo local con
      la biblioteca estándar `sqlite3`; permite probar la aplicación sin
      infraestructura.
    * `RepositorioPostgres` implementa la misma interfaz sobre PostgreSQL
      (con `psycopg2`), listo para desplegar en un contenedor Docker.

Para activar PostgreSQL basta con que exista una variable de entorno
``POSTGRES_URL`` (p. ej. ``postgresql://usuario:clave@localhost:5432/calculus``)
y tener instalado ``psycopg2-binary``. De lo contrario la aplicación usa
SQLite de forma transparente.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from .matematicas import Ejercicio


# Orto schema compartido por ambos motores.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS estudiantes (
    id        TEXT PRIMARY KEY,
    nombre    TEXT NOT NULL,
    email     TEXT UNIQUE,
    creado_el TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS intentos (
    id             TEXT PRIMARY KEY,
    estudiante_id  TEXT NOT NULL REFERENCES estudiantes(id),
    enunciado      TEXT NOT NULL,
    respuesta      TEXT NOT NULL,
    correcta       TEXT NOT NULL,
    es_correcta    INTEGER NOT NULL,
    dificultad     TEXT,
    resuelto_el    TEXT NOT NULL
);
"""


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expr_a_texto(valor) -> str:
    """Serializa una expresión SymPy (o cadena) como texto legible."""
    if isinstance(valor, str):
        return valor
    try:
        import sympy as sp

        if valor is sp.oo:
            return "inf"
        if valor == -sp.oo:
            return "-inf"
        return str(valor)
    except Exception:
        return str(valor)


class RepositorioProgreso(ABC):
    """Interfaz que toda implementación de almacenamiento debe satisfacer."""

    @abstractmethod
    def crear_estudiante(self, nombre: str, email: str) -> str:
        """Registra un estudiante y devuelve su id."""

    @abstractmethod
    def obtener_estudiante(self, email: str) -> dict | None:
        """Busca un estudiante por email; None si no existe."""

    @abstractmethod
    def guardar_intento(
        self,
        estudiante_id: str,
        ejercicio: Ejercicio,
        respuesta: str,
        es_correcta: bool,
    ) -> None:
        """Registra un intento de resolución."""

    @abstractmethod
    def guardar_registro(
        self,
        estudiante_id: str,
        tema: str,
        enunciado: str,
        respuesta: str,
        correcta: str,
        es_correcta: bool,
        dificultad: str = "general",
    ) -> None:
        """Registra un intento genérico (de cualquier unidad)."""

    @abstractmethod
    def progreso_por_estudiante(self, estudiante_id: str) -> list[dict]:
        """Devuelve los intentos de un estudiante, del más reciente al más viejo."""

    @abstractmethod
    def cerrar(self) -> None:
        """Libera recursos (si aplica)."""


# ---------------------------------------------------------------------------
# Implementación predeterminada: SQLite (archivo local)
# ---------------------------------------------------------------------------

class RepositorioSQLite(RepositorioProgreso):
    def __init__(self, ruta: str = "progreso.db") -> None:
        self.ruta = ruta
        self._conn = sqlite3.connect(ruta)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def crear_estudiante(self, nombre: str, email: str) -> str:
        if not nombre.strip() or not email.strip():
            raise ValueError("Nombre y email son obligatorios.")
        existe = self.obtener_estudiante(email)
        if existe:
            return existe["id"]
        id_estudiante = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO estudiantes (id, nombre, email, creado_el) VALUES (?, ?, ?, ?)",
            (id_estudiante, nombre.strip(), email.strip(), _ahora()),
        )
        self._conn.commit()
        return id_estudiante

    def obtener_estudiante(self, email: str) -> dict | None:
        fila = self._conn.execute(
            "SELECT * FROM estudiantes WHERE email = ?", (email.strip(),)
        ).fetchone()
        return dict(fila) if fila else None

    def guardar_intento(
        self,
        estudiante_id: str,
        ejercicio: Ejercicio,
        respuesta: str,
        es_correcta: bool,
    ) -> None:
        detalles = ejercicio.a_diccionario()
        from .matematicas import nivel_dificultad

        self.guardar_registro(
            estudiante_id,
            tema="derivada",
            enunciado=detalles["enunciado"],
            respuesta=respuesta,
            correcta=ejercicio.derivada,
            es_correcta=es_correcta,
            dificultad=nivel_dificultad(ejercicio.funcion),
        )

    def guardar_registro(
        self,
        estudiante_id: str,
        tema: str,
        enunciado: str,
        respuesta: str,
        correcta: str,
        es_correcta: bool,
        dificultad: str = "general",
    ) -> None:
        correcta_txt = _expr_a_texto(correcta)
        self._conn.execute(
            """INSERT INTO intentos
               (id, estudiante_id, enunciado, respuesta, correcta, es_correcta,
                dificultad, resuelto_el)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                estudiante_id,
                f"[{tema}] {enunciado}",
                respuesta,
                correcta_txt,
                1 if es_correcta else 0,
                dificultad,
                _ahora(),
            ),
        )
        self._conn.commit()

    def progreso_por_estudiante(self, estudiante_id: str) -> list[dict]:
        filas = self._conn.execute(
            "SELECT * FROM intentos WHERE estudiante_id = ? ORDER BY resuelto_el DESC",
            (estudiante_id,),
        ).fetchall()
        return [dict(f) for f in filas]

    def cerrar(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Implementación opcional: PostgreSQL (activa si POSTGRES_URL está definida)
# ---------------------------------------------------------------------------

class RepositorioPostgres(RepositorioProgreso):
    """Igual interfaz que sobre SQLite, pero con psycopg2 + PostgreSQL.

    Uso previsto (contenedor Docker):

        docker run --name calc-postgres -e POSTGRES_PASSWORD=admin \\
                   -e POSTGRES_DB=calculus -p 5432:5432 -d postgres:16

        POSTGRES_URL='postgresql://postgres:admin@localhost:5432/calculus' \\
            streamlit run app.py
    """

    def __init__(self, url: str) -> None:
        try:
            import psycopg2  # noqa: F401  # import opcional
        except ImportError as exc:
            raise RuntimeError(
                "Para usar PostgreSQL instala 'psycopg2-binary' (ver requirements.txt)."
            ) from exc

        import psycopg2

        self.url = url
        self._conn = psycopg2.connect(url)
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA)
        self._conn.commit()

    def crear_estudiante(self, nombre: str, email: str) -> str:
        if not nombre.strip() or not email.strip():
            raise ValueError("Nombre y email son obligatorios.")
        con = self._conn.cursor()
        con.execute("SELECT id FROM estudiantes WHERE email = %s", (email.strip(),))
        fila = con.fetchone()
        if fila:
            con.close()
            return fila[0]
        id_estudiante = str(uuid.uuid4())
        con.execute(
            "INSERT INTO estudiantes (id, nombre, email, creado_el) VALUES (%s, %s, %s, %s)",
            (id_estudiante, nombre.strip(), email.strip(), _ahora()),
        )
        self._conn.commit()
        con.close()
        return id_estudiante

    def obtener_estudiante(self, email: str) -> dict | None:
        con = self._conn.cursor()
        con.execute(
            "SELECT id, nombre, email, creado_el FROM estudiantes WHERE email = %s",
            (email.strip(),),
        )
        fila = con.fetchone()
        con.close()
        if not fila:
            return None
        id_, nombre, email_, creado = fila
        return {"id": id_, "nombre": nombre, "email": email_, "creado_el": creado}

    def guardar_intento(
        self,
        estudiante_id: str,
        ejercicio: Ejercicio,
        respuesta: str,
        es_correcta: bool,
    ) -> None:
        detalles = ejercicio.a_diccionario()
        from .matematicas import nivel_dificultad

        self.guardar_registro(
            estudiante_id,
            tema="derivada",
            enunciado=detalles["enunciado"],
            respuesta=respuesta,
            correcta=ejercicio.derivada,
            es_correcta=es_correcta,
            dificultad=nivel_dificultad(ejercicio.funcion),
        )

    def guardar_registro(
        self,
        estudiante_id: str,
        tema: str,
        enunciado: str,
        respuesta: str,
        correcta: str,
        es_correcta: bool,
        dificultad: str = "general",
    ) -> None:
        correcta_txt = _expr_a_texto(correcta)
        con = self._conn.cursor()
        con.execute(
            """INSERT INTO intentos
               (id, estudiante_id, enunciado, respuesta, correcta, es_correcta,
                dificultad, resuelto_el)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                str(uuid.uuid4()),
                estudiante_id,
                f"[{tema}] {enunciado}",
                respuesta,
                correcta_txt,
                1 if es_correcta else 0,
                dificultad,
                _ahora(),
            ),
        )
        self._conn.commit()
        con.close()

    def progreso_por_estudiante(self, estudiante_id: str) -> list[dict]:
        con = self._conn.cursor()
        con.execute(
            "SELECT * FROM intentos WHERE estudiante_id = %s ORDER BY resuelto_el DESC",
            (estudiante_id,),
        )
        columnas = [d[0] for d in con.description] if con.description else []
        filas = con.fetchall()
        con.close()
        return [dict(zip(columnas, f)) for f in filas]

    def cerrar(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Fábrica: elige el motor según el entorno
# ---------------------------------------------------------------------------

def crear_repositorio() -> RepositorioProgreso:
    """Devuelve un repositorio configurado.

    Prioridad:
        1. PostgreSQL si la variable de entorno ``POSTGRES_URL`` está definida.
        2. SQLite local (predeterminado, sin configuración adicional).
    """
    url = os.environ.get("POSTGRES_URL", "").strip()
    if url:
        return RepositorioPostgres(url)
    return RepositorioSQLite()