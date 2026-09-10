"""Conexión a SQL Server para el pipeline de scoring de morosidad.

El servidor, la base y las credenciales SIEMPRE se leen de variables de
entorno (DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD, cargadas desde el
archivo .env de la raíz del proyecto, excluido por .gitignore) — nunca se
escriben en config.yaml ni en el código. config.yaml solo aporta ajustes
no sensibles (driver ODBC, timeout).
"""

import os
import urllib.parse

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.config import RAIZ_PROYECTO, cargar_config

load_dotenv(RAIZ_PROYECTO / ".env")


class ConfiguracionIncompleta(Exception):
    """Falta una variable de entorno requerida para conectarse a la base."""


def _variable_entorno(nombre: str) -> str:
    valor = os.environ.get(nombre)
    if not valor:
        raise ConfiguracionIncompleta(
            f"Falta la variable de entorno {nombre}. Defínela en el archivo .env "
            f"de la raíz del proyecto (ver .env.example)."
        )
    return valor


def _connection_string() -> str:
    server = _variable_entorno("DB_SERVER")
    database = _variable_entorno("DB_NAME")
    user = _variable_entorno("DB_USER")
    password = _variable_entorno("DB_PASSWORD")
    driver = cargar_config().get("database", {}).get("driver", "ODBC Driver 17 for SQL Server")
    return (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
        f"UID={user};PWD={password};TrustServerCertificate=yes;"
    )


def get_engine() -> Engine:
    """Crea el engine de SQLAlchemy (vía pyodbc) para conectarse a SQL Server."""
    timeout = cargar_config().get("database", {}).get("timeout", 30)
    params = urllib.parse.quote_plus(_connection_string())
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}", connect_args={"timeout": timeout})


def read_sql(query: str, params: tuple | dict | None = None) -> pd.DataFrame:
    """Ejecuta `query` contra SQL Server y devuelve un pandas.DataFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


def ejecutar_multi_resultset(query: str, params: tuple | None = None) -> list[pd.DataFrame]:
    """Ejecuta una sentencia que devuelve varios *result sets* (p. ej. un
    procedimiento almacenado con más de un SELECT, como
    dbo.sp_ValidarEsquema o dbo.sp_ValidarCalidadDatos sin @SoloResumen)
    y los devuelve todos como una lista de DataFrames, en el mismo orden
    en que los emite SQL Server — pandas.read_sql() por sí solo solo
    captura el primero."""
    engine = get_engine()
    resultados: list[pd.DataFrame] = []
    with engine.connect() as conn:
        cursor_result = conn.exec_driver_sql(query, params or ())
        cursor = cursor_result.cursor
        while True:
            columnas = [c[0] for c in cursor.description]
            filas = cursor.fetchall()
            resultados.append(pd.DataFrame.from_records(filas, columns=columnas))
            if not cursor.nextset():
                break
        # Necesario incluso para procedimientos de solo lectura como
        # sp_ValidarEsquema (confirmar una transacción de solo lectura no
        # tiene efecto): sin este commit, un procedimiento que además
        # inserta (p. ej. sp_RegistrarEvaluacion) perdería la escritura al
        # cerrarse la conexión sin confirmar.
        conn.commit()
    return resultados


def ejecutar(query: str, params: tuple | None = None) -> None:
    """Ejecuta una sentencia de escritura (INSERT/UPDATE) suelta, sin
    pasar por un procedimiento almacenado, y confirma la transacción."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.exec_driver_sql(query, params or ())
        conn.commit()


def ejecutar_muchos(query: str, filas: list[tuple]) -> None:
    """Ejecuta la misma sentencia de escritura una vez por cada tupla de
    `filas` (p. ej. un INSERT de una fila por tramo del baseline) dentro
    de una única transacción. Los volúmenes aquí son de decenas de filas
    (tramos por variable), no miles, así que un bucle simple es
    suficiente y evita depender de detalles internos del driver."""
    if not filas:
        return
    engine = get_engine()
    with engine.connect() as conn:
        for fila in filas:
            conn.exec_driver_sql(query, fila)
        conn.commit()


def exec_sp(nombre: str, params: dict) -> None:
    """Ejecuta un procedimiento almacenado con parámetros nombrados.

    Ejemplo: exec_sp("dbo.sp_ActualizarScoringCliente", {"ClienteId": 1, "ScoreRiesgo": 415})
    """
    asignaciones = ", ".join(f"@{clave} = ?" for clave in params)
    engine = get_engine()
    with engine.connect() as conn:
        conn.exec_driver_sql(f"EXEC {nombre} {asignaciones}", tuple(params.values()))
        conn.commit()
