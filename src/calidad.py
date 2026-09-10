"""Validación de calidad de datos, compuerta obligatoria antes de
entrenar el modelo (dbo.sp_ValidarCalidadDatos, ya construido en la base
de datos: 21 controles de Integridad, Consistencia, Completitud y
Representatividad, cada uno registrado en dbo.ValidacionCalidadDatos con
una Severidad OK/ADVERTENCIA/CRITICO).

Antes de extraer el dataset de entrenamiento se ejecuta primero
`EXEC dbo.sp_ValidarCalidadDatos @SoloResumen = 1` (la compuerta rápida:
el procedimiento igual registra los 21 controles completos en la tabla,
@SoloResumen solo omite el detalle en lo que devuelve). Si el conteo de
Críticos es mayor que cero, se aborta con el mensaje exacto del
Veredicto del procedimiento, sin llegar a extraer datos. Las
advertencias no detienen el proceso, pero quedan registradas en el log
— el "documento de trazabilidad" que el propio procedimiento exige
documentar cuando el veredicto es "APTO CON OBSERVACIONES"."""

import logging

import pandas as pd

from src.config import RAIZ_PROYECTO
from src.db import read_sql

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.calidad")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "calidad.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())


class CalidadDatosError(Exception):
    """Al menos un hallazgo CRÍTICO en dbo.sp_ValidarCalidadDatos: el
    pipeline se detiene antes de extraer datos. El mensaje es el
    Veredicto textual exacto que devuelve el procedimiento."""


def obtener_veredicto() -> dict:
    """EXEC dbo.sp_ValidarCalidadDatos @SoloResumen = 1 — la compuerta
    rápida: un único resultado con EjecucionId, Criticos, Advertencias,
    Conformes y Veredicto."""
    df = read_sql("EXEC dbo.sp_ValidarCalidadDatos @SoloResumen = 1")
    return df.iloc[0].to_dict()


def obtener_advertencias(ejecucion_id) -> pd.DataFrame:
    """Hallazgos con Severidad ADVERTENCIA de la ejecución dada.
    dbo.vw_UltimaValidacion ya apunta a la más reciente por Fecha, que es
    justo la que acaba de insertar obtener_veredicto()."""
    df = read_sql("SELECT * FROM dbo.vw_UltimaValidacion WHERE Severidad = 'ADVERTENCIA'")
    return df


def validar_antes_de_entrenar() -> dict:
    """Compuerta obligatoria antes de extraer datos de entrenamiento.

    Lanza CalidadDatosError con el Veredicto del procedimiento si hay al
    menos un hallazgo CRÍTICO — el pipeline se detiene sin llegar a
    extraer datos. Las advertencias no detienen el proceso, pero cada
    una queda registrada en logs/calidad.log con su código, descripción
    y valores observado/esperado."""
    veredicto = obtener_veredicto()
    criticos = int(veredicto["Criticos"])
    advertencias = int(veredicto["Advertencias"])

    logger.info(
        "Validación de calidad de datos (ejecución %s): %d críticos, %d advertencias, %d conformes. Veredicto: %s",
        veredicto["EjecucionId"],
        criticos,
        advertencias,
        veredicto["Conformes"],
        veredicto["Veredicto"],
    )

    if criticos > 0:
        logger.error("Entrenamiento abortado por calidad de datos: %s", veredicto["Veredicto"])
        raise CalidadDatosError(veredicto["Veredicto"])

    if advertencias > 0:
        for _, fila in obtener_advertencias(veredicto["EjecucionId"]).iterrows():
            logger.warning(
                "Advertencia de calidad de datos [%s/%s]: %s (%s registros; observado=%s, umbral=%s).",
                fila["Dimension"],
                fila["Codigo"],
                fila["Descripcion"],
                fila["RegistrosAfectados"],
                fila["ValorObservado"],
                fila["UmbralEsperado"],
            )

    return veredicto
