"""Verificación de esquema — Control 1 de 3 (documento "Controles de
Esquema, Calidad y Deriva de Datos"), previo incluso a la validación de
calidad: no tiene sentido validar reglas de negocio sobre una
estructura que cambió.

Compara `dbo.ContratoEsquema` (la expectativa declarada: qué columnas
debe tener cada objeto, su tipo y su nulabilidad) contra el catálogo
real del motor (`dbo.sp_ValidarEsquema`, ya construido en la base de
datos). A diferencia de la calidad de los datos (que solo bloquea ante
críticos) o la deriva (que nunca bloquea), el esquema **siempre**
detiene la ejecución ante cualquier hallazgo crítico: un tipo de dato
distinto o una columna obligatoria ausente romperían el código de
todas formas más adelante, con un error críptico a mitad de proceso —
detenerse aquí, antes de leer un solo registro, produce un diagnóstico
claro en su lugar.

Las advertencias (columna nueva no declarada, nulabilidad distinta) no
detienen el proceso, pero quedan registradas en el log."""

import logging

import pandas as pd

from src.config import RAIZ_PROYECTO
from src.db import ejecutar_multi_resultset

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.esquema")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "esquema.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())

VERSION_CONTRATO_DEFECTO = "1.0"


class EsquemaInvalidoError(Exception):
    """Al menos un hallazgo CRÍTICO al comparar el esquema real contra
    dbo.ContratoEsquema: el pipeline se detiene antes de leer datos."""


def obtener_validacion(version: str = VERSION_CONTRATO_DEFECTO) -> tuple[pd.DataFrame, dict]:
    """EXEC dbo.sp_ValidarEsquema @Version — dos result sets: el detalle
    de hallazgos por objeto/columna, y el conteo agregado de críticos y
    advertencias que consume la compuerta."""
    detalle, veredicto_df = ejecutar_multi_resultset("EXEC dbo.sp_ValidarEsquema @Version = ?", (version,))
    veredicto = veredicto_df.iloc[0].to_dict()
    veredicto["Criticos"] = int(veredicto["Criticos"] or 0)
    veredicto["Advertencias"] = int(veredicto["Advertencias"] or 0)
    return detalle, veredicto


def validar_esquema(version: str = VERSION_CONTRATO_DEFECTO) -> dict:
    """Compuerta obligatoria: primer paso de toda ejecución del pipeline,
    anterior a cualquier lectura de datos de negocio. Lanza
    EsquemaInvalidoError si hay al menos un hallazgo CRÍTICO."""
    detalle, veredicto = obtener_validacion(version)
    criticos = veredicto["Criticos"]
    advertencias = veredicto["Advertencias"]

    logger.info(
        "Validación de esquema (versión %s): %d críticos, %d advertencias sobre %d columnas revisadas.",
        version,
        criticos,
        advertencias,
        len(detalle),
    )

    if criticos > 0:
        for _, fila in detalle[detalle["Severidad"] == "CRITICO"].iterrows():
            logger.error(
                "Esquema CRÍTICO [%s.%s]: %s (esperado=%s, actual=%s).",
                fila["Objeto"],
                fila["Columna"],
                fila["Hallazgo"],
                fila.get("TipoDatoEsperado"),
                fila.get("TipoDatoActual"),
            )
        mensaje = f"{criticos} hallazgo(s) crítico(s) de esquema contra el contrato versión {version}; ver logs/esquema.log."
        logger.error("Ejecución abortada por validación de esquema: %s", mensaje)
        raise EsquemaInvalidoError(mensaje)

    if advertencias > 0:
        for _, fila in detalle[detalle["Severidad"] == "ADVERTENCIA"].iterrows():
            logger.warning(
                "Esquema ADVERTENCIA [%s.%s]: %s.",
                fila["Objeto"],
                fila["Columna"],
                fila["Hallazgo"],
            )

    return veredicto
