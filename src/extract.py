"""Extracción de los datasets de entrenamiento e inferencia desde SQL Server.

Dos fuentes distintas, a propósito (ver notas de "leakage" tras el paso 8.4):

- dbo.vw_FeaturesScoringCliente: una fila por CLIENTE, usada en INFERENCIA
  (score.py). Al scorear hoy, todo el historial de un cliente es
  legítimamente pasado; no hay fuga posible.
- dbo.vw_DatasetEntrenamientoFactura: una fila por FACTURA, usada en
  ENTRENAMIENTO (train.py). Cada variable se calcula point-in-time (solo
  con información anterior a la FechaEmision de esa factura), porque en
  entrenamiento la etiqueta si se derivaba de la misma ventana que las
  variables de la vista de cliente, y eso sí filtraba la etiqueta hacia
  las variables (IV artificialmente altísimo).
"""

import logging
from datetime import datetime

import pandas as pd

from src.config import RAIZ_PROYECTO, cargar_config
from src.db import read_sql

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.extract")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "extract.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())


class DatosInsuficientesError(Exception):
    """La vista devolvió menos filas de las necesarias para que un
    entrenamiento sea representativo."""


def _extraer_vista(vista: str, minimo_filas: int, unidad: str, indice: str) -> pd.DataFrame:
    df = read_sql(f"SELECT * FROM {vista}")
    fecha_extraccion = datetime.now()

    if len(df) < minimo_filas:
        mensaje = (
            f"La vista {vista} devolvió {len(df)} {unidad}, menos que el mínimo "
            f"requerido ({minimo_filas}). El entrenamiento no sería representativo."
        )
        logger.error(mensaje)
        raise DatosInsuficientesError(mensaje)

    logger.info("Extracción de %s: %d filas obtenidas el %s", vista, len(df), fecha_extraccion.isoformat())
    return df.set_index(indice)


def extraer_features() -> pd.DataFrame:
    """Consulta dbo.vw_FeaturesScoringCliente (una fila por cliente) y
    devuelve un DataFrame indexado por ClienteId. Uso: inferencia
    (score.py).

    Registra en el log el número de filas obtenidas y la fecha de
    extracción. Aborta con DatosInsuficientesError si la vista devuelve
    menos clientes que el mínimo configurado (config.extract.minimo_clientes,
    30 por defecto)."""
    cfg = cargar_config().get("extract", {})
    vista = cfg.get("vista_features", "dbo.vw_FeaturesScoringCliente")
    minimo = cfg.get("minimo_clientes", 30)
    return _extraer_vista(vista, minimo, "clientes", "ClienteId")


def extraer_dataset_entrenamiento() -> pd.DataFrame:
    """Consulta dbo.vw_DatasetEntrenamientoFactura (una fila por factura,
    variables point-in-time) y devuelve un DataFrame indexado por
    FacturaId. Uso: entrenamiento (train.py).

    Mismo criterio de registro/aborto que extraer_features(), sobre
    config.extract.minimo_facturas (30 por defecto)."""
    cfg = cargar_config().get("extract", {})
    vista = cfg.get("vista_entrenamiento", "dbo.vw_DatasetEntrenamientoFactura")
    minimo = cfg.get("minimo_facturas", 30)
    return _extraer_vista(vista, minimo, "facturas", "FacturaId")


def excluir_sin_historial(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta las filas con SinHistorialPago = 1: un cliente sin
    facturas previas ya vencidas no tiene comportamiento de pago que
    aprender. Esos casos los resuelve la regla de negocio NUEVO en la
    Capa 2 (src/rules.py), no el modelo estadístico."""
    if "SinHistorialPago" not in df.columns:
        return df
    descartadas = int(df["SinHistorialPago"].sum())
    if descartadas:
        logger.info(
            "Excluidas %d de %d filas por SinHistorialPago = 1 (sin comportamiento previo que aprender).",
            descartadas,
            len(df),
        )
    return df[df["SinHistorialPago"] == 0]
