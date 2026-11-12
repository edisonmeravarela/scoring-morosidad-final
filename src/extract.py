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


COLUMNAS_HISTORIAL = "ClienteId, FacturasResueltas, UmbralVigente, TieneHistorial"


def _origen_dataset(vista: str, fecha_corte: datetime | None = None) -> tuple[str, str, tuple | None]:
    """Consulta del conjunto de entrenamiento: (sql, nombre para el log,
    parámetros). Sin fecha de corte es la vista de siempre, que toma las
    facturas vencidas a la fecha del sistema (GETDATE()). Con fecha de
    corte se usa dbo.fn_DatasetEntrenamientoFactura(@FechaCorte), creada
    en FechaCorte_Parametrizacion.sql con la misma lógica que la vista
    (probada fila a fila: con la fecha actual devuelven exactamente lo
    mismo)."""
    if fecha_corte is None:
        return f"SELECT * FROM {vista}", vista, None
    return (
        "SELECT * FROM dbo.fn_DatasetEntrenamientoFactura(?)",
        f"dbo.fn_DatasetEntrenamientoFactura({fecha_corte.isoformat(sep=' ')})",
        (fecha_corte,),
    )


def _origen_historial(fecha_corte: datetime | None = None) -> tuple[str, tuple | None]:
    """Igual que _origen_dataset() para el historial evaluable:
    dbo.vw_HistorialClienteActual (fecha del sistema) o
    dbo.fn_HistorialClientesACorte(@FechaCorte)."""
    if fecha_corte is None:
        return f"SELECT {COLUMNAS_HISTORIAL} FROM dbo.vw_HistorialClienteActual", None
    return f"SELECT {COLUMNAS_HISTORIAL} FROM dbo.fn_HistorialClientesACorte(?)", (fecha_corte,)


def _extraer_vista(
    vista: str,
    minimo_filas: int,
    unidad: str,
    indice: str,
    sql: str | None = None,
    params: tuple | None = None,
) -> pd.DataFrame:
    consulta = sql or f"SELECT * FROM {vista}"
    df = read_sql(consulta, params) if params else read_sql(consulta)
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


def extraer_dataset_entrenamiento(fecha_corte: datetime | None = None) -> pd.DataFrame:
    """Consulta dbo.vw_DatasetEntrenamientoFactura (una fila por factura,
    variables point-in-time) y devuelve un DataFrame indexado por
    FacturaId. Uso: entrenamiento (train.py).

    `fecha_corte` (opcional) reconstruye el conjunto tal como estaba a
    esa fecha vía dbo.fn_DatasetEntrenamientoFactura; sin ella, el
    comportamiento es el de siempre.

    Mismo criterio de registro/aborto que extraer_features(), sobre
    config.extract.minimo_facturas (30 por defecto)."""
    cfg = cargar_config().get("extract", {})
    vista = cfg.get("vista_entrenamiento", "dbo.vw_DatasetEntrenamientoFactura")
    minimo = cfg.get("minimo_facturas", 30)
    sql, origen, params = _origen_dataset(vista, fecha_corte)
    return _extraer_vista(origen, minimo, "facturas", "FacturaId", sql=sql, params=params)


def agregar_historial_evaluable(df: pd.DataFrame, fecha_corte: datetime | None = None) -> pd.DataFrame:
    """Incorpora FacturasResueltas/UmbralVigente/TieneHistorial desde
    dbo.vw_HistorialClienteActual, indexado por ClienteId igual que
    extraer_features() — fuente única del criterio de "historial
    evaluable" (dbo.ParametroScoring.MIN_FACTURAS_HISTORIAL vía
    dbo.fn_HistorialCliente), compartida con el entrenamiento
    (SinHistorialPago de vw_DatasetEntrenamientoFactura) y la cola de
    reevaluación bajo demanda (TR_Abono_EncolarRescoreo). Antes de esto,
    src/rules.py contaba `NumFacturas` por su cuenta con su propio umbral
    (config.reglas.minimo_facturas_para_evaluar) sin exigir que el
    vencimiento ya hubiera transcurrido — un tercer criterio divergente
    de los otros dos. Paso aparte, igual que excluir_sin_historial(), no
    integrado en extraer_features(): mantiene esa función simple y
    permite probarlo por separado.

    `fecha_corte` (opcional) evalúa el historial a esa fecha vía
    dbo.fn_HistorialClientesACorte en vez de a la fecha del sistema."""
    sql, params = _origen_historial(fecha_corte)
    historial = (read_sql(sql, params) if params else read_sql(sql)).set_index("ClienteId")
    return df.join(historial, how="left")


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
