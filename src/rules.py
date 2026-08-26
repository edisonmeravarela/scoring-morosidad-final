"""Capa 2: reglas de negocio (Paso 8.7).

Traduce el score de la Capa 1 (src/score.py) en decisiones de crédito
concretas — SegmentoRiesgoId, plazo recomendado y clasificación
comercial — de forma determinística y auditable (documento, sección 3):
el mismo score y los mismos datos de cliente siempre producen la misma
decisión, sin caja negra.

Los umbrales que YA viven en la base de datos (LimiteCredito y
DiasMaxMora por segmento en dbo.SegmentoRiesgo, Dias por plazo en
dbo.PlazoPago — sección 3.1: "modificar la política es un UPDATE en base
de datos, no un cambio de código") se consultan en vivo contra esas
tablas, no se duplican en config.yaml. Los umbrales que este módulo
introduce y que ningún catálogo contiene (cortes de score, condiciones
de anulación, clasificación RFM, mapeo segmento -> nombre de plazo) sí
se leen de config.yaml, nunca fijos en el código (criterio de 8.7).

Dos tipos de "segmento" conviven a propósito en este módulo:
  - SegmentoRiesgoId: el id real de dbo.SegmentoRiesgo. Es lo único que
    asignar_segmento() devuelve y lo único que se escribe en base de
    datos (paso 8.8).
  - Codigo (BAJO/MEDIO/ALTO/CRITICO/NUEVO): el resultado de
    aplicar_overrides() en adelante. "NUEVO" no es una fila de
    dbo.SegmentoRiesgo — es un marcador interno para "sin historial
    suficiente para evaluar riesgo" (documento, sección 3.2, "menos de
    tres facturas fuerza NUEVO con plazo contado"); solo lo consumen
    recomendar_plazo() (fuerza CONTADO) y, de forma independiente,
    clasificar_cliente() (que ya llega a NUEVO por su propio criterio de
    historial insuficiente, sección 7).
"""

import logging

import pandas as pd

from src.config import RAIZ_PROYECTO, cargar_config
from src.db import read_sql

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.rules")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "rules.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())

NUEVO = "NUEVO"

_cache_catalogos: dict[str, pd.DataFrame] = {}


def _config_reglas() -> dict:
    return cargar_config().get("reglas", {})


def invalidar_cache_catalogos() -> None:
    """Limpia la caché de dbo.SegmentoRiesgo/dbo.PlazoPago (pruebas, o
    tras un UPDATE de política en esas tablas dentro del mismo proceso)."""
    _cache_catalogos.clear()


def _catalogo_segmentos() -> pd.DataFrame:
    if "segmentos" not in _cache_catalogos:
        df = read_sql("SELECT SegmentoRiesgoId, Codigo, LimiteCredito, DiasMaxMora, BloqueaVentas FROM dbo.SegmentoRiesgo")
        _cache_catalogos["segmentos"] = df.set_index("Codigo")
    return _cache_catalogos["segmentos"]


def _catalogo_plazos() -> pd.DataFrame:
    if "plazos" not in _cache_catalogos:
        df = read_sql("SELECT PlazoPagoId, Nombre FROM dbo.PlazoPago")
        _cache_catalogos["plazos"] = df.set_index("Nombre")
    return _cache_catalogos["plazos"]


def _catalogo_clasificaciones() -> pd.DataFrame:
    if "clasificaciones" not in _cache_catalogos:
        df = read_sql("SELECT ClasificacionId, Codigo FROM dbo.ClasificacionCliente")
        _cache_catalogos["clasificaciones"] = df.set_index("Codigo")
    return _cache_catalogos["clasificaciones"]


def resolver_clasificacion_id(codigo: str) -> int:
    """Codigo de clasificar_cliente() (A/B/C/NUEVO, todos con fila propia
    en dbo.ClasificacionCliente) -> ClasificacionId real."""
    return int(_catalogo_clasificaciones().loc[codigo, "ClasificacionId"])


def resolver_segmento_final(segmento_id_asignado: int, segmento_codigo: str) -> int:
    """Codigo final de aplicar_overrides() -> SegmentoRiesgoId que se
    persiste en base de datos (paso 8.8).

    Para BAJO/MEDIO/ALTO/CRITICO es una resolución directa contra el
    catálogo. Para NUEVO no hay fila propia, y a propósito NO se resuelve
    contra CRITICO (a diferencia de marcar_revision): CRITICO implica
    BloqueaVentas=1 en dbo.SegmentoRiesgo, un bloqueo total de ventas a
    crédito mucho más severo que el "plazo contado" que pide el
    documento para un cliente simplemente sin historial (sección 3.2:
    "aunque su score sea alto", es decir, sin negarle el segmento que su
    score sí sustenta). Por eso se conserva el segmento_id_asignado
    original, el que asignar_segmento() calculó a partir del score antes
    de aplicar_overrides()."""
    if segmento_codigo == NUEVO:
        return int(segmento_id_asignado)
    return int(_catalogo_segmentos().loc[segmento_codigo, "SegmentoRiesgoId"])


def _fila_segmento(codigo: str) -> pd.Series:
    """Fila del catálogo para `codigo`. NUEVO no tiene fila propia: se
    usa la de CRITICO (límite de crédito y tolerancia de mora más
    conservadores) como tope prudente ante la falta de historial."""
    catalogo = _catalogo_segmentos()
    return catalogo.loc[codigo if codigo in catalogo.index else "CRITICO"]


def asignar_segmento(score: float) -> int:
    """Paso 1 (sección 3.2): aplica los cortes de score (config.reglas.cortes_score)
    y devuelve el SegmentoRiesgoId real de dbo.SegmentoRiesgo."""
    cortes = _config_reglas().get("cortes_score", [])
    for corte in cortes:
        if corte["hasta"] is None or score <= corte["hasta"]:
            codigo = corte["segmento"]
            return int(_catalogo_segmentos().loc[codigo, "SegmentoRiesgoId"])
    raise ValueError(f"config.reglas.cortes_score no cubre el score {score} (falta un tramo con hasta=null).")


def aplicar_overrides(cliente: pd.Series | dict, segmento_id: int) -> str:
    """Paso 2 (sección 3.2): condiciones duras que sobrescriben el
    segmento del paso 1 sin importar el score. Devuelve el Codigo final
    (BAJO/MEDIO/ALTO/CRITICO/NUEVO), no el id, porque NUEVO no tiene id
    en el catálogo."""
    config = _config_reglas()
    dias_mora_critico = config.get("dias_mora_critico", 90)
    minimo_facturas = config.get("minimo_facturas_para_evaluar", 3)

    mora_maxima = cliente.get("DiasMoraMaxima", 0) or 0
    num_facturas = cliente.get("NumFacturas", 0) or 0

    if mora_maxima > dias_mora_critico:
        logger.info("Override CRITICO: mora máxima %.0f días > %d.", mora_maxima, dias_mora_critico)
        return "CRITICO"

    if num_facturas < minimo_facturas:
        logger.info("Override NUEVO: %d facturas < mínimo %d para evaluar riesgo.", num_facturas, minimo_facturas)
        return NUEVO

    catalogo = _catalogo_segmentos()
    fila = catalogo[catalogo["SegmentoRiesgoId"] == segmento_id]
    if fila.empty:
        raise ValueError(f"SegmentoRiesgoId {segmento_id} no existe en dbo.SegmentoRiesgo.")
    return str(fila.index[0])


def recomendar_plazo(segmento_codigo: str) -> int:
    """Paso 3 (sección 3.2/7): mapea el Codigo final (salida de
    aplicar_overrides) al PlazoPagoId correspondiente."""
    mapa = _config_reglas().get("plazo_por_segmento", {})
    nombre_plazo = mapa.get(segmento_codigo)
    if nombre_plazo is None:
        raise ValueError(f"config.reglas.plazo_por_segmento no define un plazo para el segmento '{segmento_codigo}'.")
    return int(_catalogo_plazos().loc[nombre_plazo, "PlazoPagoId"])


def clasificar_cliente(cliente: pd.Series | dict) -> str:
    """Paso 4 (sección 3.2/7): clasificación comercial A/B/C/NUEVO por
    RFM (Recencia, Frecuencia, Monto), independiente del riesgo. Sin
    historial suficiente, el cliente queda en NUEVO por defecto (sección
    7), con el mismo umbral de minimo_facturas_para_evaluar que usa el
    override de riesgo."""
    config = _config_reglas()
    minimo_facturas = config.get("minimo_facturas_para_evaluar", 3)
    num_facturas = cliente.get("NumFacturas", 0) or 0
    if num_facturas < minimo_facturas:
        return NUEVO

    rfm = config.get("rfm", {})
    recencia_dias = cliente.get("DiasDesdeUltimaFactura", None)
    monto = cliente.get("ValorFacturaPromedio", 0) or 0

    if recencia_dias is not None and recencia_dias <= rfm.get("recencia_dias_buena", 30):
        puntos_r = 3
    elif recencia_dias is not None and recencia_dias <= rfm.get("recencia_dias_regular", 90):
        puntos_r = 2
    else:
        puntos_r = 1

    if num_facturas >= rfm.get("frecuencia_alta", 10):
        puntos_f = 3
    elif num_facturas >= rfm.get("frecuencia_regular", 4):
        puntos_f = 2
    else:
        puntos_f = 1

    if monto >= rfm.get("monto_alto", 2000):
        puntos_m = 3
    elif monto >= rfm.get("monto_regular", 500):
        puntos_m = 2
    else:
        puntos_m = 1

    total = puntos_r + puntos_f + puntos_m
    if total >= 7:
        return "A"
    if total >= 5:
        return "B"
    return "C"


def marcar_revision(cliente: pd.Series | dict, segmento_codigo: str) -> bool:
    """Paso adicional (sección 3.3, "observación relevante"): marca para
    revisión manual a los clientes cuyo saldo pendiente supere el límite
    de crédito de su segmento, o cuya mora máxima exceda el DiasMaxMora
    del catálogo — aun cuando el segmento asignado no haya llegado a
    CRITICO."""
    fila = _fila_segmento(segmento_codigo)
    saldo_actual = cliente.get("SaldoActualTotal", 0) or 0
    mora_maxima = cliente.get("DiasMoraMaxima", 0) or 0

    excede_limite = saldo_actual > float(fila["LimiteCredito"])
    excede_mora = mora_maxima > float(fila["DiasMaxMora"])
    if excede_limite or excede_mora:
        logger.info(
            "Cliente marcado para revisión manual (segmento %s): saldo=%.2f (límite %.2f), mora=%.0f (tolerancia %.0f).",
            segmento_codigo,
            saldo_actual,
            fila["LimiteCredito"],
            mora_maxima,
            fila["DiasMaxMora"],
        )
    return bool(excede_limite or excede_mora)
