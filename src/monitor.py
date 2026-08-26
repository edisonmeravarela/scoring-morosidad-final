"""Monitoreo del modelo en producción (Paso 8.9).

Dos controles independientes, ambos comparando "lo que el modelo asumió
al entrenar" contra "lo que efectivamente está pasando ahora":

  - evaluar_desempeno(): cruza el score con el que se reclasificó a cada
    cliente (dbo.ClienteRiesgoHistorial, vía dbo.vw_MonitoreoScoring —
    "seguimiento del modelo en producción") contra la mora realmente
    observada después, y calcula el KS actual del modelo en producción.
    dbo.vw_MonitoreoScoring no guarda el score como columna propia: lo
    incluye dentro de Motivo (dbo.sp_ActualizarScoringCliente concatena
    "... (score: 415.00)"), así que se extrae de ahí.

  - detectar_drift(): Population Stability Index (PSI) de cada variable
    del modelo, comparando su distribución actual (vw_FeaturesScoringCliente)
    contra los tramos guardados en el mapeo WOE del entrenamiento — el
    mismo mecanismo de tramos que ya usan calcular_woe_variable/
    calcular_woe_categoria en features.py, aplicado ahora para comparar
    dos poblaciones en vez de dos clases. Reutiliza la traducción de
    nombres de variable entre vistas de config.score.mapa_variables_inferencia
    (el mismo problema que resolvió src/score.py).

Criterio de aceptación: emitir una alerta de reentrenamiento cuando el
KS en producción caiga por debajo del 80% (config.monitor.umbral_ks_relativo)
del valor obtenido en validación (src/train.py).
"""

import logging
import math
import re

import pandas as pd

from src.config import RAIZ_PROYECTO, cargar_config
from src.db import read_sql
from src.extract import extraer_features
from src.features import CATEGORIA_OTRAS, TRAMO_FALTANTE, _clasificar_valor
from src.train import cargar_modelo, calcular_ks

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.monitor")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "monitor.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())


class DatosInsuficientesError(Exception):
    """No hay suficientes reclasificaciones con desenlace observable en
    dbo.vw_MonitoreoScoring para calcular un KS de producción confiable."""


def _config_monitor() -> dict:
    return cargar_config().get("monitor", {})


def _extraer_score_de_motivo(motivo: str) -> float | None:
    """El score con el que se reclasificó al cliente va embebido en
    Motivo (dbo.sp_ActualizarScoringCliente: "... (score: 415.00)").
    CONCAT de un DECIMAL en T-SQL siempre usa punto decimal, nunca coma
    de miles ni separador regional."""
    coincidencia = re.search(r"score:\s*([\d.]+)", motivo or "", re.IGNORECASE)
    if not coincidencia:
        return None
    try:
        return float(coincidencia.group(1))
    except ValueError:
        return None


def evaluar_desempeno(df_monitoreo: pd.DataFrame | None = None, minimo_registros: int | None = None) -> dict:
    """KS actual del modelo en producción, comparado contra el KS de
    validación del último modelo entrenado. Solo usa reclasificaciones
    con al menos una factura posterior (desenlace observable) y con un
    score identificable en Motivo."""
    if df_monitoreo is None:
        df_monitoreo = read_sql("SELECT * FROM dbo.vw_MonitoreoScoring")

    config = _config_monitor()
    minimo_registros = minimo_registros if minimo_registros is not None else config.get("minimo_registros", 20)

    observables = df_monitoreo[df_monitoreo["FacturasPosteriores"].fillna(0) > 0].copy()
    observables["ScoreExtraido"] = observables["Motivo"].apply(_extraer_score_de_motivo)
    observables = observables.dropna(subset=["ScoreExtraido"])

    if len(observables) < minimo_registros:
        raise DatosInsuficientesError(
            f"Solo hay {len(observables)} reclasificaciones con facturación posterior y score identificable "
            f"en dbo.vw_MonitoreoScoring; se requieren al menos {minimo_registros} para un KS de producción confiable."
        )

    y_true = (observables["MorosasPosteriores"].fillna(0) > 0).astype(int)
    ks_actual = calcular_ks(y_true, observables["ScoreExtraido"])

    ks_validacion = cargar_modelo()["metricas"]["ks"]
    umbral_relativo = config.get("umbral_ks_relativo", 0.8)
    umbral_alerta = ks_validacion * umbral_relativo
    alerta = ks_actual < umbral_alerta

    if alerta:
        logger.warning(
            "Alerta de reentrenamiento: KS en producción (%.4f) por debajo del %.0f%% del KS de validación (%.4f; umbral %.4f).",
            ks_actual,
            umbral_relativo * 100,
            ks_validacion,
            umbral_alerta,
        )
    else:
        logger.info(
            "KS en producción (%.4f) dentro del rango esperado (umbral %.4f, validación %.4f).",
            ks_actual,
            umbral_alerta,
            ks_validacion,
        )

    return {
        "ks_actual": ks_actual,
        "ks_validacion": ks_validacion,
        "umbral_alerta": umbral_alerta,
        "alerta_reentrenamiento": alerta,
        "n_observaciones": len(observables),
    }


def calcular_psi(serie_actual: pd.Series, info: dict) -> float:
    """Population Stability Index de una variable frente a su
    distribución de entrenamiento, reutilizando los mismos tramos
    guardados en el mapeo WOE (info["tramos"], con el conteo `n` de cada
    uno en entrenamiento) — el mismo suavizado que evita ln(0) o
    división por cero que usa el cálculo de IV."""
    if info.get("tipo") == "categorica":
        valor_a_bucket = info["valor_a_bucket"]

        def _bucket(valor):
            if pd.isna(valor):
                return TRAMO_FALTANTE
            return valor_a_bucket.get(valor, CATEGORIA_OTRAS)

        conteo_actual = serie_actual.map(_bucket).value_counts().to_dict()
    else:
        bordes = info["bordes"]

        def _tramo(valor):
            return -1 if pd.isna(valor) else _clasificar_valor(valor, bordes)

        conteo_actual = serie_actual.map(_tramo).value_counts().to_dict()

    total_actual = max(len(serie_actual), 1)
    total_entrenamiento = max(sum(t["n"] for t in info["tramos"]), 1)

    psi = 0.0
    for tramo in info["tramos"]:
        n_entrenamiento = max(tramo["n"], 1)
        n_actual = max(conteo_actual.get(tramo["tramo_id"], 0), 1)
        prop_entrenamiento = n_entrenamiento / total_entrenamiento
        prop_actual = n_actual / total_actual
        psi += (prop_actual - prop_entrenamiento) * math.log(prop_actual / prop_entrenamiento)
    return psi


def detectar_drift(df_actual: pd.DataFrame | None = None, mapeo: dict | None = None, psi_alerta: float | None = None) -> pd.DataFrame:
    """PSI de cada variable del modelo entre la distribución actual
    (dbo.vw_FeaturesScoringCliente) y la de entrenamiento (mapeo WOE
    guardado junto al modelo). Igual que en src/score.py, traduce
    nombres de variable entre la vista de entrenamiento y la de
    inferencia (config.score.mapa_variables_inferencia); las variables
    sin equivalente en inferencia no pueden monitorearse y se omiten."""
    if df_actual is None:
        df_actual = extraer_features()
    if mapeo is None:
        mapeo = cargar_modelo()["mapeo"]
    psi_alerta = psi_alerta if psi_alerta is not None else _config_monitor().get("psi_alerta", 0.25)

    mapa_variables = cargar_config().get("score", {}).get("mapa_variables_inferencia", {})

    filas = []
    for variable_entrenamiento, info in mapeo.items():
        variable_actual = mapa_variables.get(variable_entrenamiento, variable_entrenamiento)
        if not variable_actual or variable_actual not in df_actual.columns:
            continue
        psi = calcular_psi(df_actual[variable_actual], info)
        filas.append({"variable": variable_entrenamiento, "psi": psi, "alerta": psi > psi_alerta})

    resultado = pd.DataFrame(filas, columns=["variable", "psi", "alerta"])
    variables_con_alerta = resultado.loc[resultado["alerta"], "variable"].tolist() if not resultado.empty else []
    if variables_con_alerta:
        logger.warning("Desviación de datos (drift) significativa (PSI > %.2f) en: %s", psi_alerta, ", ".join(variables_con_alerta))
    else:
        logger.info("Sin desviación de datos significativa (PSI <= %.2f) en las %d variables evaluadas.", psi_alerta, len(resultado))
    return resultado
