"""Scoring por lote de clientes (Paso 8.6).

Carga el modelo entrenado por src/train.py (sobre
dbo.vw_DatasetEntrenamientoFactura, point-in-time por factura) y lo aplica
a dbo.vw_FeaturesScoringCliente (inferencia: una fila por cliente, estado
actual — ver docstring de src/extract.py).

Las dos vistas nombran de forma distinta variables conceptualmente
equivalentes (documento, sección 4.3: "las definiciones de cada variable
deben coincidir entre ambas vistas"); config.score.mapa_variables_inferencia
traduce entre ambas. Las variables de entrenamiento sin equivalente real a
nivel de cliente (propias de una factura puntual, como su monto exacto, o
del vendedor que la emitió) no pueden calcularse en el scoring por cliente:
su contribución al score se fija en WOE=0 (neutra), en vez de inventar un
valor o fallar, y se deja constancia en el log de cuáles fueron.

Criterio de aceptación del paso: el score debe ser reproducible —
ejecutar el proceso dos veces sobre los mismos datos produce exactamente
el mismo resultado (no hay aleatoriedad en la inferencia: el modelo ya
está entrenado y serializado, y la transformación WOE es determinista).
"""

import logging

import pandas as pd

from src.config import RAIZ_PROYECTO, cargar_config
from src.extract import extraer_features
from src.features import transformar_a_woe
from src.train import cargar_modelo, factor_offset

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.score")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "score.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())


def _config_score() -> dict:
    return cargar_config().get("score", {})


def _construir_features_inferencia(df: pd.DataFrame, mapeo: dict, columnas_modelo: list[str]) -> pd.DataFrame:
    """Traduce las columnas de la vista de cliente a los nombres que
    espera el modelo (entrenado sobre la vista de factura) y calcula su
    WOE con el mismo mapeo del entrenamiento. Las columnas del modelo sin
    equivalente disponible en `df` quedan en WOE=0 (sin contribución al
    score), preservando el orden y la cantidad de columnas que el modelo
    espera."""
    mapa_variables = _config_score().get("mapa_variables_inferencia", {})

    columnas_traducidas = {}
    sin_equivalente = []
    for variable_modelo in columnas_modelo:
        variable_vista = mapa_variables.get(variable_modelo, variable_modelo)
        if variable_vista and variable_vista in df.columns:
            columnas_traducidas[variable_modelo] = df[variable_vista]
        else:
            sin_equivalente.append(variable_modelo)

    if sin_equivalente:
        logger.info(
            "Variables del modelo sin equivalente en la vista de inferencia (WOE=0, sin contribución al score): %s",
            ", ".join(sin_equivalente),
        )

    df_traducido = pd.DataFrame(columnas_traducidas, index=df.index)
    mapeo_disponible = {v: info for v, info in mapeo.items() if v in df_traducido.columns}
    x_parcial = (
        transformar_a_woe(df_traducido, mapeo_disponible)
        if mapeo_disponible
        else pd.DataFrame(index=df.index)
    )

    x_completo = pd.DataFrame(0.0, index=df.index, columns=columnas_modelo)
    for columna in x_parcial.columns:
        x_completo[columna] = x_parcial[columna]
    return x_completo


def calcular_scores(df: pd.DataFrame | None = None, ruta_modelo=None) -> pd.DataFrame:
    """Calcula, para cada fila de `df` (por defecto, extraer_features()
    sobre dbo.vw_FeaturesScoringCliente, indexado por ClienteId), la
    probabilidad de incumplimiento y el score escalado (config.score.score_min
    a score_max, 0-1000 por defecto).

    Reproducible: con el mismo modelo serializado y los mismos datos de
    entrada, el resultado es siempre idéntico."""
    if df is None:
        df = extraer_features()

    bundle = cargar_modelo(ruta_modelo)
    modelo = bundle["modelo"]
    mapeo = bundle["mapeo"]
    columnas_modelo = bundle["columnas"]

    x_woe = _construir_features_inferencia(df, mapeo, columnas_modelo)
    proba_incumplimiento = modelo.predict_proba(x_woe)[:, 1]
    logit_incumplimiento = modelo.decision_function(x_woe)

    factor, offset = factor_offset()
    score = offset - factor * logit_incumplimiento

    config = _config_score()
    score_min = config.get("score_min", 0)
    score_max = config.get("score_max", 1000)
    score = pd.Series(score, index=df.index).clip(score_min, score_max).round().astype(int)

    logger.info(
        "Scores calculados para %d clientes (modelo entrenado el %s).",
        len(df),
        bundle.get("fecha_entrenamiento"),
    )

    return pd.DataFrame(
        {"probabilidad_incumplimiento": proba_incumplimiento, "score": score},
        index=df.index,
    )
