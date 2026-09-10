"""Entrenamiento y validación del modelo de scoring (Paso 8.5).

Dataset: dbo.vw_DatasetEntrenamientoFactura (point-in-time, una fila por
factura), excluyendo SinHistorialPago=1 — nunca dbo.vw_FeaturesScoringCliente,
que es solo para inferencia (ver docstring de src/extract.py).

Split: temporal por FechaEmision (nunca aleatorio) y, sobre ese resultado,
purga de validación de cualquier cliente que también aparezca en
entrenamiento (documento, sección 4.4, "Agrupación por cliente") — así
ningún cliente está en ambos conjuntos.

Controles de fuga aplicados en este paso (además de los de features.py):
KS mínimo de validación y techo de AUC sospechoso. Si el modelo no los
cumple, se rechaza y NO se sobrescribe el modelo anterior en models/.
"""

import logging
import math
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.calidad import validar_antes_de_entrenar
from src.config import RAIZ_PROYECTO, cargar_config
from src.drift import registrar_baseline
from src.esquema import validar_esquema
from src.extract import excluir_sin_historial, extraer_dataset_entrenamiento
from src.features import preparar_features, transformar_a_woe

MODELS_DIR = RAIZ_PROYECTO / "models"
MODELS_DIR.mkdir(exist_ok=True)

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.train")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "train.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())


class ModeloRechazadoError(Exception):
    """El modelo no cumplió los criterios mínimos de aceptación (KS bajo o
    AUC sospechosamente alto, indicio de fuga) y no se sobrescribió el
    modelo anterior en models/."""


class DatosInsuficientesError(Exception):
    """El split temporal + purga por cliente dejó un conjunto de
    validación vacío o con una sola clase; no se puede calcular AUC/KS."""


def _config_entrenamiento() -> dict:
    return cargar_config().get("entrenamiento", {})


def dividir_temporal_por_cliente(
    df: pd.DataFrame,
    fecha_col: str | None = None,
    cliente_col: str | None = None,
    meses_validacion: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal (nunca aleatorio): las facturas de los últimos
    `meses_validacion` meses van a validación, el resto a entrenamiento.
    Después purga de validación cualquier fila cuyo cliente también
    aparezca en entrenamiento, para que ningún cliente esté en ambos
    conjuntos a la vez (sección 4.4, "Agrupación por cliente") — de lo
    contrario el modelo podría "recordar" el comportamiento pasado de ese
    cliente en vez de generalizar a comportamiento genuinamente nuevo."""
    config = _config_entrenamiento()
    fecha_col = fecha_col or config.get("columna_fecha", "FechaEmision")
    cliente_col = cliente_col or config.get("columna_cliente", "ClienteId")
    meses_validacion = meses_validacion if meses_validacion is not None else config.get("meses_validacion", 6)

    fechas = pd.to_datetime(df[fecha_col])
    fecha_corte = fechas.max() - pd.DateOffset(months=meses_validacion)

    train_df = df[fechas < fecha_corte].copy()
    val_df = df[fechas >= fecha_corte].copy()

    clientes_entrenamiento = set(train_df[cliente_col])
    mascara_purga = val_df[cliente_col].isin(clientes_entrenamiento)
    if mascara_purga.any():
        logger.info(
            "Agrupación por cliente: %d facturas de validación purgadas (%d clientes ya vistos en entrenamiento).",
            int(mascara_purga.sum()),
            val_df.loc[mascara_purga, cliente_col].nunique(),
        )
    val_df = val_df[~mascara_purga]

    logger.info(
        "Split temporal (corte %s): %d facturas entrenamiento, %d facturas validación.",
        fecha_corte.date().isoformat(),
        len(train_df),
        len(val_df),
    )
    return train_df, val_df


def calcular_ks(y_true, y_score) -> float:
    """Estadístico Kolmogorov-Smirnov: máxima distancia entre las
    distribuciones acumuladas de los scores de la clase 1 (morosos) y la
    clase 0 (no morosos). 0 = no separa nada; 1 = separación perfecta."""
    y_true = pd.Series(y_true).to_numpy()
    y_score = pd.Series(y_score).to_numpy()
    malos = y_score[y_true == 1]
    buenos = y_score[y_true == 0]
    if len(malos) == 0 or len(buenos) == 0:
        raise DatosInsuficientesError(
            "No se puede calcular KS: el conjunto de validación no tiene ambas clases representadas."
        )
    return float(ks_2samp(malos, buenos).statistic)


def entrenar_regresion_logistica(x_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    class_weight = _config_entrenamiento().get("class_weight", "balanced")
    modelo = LogisticRegression(class_weight=class_weight, max_iter=1000)
    modelo.fit(x_train, y_train)
    return modelo


def entrenar_xgboost(x_train: pd.DataFrame, y_train: pd.Series):
    """Modelo alternativo de mayor poder predictivo, solo para comparar
    (sección 6): no es el que se sirve ni el que decide aceptación/rechazo,
    la regresión logística WOE sigue siendo la línea base recomendada."""
    from xgboost import XGBClassifier

    n_malos = int((y_train == 1).sum())
    n_buenos = int((y_train == 0).sum())
    scale_pos_weight = (n_buenos / n_malos) if n_malos else 1.0

    modelo = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    modelo.fit(x_train, y_train)
    return modelo


def factor_offset() -> tuple[float, float]:
    """Constantes del escalado tipo FICO (score = offset + factor *
    ln(odds_bueno)). Las usa tanto generar_scorecard() aquí como
    src/score.py al calcular el score real de un cliente — deben ser
    exactamente las mismas en los dos lugares para que los puntos del
    scorecard sumen al score final."""
    config = _config_entrenamiento()
    pdo = config.get("pdo", 20)
    score_base = config.get("score_base", 600)
    odds_base = config.get("odds_base", 50)
    factor = pdo / math.log(2)
    offset = score_base - factor * math.log(odds_base)
    return factor, offset


def generar_scorecard(modelo: LogisticRegression, mapeo: dict, columnas: list[str]) -> pd.DataFrame:
    """Tabla de puntos del scorecard (score base + puntos por tramo),
    lista para exportar a CSV y revisar en el área de crédito (sección
    2.3/8.5): cada variable aporta una cantidad de puntos conocida y
    auditable, no es una caja negra.

    El target del modelo es Target=1 (moroso); WOE está definido como
    ln(%buenos/%malos), así que ln(odds_bueno) = -(intercepto + sum(beta_i
    * WOE_i)). Un tramo "seguro" (WOE alto) con un beta de signo normal
    (negativo) aporta puntos positivos, como es de esperar."""
    factor, offset = factor_offset()
    beta_por_variable = dict(zip(columnas, modelo.coef_[0]))

    filas = [{"variable": "(base)", "tramo": "—", "beta": None, "woe": None, "puntos": round(offset - factor * modelo.intercept_[0], 1)}]
    for variable, info in mapeo.items():
        if variable not in beta_por_variable:
            continue
        beta = beta_por_variable[variable]
        for tramo in info["tramos"]:
            puntos = -factor * beta * tramo["woe"]
            filas.append(
                {
                    "variable": variable,
                    "tramo": tramo["etiqueta"],
                    "beta": round(beta, 6),
                    "woe": round(tramo["woe"], 4),
                    "puntos": round(puntos, 1),
                }
            )
    return pd.DataFrame(filas)


def _resolver_ruta(clave: str, valor_por_defecto: str, ruta_override=None) -> Path:
    ruta = Path(ruta_override) if ruta_override is not None else Path(_config_entrenamiento().get(clave, valor_por_defecto))
    return ruta if ruta.is_absolute() else RAIZ_PROYECTO / ruta


def guardar_modelo(resultado: dict, ruta_modelo=None, marca_tiempo: str | None = None) -> Path:
    """Serializa el modelo con joblib, versionado con la fecha de
    entrenamiento, y además lo guarda/sobrescribe en la ruta estable
    (models/modelo.pkl) que usará src/score.py — solo se llega aquí si el
    modelo ya pasó los criterios de aceptación.

    `marca_tiempo`, si se indica, es la misma etiqueta de versión que
    entrenar() usa para dbo.BaselineVariable (src/drift.py) — el
    artefacto serializado y el baseline deben compartirla."""
    ruta_estable = _resolver_ruta("ruta_modelo", "models/modelo.pkl", ruta_modelo)
    ruta_estable.parent.mkdir(exist_ok=True, parents=True)

    marca_tiempo = marca_tiempo or datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_versionada = ruta_estable.parent / f"{ruta_estable.stem}_{marca_tiempo}{ruta_estable.suffix}"

    joblib.dump(resultado, ruta_versionada)
    joblib.dump(resultado, ruta_estable)
    logger.info("Modelo guardado en %s (y versionado en %s).", ruta_estable, ruta_versionada)
    return ruta_estable


def cargar_modelo(ruta_modelo=None) -> dict:
    """Carga el bundle guardado por guardar_modelo(): {modelo, mapeo,
    columnas, metricas, fecha_entrenamiento}. Lo usa src/score.py para
    aplicar el modelo entrenado a clientes nuevos."""
    ruta = _resolver_ruta("ruta_modelo", "models/modelo.pkl", ruta_modelo)
    return joblib.load(ruta)


def entrenar() -> dict:
    """Orquesta el paso 8.5 completo. Devuelve el dict con el modelo, el
    mapeo WOE, las columnas del modelo y las métricas de validación.
    Lanza ModeloRechazadoError sin escribir nada en models/ si el modelo
    no cumple los criterios de aceptación (KS mínimo, techo de AUC).

    Antes de extraer nada, corre los dos primeros controles del
    documento "Controles de Esquema, Calidad y Deriva de Datos", en su
    orden de dependencia obligatorio: 1) esquema (src.esquema.validar_esquema,
    SIEMPRE bloquea ante un hallazgo crítico — no tiene sentido validar
    reglas de negocio sobre una estructura que cambió) y 2) calidad
    (src.calidad.validar_antes_de_entrenar, bloquea solo ante críticos).
    Cualquiera de los dos puede lanzar antes de tocar los datos de
    entrenamiento."""
    validar_esquema()
    validar_antes_de_entrenar()

    config = _config_entrenamiento()
    target_col = config.get("columna_target", "Target")

    df = extraer_dataset_entrenamiento()
    if config.get("excluir_sin_historial", True):
        df = excluir_sin_historial(df)

    train_df, val_df = dividir_temporal_por_cliente(df)
    if val_df.empty:
        raise DatosInsuficientesError("El conjunto de validación quedó vacío tras el split temporal y la purga por cliente.")

    x_train, mapeo = preparar_features(train_df, target_col=target_col)
    x_val = transformar_a_woe(val_df, mapeo)
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    if y_val.nunique() < 2:
        raise DatosInsuficientesError(
            "El conjunto de validación quedó con una sola clase tras el split temporal y la purga por "
            "cliente; no se puede calcular AUC/KS. Prueba con menos meses_validacion."
        )

    modelo = entrenar_regresion_logistica(x_train, y_train)
    proba_val = modelo.predict_proba(x_val)[:, 1]

    auc = float(roc_auc_score(y_val, proba_val))
    ks = calcular_ks(y_val, proba_val)

    ks_minimo = config.get("ks_minimo", 0.25)
    auc_maximo = config.get("auc_maximo_sospechoso", 0.95)
    logger.info(
        "Validación (regresión logística): AUC=%.4f, KS=%.4f (umbrales: KS>=%.2f, AUC<=%.2f).",
        auc,
        ks,
        ks_minimo,
        auc_maximo,
    )

    if config.get("entrenar_xgboost", True):
        try:
            modelo_xgb = entrenar_xgboost(x_train, y_train)
            proba_xgb = modelo_xgb.predict_proba(x_val)[:, 1]
            auc_xgb = float(roc_auc_score(y_val, proba_xgb))
            ks_xgb = calcular_ks(y_val, proba_xgb)
            logger.info("Validación (XGBoost, solo comparación): AUC=%.4f, KS=%.4f.", auc_xgb, ks_xgb)
        except DatosInsuficientesError:
            pass

    if ks < ks_minimo:
        mensaje = f"KS de validación ({ks:.4f}) inferior al mínimo requerido ({ks_minimo}); modelo rechazado."
        logger.error(mensaje)
        raise ModeloRechazadoError(mensaje)

    if auc > auc_maximo:
        mensaje = (
            f"AUC de validación ({auc:.4f}) supera el techo de sospecha de fuga ({auc_maximo}); "
            "modelo rechazado. Revisar la construcción de las variables antes de reentrenar."
        )
        logger.error(mensaje)
        raise ModeloRechazadoError(mensaje)

    scorecard = generar_scorecard(modelo, mapeo, list(x_train.columns))
    ruta_scorecard = _resolver_ruta("ruta_scorecard", "models/scorecard.csv")
    scorecard.to_csv(ruta_scorecard, index=False, encoding="utf-8")
    logger.info("Scorecard exportado a %s (%d filas).", ruta_scorecard, len(scorecard))

    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    resultado = {
        "modelo": modelo,
        "mapeo": mapeo,
        "columnas": list(x_train.columns),
        "metricas": {"auc": auc, "ks": ks},
        "fecha_entrenamiento": datetime.now().isoformat(),
        "version_modelo": marca_tiempo,
        "n_entrenamiento": len(train_df),
        "n_validacion": len(val_df),
    }
    guardar_modelo(resultado, marca_tiempo=marca_tiempo)

    try:
        registrar_baseline(marca_tiempo, datetime.now().date(), train_df, mapeo)
    except Exception as exc:
        # El modelo ya quedó guardado y es válido; perder el baseline
        # degrada la medición de deriva pero no invalida el entrenamiento.
        logger.error("No se pudo registrar el baseline de deriva para la versión %s: %s", marca_tiempo, exc)

    return resultado
