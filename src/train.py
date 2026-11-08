"""Entrenamiento y validación del modelo de scoring (Paso 8.5).

Dataset: dbo.vw_DatasetEntrenamientoFactura (point-in-time, una fila por
factura), excluyendo SinHistorialPago=1 — nunca dbo.vw_FeaturesScoringCliente,
que es solo para inferencia (ver docstring de src/extract.py).

Split: partición 80/20 agrupada por cliente (src/ajustes_evaluacion.py:
dividir_particion_agrupada — nunca aleatoria por fila), sin solapamiento
y sin necesidad de purgar (documento, sección 4.4, "Agrupación por
cliente") — ningún cliente está en ambos conjuntos a la vez. La función
`dividir_temporal_por_cliente()` (corte por FechaEmision + purga) se deja
en este archivo sin usar, solo para comparar.

Controles de fuga aplicados en este paso (además de los de features.py):
KS y AUC mínimos, y techo de AUC sospechoso, todos calculados sobre la
validación cruzada agrupada fuera de muestra (metricas_fuera_de_muestra,
en src/ajustes_evaluacion.py; UmbralesAceptacion/cargar_umbrales/
modelo_aceptado, aquí mismo, más abajo) — el holdout temporal solo veta
(decidir_promocion, src/ajustes_evaluacion.py). Si el modelo no cumple, se
rechaza y NO se sobrescribe el modelo anterior en models/.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.ajustes_evaluacion import (
    decidir_promocion,
    dividir_particion_agrupada,
    metricas_fuera_de_muestra,
)
from src.calidad import validar_antes_de_entrenar
from src.config import RAIZ_PROYECTO, cargar_config
from src.db import consultar_escalar
from src.drift import registrar_baseline
from src.esquema import validar_esquema
from src.evaluacion import evaluar_modelo
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


def _consultar_parametro_scoring(codigo: str) -> float | None:
    """Wrapper de acceso a base para cargar_umbrales(): lee un umbral de
    dbo.ParametroScoring vía dbo.fn_ParametroScoring."""
    fila = consultar_escalar("SELECT dbo.fn_ParametroScoring(?)", (codigo,))
    return float(fila) if fila is not None else None


# ============================================================================
# Piso de AUC en el criterio de aceptación (incidente 20261007_183452 y
# corridas posteriores: KS 0.75/AUC 0.25 quedaba aceptado porque solo había
# piso de KS y techo de AUC, nunca piso de AUC). Integrado de
# ajustes/piso_auc.py, conservando la lógica tal cual estaba decidida.
# ============================================================================

# Valores de respaldo de los tres umbrales de aceptación. Se usan solo si
# dbo.ParametroScoring no responde: un entrenamiento nunca debe quedarse sin
# criterio de aceptación, y tampoco debe aceptar por ausencia de criterio.
# Son deliberadamente iguales a los de la base.
RESPALDO_UMBRALES = {"KS_MINIMO": 0.25, "AUC_MINIMO": 0.60, "AUC_MAXIMO": 0.95}

# Por debajo de este valor el ordenamiento está literalmente invertido.
# No es un umbral configurable: es la definición del AUC.
AZAR = 0.50


@dataclass(frozen=True)
class UmbralesAceptacion:
    """Los tres umbrales que deciden si un modelo puede promoverse."""

    ks_minimo: float
    auc_minimo: float
    auc_maximo: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.ks_minimo <= 1.0:
            raise ValueError(f"ks_minimo fuera de rango: {self.ks_minimo}")
        if not 0.0 <= self.auc_minimo < self.auc_maximo <= 1.0:
            raise ValueError(
                f"umbrales de AUC incoherentes: minimo {self.auc_minimo}, "
                f"maximo {self.auc_maximo}"
            )
        if self.auc_minimo < AZAR:
            # No se bloquea, pero debe quedar dicho: un piso por debajo del
            # azar no protege de nada.
            logger.warning(
                "El piso de AUC (%.4f) esta por debajo del azar (%.2f): "
                "un modelo con el ordenamiento invertido pasaria el filtro.",
                self.auc_minimo, AZAR,
            )


def cargar_umbrales(consultar=None) -> UmbralesAceptacion:
    """Lee los tres umbrales de dbo.ParametroScoring.

    `consultar(codigo) -> float | None` es la función de acceso a base del
    proyecto — en este módulo, `_consultar_parametro_scoring`, que envuelve
    `src/db.py::consultar_escalar` sobre `dbo.fn_ParametroScoring`. Si la
    consulta falla o devuelve NULL se usa el valor de respaldo y se deja
    constancia en el log. Nunca se omite la comprobación.
    """
    valores = dict(RESPALDO_UMBRALES)
    if consultar is None:
        logger.warning("Sin acceso a dbo.ParametroScoring: se usan los umbrales de respaldo.")
        return UmbralesAceptacion(valores["KS_MINIMO"], valores["AUC_MINIMO"],
                                  valores["AUC_MAXIMO"])

    for codigo in RESPALDO_UMBRALES:
        try:
            valor = consultar(codigo)
        except Exception:
            logger.exception("Fallo al leer %s; se usa el respaldo %.4f.",
                             codigo, RESPALDO_UMBRALES[codigo])
            continue
        if valor is None:
            logger.warning("%s no esta definido en dbo.ParametroScoring; "
                           "se usa el respaldo %.4f.", codigo, RESPALDO_UMBRALES[codigo])
            continue
        valores[codigo] = float(valor)

    return UmbralesAceptacion(valores["KS_MINIMO"], valores["AUC_MINIMO"],
                              valores["AUC_MAXIMO"])


def modelo_aceptado(ks: float, auc: float,
                    umbrales: UmbralesAceptacion) -> tuple[bool, str | None]:
    """Decide si el modelo puede sobrescribir al vigente, a partir de KS/AUC.

    Devuelve (aceptado, motivo_rechazo). El motivo es None cuando se acepta.

    El orden de las comprobaciones es deliberado: KS, piso de AUC, techo de
    AUC. Cuando una corrida falla por varias razones a la vez, el motivo que
    queda registrado es el más fundamental y no el primero que aparezca por
    casualidad.
    """
    if ks != ks or auc != auc:          # NaN: ninguna comparación sería válida
        return False, "KS o AUC no son numeros: la evaluacion no se completo"

    if ks < umbrales.ks_minimo:
        return False, (f"KS {ks:.4f} por debajo del minimo "
                       f"{umbrales.ks_minimo:.4f}")

    if auc < umbrales.auc_minimo:
        detalle = ("ordenamiento invertido" if auc < AZAR
                   else "poder predictivo insuficiente")
        return False, (f"AUC {auc:.4f} por debajo del minimo "
                       f"{umbrales.auc_minimo:.4f} ({detalle})")

    if auc > umbrales.auc_maximo:
        return False, (f"AUC {auc:.4f} sobre el techo "
                       f"{umbrales.auc_maximo:.4f}: indicio de fuga de informacion")

    return True, None


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
    no cumple los criterios de aceptación de decidir_promocion() (KS y AUC
    mínimos, techo de AUC) sobre la validación cruzada fuera de muestra.

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
    umbrales = cargar_umbrales(consultar=_consultar_parametro_scoring)

    df = extraer_dataset_entrenamiento()
    if config.get("excluir_sin_historial", True):
        df = excluir_sin_historial(df)

    train_df, val_df = dividir_particion_agrupada(
        df,
        fecha_col=config.get("columna_fecha", "FechaEmision"),
        cliente_col=config.get("columna_cliente", "ClienteId"),
        target_col=target_col,
        prop_validacion=config.get("prop_validacion", 0.20),
        min_filas_validacion=config.get("min_filas_validacion", 30),
        min_positivos_validacion=config.get("min_positivos_validacion", 10),
        estricto=config.get("particion_estricta", False),
    )
    if val_df.empty:
        raise DatosInsuficientesError("El conjunto de validación quedó vacío tras el split temporal y la purga por cliente.")

    x_train, mapeo = preparar_features(train_df, target_col=target_col)
    x_val = transformar_a_woe(val_df, mapeo)
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    if y_val.nunique() < 2:
        raise DatosInsuficientesError(
            "El conjunto de validación quedó con una sola clase tras la partición agrupada por cliente; "
            "no se puede calcular AUC/KS. Prueba con una prop_validacion mayor."
        )

    modelo = entrenar_regresion_logistica(x_train, y_train)
    proba_val = modelo.predict_proba(x_val)[:, 1]

    auc = float(roc_auc_score(y_val, proba_val))
    ks = calcular_ks(y_val, proba_val)

    logger.info(
        "Validación (regresión logística): AUC=%.4f, KS=%.4f (umbrales: KS>=%.2f, AUC<=%.2f).",
        auc,
        ks,
        umbrales.ks_minimo,
        umbrales.auc_maximo,
    )

    # Etiqueta de versión calculada ya aquí (y no más abajo, tras el
    # criterio de aceptación) porque el módulo de evaluación (sección 8
    # del documento "Métricas de evaluación del modelo") registra la
    # evaluación de AMBOS desenlaces, aceptado o rechazado — el propio
    # dbo.sp_RegistrarEvaluacion deja constancia del motivo de rechazo,
    # es la bitácora de evaluaciones, no solo de modelos aceptados.
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    factor, offset = factor_offset()
    score_config = cargar_config().get("score", {})
    score_min = score_config.get("score_min", 0)
    score_max = score_config.get("score_max", 1000)
    cliente_col = config.get("columna_cliente", "ClienteId")

    try:
        evaluar_modelo(
            version_modelo=marca_tiempo,
            alternativa="REGRESION_LOGISTICA_WOE",
            es_modelo_servido=True,
            modelo=modelo,
            x_val=x_val,
            y_val=y_val,
            df_cv=df,
            target_col=target_col,
            cliente_col=cliente_col,
            n_entrenamiento=len(train_df),
            entrenador_cv=entrenar_regresion_logistica,
            factor=factor,
            offset=offset,
            score_min=score_min,
            score_max=score_max,
        )
    except Exception as exc:
        # La evaluación es un registro de auditoría adicional; un fallo
        # aquí no debe impedir que el modelo se acepte o se rechace por
        # sus propios criterios (ks_minimo/auc_maximo), calculados aparte.
        logger.error("No se pudo registrar la evaluación de REGRESION_LOGISTICA_WOE para la versión %s: %s", marca_tiempo, exc)

    if config.get("entrenar_xgboost", True):
        try:
            modelo_xgb = entrenar_xgboost(x_train, y_train)
            proba_xgb = modelo_xgb.predict_proba(x_val)[:, 1]
            auc_xgb = float(roc_auc_score(y_val, proba_xgb))
            ks_xgb = calcular_ks(y_val, proba_xgb)
            logger.info("Validación (XGBoost, solo comparación): AUC=%.4f, KS=%.4f.", auc_xgb, ks_xgb)
            try:
                evaluar_modelo(
                    version_modelo=marca_tiempo,
                    alternativa="XGBOOST",
                    es_modelo_servido=False,
                    modelo=modelo_xgb,
                    x_val=x_val,
                    y_val=y_val,
                    df_cv=df,
                    target_col=target_col,
                    cliente_col=cliente_col,
                    n_entrenamiento=len(train_df),
                    entrenador_cv=entrenar_xgboost,
                    factor=factor,
                    offset=offset,
                    score_min=score_min,
                    score_max=score_max,
                )
            except Exception as exc:
                logger.error("No se pudo registrar la evaluación de XGBOOST para la versión %s: %s", marca_tiempo, exc)
        except DatosInsuficientesError:
            pass

    # Criterio de aceptación: la validación cruzada agrupada por cliente
    # gobierna (oof), calculada aquí de forma independiente de
    # evaluar_modelo() —que necesita base de datos real y cuyo fallo nunca
    # debe impedir la aceptación— para que la decisión siga siendo
    # verificable sin conexión a SQL Server. El holdout temporal (ventana
    # x_val/y_val, ya calculada arriba) solo veta, nunca promueve por sí
    # solo (decidir_promocion, src/ajustes_evaluacion.py). El piso/techo de
    # KS/AUC propiamente dichos los decide modelo_aceptado(), aquí mismo.
    def _ajustar_y_predecir(df_train, df_test):
        x_train_fold, mapeo_fold = preparar_features(df_train, target_col=target_col)
        x_test_fold = transformar_a_woe(df_test, mapeo_fold)
        modelo_fold = entrenar_regresion_logistica(x_train_fold, df_train[target_col])
        return modelo_fold.predict_proba(x_test_fold)[:, 1]

    oof = metricas_fuera_de_muestra(
        df,
        ajustar_y_predecir=_ajustar_y_predecir,
        target_col=target_col,
        cliente_col=cliente_col,
        n_particiones=config.get("n_particiones_crossval", 5),
    )
    holdout = {"auc": auc, "n_evaluados": len(y_val)}
    logger.info(
        "Fuera de muestra (gobierna la aceptación): AUC=%.4f, KS=%.4f sobre %d casos, %d particiones válidas.",
        oof["auc"], oof["ks"], oof["n_evaluados"], oof["n_particiones_validas"],
    )
    cfg_decision = SimpleNamespace(train=SimpleNamespace(**config))
    aceptado_ks_auc, motivo_ks_auc = modelo_aceptado(oof["ks"], oof["auc"], umbrales)
    aceptado, motivo = decidir_promocion(oof, holdout, cfg_decision, aceptado_ks_auc, motivo_ks_auc)
    if not aceptado:
        logger.error(motivo)
        raise ModeloRechazadoError(motivo)

    scorecard = generar_scorecard(modelo, mapeo, list(x_train.columns))
    ruta_scorecard = _resolver_ruta("ruta_scorecard", "models/scorecard.csv")
    scorecard.to_csv(ruta_scorecard, index=False, encoding="utf-8")
    logger.info("Scorecard exportado a %s (%d filas).", ruta_scorecard, len(scorecard))

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
