"""Evaluación del modelo — documento "Métricas de evaluación del modelo".

Módulo independiente que corre al concluir el entrenamiento (paso 8.4 del
documento) y sustenta tanto la matriz comparativa entre la línea base
interpretable (regresión logística WOE, la que efectivamente se sirve) y
las alternativas de mayor complejidad (XGBoost, solo para comparar — ver
`train.entrenar_xgboost`), como el contraste posterior con el desempeño
observado en producción.

Se apoya en objetos ya construidos en la base de datos:

  - `dbo.CorteSegmento` / `dbo.fn_UmbralOperativo()`: fuente única del
    corte operativo (el score más alto que aún restringe crédito), para
    que la matriz de confusión de este módulo se construya sobre el mismo
    punto de corte que usa la capa de reglas para decidir, y no sobre uno
    propio que podría desalinearse con silencio.
  - `dbo.sp_RegistrarEvaluacion`: persiste una fila en `dbo.EvaluacionModelo`
    (una por versión+alternativa) y una en `dbo.EvaluacionMetrica` por cada
    `Origen` (`VALIDACION` — el corte único sobre el conjunto de
    validación — y `CROSSVAL` — el resumen de la validación cruzada). El propio
    procedimiento deriva precisión/sensibilidad/F1/F2 a partir de los
    cuatro conteos que se le pasan; aquí solo se calculan esos conteos.
  - `dbo.vw_ComparativoAlternativas` / `dbo.vw_DesempenoProduccion` /
    `dbo.vw_DeclaradoVsObservado`: lectura, expuestas también en el menú
    "Métricas de evaluación del modelo" de GestionCartera.

Hallazgo corregido: el corte declarado en `dbo.CorteSegmento`
(ALTO/CRITICO hasta score=499) estaba desalineado en 1 punto respecto a
`config.reglas.cortes_score` (hasta 500), la fuente real que usa
`rules.py` para asignar el segmento a cada cliente. Se corrigió
`dbo.CorteSegmento` (ALTO ahora hasta 500, MEDIO desde 501) para que
`dbo.fn_UmbralOperativo()` devuelva 500, igual que `rules.py` — sin tocar
`rules.py` ni `config.yaml`, que ya tenían el valor correcto.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.config import RAIZ_PROYECTO, cargar_config
from src.db import ejecutar_multi_resultset, read_sql

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.evaluacion")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "evaluacion.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())

_EPS = 1e-6


class EvaluacionInsuficienteError(Exception):
    """No se pudo completar la validación cruzada: ninguna partición tuvo
    ambas clases representadas tanto en entrenamiento como en prueba."""


def obtener_umbral_operativo() -> int:
    """dbo.fn_UmbralOperativo(): el score más alto que aún restringe
    crédito, según dbo.CorteSegmento — alineado con config.reglas.cortes_score
    (ver el hallazgo corregido en el docstring del módulo)."""
    return int(read_sql("SELECT dbo.fn_UmbralOperativo() AS umbral").iloc[0]["umbral"])


def _score_desde_proba(proba, factor: float, offset: float, score_min: int, score_max: int) -> np.ndarray:
    """Traduce una probabilidad de incumplimiento al score tipo FICO
    (misma escala que src/score.py), pasando por el logit — válido tanto
    para la regresión logística (equivalente exacto de su
    `decision_function`) como para cualquier otro clasificador con
    `predict_proba`, lo que permite comparar alternativas sobre el mismo
    corte operativo."""
    proba = np.clip(np.asarray(proba, dtype=float), _EPS, 1 - _EPS)
    logit = np.log(proba / (1 - proba))
    score = offset - factor * logit
    return np.clip(score, score_min, score_max).round().astype(int)


def matriz_confusion(y_true, predicho_positivo) -> dict:
    """Cuenta VP/FP/FN/VN. "Positivo" = el sistema restringiría crédito
    (score <= umbral operativo); y_true=1 = el cliente incumplió."""
    y_true = pd.Series(y_true).to_numpy()
    predicho_positivo = pd.Series(predicho_positivo).to_numpy().astype(bool)
    return {
        "VP": int((predicho_positivo & (y_true == 1)).sum()),
        "FP": int((predicho_positivo & (y_true == 0)).sum()),
        "FN": int((~predicho_positivo & (y_true == 1)).sum()),
        "VN": int((~predicho_positivo & (y_true == 0)).sum()),
    }


def evaluar_validacion_cruzada(
    df: pd.DataFrame,
    target_col: str,
    cliente_col: str,
    umbral: int,
    entrenador,
    factor: float,
    offset: float,
    score_min: int,
    score_max: int,
    n_splits: int = 5,
) -> dict:
    """Paso 5 (sección 8.3/8.4): validación cruzada agrupada por cliente
    (GroupKFold, nunca aleatoria por fila, mismo criterio que el split
    temporal de train.py) — refita WOE + modelo en cada partición para no
    filtrar información entre folds. Devuelve los conteos VP/FP/FN/VN
    agrupados (out-of-fold, sobre la totalidad de `df`) junto con la
    desviación de F1 y sensibilidad entre particiones, que es lo que
    `sp_RegistrarEvaluacion` no puede derivar por sí solo de un conteo
    agregado."""
    from src.features import preparar_features, transformar_a_woe

    gkf = GroupKFold(n_splits=n_splits)
    conteos_totales = {"VP": 0, "FP": 0, "FN": 0, "VN": 0}
    f1_por_particion, sensibilidad_por_particion = [], []

    for train_idx, test_idx in gkf.split(df, groups=df[cliente_col]):
        train_fold, test_fold = df.iloc[train_idx], df.iloc[test_idx]
        if train_fold[target_col].nunique() < 2 or test_fold[target_col].nunique() < 2:
            logger.warning("Partición de validación cruzada omitida: una clase ausente en entrenamiento o prueba.")
            continue

        x_train_fold, mapeo_fold = preparar_features(train_fold, target_col=target_col)
        x_test_fold = transformar_a_woe(test_fold, mapeo_fold)
        modelo_fold = entrenador(x_train_fold, train_fold[target_col])
        proba_fold = modelo_fold.predict_proba(x_test_fold)[:, 1]
        score_fold = _score_desde_proba(proba_fold, factor, offset, score_min, score_max)

        conteos = matriz_confusion(test_fold[target_col], score_fold <= umbral)
        for clave in conteos_totales:
            conteos_totales[clave] += conteos[clave]

        precision = conteos["VP"] / (conteos["VP"] + conteos["FP"]) if (conteos["VP"] + conteos["FP"]) else None
        sensibilidad = conteos["VP"] / (conteos["VP"] + conteos["FN"]) if (conteos["VP"] + conteos["FN"]) else None
        if sensibilidad is not None:
            sensibilidad_por_particion.append(sensibilidad)
        if precision is not None and sensibilidad is not None and (precision + sensibilidad) > 0:
            f1_por_particion.append(2 * precision * sensibilidad / (precision + sensibilidad))

    if not sensibilidad_por_particion:
        raise EvaluacionInsuficienteError(
            "Ninguna partición de la validación cruzada tuvo ambas clases representadas en entrenamiento y prueba."
        )

    return {
        "conteos": conteos_totales,
        "num_particiones": len(sensibilidad_por_particion),
        "desviacion_f1": float(np.std(f1_por_particion, ddof=1)) if len(f1_por_particion) > 1 else 0.0,
        "desviacion_sensibilidad": float(np.std(sensibilidad_por_particion, ddof=1)) if len(sensibilidad_por_particion) > 1 else 0.0,
    }


def registrar_evaluacion(
    version_modelo: str,
    alternativa: str,
    es_modelo_servido: bool,
    ks: float,
    auc: float,
    n_entrenamiento: int,
    n_validacion: int,
    n_positivos_validacion: int,
    conteos_validacion: dict,
    cv: dict | None = None,
) -> dict:
    """EXEC dbo.sp_RegistrarEvaluacion (paso 7): una llamada con
    Origen='VALIDACION' (siempre) y, si hay resultado de validación
    cruzada, una segunda con Origen='CROSSVAL' (único valor de esa forma
    permitido por CK_EvalMetrica_Origen, junto con 'VALIDACION' y
    'ENTRENAMIENTO') — ambas comparten la misma
    cabecera en dbo.EvaluacionModelo (el procedimiento solo la inserta la
    primera vez que ve esa versión+alternativa)."""
    resultado_sets = ejecutar_multi_resultset(
        "EXEC dbo.sp_RegistrarEvaluacion "
        "@VersionModelo=?, @Alternativa=?, @EsModeloServido=?, @KS=?, @AUC=?, "
        "@RegEntrenamiento=?, @RegValidacion=?, @PositivosValidacion=?, @Origen=?, "
        "@VP=?, @FP=?, @FN=?, @VN=?",
        (
            version_modelo, alternativa, es_modelo_servido, ks, auc,
            n_entrenamiento, n_validacion, n_positivos_validacion, "VALIDACION",
            conteos_validacion["VP"], conteos_validacion["FP"], conteos_validacion["FN"], conteos_validacion["VN"],
        ),
    )
    resultado = resultado_sets[0].iloc[0].to_dict()

    if cv is not None:
        ejecutar_multi_resultset(
            "EXEC dbo.sp_RegistrarEvaluacion "
            "@VersionModelo=?, @Alternativa=?, @EsModeloServido=?, @Origen=?, "
            "@VP=?, @FP=?, @FN=?, @VN=?, @NumParticiones=?, @DesviacionF1=?, @DesviacionSens=?",
            (
                version_modelo, alternativa, es_modelo_servido, "CROSSVAL",
                cv["conteos"]["VP"], cv["conteos"]["FP"], cv["conteos"]["FN"], cv["conteos"]["VN"],
                cv["num_particiones"], cv["desviacion_f1"], cv["desviacion_sensibilidad"],
            ),
        )

    return {
        "EvaluacionId": int(resultado["EvaluacionId"]),
        "Aceptado": bool(resultado["Aceptado"]),
        "MotivoRechazo": resultado["MotivoRechazo"],
    }


def evaluar_modelo(
    version_modelo: str,
    alternativa: str,
    es_modelo_servido: bool,
    modelo,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    df_cv: pd.DataFrame,
    target_col: str,
    cliente_col: str,
    n_entrenamiento: int,
    entrenador_cv,
    factor: float,
    offset: float,
    score_min: int = 0,
    score_max: int = 1000,
    n_splits_cv: int = 5,
) -> dict:
    """Orquesta el paso 8.4 completo para una alternativa: métricas de
    ordenamiento (paso 2, reutilizando KS/AUC ya calculados por quien
    llama no es necesario — se recalculan aquí sobre `x_val`/`y_val` para
    que el módulo sea autosuficiente), matriz de confusión en el corte
    operativo (pasos 3-4), estabilidad por validación cruzada (paso 5) y
    registro (paso 7). Nunca lanza por un fallo de la validación cruzada
    en sí (se registra sin ese componente); si falla la conexión a la
    base al registrar, sí propaga — quien entrena decide si eso debe
    interrumpir el resto del proceso."""
    umbral = obtener_umbral_operativo()

    proba_val = modelo.predict_proba(x_val)[:, 1]
    from src.train import calcular_ks  # import diferido: evita el ciclo evaluacion<->train

    # KS/AUC: preferir el acumulado fuera de muestra de la validación
    # cruzada agrupada (src/ajustes_evaluacion.py:metricas_fuera_de_muestra)
    # sobre la ventana de validación — antes convivían un AUC de la ventana
    # (pocas filas) con VP/FP/FN/VN de la validación cruzada, dos
    # poblaciones distintas en la misma fila de resultados. Si no se puede
    # calcular (p. ej. df_cv vacío o sin ambas clases), se usa la ventana
    # como respaldo — nunca debe impedir el registro de esta fila.
    auc_ventana = float(roc_auc_score(y_val, proba_val))
    ks_ventana = calcular_ks(y_val, proba_val)
    try:
        from src.ajustes_evaluacion import metricas_fuera_de_muestra
        from src.features import preparar_features, transformar_a_woe

        def _ajustar_y_predecir(df_train, df_test):
            x_train_fold, mapeo_fold = preparar_features(df_train, target_col=target_col)
            x_test_fold = transformar_a_woe(df_test, mapeo_fold)
            modelo_fold = entrenador_cv(x_train_fold, df_train[target_col])
            return modelo_fold.predict_proba(x_test_fold)[:, 1]

        oof = metricas_fuera_de_muestra(
            df_cv, ajustar_y_predecir=_ajustar_y_predecir,
            target_col=target_col, cliente_col=cliente_col, n_particiones=n_splits_cv,
        )
        auc, ks = oof["auc"], oof["ks"]
    except Exception as exc:
        logger.warning(
            "No se pudo calcular KS/AUC fuera de muestra para %s/%s (%s); se usa la ventana de validación.",
            version_modelo, alternativa, exc,
        )
        auc, ks = auc_ventana, ks_ventana

    score_val = _score_desde_proba(proba_val, factor, offset, score_min, score_max)
    conteos_val = matriz_confusion(y_val, score_val <= umbral)

    try:
        cv = evaluar_validacion_cruzada(
            df_cv, target_col, cliente_col, umbral, entrenador_cv, factor, offset, score_min, score_max, n_splits_cv
        )
    except EvaluacionInsuficienteError as exc:
        logger.warning("Validación cruzada omitida para %s/%s: %s", version_modelo, alternativa, exc)
        cv = None

    registro = registrar_evaluacion(
        version_modelo=version_modelo,
        alternativa=alternativa,
        es_modelo_servido=es_modelo_servido,
        ks=ks,
        auc=auc,
        n_entrenamiento=n_entrenamiento,
        n_validacion=len(y_val),
        n_positivos_validacion=int((pd.Series(y_val) == 1).sum()),
        conteos_validacion=conteos_val,
        cv=cv,
    )

    logger.info(
        "Evaluación registrada [%s/%s]: KS=%.4f AUC=%.4f umbral=%d VP=%d FP=%d FN=%d VN=%d -> %s%s",
        version_modelo,
        alternativa,
        ks,
        auc,
        umbral,
        conteos_val["VP"], conteos_val["FP"], conteos_val["FN"], conteos_val["VN"],
        "ACEPTADO" if registro["Aceptado"] else f"RECHAZADO: {registro['MotivoRechazo']}",
        f" (CV: {cv['num_particiones']} particiones, σF1={cv['desviacion_f1']:.4f})" if cv else " (sin validación cruzada)",
    )

    return {
        "ks": ks,
        "auc": auc,
        "umbral_operativo": umbral,
        "conteos_validacion": conteos_val,
        "validacion_cruzada": cv,
        **registro,
    }
