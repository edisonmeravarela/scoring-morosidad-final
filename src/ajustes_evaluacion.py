"""
ajustes_evaluacion.py — Correcciones a la partición y a la evaluación del modelo
================================================================================
Módulo único con los tres ajustes propuestos para `scoring-morosidad`.
Colocar en `src/ajustes_evaluacion.py` y conectarlo según la sección
"CÓMO CONECTARLO" de abajo.

POR QUÉ
-------
La corrida registrada 20261007_183452 dejó tres problemas a la vista:

1. La ventana de validación temporal quedó en 5 registros (de 22 en la corrida
   anterior) porque la purga de clientes vacía el conjunto de prueba cuando la
   cartera está concentrada. Con 5 casos, KS y AUC no significan nada.

2. KS y AUC se calculan sobre esa ventana mientras que VP/FP/FN/VN salen de la
   validación cruzada. Son dos poblaciones distintas en la misma fila de
   resultados: por eso convive un AUC de 0,2500 con una sensibilidad de 0,8108.

3. El criterio de aceptación solo impone piso de KS (0,25) y techo de AUC
   (0,95). Un AUC de 0,25 —un modelo que ordena al revés— pasa el filtro y
   puede sobrescribir al modelo vigente. Falta el piso de AUC.

QUÉ HACE ESTE MÓDULO
--------------------
  dividir_particion_agrupada()   reparto 80/20 por cliente, sin solapamiento
                                 y sin necesidad de purgar
  metricas_fuera_de_muestra()    KS y AUC sobre el acumulado fuera de muestra,
                                 la misma fuente de la que ya salen los conteos
  decidir_promocion()            criterio de aceptación con piso de AUC; la
                                 validación cruzada gobierna, el holdout veta

CÓMO CONECTARLO
---------------
  src/train.py
      - Reemplazar la llamada a `dividir_temporal_por_cliente(...)` por
        `dividir_particion_agrupada(df, prop_validacion=cfg.train.prop_validacion,
         estricto=cfg.train.particion_estricta)`.
      - Sustituir el bloque de criterios de aceptación por `decidir_promocion(...)`.

  src/evaluacion.py
      - En `evaluar_modelo(...)`, obtener KS y AUC de `metricas_fuera_de_muestra(...)`
        en lugar de calcularlos sobre la ventana de validación.
      - Seguir registrando las dos filas (Origen VALIDACION y CROSSVAL); la que
        gobierna la decisión es CROSSVAL.

  config/config.yaml
      - Ver CONFIG_SUGERIDA al final de este archivo.

Los tres umbrales de aceptación (KS mínimo, AUC mínimo, AUC máximo) ya NO
viven en config.yaml: `UmbralesAceptacion`/`cargar_umbrales`/`modelo_aceptado`
viven en `src/train.py` (junto al resto del criterio de aceptación) y leen
de dbo.ParametroScoring vía dbo.fn_ParametroScoring (envolviendo
src/db.py::consultar_escalar), con un respaldo si la base no responde.
`decidir_promocion()` de este módulo ya no calcula KS/AUC: recibe el
resultado de `modelo_aceptado()` ya calculado por `train.py`, y solo añade
la comprobación de particiones válidas y el veto del holdout temporal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

log = logging.getLogger(__name__)


class ParticionInsuficienteError(Exception):
    """El conjunto de validación no alcanza el tamaño mínimo exigible."""


# ============================================================================
# 1. Partición 80/20 agrupada por cliente
# ============================================================================
def dividir_particion_agrupada(
    df: pd.DataFrame,
    *,
    fecha_col: str = "FechaEmision",
    cliente_col: str = "ClienteId",
    target_col: str = "Target",
    prop_validacion: float = 0.20,
    min_filas_validacion: int = 30,
    min_positivos_validacion: int = 10,
    estricto: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en entrenamiento y validación por cliente.

    Parameters
    ----------
    prop_validacion
        Proporción objetivo de FILAS para validación (0.20 = 20 %).
        El reparto se hace por cliente, así que la proporción final cae
        en el cliente más cercano y rara vez es exacta.
    min_filas_validacion, min_positivos_validacion
        Guardas duras. Por debajo de ellas las métricas de la ventana de
        validación no son interpretables.
    estricto
        Si es True, lanza ParticionInsuficienteError cuando no se cumplen
        las guardas. Si es False, registra la advertencia y continúa — el
        llamador debe entonces reportar solo la validación cruzada.

    Returns
    -------
    (entrenamiento, validacion)
    """
    if not 0.05 <= prop_validacion <= 0.50:
        raise ValueError("prop_validacion debe estar entre 0.05 y 0.50")

    d = df.copy()
    d[fecha_col] = pd.to_datetime(d[fecha_col])

    # Fecha de referencia de cada cliente: su factura más reciente.
    # Ordena de más antiguo a más reciente.
    ref = d.groupby(cliente_col)[fecha_col].max().sort_values()
    filas = d.groupby(cliente_col).size()

    objetivo = max(min_filas_validacion, int(round(len(d) * prop_validacion)))

    # Llena validación desde el cliente más reciente hacia atrás.
    clientes_val: list = []
    acumulado = 0
    for cid in reversed(ref.index.tolist()):
        if acumulado >= objetivo:
            break
        clientes_val.append(cid)
        acumulado += int(filas[cid])

    mask = d[cliente_col].isin(clientes_val)
    entrenamiento = d.loc[~mask].copy()
    validacion = d.loc[mask].copy()

    # ---- diagnóstico, siempre a la bitácora -----------------------------
    pos_val = int(validacion[target_col].sum()) if target_col in validacion else -1
    pct = 100 * len(validacion) / max(len(d), 1)
    log.info(
        "Partición agrupada por cliente: %d entrenamiento / %d validación "
        "(%.1f %% del total, %d clientes, %d positivos). "
        "Corte temporal efectivo: %s",
        len(entrenamiento), len(validacion), pct, len(clientes_val), pos_val,
        ref.loc[clientes_val].min().date() if clientes_val else "n/d",
    )

    # ---- guardas --------------------------------------------------------
    problemas = []
    if len(validacion) < min_filas_validacion:
        problemas.append(
            f"validación con {len(validacion)} filas, mínimo exigido {min_filas_validacion}"
        )
    if pos_val >= 0 and pos_val < min_positivos_validacion:
        problemas.append(
            f"validación con {pos_val} casos positivos, mínimo exigido {min_positivos_validacion}"
        )
    if len(entrenamiento) < 3 * len(validacion):
        problemas.append(
            f"entrenamiento ({len(entrenamiento)}) desproporcionado frente a validación ({len(validacion)})"
        )

    if problemas:
        msg = (
            "La partición no permite evaluar con la ventana temporal: "
            + "; ".join(problemas)
            + ". Reportar únicamente la validación cruzada agrupada."
        )
        if estricto:
            raise ParticionInsuficienteError(msg)
        log.warning(msg)

    # Comprobación de la invariante que motiva todo el diseño.
    solapados = set(entrenamiento[cliente_col]) & set(validacion[cliente_col])
    assert not solapados, f"clientes a ambos lados de la partición: {solapados}"

    return entrenamiento, validacion



# ============================================================================
# 2. KS y AUC fuera de muestra
# ============================================================================
def calcular_ks(y_true: np.ndarray, score: np.ndarray) -> float:
    """KS entre la distribución de score de buenos y de malos."""
    from scipy.stats import ks_2samp
    buenos = score[y_true == 0]
    malos = score[y_true == 1]
    if len(buenos) == 0 or len(malos) == 0:
        return float("nan")
    return float(ks_2samp(buenos, malos).statistic)


def metricas_fuera_de_muestra(
    df: pd.DataFrame,
    *,
    ajustar_y_predecir,
    target_col: str = "Target",
    cliente_col: str = "ClienteId",
    n_particiones: int = 5,
) -> dict:
    """KS y AUC calculados sobre predicciones fuera de muestra.

    Parameters
    ----------
    ajustar_y_predecir
        Función `(df_train, df_test) -> np.ndarray` que reajusta el mapeo WOE
        y el modelo sobre `df_train` y devuelve la probabilidad de
        incumplimiento de cada fila de `df_test`. Reajustar dentro de cada
        partición es obligatorio: calcular el WOE una sola vez sobre todo el
        conjunto filtraría información entre particiones.

    Returns
    -------
    dict con las claves:
        auc, ks                      → sobre el acumulado fuera de muestra
        auc_por_particion, ks_...    → una entrada por partición
        desviacion_auc, desviacion_ks
        n_evaluados, n_positivos, n_particiones_validas
    """
    y = df[target_col].to_numpy()
    grupos = df[cliente_col].to_numpy()

    n_clientes = df[cliente_col].nunique()
    if n_particiones > n_clientes:
        raise ValueError(
            f"{n_particiones} particiones con solo {n_clientes} clientes distintos"
        )

    proba_oof = np.full(len(df), np.nan)
    auc_fold, ks_fold = [], []

    for k, (idx_tr, idx_te) in enumerate(
        GroupKFold(n_splits=n_particiones).split(df, y, groups=grupos), start=1
    ):
        d_tr = df.iloc[idx_tr]
        d_te = df.iloc[idx_te]
        proba = np.asarray(ajustar_y_predecir(d_tr, d_te), dtype=float)
        proba_oof[idx_te] = proba

        y_te = y[idx_te]
        if len(np.unique(y_te)) < 2:
            log.warning("Partición %d sin ambas clases: no aporta AUC ni KS.", k)
            continue
        auc_fold.append(float(roc_auc_score(y_te, proba)))
        ks_fold.append(calcular_ks(y_te, proba))

    validos = ~np.isnan(proba_oof)
    y_val = y[validos]
    p_val = proba_oof[validos]

    if len(np.unique(y_val)) < 2:
        raise ValueError("El acumulado fuera de muestra no contiene ambas clases.")

    resultado = {
        "auc": float(roc_auc_score(y_val, p_val)),
        "ks": calcular_ks(y_val, p_val),
        "auc_por_particion": auc_fold,
        "ks_por_particion": ks_fold,
        "desviacion_auc": float(np.std(auc_fold, ddof=1)) if len(auc_fold) > 1 else float("nan"),
        "desviacion_ks": float(np.std(ks_fold, ddof=1)) if len(ks_fold) > 1 else float("nan"),
        "n_evaluados": int(validos.sum()),
        "n_positivos": int(y_val.sum()),
        "n_particiones_validas": len(auc_fold),
        "proba_fuera_de_muestra": proba_oof,
    }

    log.info(
        "Fuera de muestra: AUC %.4f (±%.4f) · KS %.4f (±%.4f) sobre %d casos, "
        "%d positivos, %d particiones válidas",
        resultado["auc"], resultado["desviacion_auc"],
        resultado["ks"], resultado["desviacion_ks"],
        resultado["n_evaluados"], resultado["n_positivos"],
        resultado["n_particiones_validas"],
    )
    return resultado



# ============================================================================
# 3. Criterio de promoción
# ============================================================================
def decidir_promocion(oof: dict, holdout: dict | None, cfg,
                      aceptado_ks_auc: bool, motivo_ks_auc: str | None) -> tuple[bool, str | None]:
    """Decide si el modelo se promueve.

    La métrica que gobierna es la de validación cruzada, porque evalúa a
    todos los clientes. El holdout temporal solo puede VETAR —nunca promover
    por sí solo— y únicamente si tiene tamaño suficiente para significar algo.

    `aceptado_ks_auc`/`motivo_ks_auc` son el resultado ya calculado por
    `train.py::modelo_aceptado(oof["ks"], oof["auc"], umbrales)` — este
    módulo no importa `train.py` (evitaría un import circular, ya que
    `train.py` importa `decidir_promocion` de aquí) ni conoce los umbrales.
    """
    min_filas_holdout = getattr(cfg.train, "min_filas_validacion", 30)

    if oof["n_particiones_validas"] < 3:
        return False, "menos de 3 particiones válidas: el resultado no es estimable"

    if not aceptado_ks_auc:
        return False, f"{motivo_ks_auc} (fuera de muestra)"

    # El holdout temporal solo veta, y solo si es interpretable.
    if holdout and holdout.get("n_evaluados", 0) >= min_filas_holdout:
        if holdout.get("auc", 1.0) < 0.50:
            return False, (
                f"el holdout temporal da AUC {holdout['auc']:.4f} sobre "
                f"{holdout['n_evaluados']} casos: el modelo no anticipa, solo interpola"
            )
    elif holdout:
        log.warning(
            "Holdout temporal con %d casos: por debajo del mínimo de %d, no se usa para decidir.",
            holdout.get("n_evaluados", 0), min_filas_holdout,
        )

    return True, None



# ==============================================================================
# 4. Parámetros nuevos para config/config.yaml
# ==============================================================================
CONFIG_SUGERIDA = """
train:
  # --- partición ---------------------------------------------------------
  # Reemplaza a meses_validacion: la proporción ya no depende del calendario.
  prop_validacion: 0.20
  min_filas_validacion: 30
  min_positivos_validacion: 10
  particion_estricta: false      # true = aborta; false = avisa y sigue con crossval

  # --- evaluación --------------------------------------------------------
  n_particiones_crossval: 5

  # Los criterios de aceptación (KS mínimo, AUC mínimo, AUC máximo) NO van
  # aquí: viven en dbo.ParametroScoring, leídos por cargar_umbrales().
"""
