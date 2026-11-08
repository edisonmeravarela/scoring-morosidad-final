"""
evaluacion_walkforward.py
==========================
Validación temporal con varios orígenes de corte y refit COMPLETO de
artefactos dentro de cada fold (discretización, WOE y selección de
variables — src/features_particion.py), con purga por cliente en cada
origen (misma regla que src/train.py::dividir_particion_agrupada, aplicada
aquí sobre el corte temporal en vez de sobre la proporción fija).

Por qué hace falta: train.py hace el corte TEMPORAL correcto, pero ajusta
las variables sobre el dataset ya transformado a WOE (ajuste global, sin
refit por fold); evaluacion.py::evaluar_validacion_cruzada() SÍ refita por
fold, pero con GroupKFold por cliente, que ignora el orden temporal. Este
módulo combina ambas cosas, y además repite la medición con varios
orígenes de corte en vez de uno solo: con el volumen de datos disponible,
una sola lectura no distingue el desempeño real del modelo de la
variabilidad del muestreo.

Uso:  python -m src.evaluacion_walkforward
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score

from src import features_particion as fp
from src.train import calcular_ks, entrenar_regresion_logistica


@dataclass
class ResultadoParticion:
    origen: int
    corte: pd.Timestamp
    n_train: int
    n_valid: int
    n_variables: int
    auc: float
    ks: float
    purgados: int


def dividir_temporal(
    df: pd.DataFrame,
    corte: pd.Timestamp,
    fecha_col: str,
    cliente_col: str,
    meses_validacion: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Corte temporal + purga por cliente, mismo criterio que
    src/train.py::dividir_temporal_por_cliente: entrena con lo anterior al
    corte, valida con la ventana posterior, y retira de validación a todo
    cliente que ya haya aparecido en entrenamiento (ningún cliente puede
    estar en ambos lados a la vez)."""
    fin_valid = corte + pd.DateOffset(months=meses_validacion)
    train = df[df[fecha_col] < corte]
    valid = df[(df[fecha_col] >= corte) & (df[fecha_col] < fin_valid)]

    clientes_train = set(train[cliente_col])
    antes = len(valid)
    valid = valid[~valid[cliente_col].isin(clientes_train)]
    return train, valid, antes - len(valid)


def evaluar_walkforward(
    df: pd.DataFrame,
    target_col: str = "Target",
    fecha_col: str = "FechaEmision",
    cliente_col: str = "ClienteId",
    n_origenes: int = 3,
    meses_validacion: int = 3,
    min_filas_train: int = 60,
    min_filas_valid: int = 15,
) -> tuple[pd.DataFrame, dict]:
    """Validación temporal con varios orígenes y refit completo por
    partición. Devuelve (detalle_por_origen, resumen_agregado).

    `df` debe conservar el índice que trae extraer_dataset_entrenamiento()
    (FacturaId) — no se reindexa aquí, para que
    features.variables_candidatas() siga excluyendo FacturaId/ClienteId
    exactamente como ya lo hace en producción (vía dtype y
    config/config.yaml:features.variables_excluidas), sin necesitar una
    lista de exclusiones paralela."""
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(fecha_col)

    fecha_min, fecha_max = df[fecha_col].min(), df[fecha_col].max()
    # Orígenes repartidos en la segunda mitad del histórico, para que el
    # primer fold de entrenamiento no quede demasiado corto.
    inicio = fecha_min + (fecha_max - fecha_min) * 0.5
    cortes = pd.date_range(
        start=inicio,
        end=fecha_max - pd.DateOffset(months=meses_validacion),
        periods=n_origenes,
    )

    resultados: list[ResultadoParticion] = []

    for i, corte in enumerate(cortes, start=1):
        train, valid, purgados = dividir_temporal(df, corte, fecha_col, cliente_col, meses_validacion)

        if len(train) < min_filas_train or len(valid) < min_filas_valid:
            print(f"[origen {i}] descartado: train={len(train)} valid={len(valid)}")
            continue
        if valid[target_col].nunique() < 2:
            print(f"[origen {i}] descartado: validación sin ambas clases")
            continue

        # ---- EL PUNTO CLAVE: fit solo con train, transform en ambos -------
        art = fp.ajustar_artefactos(train, target_col)
        if not art:
            print(f"[origen {i}] descartado: ninguna variable superó los controles en este fold")
            continue
        x_train = fp.aplicar_artefactos(train, art)
        x_valid = fp.aplicar_artefactos(valid, art)
        # --------------------------------------------------------------------

        modelo = entrenar_regresion_logistica(x_train, train[target_col])

        proba = modelo.predict_proba(x_valid)[:, 1]
        auc = roc_auc_score(valid[target_col], proba)
        ks = calcular_ks(valid[target_col].to_numpy(), proba)

        resultados.append(
            ResultadoParticion(
                origen=i,
                corte=corte,
                n_train=len(train),
                n_valid=len(valid),
                n_variables=len(art),
                auc=float(auc),
                ks=float(ks),
                purgados=purgados,
            )
        )
        print(
            f"[origen {i}] corte={corte:%Y-%m-%d} train={len(train)} "
            f"valid={len(valid)} purgados={purgados} vars={len(art)} "
            f"AUC={auc:.4f} KS={ks:.4f}"
        )

    if not resultados:
        raise RuntimeError(
            "Ninguna partición fue válida. Con pocos datos, reducir n_origenes o meses_validacion."
        )

    detalle = pd.DataFrame([r.__dict__ for r in resultados])
    resumen = {
        "n_particiones": len(detalle),
        "auc_media": float(detalle["auc"].mean()),
        "auc_desv": float(detalle["auc"].std(ddof=1)) if len(detalle) > 1 else 0.0,
        "auc_min": float(detalle["auc"].min()),
        "auc_max": float(detalle["auc"].max()),
        "ks_media": float(detalle["ks"].mean()),
        "ks_desv": float(detalle["ks"].std(ddof=1)) if len(detalle) > 1 else 0.0,
        "ks_min": float(detalle["ks"].min()),
        "ks_max": float(detalle["ks"].max()),
        "variables_media": float(detalle["n_variables"].mean()),
    }
    return detalle, resumen


def formatear_para_tesis(resumen: dict) -> str:
    """Línea lista para pegar en la sección 7.3 del documento."""
    return (
        f"Bajo validación temporal con {resumen['n_particiones']} orígenes de corte "
        f"y purga por cliente, con discretización, WOE y selección de variables "
        f"ajustados dentro de cada partición, el modelo obtuvo un AUC de "
        f"{resumen['auc_media']:.4f} (desviación {resumen['auc_desv']:.4f}; "
        f"rango {resumen['auc_min']:.4f} a {resumen['auc_max']:.4f}) y un KS de "
        f"{resumen['ks_media']:.4f} (desviación {resumen['ks_desv']:.4f}; "
        f"rango {resumen['ks_min']:.4f} a {resumen['ks_max']:.4f})."
    )


if __name__ == "__main__":
    from src.extract import excluir_sin_historial, extraer_dataset_entrenamiento

    datos = excluir_sin_historial(extraer_dataset_entrenamiento())
    detalle, resumen = evaluar_walkforward(datos)
    print()
    print(detalle.to_string(index=False))
    print()
    print(formatear_para_tesis(resumen))
