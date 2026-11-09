"""
features_particion.py
======================
Separa el ajuste (`fit`) y la aplicación (`transform`) de la ingeniería de
variables (paso 8.4), para que la validación por partición (walk-forward,
src/evaluacion_walkforward.py) pueda ajustar discretización, tablas WOE y
selección de variables por Information Value SOLO con el fold de
entrenamiento, y aplicar ese mismo ajuste — sin volver a mirar la etiqueta —
al fold de validación.

No reimplementa nada: `ajustar_artefactos()` ejecuta exactamente los mismos
pasos que ya encadena `features.preparar_features()` (variables_candidatas →
identificar_nulos → detectar_separacion_perfecta → calcular_mapeo_woe →
filtrar_por_iv), deteniéndose justo antes de transformar/guardar — eso ya
no es parte del `fit`. El "artefacto" es el mismo dict `{variable: info}`
que `preparar_features()` ya produce y que `transformar_a_woe()` ya
consume; no hace falta ningún tipo nuevo.
"""

from __future__ import annotations

import logging

import pandas as pd

from src import features as fx

logger = logging.getLogger("scoring_morosidad.features_particion")


def ajustar_artefactos(df_train: pd.DataFrame, target_col: str) -> dict:
    """FIT: aprende tramos, tablas WOE y selección de variables SOLO con
    `df_train`. Todo lo que mira la etiqueta vive aquí; nunca recibe el
    fold de validación."""
    variables = fx.variables_candidatas(df_train, target_col)
    variables_cat = fx.variables_categoricas_candidatas(df_train)

    nulos = fx.identificar_nulos(df_train, variables)
    if nulos.any():
        logger.info(
            "Variables con valores nulos en este fold (tratados como tramo MISSING):\n%s",
            nulos[nulos > 0].to_string(),
        )

    separadoras = fx.detectar_separacion_perfecta(df_train, target_col, variables)
    variables_seguras = [v for v in variables if v not in separadoras]

    mapeo_completo = fx.calcular_mapeo_woe(df_train, target_col, variables_seguras, variables_cat)
    return fx.filtrar_por_iv(mapeo_completo)


def aplicar_artefactos(df: pd.DataFrame, artefactos: dict) -> pd.DataFrame:
    """TRANSFORM: nunca mira la etiqueta. Alias explícito de
    `features.transformar_a_woe()` — separado para que el llamador
    (evaluacion_walkforward.py) deje constancia de en qué paso del
    pipeline está, sin tener que saber que el artefacto es, en este
    proyecto, el mismo mapeo WOE que ya usa el resto del código."""
    return fx.transformar_a_woe(df, artefactos)
