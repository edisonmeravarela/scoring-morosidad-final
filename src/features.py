"""Preparación de variables (Paso 8.4): imputación, discretización en
tramos, transformación WOE (Weight of Evidence) e Information Value (IV).

Convención de WOE usada (estándar de scoring crediticio):
    Bueno = cliente sin mora histórica (TuvoMoraHistorica == 0)
    Malo  = cliente con mora histórica (TuvoMoraHistorica == 1)
    WOE_tramo = ln( %Buenos_en_tramo / %Malos_en_tramo )
Un WOE positivo indica un tramo más seguro que el promedio; uno negativo,
más riesgoso. El Information Value (IV) de una variable es la suma, sobre
todos sus tramos, de (%Buenos - %Malos) * WOE, y mide su poder predictivo.

Los clientes sin abonos generan nulos en las variables de comportamiento de
pago. Esos nulos NO se rellenan con cero (cambiaría su significado): se
tratan como una categoría propia, el tramo "MISSING", con su propio WOE.
"""

import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import RAIZ_PROYECTO, cargar_config

TRAMO_FALTANTE = "MISSING"

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.features")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "features.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())


def _config_features() -> dict:
    return cargar_config().get("features", {})


def variables_candidatas(df: pd.DataFrame, target_col: str, variables_excluidas: list[str] | None = None) -> list[str]:
    """Columnas numéricas CONTINUAS de `df` aptas para discretizar en
    tramos de cuantiles: excluye la variable objetivo, las columnas de
    estado/catálogo/texto configuradas en features.variables_excluidas
    (p.ej. NombreCompleto) y las configuradas en features.variables_categoricas
    (p.ej. VendedorId, CiudadId) — esas van por variables_categoricas_candidatas."""
    config = _config_features()
    excluidas = set(variables_excluidas if variables_excluidas is not None else config.get("variables_excluidas", []))
    excluidas.add(target_col)
    excluidas.update(config.get("variables_categoricas", []))
    numericas = df.select_dtypes(include="number").columns
    return [c for c in numericas if c not in excluidas]


def variables_categoricas_candidatas(df: pd.DataFrame, variables_categoricas: list[str] | None = None) -> list[str]:
    """Columnas configuradas como categóricas (features.variables_categoricas,
    p.ej. VendedorId, CiudadId) que efectivamente están presentes en `df` —
    no todas las vistas tienen las mismas columnas (VendedorId solo existe
    en la vista de entrenamiento, no en la de inferencia por cliente)."""
    categoricas = variables_categoricas if variables_categoricas is not None else _config_features().get("variables_categoricas", [])
    return [c for c in categoricas if c in df.columns]


def identificar_nulos(df: pd.DataFrame, variables: list[str]) -> pd.Series:
    """Cuenta valores nulos por variable (diagnóstico previo a discretizar).

    Estos nulos —típicamente clientes sin abonos, que no tienen cómo tener
    'días de pago' o 'brecha de plazo'— no se imputan con cero: quedan
    como su propia categoría dentro de la discretización (tramo MISSING)."""
    return df[variables].isna().sum()


def _bordes_interiores(serie_no_nula: pd.Series, n_tramos: int) -> list[float]:
    """Cortes de cuantiles sobre los valores no nulos. Con pocos datos o
    valores muy repetidos, qcut puede colapsar tramos duplicados y devolver
    menos cortes de los pedidos; eso es aceptable."""
    if serie_no_nula.nunique() <= 1:
        return []
    _, bordes = pd.qcut(serie_no_nula, q=n_tramos, retbins=True, duplicates="drop")
    return [float(b) for b in bordes[1:-1]]  # sin -inf/+inf: esos se agregan al clasificar


def _etiqueta_tramo(inferior: float, superior: float) -> str:
    if inferior == -np.inf:
        return f"<= {superior:.4g}"
    if superior == np.inf:
        return f"> {inferior:.4g}"
    return f"{inferior:.4g} a {superior:.4g}"


def _clasificar_valor(valor: float, bordes_completos: list[float]) -> int:
    """Índice del tramo (0-based) al que pertenece `valor`, dado
    bordes_completos = [-inf, corte_1, ..., corte_n, +inf]. Con -inf/+inf
    como bordes exteriores, todo valor real cae en algún tramo."""
    return int(np.digitize([valor], bordes_completos[1:-1], right=True)[0])


def calcular_woe_variable(serie: pd.Series, target: pd.Series, n_tramos: int | None = None) -> dict:
    """Discretiza `serie` en tramos, calcula el WOE de cada tramo (incluido
    el tramo MISSING para los nulos) y el IV total de la variable."""
    n_tramos = n_tramos or _config_features().get("n_tramos", 5)

    mascara_no_nula = serie.notna()
    bordes_interiores = _bordes_interiores(serie[mascara_no_nula], n_tramos)
    bordes_completos = [-np.inf, *bordes_interiores, np.inf]

    total_buenos = int((target == 0).sum())
    total_malos = int((target == 1).sum())

    tramo_idx = pd.Series(np.nan, index=serie.index)
    tramo_idx.loc[mascara_no_nula] = serie[mascara_no_nula].apply(lambda v: _clasificar_valor(v, bordes_completos))

    tramos = []
    iv_total = 0.0
    n_tramos_reales = len(bordes_completos) - 1
    for i in range(n_tramos_reales):
        en_tramo = tramo_idx == i
        _agregar_tramo(
            tramos,
            tramo_id=i,
            etiqueta=_etiqueta_tramo(bordes_completos[i], bordes_completos[i + 1]),
            en_tramo=en_tramo,
            target=target,
            total_buenos=total_buenos,
            total_malos=total_malos,
        )
        iv_total += tramos[-1]["iv_parcial"]

    if mascara_no_nula.sum() < len(serie):
        en_tramo = ~mascara_no_nula
        _agregar_tramo(
            tramos,
            tramo_id=-1,
            etiqueta=TRAMO_FALTANTE,
            en_tramo=en_tramo,
            target=target,
            total_buenos=total_buenos,
            total_malos=total_malos,
        )
        iv_total += tramos[-1]["iv_parcial"]

    return {
        "tipo": "numerica",
        "bordes": bordes_completos,
        "tramos": tramos,
        "iv": iv_total,
        "min_observado": float(serie[mascara_no_nula].min()) if mascara_no_nula.any() else None,
        "max_observado": float(serie[mascara_no_nula].max()) if mascara_no_nula.any() else None,
    }


CATEGORIA_OTRAS = "OTRAS"


def calcular_woe_categoria(serie: pd.Series, target: pd.Series, min_conteo: int | None = None) -> dict:
    """WOE para una variable CATEGÓRICA sin orden natural (p.ej. VendedorId,
    CiudadId — sección 5 del documento, "Propias de la factura"): cada
    valor distinto es su propio tramo, no se discretiza en cuantiles.

    Las categorías con menos de `min_conteo` observaciones se agrupan en
    una categoría "OTRAS" antes de calcular WOE (evita tramos con 1-2
    casos); esa misma categoría "OTRAS" es el destino de cualquier valor
    no visto en entrenamiento al transformar datos nuevos, con la misma
    lógica de "tramo más cercano + advertencia" que usan las numéricas."""
    min_conteo = min_conteo if min_conteo is not None else _config_features().get("min_conteo_categoria", 5)

    total_buenos = int((target == 0).sum())
    total_malos = int((target == 1).sum())

    conteos = serie.value_counts(dropna=True)
    categorias_frecuentes = set(conteos[conteos >= min_conteo].index)
    valor_a_bucket = {valor: (valor if valor in categorias_frecuentes else CATEGORIA_OTRAS) for valor in conteos.index}

    buckets = serie.map(valor_a_bucket)  # los nulos quedan como NaN (map no los toca)

    tramos = []
    iv_total = 0.0
    for bucket in sorted(buckets.dropna().unique(), key=str):
        en_tramo = buckets == bucket
        _agregar_tramo(
            tramos,
            tramo_id=bucket,
            etiqueta=str(bucket),
            en_tramo=en_tramo,
            target=target,
            total_buenos=total_buenos,
            total_malos=total_malos,
        )
        iv_total += tramos[-1]["iv_parcial"]

    if serie.isna().any():
        en_tramo = serie.isna()
        _agregar_tramo(
            tramos,
            tramo_id=TRAMO_FALTANTE,
            etiqueta=TRAMO_FALTANTE,
            en_tramo=en_tramo,
            target=target,
            total_buenos=total_buenos,
            total_malos=total_malos,
        )
        iv_total += tramos[-1]["iv_parcial"]

    return {
        "tipo": "categorica",
        "valor_a_bucket": valor_a_bucket,
        "tramos": tramos,
        "iv": iv_total,
    }


def _agregar_tramo(tramos, tramo_id, etiqueta, en_tramo, target, total_buenos, total_malos):
    n = int(en_tramo.sum())
    buenos = int(((target == 0) & en_tramo).sum())
    malos = int(((target == 1) & en_tramo).sum())

    # Suavizado de Laplace: evita ln(0) o división por cero cuando un tramo
    # no tiene buenos o malos (frecuente con datasets chicos).
    dist_buenos = max(buenos, 0.5) / total_buenos if total_buenos else 0.0
    dist_malos = max(malos, 0.5) / total_malos if total_malos else 0.0
    woe = math.log(dist_buenos / dist_malos) if dist_buenos and dist_malos else 0.0
    iv_parcial = (dist_buenos - dist_malos) * woe

    tramos.append(
        {
            "tramo_id": tramo_id,
            "etiqueta": etiqueta,
            "n": n,
            "buenos": buenos,
            "malos": malos,
            "woe": woe,
            "iv_parcial": iv_parcial,
        }
    )


def detectar_separacion_perfecta(df: pd.DataFrame, target_col: str, variables: list[str] | None = None) -> list[str]:
    """Prueba de fuga (documento V1, sección 4.4): para cada variable
    numérica, compara el mínimo de los valores de una clase contra el
    máximo de la otra. Si una clase queda enteramente por encima (o por
    debajo) de la otra, la variable separa perfectamente el target — señal
    casi segura de fuga (o de una identidad matemática con la etiqueta),
    no de poder predictivo genuino, sin importar cuánto o poco IV tenga.

    Filas con valor nulo en la variable no participan (no hay con qué
    comparar); si a alguna de las dos clases no le queda ningún valor no
    nulo, la variable no puede evaluarse y se omite."""
    variables = variables or variables_candidatas(df, target_col)
    target = df[target_col]
    sospechosas = []
    for variable in variables:
        serie = df[variable]
        mascara = serie.notna()
        valores_malos = serie[mascara & (target == 1)]
        valores_buenos = serie[mascara & (target == 0)]
        if valores_malos.empty or valores_buenos.empty:
            continue
        if valores_malos.min() > valores_buenos.max() or valores_buenos.min() > valores_malos.max():
            sospechosas.append(variable)
    if sospechosas:
        logger.warning("Separación perfecta detectada (posible fuga o identidad con la etiqueta): %s", ", ".join(sospechosas))
    return sospechosas


def calcular_mapeo_woe(
    df: pd.DataFrame,
    target_col: str,
    variables: list[str] | None = None,
    variables_categoricas: list[str] | None = None,
) -> dict:
    """Calcula el mapeo WOE/IV de las `variables` numéricas continuas
    (por defecto, las detectadas vía variables_candidatas) y de las
    `variables_categoricas` (por defecto, vía variables_categoricas_candidatas
    — p.ej. VendedorId, CiudadId). Devuelve un dict {variable: resultado},
    incluyendo también las de IV bajo — el filtrado por umbral es un paso
    aparte (filtrar_por_iv) para poder inspeccionar el IV de todas antes de
    decidir."""
    variables = variables if variables is not None else variables_candidatas(df, target_col)
    variables_categoricas = (
        variables_categoricas if variables_categoricas is not None else variables_categoricas_candidatas(df)
    )
    target = df[target_col]
    mapeo = {var: calcular_woe_variable(df[var], target) for var in variables}
    mapeo.update({var: calcular_woe_categoria(df[var], target) for var in variables_categoricas})
    return mapeo


def filtrar_por_iv(mapeo: dict, iv_minimo: float | None = None, iv_maximo: float | None = None) -> dict:
    """Descarta automáticamente las variables con IV por debajo del umbral
    mínimo (0.02 por defecto) por carecer de poder predictivo suficiente,
    y las variables con IV por encima del umbral máximo (1.0 por defecto):
    un IV así de alto casi nunca es señal genuina, es indicio de que la
    variable filtra la etiqueta (leakage) — prueba automática de fuga."""
    config = _config_features()
    iv_minimo = iv_minimo if iv_minimo is not None else config.get("iv_minimo", 0.02)
    iv_maximo = iv_maximo if iv_maximo is not None else config.get("iv_maximo", 1.0)

    descartadas_bajo = [v for v, info in mapeo.items() if info["iv"] < iv_minimo]
    if descartadas_bajo:
        logger.info("Variables descartadas por IV < %.4f: %s", iv_minimo, ", ".join(descartadas_bajo))

    descartadas_fuga = [v for v, info in mapeo.items() if info["iv"] > iv_maximo]
    if descartadas_fuga:
        logger.warning(
            "Variables descartadas por IV > %.4f (posible fuga de la etiqueta): %s",
            iv_maximo,
            ", ".join(descartadas_fuga),
        )

    return {v: info for v, info in mapeo.items() if iv_minimo <= info["iv"] <= iv_maximo}


def transformar_a_woe(df: pd.DataFrame, mapeo: dict) -> pd.DataFrame:
    """Reemplaza cada variable de `mapeo` por el WOE del tramo al que cae
    cada fila. Si un valor cae fuera del rango observado en entrenamiento
    (min_observado/max_observado), igual se asigna al tramo más cercano
    (el primero o el último, gracias a los bordes -inf/+inf) pero se
    registra una advertencia en el log."""
    resultado = pd.DataFrame(index=df.index)
    for variable, info in mapeo.items():
        resultado[variable] = _transformar_variable(df, variable, info)
    return resultado


def _transformar_variable(df: pd.DataFrame, variable: str, info: dict) -> pd.Series:
    if info.get("tipo") == "categorica":
        return _transformar_variable_categorica(df, variable, info)
    return _transformar_variable_numerica(df, variable, info)


def _transformar_variable_numerica(df: pd.DataFrame, variable: str, info: dict) -> pd.Series:
    woe_por_tramo = {t["tramo_id"]: t["woe"] for t in info["tramos"]}
    woe_faltante = woe_por_tramo.get(-1, 0.0)
    bordes = info["bordes"]
    minimo, maximo = info["min_observado"], info["max_observado"]

    valores_woe = []
    for idx, valor in df[variable].items():
        if pd.isna(valor):
            valores_woe.append(woe_faltante)
            continue
        if minimo is not None and (valor < minimo or valor > maximo):
            logger.warning(
                "Cliente %s: %s=%.4g fuera del rango de entrenamiento [%.4g, %.4g]; "
                "asignado al tramo más cercano.",
                idx,
                variable,
                valor,
                minimo,
                maximo,
            )
        tramo_id = _clasificar_valor(valor, bordes)
        valores_woe.append(woe_por_tramo.get(tramo_id, 0.0))
    return pd.Series(valores_woe, index=df.index, name=variable)


def _transformar_variable_categorica(df: pd.DataFrame, variable: str, info: dict) -> pd.Series:
    """Análogo categórico de _transformar_variable_numerica: un valor no
    visto en entrenamiento no tiene un "tramo más cercano" ordenado al que
    asignarse, así que va a la categoría OTRAS (el mismo destino que ya
    reciben las categorías infrecuentes al entrenar) — es la interpretación
    categórica del criterio de aceptación 8.4, con la misma advertencia."""
    valor_a_bucket = info["valor_a_bucket"]
    woe_por_bucket = {t["tramo_id"]: t["woe"] for t in info["tramos"]}
    woe_otras = woe_por_bucket.get(CATEGORIA_OTRAS, 0.0)
    woe_faltante = woe_por_bucket.get(TRAMO_FALTANTE, woe_otras)

    valores_woe = []
    for idx, valor in df[variable].items():
        if pd.isna(valor):
            valores_woe.append(woe_faltante)
            continue
        bucket = valor_a_bucket.get(valor)
        if bucket is None:
            logger.warning(
                "Cliente %s: %s=%r es una categoría no vista en entrenamiento; asignada a OTRAS.",
                idx,
                variable,
                valor,
            )
            bucket = CATEGORIA_OTRAS
        valores_woe.append(woe_por_bucket.get(bucket, woe_otras))
    return pd.Series(valores_woe, index=df.index, name=variable)


def _resolver_ruta_mapeo(ruta) -> Path:
    if ruta is None:
        ruta = _config_features().get("ruta_mapeo", "models/mapeo_woe.pkl")
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else RAIZ_PROYECTO / ruta


def guardar_mapeo(mapeo: dict, ruta=None) -> None:
    ruta = _resolver_ruta_mapeo(ruta)
    ruta.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(mapeo, ruta)
    logger.info("Mapeo WOE guardado en %s (%d variables).", ruta, len(mapeo))


def cargar_mapeo(ruta=None) -> dict:
    return joblib.load(_resolver_ruta_mapeo(ruta))


def preparar_features(df: pd.DataFrame, target_col: str = "TuvoMoraHistorica") -> tuple[pd.DataFrame, dict]:
    """Orquesta el paso 8.4 completo para entrenamiento: aplica la prueba
    de separación perfecta (solo aplicable a variables numéricas continuas),
    calcula WOE/IV de las variables numéricas y categóricas, descarta las
    de IV fuera de rango (bajo poder predictivo o fuga), transforma el
    dataset a WOE y persiste el mapeo (el mismo que debe reutilizarse en
    el scoring, vía cargar_mapeo + transformar_a_woe)."""
    variables = variables_candidatas(df, target_col)
    variables_cat = variables_categoricas_candidatas(df)
    nulos = identificar_nulos(df, variables)
    if nulos.any():
        logger.info("Variables con valores nulos (tratados como tramo MISSING):\n%s", nulos[nulos > 0].to_string())

    separadoras = detectar_separacion_perfecta(df, target_col, variables)
    variables_seguras = [v for v in variables if v not in separadoras]

    mapeo_completo = calcular_mapeo_woe(df, target_col, variables_seguras, variables_cat)
    mapeo_filtrado = filtrar_por_iv(mapeo_completo)
    x_woe = transformar_a_woe(df, mapeo_filtrado)
    guardar_mapeo(mapeo_filtrado)
    return x_woe, mapeo_filtrado
