"""Medición de deriva — Control 3 de 3 (documento "Controles de
Esquema, Calidad y Deriva de Datos"). A diferencia de esquema (siempre
bloquea) y calidad (bloquea ante críticos), la deriva **nunca** detiene
la ejecución: es una señal de vigencia del modelo, no de validez de los
datos — suspender el scoring justo cuando la población cambió privaría
al negocio de la herramienta en el momento de mayor necesidad.

El cálculo se reparte según la naturaleza de la variable, sin duplicar
lógica, pero ambos orígenes escriben en la misma bitácora
(dbo.DriftMedicion/dbo.DriftDetalleTramo) para que el histórico quede
unificado y comparable:

  - Categóricas (CiudadId): cada valor es su propia categoría, no
    requiere tramos — se calcula enteramente en el motor de base de
    datos (dbo.sp_CalcularDriftCategorico), que además persiste el
    resultado.
  - Numéricas (todas las demás variables del modelo): necesitan
    exactamente los tramos del mapeo WOE serializado — se calculan
    aquí, en Python, y se insertan con Origen='PYTHON'.

La referencia (dbo.BaselineVariable) se persiste una sola vez, al
terminar el entrenamiento, con la misma etiqueta de versión que el
modelo serializado (train.entrenar() la genera y la pasa) — sin esa
correspondencia, medir la deriva más adelante compararía contra una
referencia equivocada.

Limitación conocida: `PlazoPagoId`, que sp_CalcularDriftCategorico sí
soporta, no tiene una columna equivalente en
dbo.vw_DatasetEntrenamientoFactura (la vista de entrenamiento expone
`PlazoOtorgadoDias`, un número de días, no el catálogo). Registrar un
baseline para esa variable exigiría una fuente adicional que hoy no
existe de forma confiable point-in-time; por eso este módulo solo
registra y mide la deriva categórica de `CiudadId`."""

import logging
import math
import uuid

import pandas as pd

from src.config import RAIZ_PROYECTO, cargar_config
from src.db import ejecutar, ejecutar_muchos, read_sql
from src.extract import extraer_features
from src.features import _clasificar_valor

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.drift")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "drift.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())

# Variable de deriva -> columna de dbo.vw_DatasetEntrenamientoFactura de
# la que se deriva su baseline. Debe coincidir con lo que soporta
# dbo.sp_CalcularDriftCategorico (ver limitación en el docstring del módulo).
VARIABLES_CATEGORICAS_SQL = {"CiudadId": "CiudadId"}

_EPS = 0.0001  # suaviza ln(0) / división por cero, igual que el procedimiento SQL


class DriftError(Exception):
    """No se pudo medir la deriva de una variable (p. ej. sin baseline
    registrado). Nunca debe detener el pipeline: quien la reciba debe
    registrarla y continuar con el resto de las variables."""


def registrar_baseline_categoricas(version_modelo: str, fecha_entrenamiento, df_entrenamiento: pd.DataFrame) -> int:
    """Distribución de valores CRUDOS (sin agrupar en 'OTRAS': ese
    agrupamiento es propio de la codificación WOE, no de la medición de
    deriva) de cada variable categórica soportada, sobre el dataset de
    entrenamiento — el mismo criterio que ilustra la documentación."""
    filas = []
    for variable, columna in VARIABLES_CATEGORICAS_SQL.items():
        if columna not in df_entrenamiento.columns:
            logger.warning("No se registró baseline de '%s': la columna '%s' no está en el dataset de entrenamiento.", variable, columna)
            continue
        conteos = df_entrenamiento[columna].dropna().astype(int).astype(str).value_counts()
        total = int(conteos.sum())
        if total == 0:
            continue
        for valor, n in conteos.items():
            filas.append((version_modelo, fecha_entrenamiento, variable, valor, int(n), float(n) / total))

    if filas:
        ejecutar_muchos(
            "INSERT INTO dbo.BaselineVariable (VersionModelo, FechaEntrenamiento, Variable, Tramo, NumRegistros, Proporcion) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            filas,
        )
    return len(filas)


def registrar_baseline_numericas(version_modelo: str, fecha_entrenamiento, mapeo: dict) -> int:
    """Proporción de cada tramo, para cada variable NUMÉRICA del mapeo
    WOE serializado — exactamente los mismos tramos que usa el modelo,
    sin volver a discretizar nada."""
    filas = []
    for variable, info in mapeo.items():
        if info.get("tipo") == "categorica":
            continue
        total = sum(t["n"] for t in info["tramos"])
        if total == 0:
            continue
        for tramo in info["tramos"]:
            filas.append((version_modelo, fecha_entrenamiento, variable, str(tramo["etiqueta"]), tramo["n"], tramo["n"] / total))

    if filas:
        ejecutar_muchos(
            "INSERT INTO dbo.BaselineVariable (VersionModelo, FechaEntrenamiento, Variable, Tramo, NumRegistros, Proporcion) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            filas,
        )
    return len(filas)


def registrar_baseline(version_modelo: str, fecha_entrenamiento, df_entrenamiento: pd.DataFrame, mapeo: dict) -> dict:
    """Orquesta el registro completo del baseline al finalizar el
    entrenamiento, en el mismo punto donde se serializa el modelo
    (train.guardar_modelo) — misma etiqueta de versión para ambos."""
    n_cat = registrar_baseline_categoricas(version_modelo, fecha_entrenamiento, df_entrenamiento)
    n_num = registrar_baseline_numericas(version_modelo, fecha_entrenamiento, mapeo)
    logger.info("Baseline registrado (versión %s): %d filas categóricas, %d filas numéricas.", version_modelo, n_cat, n_num)
    return {"version_modelo": version_modelo, "filas_categoricas": n_cat, "filas_numericas": n_num}


def calcular_drift_categorico(version_modelo: str, variable: str) -> dict:
    """EXEC dbo.sp_CalcularDriftCategorico: calcula, inserta (medición +
    detalle por tramo) y devuelve el diagnóstico ya resuelto
    (dbo.vw_DiagnosticoDrift). Requiere que exista baseline para esa
    variable/versión.

    Nota: el RAISERROR "no existe baseline" del propio procedimiento no
    dispara de forma confiable cuando la población actual no está vacía
    (el FULL OUTER JOIN interno conserva sus filas aunque el baseline no
    tenga ninguna), así que se verifica aquí primero, en vez de confiar
    en que el procedimiento lo señale con un error SQL claro."""
    existe_baseline = read_sql(
        "SELECT TOP 1 1 AS existe FROM dbo.BaselineVariable WHERE VersionModelo = ? AND Variable = ?",
        (version_modelo, variable),
    )
    if existe_baseline.empty:
        raise DriftError(f"No existe baseline registrado para la variable '{variable}' en la versión '{version_modelo}'.")

    try:
        resultado = read_sql(
            "EXEC dbo.sp_CalcularDriftCategorico @VersionModelo = ?, @Variable = ?",
            (version_modelo, variable),
        )
    except Exception as exc:
        raise DriftError(f"No se pudo calcular la deriva categórica de '{variable}': {exc}") from exc

    if resultado.empty:
        raise DriftError(f"dbo.sp_CalcularDriftCategorico no devolvió diagnóstico para '{variable}'.")

    fila = resultado.iloc[0].to_dict()
    logger.info(
        "Deriva categórica [%s]: PSI=%.4f, %.1f%% fuera de rango, diagnóstico: %s.",
        variable,
        fila.get("PSI") or 0.0,
        fila.get("PctFueraDeRango") or 0.0,
        fila.get("Diagnostico"),
    )
    return fila


def calcular_drift_numerico(version_modelo: str, variable: str, info_mapeo: dict, serie_actual: pd.Series) -> dict:
    """PSI de una variable NUMÉRICA, reutilizando exactamente los tramos
    del mapeo WOE serializado (los mismos bordes de corte que usa el
    modelo), e insertando el resultado en dbo.DriftMedicion/
    dbo.DriftDetalleTramo con Origen='PYTHON' — la misma bitácora que
    usan las categóricas calculadas en SQL."""
    valores = serie_actual.dropna()
    total_actual = len(valores)
    if total_actual == 0:
        raise DriftError(f"No hay valores no nulos de '{variable}' en la población actual para medir su deriva.")

    bordes = info_mapeo["bordes"]
    minimo, maximo = info_mapeo.get("min_observado"), info_mapeo.get("max_observado")

    conteo_actual: dict = {}
    fuera_de_rango = 0
    for valor in valores:
        if minimo is not None and (valor < minimo or valor > maximo):
            fuera_de_rango += 1
        idx = _clasificar_valor(valor, bordes)
        conteo_actual[idx] = conteo_actual.get(idx, 0) + 1

    total_base = sum(t["n"] for t in info_mapeo["tramos"])
    psi = 0.0
    detalle = []
    for tramo in info_mapeo["tramos"]:
        prop_base = max((tramo["n"] / total_base) if total_base else 0.0, _EPS)
        n_actual = conteo_actual.get(tramo["tramo_id"], 0)
        prop_actual = max(n_actual / total_actual, _EPS)
        aporte = (prop_actual - prop_base) * math.log(prop_actual / prop_base)
        psi += aporte
        detalle.append(
            {
                "tramo": str(tramo["etiqueta"]),
                "prop_base": prop_base,
                "prop_actual": prop_actual,
                "aporte": aporte,
            }
        )

    pct_fuera = 100.0 * fuera_de_rango / total_actual

    ejecucion_id = str(uuid.uuid4())
    ejecutar(
        "INSERT INTO dbo.DriftMedicion (EjecucionId, VersionModelo, Variable, PSI, PctFueraDeRango, Origen) "
        "VALUES (?, ?, ?, ?, ?, 'PYTHON')",
        (ejecucion_id, version_modelo, variable, psi, pct_fuera),
    )
    medicion = read_sql("SELECT MedicionId FROM dbo.DriftMedicion WHERE EjecucionId = ?", (ejecucion_id,))
    medicion_id = int(medicion.iloc[0]["MedicionId"])

    filas_detalle = [
        (medicion_id, d["tramo"], d["prop_base"], d["prop_actual"], d["aporte"], True) for d in detalle
    ]
    ejecutar_muchos(
        "INSERT INTO dbo.DriftDetalleTramo (MedicionId, Tramo, ProporcionBase, ProporcionActual, AportePSI, EnBaseline) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        filas_detalle,
    )

    logger.info("Deriva numérica [%s]: PSI=%.4f, %.1f%% fuera de rango, %d tramos.", variable, psi, pct_fuera, len(detalle))
    return {"MedicionId": medicion_id, "Variable": variable, "PSI": psi, "PctFueraDeRango": pct_fuera, "Origen": "PYTHON"}


def medir_deriva(version_modelo: str, mapeo: dict, df_actual: pd.DataFrame | None = None) -> pd.DataFrame:
    """Orquesta la medición completa de deriva contra un baseline ya
    registrado: numéricas vía Python, CiudadId vía SQL. Nunca lanza —
    un fallo al medir una variable puntual se registra como advertencia
    y se continúa con el resto (la deriva nunca detiene el pipeline)."""
    if df_actual is None:
        df_actual = extraer_features()

    mapa_variables = cargar_config().get("score", {}).get("mapa_variables_inferencia", {})
    resultados = []

    for variable, info in mapeo.items():
        if info.get("tipo") == "categorica":
            continue  # las categóricas del modelo (p. ej. VendedorId) no tienen soporte SQL de deriva
        variable_actual = mapa_variables.get(variable, variable)
        if not variable_actual or variable_actual not in df_actual.columns:
            logger.info("Deriva: '%s' sin equivalente en la vista de inferencia, se omite.", variable)
            continue
        try:
            fila = calcular_drift_numerico(version_modelo, variable, info, df_actual[variable_actual])
            resultados.append({"Variable": fila["Variable"], "PSI": fila["PSI"], "PctFueraDeRango": fila["PctFueraDeRango"], "Origen": "PYTHON"})
        except DriftError as exc:
            logger.warning("No se pudo medir la deriva de '%s': %s", variable, exc)

    for variable in VARIABLES_CATEGORICAS_SQL:
        try:
            fila = calcular_drift_categorico(version_modelo, variable)
            resultados.append({"Variable": fila["Variable"], "PSI": fila["PSI"], "PctFueraDeRango": fila.get("PctFueraDeRango"), "Origen": "SQL"})
        except DriftError as exc:
            logger.warning("No se pudo medir la deriva categórica de '%s': %s", variable, exc)

    logger.info("Medición de deriva completa (versión %s): %d variables evaluadas.", version_modelo, len(resultados))
    return pd.DataFrame(resultados)
