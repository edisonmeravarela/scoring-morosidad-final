import math

import numpy as np
import pandas as pd
import pytest

from src import drift
from src.db import ejecutar, read_sql


# ---------- registrar_baseline_categoricas / numericas (unitarias) ----------


def test_registrar_baseline_categoricas_usa_valores_crudos_no_agrupados(monkeypatch):
    filas_insertadas = []
    monkeypatch.setattr(drift, "ejecutar_muchos", lambda query, filas: filas_insertadas.extend(filas))
    monkeypatch.setattr(drift, "VARIABLES_CATEGORICAS_SQL", {"CiudadId": "CiudadId"})

    df = pd.DataFrame({"CiudadId": [1, 1, 2, 3, 3, 3]})
    n = drift.registrar_baseline_categoricas("v1", "2026-01-01", df)

    assert n == 3  # 3 valores distintos, sin agrupar en "OTRAS"
    tramos = {f[3] for f in filas_insertadas}
    assert tramos == {"1", "2", "3"}
    fila_1 = next(f for f in filas_insertadas if f[3] == "1")
    assert fila_1[4] == 2  # NumRegistros
    assert fila_1[5] == pytest.approx(2 / 6)  # Proporcion


def test_registrar_baseline_categoricas_omite_variable_sin_columna(monkeypatch):
    monkeypatch.setattr(drift, "ejecutar_muchos", lambda query, filas: None)
    monkeypatch.setattr(drift, "VARIABLES_CATEGORICAS_SQL", {"CiudadId": "CiudadId"})
    n = drift.registrar_baseline_categoricas("v1", "2026-01-01", pd.DataFrame({"OtraCosa": [1, 2]}))
    assert n == 0


def test_registrar_baseline_numericas_usa_los_tramos_del_mapeo(monkeypatch):
    filas_insertadas = []
    monkeypatch.setattr(drift, "ejecutar_muchos", lambda query, filas: filas_insertadas.extend(filas))

    mapeo = {
        "VarA": {
            "tipo": "numerica",
            "tramos": [{"tramo_id": 0, "etiqueta": "<= 50", "n": 30}, {"tramo_id": 1, "etiqueta": "> 50", "n": 70}],
        },
        "VarCategorica": {"tipo": "categorica", "tramos": [{"tramo_id": "X", "etiqueta": "X", "n": 100}]},
    }
    n = drift.registrar_baseline_numericas("v1", "2026-01-01", mapeo)

    assert n == 2  # solo los 2 tramos de VarA; la categórica se omite aquí
    variables = {f[2] for f in filas_insertadas}
    assert variables == {"VarA"}
    fila_50 = next(f for f in filas_insertadas if f[3] == "<= 50")
    assert fila_50[5] == pytest.approx(0.30)


# ---------- calcular_drift_numerico (unitaria, PSI verificado a mano) ----------


def _mapeo_numerico_prueba():
    return {
        "bordes": [-np.inf, 50.0, np.inf],
        "min_observado": 0.0,
        "max_observado": 100.0,
        "tramos": [{"tramo_id": 0, "etiqueta": "<= 50", "n": 50}, {"tramo_id": 1, "etiqueta": "> 50", "n": 50}],
    }


def test_calcular_drift_numerico_psi_identico_es_cercano_a_cero(monkeypatch):
    monkeypatch.setattr(drift, "ejecutar", lambda *a, **k: None)
    monkeypatch.setattr(drift, "read_sql", lambda *a, **k: pd.DataFrame({"MedicionId": [1]}))
    monkeypatch.setattr(drift, "ejecutar_muchos", lambda *a, **k: None)

    serie = pd.Series([10.0] * 50 + [90.0] * 50)  # misma proporción 50/50 que el baseline
    resultado = drift.calcular_drift_numerico("v1", "VarA", _mapeo_numerico_prueba(), serie)
    assert resultado["PSI"] == pytest.approx(0.0, abs=1e-6)
    assert resultado["PctFueraDeRango"] == 0.0


def test_calcular_drift_numerico_detecta_valores_fuera_de_rango(monkeypatch):
    monkeypatch.setattr(drift, "ejecutar", lambda *a, **k: None)
    monkeypatch.setattr(drift, "read_sql", lambda *a, **k: pd.DataFrame({"MedicionId": [1]}))
    monkeypatch.setattr(drift, "ejecutar_muchos", lambda *a, **k: None)

    serie = pd.Series([10.0] * 50 + [90.0] * 40 + [200.0] * 10)  # 200 está fuera de [0,100]
    resultado = drift.calcular_drift_numerico("v1", "VarA", _mapeo_numerico_prueba(), serie)
    assert resultado["PctFueraDeRango"] == pytest.approx(10.0)


def test_calcular_drift_numerico_sin_valores_lanza_drifterror(monkeypatch):
    with pytest.raises(drift.DriftError, match="no nulos"):
        drift.calcular_drift_numerico("v1", "VarA", _mapeo_numerico_prueba(), pd.Series([None, None]))


def test_calcular_drift_numerico_inserta_medicion_y_detalle(monkeypatch):
    inserts_medicion = []
    inserts_detalle = []
    monkeypatch.setattr(drift, "ejecutar", lambda query, params: inserts_medicion.append(params))
    monkeypatch.setattr(drift, "read_sql", lambda *a, **k: pd.DataFrame({"MedicionId": [42]}))
    monkeypatch.setattr(drift, "ejecutar_muchos", lambda query, filas: inserts_detalle.extend(filas))

    serie = pd.Series([10.0] * 50 + [90.0] * 50)
    resultado = drift.calcular_drift_numerico("v1", "VarA", _mapeo_numerico_prueba(), serie)

    assert resultado["MedicionId"] == 42
    assert inserts_medicion[0][1] == "v1"  # VersionModelo
    assert inserts_medicion[0][2] == "VarA"  # Variable
    assert len(inserts_detalle) == 2  # un tramo por cada uno de los 2 tramos del mapeo


# ---------- calcular_drift_categorico (unitaria) ----------


def test_calcular_drift_categorico_devuelve_el_diagnostico(monkeypatch):
    monkeypatch.setattr(
        drift,
        "read_sql",
        lambda *a, **k: pd.DataFrame([{"MedicionId": 1, "Variable": "CiudadId", "PSI": 0.4, "PctFueraDeRango": 5.0, "Diagnostico": "DERIVA REAL"}]),
    )
    resultado = drift.calcular_drift_categorico("v1", "CiudadId")
    assert resultado["Diagnostico"] == "DERIVA REAL"


def test_calcular_drift_categorico_sin_baseline_registrado_lanza_drifterror(monkeypatch):
    # El pre-chequeo en Python (no el RAISERROR del procedimiento, que no
    # dispara de forma confiable — ver docstring) es el que detecta esto.
    monkeypatch.setattr(drift, "read_sql", lambda query, params=None: pd.DataFrame(columns=["existe"]))
    with pytest.raises(drift.DriftError, match="No existe baseline"):
        drift.calcular_drift_categorico("v1", "CiudadId")


def test_calcular_drift_categorico_error_del_procedimiento_lanza_drifterror(monkeypatch):
    llamadas = {"n": 0}

    def _read_sql_falso(query, params=None):
        llamadas["n"] += 1
        if llamadas["n"] == 1:
            return pd.DataFrame({"existe": [1]})  # el pre-chequeo confirma que sí hay baseline
        raise RuntimeError("fallo simulado del procedimiento")

    monkeypatch.setattr(drift, "read_sql", _read_sql_falso)
    with pytest.raises(drift.DriftError, match="fallo simulado"):
        drift.calcular_drift_categorico("v1", "CiudadId")


# ---------- medir_deriva (orquestación, nunca lanza) ----------


def test_medir_deriva_continua_si_una_variable_falla(monkeypatch):
    mapeo = {
        "VarBuena": {"tipo": "numerica", **_mapeo_numerico_prueba()},
        "VarMala": {"tipo": "numerica", **_mapeo_numerico_prueba()},
    }
    monkeypatch.setattr(drift, "cargar_config", lambda: {"score": {"mapa_variables_inferencia": {}}})
    monkeypatch.setattr(drift, "VARIABLES_CATEGORICAS_SQL", {})

    df_actual = pd.DataFrame({"VarBuena": [10.0] * 20, "VarMala": [10.0] * 20})

    def _calcular_falso(version, variable, info, serie):
        if variable == "VarMala":
            raise drift.DriftError("fallo simulado")
        return {"MedicionId": 1, "Variable": variable, "PSI": 0.1, "PctFueraDeRango": 0.0, "Origen": "PYTHON"}

    monkeypatch.setattr(drift, "calcular_drift_numerico", _calcular_falso)

    resultado = drift.medir_deriva("v1", mapeo, df_actual)  # no debe lanzar

    assert list(resultado["Variable"]) == ["VarBuena"]


def test_medir_deriva_omite_variables_sin_equivalente_en_inferencia(monkeypatch):
    mapeo = {"VarSinEquivalente": {"tipo": "numerica", **_mapeo_numerico_prueba()}}
    monkeypatch.setattr(drift, "cargar_config", lambda: {"score": {"mapa_variables_inferencia": {"VarSinEquivalente": None}}})
    monkeypatch.setattr(drift, "VARIABLES_CATEGORICAS_SQL", {})

    resultado = drift.medir_deriva("v1", mapeo, pd.DataFrame({"Otra": [1]}))
    assert resultado.empty


def test_medir_deriva_omite_variables_categoricas_del_modelo(monkeypatch):
    mapeo = {"VendedorId": {"tipo": "categorica", "tramos": []}}
    monkeypatch.setattr(drift, "cargar_config", lambda: {"score": {"mapa_variables_inferencia": {}}})
    monkeypatch.setattr(drift, "VARIABLES_CATEGORICAS_SQL", {})
    resultado = drift.medir_deriva("v1", mapeo, pd.DataFrame({"VendedorId": [1]}))
    assert resultado.empty


# ---------- Verificación en vivo, con baseline descartable ----------


VERSION_PRUEBA = "TEST_DRIFT_8_11"


def _limpiar_version_prueba():
    ejecutar("DELETE FROM dbo.DriftDetalleTramo WHERE MedicionId IN (SELECT MedicionId FROM dbo.DriftMedicion WHERE VersionModelo = ?)", (VERSION_PRUEBA,))
    ejecutar("DELETE FROM dbo.DriftMedicion WHERE VersionModelo = ?", (VERSION_PRUEBA,))
    ejecutar("DELETE FROM dbo.BaselineVariable WHERE VersionModelo = ?", (VERSION_PRUEBA,))


@pytest.fixture
def baseline_descartable():
    _limpiar_version_prueba()
    df_entrenamiento = pd.DataFrame({"CiudadId": [1, 1, 2, 3]})
    mapeo = {
        "PorcMoraPrevia": {
            "tipo": "numerica",
            "bordes": [-math.inf, 0.5, math.inf],
            "min_observado": 0.0,
            "max_observado": 1.0,
            "tramos": [{"tramo_id": 0, "etiqueta": "<= 0.5", "n": 10}, {"tramo_id": 1, "etiqueta": "> 0.5", "n": 10}],
        }
    }
    drift.registrar_baseline(VERSION_PRUEBA, "2026-01-01", df_entrenamiento, mapeo)
    try:
        yield mapeo
    finally:
        _limpiar_version_prueba()


def test_registrar_y_calcular_drift_numerico_contra_la_base_real(baseline_descartable):
    mapeo = baseline_descartable
    serie_actual = pd.Series([0.1] * 15 + [0.9] * 5)
    resultado = drift.calcular_drift_numerico(VERSION_PRUEBA, "PorcMoraPrevia", mapeo["PorcMoraPrevia"], serie_actual)

    assert resultado["PSI"] > 0

    guardado = read_sql("SELECT * FROM dbo.DriftMedicion WHERE MedicionId = ?", (resultado["MedicionId"],))
    assert len(guardado) == 1
    assert guardado.iloc[0]["Origen"] == "PYTHON"
    assert guardado.iloc[0]["Interpretacion"]  # la calcula SQL Server como columna computada

    detalle = read_sql("SELECT * FROM dbo.DriftDetalleTramo WHERE MedicionId = ?", (resultado["MedicionId"],))
    assert len(detalle) == 2


def test_calcular_drift_categorico_contra_la_base_real(baseline_descartable):
    resultado = drift.calcular_drift_categorico(VERSION_PRUEBA, "CiudadId")
    assert resultado["PSI"] >= 0
    assert "Diagnostico" in resultado


def test_calcular_drift_categorico_sin_baseline_da_error_claro():
    with pytest.raises(drift.DriftError):
        drift.calcular_drift_categorico("VERSION_QUE_NO_EXISTE_JAMAS", "CiudadId")
