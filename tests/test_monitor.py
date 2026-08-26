import pandas as pd
import pytest

from src import monitor


# ---------- _extraer_score_de_motivo ----------


@pytest.mark.parametrize(
    "motivo,esperado",
    [
        ("Actualización automática por modelo de scoring (score: 415.00)", 415.0),
        ("Reclasificación (score: 900.50)", 900.5),
        ("SCORE: 123", 123.0),
        ("Sin score en el texto", None),
        ("", None),
        (None, None),
    ],
)
def test_extraer_score_de_motivo(motivo, esperado):
    assert monitor._extraer_score_de_motivo(motivo) == esperado


# ---------- evaluar_desempeno ----------


def _df_monitoreo_sintetico(n_buenos=15, n_malos=15) -> pd.DataFrame:
    filas = []
    for i in range(n_buenos):
        filas.append(
            {
                "FacturasPosteriores": 2,
                "MorosasPosteriores": 0,
                "Motivo": f"Reclasificación (score: {700 + i})",
            }
        )
    for i in range(n_malos):
        filas.append(
            {
                "FacturasPosteriores": 2,
                "MorosasPosteriores": 1,
                "Motivo": f"Reclasificación (score: {300 + i})",
            }
        )
    return pd.DataFrame(filas)


def test_evaluar_desempeno_lanza_error_si_hay_pocos_registros_observables():
    df = _df_monitoreo_sintetico(n_buenos=2, n_malos=2)
    with pytest.raises(monitor.DatosInsuficientesError):
        monitor.evaluar_desempeno(df, minimo_registros=20)


def test_evaluar_desempeno_ignora_registros_sin_facturacion_posterior():
    df = _df_monitoreo_sintetico(n_buenos=15, n_malos=15)
    sin_desenlace = pd.DataFrame(
        [{"FacturasPosteriores": 0, "MorosasPosteriores": None, "Motivo": "Reclasificación (score: 500.00)"}] * 50
    )
    df_con_ruido = pd.concat([df, sin_desenlace], ignore_index=True)
    # No debería contar las 50 filas sin facturación posterior ni fallar por ellas.
    resultado = monitor.evaluar_desempeno(df_con_ruido, minimo_registros=20)
    assert resultado["n_observaciones"] == 30


def test_evaluar_desempeno_calcula_ks_y_alerta_bajo_el_umbral(monkeypatch):
    monkeypatch.setattr(monitor, "cargar_modelo", lambda: {"metricas": {"ks": 0.50}})
    df = _df_monitoreo_sintetico(n_buenos=15, n_malos=15)  # separación perfecta -> KS=1.0

    resultado = monitor.evaluar_desempeno(df, minimo_registros=20)

    assert resultado["ks_actual"] == 1.0
    assert resultado["ks_validacion"] == 0.50
    assert resultado["umbral_alerta"] == pytest.approx(0.40)  # 0.50 * 0.8 (config real, no mockeada)
    assert resultado["alerta_reentrenamiento"] is False  # 1.0 no está por debajo de 0.40


def test_evaluar_desempeno_dispara_alerta_si_ks_cae(monkeypatch):
    monkeypatch.setattr(monitor, "cargar_modelo", lambda: {"metricas": {"ks": 0.90}})
    monkeypatch.setattr(monitor, "_config_monitor", lambda: {"minimo_registros": 20, "umbral_ks_relativo": 0.8})

    # KS bajo a propósito: mezclar scores de buenos y malos casi igual.
    filas = [{"FacturasPosteriores": 1, "MorosasPosteriores": i % 2, "Motivo": f"(score: {500 + i})"} for i in range(30)]
    df = pd.DataFrame(filas)

    resultado = monitor.evaluar_desempeno(df)

    assert resultado["umbral_alerta"] == pytest.approx(0.72)  # 0.90 * 0.8
    assert resultado["ks_actual"] < resultado["umbral_alerta"]
    assert resultado["alerta_reentrenamiento"] is True


# ---------- calcular_psi ----------


def _info_numerica_prueba() -> dict:
    return {
        "tipo": "numerica",
        "bordes": [float("-inf"), 50.0, float("inf")],
        "tramos": [
            {"tramo_id": 0, "n": 50},
            {"tramo_id": 1, "n": 50},
        ],
    }


def test_calcular_psi_distribucion_identica_es_cercana_a_cero():
    info = _info_numerica_prueba()
    # Misma proporción 50/50 que en entrenamiento -> PSI ~ 0.
    serie_actual = pd.Series([10.0] * 50 + [90.0] * 50)
    assert monitor.calcular_psi(serie_actual, info) == pytest.approx(0.0, abs=1e-9)


def test_calcular_psi_distribucion_desplazada_es_mayor_que_cero():
    info = _info_numerica_prueba()
    # Casi todo cae ahora en el tramo alto: distribución muy distinta.
    serie_actual = pd.Series([10.0] * 5 + [90.0] * 95)
    psi = monitor.calcular_psi(serie_actual, info)
    assert psi > 0.25  # cambio significativo por diseño del caso de prueba


def test_calcular_psi_categorica_usa_valor_a_bucket():
    info = {
        "tipo": "categorica",
        "valor_a_bucket": {1: 1, 2: 2, 3: "OTRAS"},
        "tramos": [
            {"tramo_id": 1, "n": 30},
            {"tramo_id": 2, "n": 30},
            {"tramo_id": "OTRAS", "n": 40},
        ],
    }
    serie_actual = pd.Series([1] * 30 + [2] * 30 + [3] * 40)
    assert monitor.calcular_psi(serie_actual, info) == pytest.approx(0.0, abs=1e-9)


def test_calcular_psi_categoria_no_vista_va_a_otras():
    info = {
        "tipo": "categorica",
        "valor_a_bucket": {1: 1},
        "tramos": [{"tramo_id": 1, "n": 50}, {"tramo_id": "OTRAS", "n": 50}],
    }
    # 999 nunca se vio en entrenamiento -> cae en OTRAS, no lanza error.
    serie_actual = pd.Series([1] * 50 + [999] * 50)
    assert monitor.calcular_psi(serie_actual, info) == pytest.approx(0.0, abs=1e-9)


# ---------- detectar_drift ----------


def test_detectar_drift_omite_variables_sin_equivalente_en_inferencia(monkeypatch):
    mapeo = {
        "VarConEquivalente": _info_numerica_prueba(),
        "VarSinEquivalente": _info_numerica_prueba(),
    }
    monkeypatch.setattr(monitor, "cargar_modelo", lambda: {"mapeo": mapeo})
    monkeypatch.setattr(
        monitor,
        "cargar_config",
        lambda: {"score": {"mapa_variables_inferencia": {"VarConEquivalente": "ColActual", "VarSinEquivalente": None}}},
    )
    df_actual = pd.DataFrame({"ColActual": [10.0] * 50 + [90.0] * 50})

    resultado = monitor.detectar_drift(df_actual)

    assert list(resultado["variable"]) == ["VarConEquivalente"]


def test_detectar_drift_marca_alerta_segun_umbral(monkeypatch):
    mapeo = {"VarX": _info_numerica_prueba()}
    monkeypatch.setattr(monitor, "cargar_modelo", lambda: {"mapeo": mapeo})
    monkeypatch.setattr(monitor, "cargar_config", lambda: {"score": {"mapa_variables_inferencia": {}}})
    df_actual = pd.DataFrame({"VarX": [10.0] * 5 + [90.0] * 95})  # muy desplazada

    resultado = monitor.detectar_drift(df_actual, psi_alerta=0.25)

    assert bool(resultado.loc[0, "alerta"])


# ---------- Verificación en vivo ----------


def test_evaluar_desempeno_con_datos_reales_reporta_insuficiencia():
    # dbo.vw_MonitoreoScoring solo tiene, por ahora, una reclasificación
    # de carga inicial (muy por debajo del mínimo) — el paso 8.9 debe
    # detectarlo con un error claro, no fabricar un KS con casi ningún dato.
    with pytest.raises(monitor.DatosInsuficientesError):
        monitor.evaluar_desempeno()


def test_detectar_drift_con_modelo_y_datos_reales():
    resultado = monitor.detectar_drift()

    assert isinstance(resultado, pd.DataFrame)
    assert set(resultado.columns) == {"variable", "psi", "alerta"}
    assert len(resultado) > 0
    assert (resultado["psi"] >= 0).all()  # PSI nunca es negativo por construcción
