import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import extract, features, score


def test_construir_features_inferencia_traduce_nombres_y_completa_con_cero(monkeypatch):
    target = pd.Series([0, 0, 0, 1, 1, 1, 0, 1])
    var_a = pd.Series([1, 2, 3, 8, 9, 10, 2, 9])
    mapeo = {"VarModeloA": features.calcular_woe_variable(var_a, target, n_tramos=2)}
    columnas_modelo = ["VarModeloA", "VarSinEquivalente"]

    monkeypatch.setattr(
        score,
        "_config_score",
        lambda: {"mapa_variables_inferencia": {"VarModeloA": "VarVistaA", "VarSinEquivalente": None}},
    )

    # VarVistaA=1 (bajo) y VarVistaA=9 (alto) deben caer en tramos distintos.
    df_cliente = pd.DataFrame({"VarVistaA": [1, 9]}, index=pd.Index([101, 102], name="ClienteId"))
    resultado = score._construir_features_inferencia(df_cliente, mapeo, columnas_modelo)

    assert list(resultado.columns) == columnas_modelo
    assert (resultado["VarSinEquivalente"] == 0.0).all()  # sin equivalente -> WOE neutro
    assert resultado.loc[101, "VarModeloA"] != resultado.loc[102, "VarModeloA"]


def test_construir_features_inferencia_usa_mismo_nombre_si_no_esta_mapeada(monkeypatch):
    monkeypatch.setattr(score, "_config_score", lambda: {"mapa_variables_inferencia": {}})
    df_cliente = pd.DataFrame({"CiudadId": [1, 2]}, index=pd.Index([1, 2], name="ClienteId"))
    mapeo = {}  # sin info WOE: solo interesa que la columna se localice por el mismo nombre
    resultado = score._construir_features_inferencia(df_cliente, mapeo, ["CiudadId"])
    assert list(resultado.columns) == ["CiudadId"]


def test_calcular_scores_es_reproducible(monkeypatch):
    target = pd.Series([0, 0, 0, 1, 1, 1, 0, 1])
    var_a = pd.Series([1, 2, 3, 8, 9, 10, 2, 9])
    var_b = pd.Series([5, 4, 6, 1, 2, 1, 5, 2])
    mapeo = {
        "VarModeloA": features.calcular_woe_variable(var_a, target, n_tramos=2),
        "VarModeloB": features.calcular_woe_variable(var_b, target, n_tramos=2),
    }
    columnas = ["VarModeloA", "VarModeloB"]

    x_entrenamiento = features.transformar_a_woe(pd.DataFrame({"VarModeloA": var_a, "VarModeloB": var_b}), mapeo)
    modelo = LogisticRegression().fit(x_entrenamiento, target)

    bundle = {"modelo": modelo, "mapeo": mapeo, "columnas": columnas, "fecha_entrenamiento": "2026-01-01T00:00:00"}
    monkeypatch.setattr(score, "cargar_modelo", lambda ruta_modelo=None: bundle)
    monkeypatch.setattr(
        score, "_config_score", lambda: {"mapa_variables_inferencia": {}, "score_min": 0, "score_max": 1000}
    )

    # Mismos nombres de columna que el modelo: no hace falta traducción.
    df_cliente = pd.DataFrame(
        {"VarModeloA": [1, 9, 5], "VarModeloB": [5, 2, 3]},
        index=pd.Index([201, 202, 203], name="ClienteId"),
    )

    resultado_1 = score.calcular_scores(df_cliente)
    resultado_2 = score.calcular_scores(df_cliente)

    pd.testing.assert_frame_equal(resultado_1, resultado_2)
    assert resultado_1["score"].between(0, 1000).all()
    assert resultado_1["probabilidad_incumplimiento"].between(0.0, 1.0).all()


def test_calcular_scores_recorta_al_rango_configurado(monkeypatch):
    class ModeloExtremoFalso:
        def predict_proba(self, x):
            n = len(x)
            return np.column_stack([np.zeros(n), np.ones(n)])

        def decision_function(self, x):
            return np.full(len(x), 1000.0)  # logit extremo -> score fuera de rango sin recorte

    bundle = {
        "modelo": ModeloExtremoFalso(),
        "mapeo": {},
        "columnas": ["VarX"],
        "fecha_entrenamiento": "2026-01-01T00:00:00",
    }
    monkeypatch.setattr(score, "cargar_modelo", lambda ruta_modelo=None: bundle)
    monkeypatch.setattr(
        score, "_config_score", lambda: {"mapa_variables_inferencia": {}, "score_min": 0, "score_max": 1000}
    )

    df_cliente = pd.DataFrame({"VarX": [1, 2]}, index=pd.Index([1, 2], name="ClienteId"))
    resultado = score.calcular_scores(df_cliente)

    assert (resultado["score"] == 0).all()  # logit de riesgo extremo -> score mínimo
    assert (resultado["probabilidad_incumplimiento"] == 1.0).all()


def test_calcular_scores_con_modelo_real_es_reproducible():
    # Verificación en vivo del paso 8.6: aplica el modelo ya entrenado
    # (paso 8.5) sobre dbo.vw_FeaturesScoringCliente real, dos veces, y
    # exige exactamente el mismo resultado (criterio de aceptación literal).
    df_clientes = extract.extraer_features()

    resultado_1 = score.calcular_scores(df_clientes)
    resultado_2 = score.calcular_scores(df_clientes)

    pd.testing.assert_frame_equal(resultado_1, resultado_2)
    assert len(resultado_1) == len(df_clientes)
    assert list(resultado_1.index) == list(df_clientes.index)
    assert resultado_1["probabilidad_incumplimiento"].between(0.0, 1.0).all()
    assert resultado_1["score"].between(0, 1000).all()
