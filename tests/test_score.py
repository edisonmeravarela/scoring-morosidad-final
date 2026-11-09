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


# ---------- construir_detalle_variables ----------


class _ModeloLinealFalso:
    """Doble mínimo de LogisticRegression: solo expone coef_/intercept_,
    que es todo lo que construir_detalle_variables necesita (nunca llama a
    predict_proba/decision_function)."""

    def __init__(self, coef: list[float], intercept: float):
        self.coef_ = np.array([coef])
        self.intercept_ = np.array([intercept])


def test_construir_detalle_variables_incluye_fila_base_y_una_fila_por_variable(monkeypatch):
    target = pd.Series([0, 0, 0, 1, 1, 1, 0, 1])
    var_a = pd.Series([1, 2, 3, 8, 9, 10, 2, 9])
    mapeo = {"VarModeloA": features.calcular_woe_variable(var_a, target, n_tramos=2)}
    columnas = ["VarModeloA"]
    bundle = {"modelo": _ModeloLinealFalso([-0.5], 0.1), "mapeo": mapeo, "columnas": columnas}
    monkeypatch.setattr(score, "cargar_modelo", lambda ruta_modelo=None: bundle)
    monkeypatch.setattr(score, "_config_score", lambda: {"mapa_variables_inferencia": {}})

    df_cliente = pd.DataFrame({"VarModeloA": [1, 9]}, index=pd.Index([101, 102], name="ClienteId"))
    detalle = score.construir_detalle_variables(df_cliente)

    assert set(detalle.keys()) == {101, 102}
    filas_101 = detalle[101]
    assert len(filas_101) == 2  # "(base)" + VarModeloA
    factor, offset = score.factor_offset()
    assert filas_101[0] == {
        "Variable": "(base)",
        "Etiqueta": "Puntaje base",
        "ValorOriginal": None,
        "WOE": None,
        "Puntos": round(offset - factor * 0.1, 3),
        "Orden": 0,
    }
    fila_variable = filas_101[1]
    assert fila_variable["Variable"] == "VarModeloA"
    assert fila_variable["ValorOriginal"] == 1  # mismo nombre en df: sin traducción
    assert fila_variable["Orden"] == 1


def test_construir_detalle_variables_puntos_siguen_la_formula_del_scorecard(monkeypatch):
    """puntos = -factor * beta * woe (misma fórmula que
    train.generar_scorecard usa por tramo, aquí con el WOE real del
    cliente) y la fila "(base)" = offset - factor*intercept — exactamente
    el criterio que train.py ya documenta como "auditable, no es una caja
    negra"."""
    bundle = {
        "modelo": _ModeloLinealFalso([2.0, -1.0], 0.5),
        "mapeo": {},
        "columnas": ["VarA", "VarB"],
    }
    monkeypatch.setattr(score, "cargar_modelo", lambda ruta_modelo=None: bundle)
    monkeypatch.setattr(score, "_config_score", lambda: {"mapa_variables_inferencia": {}})
    monkeypatch.setattr(
        score,
        "_construir_features_inferencia",
        lambda df, mapeo, columnas: pd.DataFrame({"VarA": [1.5], "VarB": [-2.0]}, index=df.index),
    )

    df_cliente = pd.DataFrame({"VarA": [10], "VarB": [20]}, index=pd.Index([1], name="ClienteId"))
    detalle = score.construir_detalle_variables(df_cliente)

    factor, offset = score.factor_offset()
    filas = {f["Variable"]: f for f in detalle[1]}
    assert filas["(base)"]["Puntos"] == round(offset - factor * 0.5, 3)
    assert filas["VarA"]["Puntos"] == round(-factor * 2.0 * 1.5, 3)
    assert filas["VarB"]["Puntos"] == round(-factor * -1.0 * -2.0, 3)
    # orden por |puntos| descendente (la fila base siempre va primero, Orden 0)
    assert [f["Orden"] for f in detalle[1]] == [0, 1, 2]
    assert abs(filas["VarA"]["Puntos"]) >= abs(filas["VarB"]["Puntos"])


def test_construir_detalle_variables_valor_original_none_sin_equivalente_de_cliente(monkeypatch):
    """Las variables de entrenamiento sin equivalente a nivel de cliente
    (mapa_variables_inferencia: null) quedan en WOE=0/Puntos=0 (ver
    _construir_features_inferencia) y aquí, además, con ValorOriginal=None
    — no existe un valor crudo que mostrar."""
    bundle = {
        "modelo": _ModeloLinealFalso([1.0], 0.0),
        "mapeo": {},
        "columnas": ["VendedorId"],
    }
    monkeypatch.setattr(score, "cargar_modelo", lambda ruta_modelo=None: bundle)
    monkeypatch.setattr(score, "_config_score", lambda: {"mapa_variables_inferencia": {"VendedorId": None}})

    df_cliente = pd.DataFrame({"CiudadId": [1]}, index=pd.Index([1], name="ClienteId"))
    detalle = score.construir_detalle_variables(df_cliente)

    fila = [f for f in detalle[1] if f["Variable"] == "VendedorId"][0]
    assert fila["ValorOriginal"] is None
    assert fila["WOE"] == 0.0
    assert fila["Puntos"] == 0.0


def test_variable_neutralizada_no_aporta_al_score_ni_aparece_en_el_detalle(monkeypatch):
    """score.variables_neutralizadas fija la variable en WOE=0 al inferir:
    el score deja de incluir su aporte, no genera fila en el detalle y la
    suma del detalle coincide con el score."""
    target = pd.Series([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    ciudad = pd.Series([1, 1, 1, 2, 2, 2, 1, 2, 1, 2])
    monto = pd.Series([1, 2, 3, 8, 9, 10, 2, 9, 3, 8])
    mapeo = {
        "CiudadId": features.calcular_woe_categoria(ciudad, target, min_conteo=1),
        "MontoFactura": features.calcular_woe_variable(monto, target, n_tramos=2),
    }
    columnas = ["MontoFactura", "CiudadId"]
    x = features.transformar_a_woe(pd.DataFrame({"MontoFactura": monto, "CiudadId": ciudad}), mapeo)
    modelo = LogisticRegression().fit(x[columnas], target)
    bundle = {"modelo": modelo, "mapeo": mapeo, "columnas": columnas}
    monkeypatch.setattr(score, "cargar_modelo", lambda ruta_modelo=None: bundle)

    df_cliente = pd.DataFrame({"MontoFactura": [9], "CiudadId": [2]}, index=pd.Index([7], name="ClienteId"))
    config = {"mapa_variables_inferencia": {}, "variables_neutralizadas": []}
    monkeypatch.setattr(score, "_config_score", lambda: config)
    score_con_ciudad = score.calcular_scores(df_cliente)["score"].iloc[0]
    detalle_con_ciudad = score.construir_detalle_variables(df_cliente)[7]
    ciudad_pts = [f["Puntos"] for f in detalle_con_ciudad if f["Variable"] == "CiudadId"][0]
    assert ciudad_pts != 0

    config["variables_neutralizadas"] = ["CiudadId"]
    score_sin_ciudad = score.calcular_scores(df_cliente)["score"].iloc[0]
    detalle = score.construir_detalle_variables(df_cliente)[7]

    assert [f["Variable"] for f in detalle] == ["(base)", "MontoFactura"]
    assert [f["Orden"] for f in detalle] == [0, 1]
    assert abs(score_sin_ciudad - (score_con_ciudad - ciudad_pts)) <= 1.0  # ya no incluye Ciudad
    assert abs(sum(f["Puntos"] for f in detalle) - score_sin_ciudad) <= 1.0  # el detalle cuadra


def test_config_real_neutraliza_ciudad_y_vendedor():
    neutralizadas = score._config_score().get("variables_neutralizadas", [])
    assert "CiudadId" in neutralizadas and "VendedorId" in neutralizadas


def test_construir_detalle_variables_no_deja_tipos_numpy_en_el_resultado(monkeypatch):
    """modelo.coef_/intercept_/df.at[...] son tipos numpy (np.float64,
    np.int64) — sin convertirlos a tipos nativos de Python, pyodbc puede no
    bindearlos igual que un float/int normal al insertar (writeback.py).
    Esta prueba fija ese contrato con un modelo "real" (numpy real, no un
    doble con floats de Python a mano)."""
    target = pd.Series([0, 0, 0, 1, 1, 1, 0, 1])
    var_a = pd.Series([1, 2, 3, 8, 9, 10, 2, 9])
    mapeo = {"VarModeloA": features.calcular_woe_variable(var_a, target, n_tramos=2)}
    columnas = ["VarModeloA"]
    x_entrenamiento = features.transformar_a_woe(pd.DataFrame({"VarModeloA": var_a}), mapeo)
    modelo_real = LogisticRegression().fit(x_entrenamiento, target)  # coef_/intercept_ son np.float64

    bundle = {"modelo": modelo_real, "mapeo": mapeo, "columnas": columnas}
    monkeypatch.setattr(score, "cargar_modelo", lambda ruta_modelo=None: bundle)
    monkeypatch.setattr(score, "_config_score", lambda: {"mapa_variables_inferencia": {}})

    df_cliente = pd.DataFrame({"VarModeloA": [1]}, index=pd.Index([1], name="ClienteId"))
    detalle = score.construir_detalle_variables(df_cliente)

    for fila in detalle[1]:
        assert fila["Puntos"] is None or type(fila["Puntos"]) is float
        assert fila["WOE"] is None or type(fila["WOE"]) is float
        assert fila["ValorOriginal"] is None or type(fila["ValorOriginal"]) in (int, float)


def test_construir_detalle_variables_con_modelo_real_la_suma_de_puntos_reproduce_el_score():
    # Verificación en vivo (solo lectura): la suma de "Puntos" de todas las
    # filas (base y variables) debe reproducir el score de calcular_scores()
    # para el mismo cliente, salvo el margen de redondeo (±1) del round()
    # final; las variables neutralizadas no aparecen ni aportan.
    df_clientes = extract.extraer_features()
    scores = score.calcular_scores(df_clientes)
    detalle = score.construir_detalle_variables(df_clientes)
    ocultas = set(score._config_score().get("variables_neutralizadas", []))

    for cliente_id in df_clientes.index:
        assert not any(f["Variable"] in ocultas for f in detalle[cliente_id])
        suma_puntos = sum(f["Puntos"] for f in detalle[cliente_id])
        assert abs(suma_puntos - scores.loc[cliente_id, "score"]) <= 1.0, cliente_id
