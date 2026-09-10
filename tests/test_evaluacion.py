import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import evaluacion
from src.db import ejecutar, read_sql
from src.train import entrenar_regresion_logistica, factor_offset

# ---------- matriz_confusion / _score_desde_proba (unitarias) ----------


def test_matriz_confusion_cuenta_los_cuatro_casos():
    y_true = [1, 1, 0, 0, 1, 0]
    positivo = [True, False, True, False, True, False]
    conteos = evaluacion.matriz_confusion(y_true, positivo)
    assert conteos == {"VP": 2, "FP": 1, "FN": 1, "VN": 2}


def test_score_desde_proba_es_monotono_decreciente_con_la_probabilidad():
    # A mayor probabilidad de incumplimiento, menor score (más riesgo).
    factor, offset = factor_offset()
    proba = np.array([0.01, 0.5, 0.99])
    scores = evaluacion._score_desde_proba(proba, factor, offset, 0, 1000)
    assert scores[0] > scores[1] > scores[2]


def test_score_desde_proba_respeta_los_limites():
    factor, offset = factor_offset()
    # Límites deliberadamente más estrechos que el rango natural del score
    # para estas probabilidades extremas (~886 y ~88 sin recortar),
    # de modo que la prueba ejercite el recorte en sí, no dependa del
    # rango 0-1000 real.
    proba = np.array([1e-9, 1 - 1e-9])
    scores = evaluacion._score_desde_proba(proba, factor, offset, 200, 800)
    assert scores[0] == 800
    assert scores[1] == 200


# ---------- evaluar_validacion_cruzada (unitaria, modelo sintético) ----------


def _dataset_sintetico_para_cv(seed: int = 3, n_clientes: int = 40) -> pd.DataFrame:
    """20 filas por cliente (suficientes para que GroupKFold con 5
    particiones deje ambas clases representadas en cada fold), con una
    variable claramente separadora para que el modelo logístico converja
    sin advertencias."""
    rng = np.random.default_rng(seed)
    filas = []
    for cliente_id in range(1, n_clientes + 1):
        for _ in range(5):
            x = rng.normal()
            target = int(x > 0)
            filas.append({"ClienteId": cliente_id, "VarA": x, "Target": target})
    return pd.DataFrame(filas)


def test_evaluar_validacion_cruzada_devuelve_conteos_y_desviaciones(monkeypatch):
    df = _dataset_sintetico_para_cv()
    factor, offset = factor_offset()

    # WOE real de features.py opera sobre variables numéricas reales; para
    # aislar esta prueba de esa lógica (probada aparte en test_features.py)
    # se sustituye preparar_features/transformar_a_woe por una traducción
    # directa 1:1 de VarA, suficiente para ejercitar el refit por partición.
    import src.features as features_mod

    monkeypatch.setattr(features_mod, "preparar_features", lambda d, target_col: (d[["VarA"]], {}))
    monkeypatch.setattr(features_mod, "transformar_a_woe", lambda d, mapeo: d[["VarA"]])

    umbral = 500
    resultado = evaluacion.evaluar_validacion_cruzada(
        df, "Target", "ClienteId", umbral, entrenar_regresion_logistica, factor, offset, 0, 1000, n_splits=5
    )
    assert resultado["num_particiones"] == 5
    assert sum(resultado["conteos"].values()) == len(df)
    assert resultado["desviacion_f1"] >= 0.0
    assert resultado["desviacion_sensibilidad"] >= 0.0


def test_evaluar_validacion_cruzada_sin_ambas_clases_lanza_error(monkeypatch):
    df = pd.DataFrame({"ClienteId": [1, 2, 3, 4], "VarA": [0.1, 0.2, 0.3, 0.4], "Target": [0, 0, 0, 0]})
    factor, offset = factor_offset()

    import src.features as features_mod

    monkeypatch.setattr(features_mod, "preparar_features", lambda d, target_col: (d[["VarA"]], {}))
    monkeypatch.setattr(features_mod, "transformar_a_woe", lambda d, mapeo: d[["VarA"]])

    with pytest.raises(evaluacion.EvaluacionInsuficienteError):
        evaluacion.evaluar_validacion_cruzada(
            df, "Target", "ClienteId", 500, entrenar_regresion_logistica, factor, offset, 0, 1000, n_splits=2
        )


# ---------- registrar_evaluacion / evaluar_modelo (unitarias, DB simulada) ----------


def test_registrar_evaluacion_llama_al_procedimiento_una_vez_sin_cv(monkeypatch):
    llamadas = []

    def _ejecutar_multi_falso(query, params):
        llamadas.append((query, params))
        return [pd.DataFrame([{"EvaluacionId": 1, "Aceptado": True, "MotivoRechazo": None}])]

    monkeypatch.setattr(evaluacion, "ejecutar_multi_resultset", _ejecutar_multi_falso)

    resultado = evaluacion.registrar_evaluacion(
        "v1", "REGRESION_LOGISTICA_WOE", True, 0.4, 0.7, 100, 30, 10,
        {"VP": 5, "FP": 2, "FN": 3, "VN": 20},
    )
    assert len(llamadas) == 1
    assert "VALIDACION" in llamadas[0][1]
    assert resultado == {"EvaluacionId": 1, "Aceptado": True, "MotivoRechazo": None}


def test_registrar_evaluacion_con_cv_llama_al_procedimiento_dos_veces(monkeypatch):
    llamadas = []

    def _ejecutar_multi_falso(query, params):
        llamadas.append(params)
        return [pd.DataFrame([{"EvaluacionId": 7, "Aceptado": False, "MotivoRechazo": "KS inferior a 0.25"}])]

    monkeypatch.setattr(evaluacion, "ejecutar_multi_resultset", _ejecutar_multi_falso)

    cv = {"conteos": {"VP": 20, "FP": 8, "FN": 12, "VN": 60}, "num_particiones": 5, "desviacion_f1": 0.05, "desviacion_sensibilidad": 0.03}
    resultado = evaluacion.registrar_evaluacion(
        "v1", "XGBOOST", False, 0.4, 0.7, 100, 30, 10,
        {"VP": 5, "FP": 2, "FN": 3, "VN": 20}, cv=cv,
    )
    assert len(llamadas) == 2
    assert "CROSSVAL" in llamadas[1]
    assert resultado["Aceptado"] is False


def test_evaluar_modelo_registra_incluso_si_falla_la_validacion_cruzada(monkeypatch):
    """Un fallo puntual de la validación cruzada (p. ej. muy pocos datos)
    no debe impedir el registro de la evaluación sobre el corte único de
    validación."""
    monkeypatch.setattr(evaluacion, "obtener_umbral_operativo", lambda: 500)

    def _cv_falla(*a, **k):
        raise evaluacion.EvaluacionInsuficienteError("sin datos suficientes")

    monkeypatch.setattr(evaluacion, "evaluar_validacion_cruzada", _cv_falla)

    registros = []
    monkeypatch.setattr(
        evaluacion,
        "registrar_evaluacion",
        lambda **k: registros.append(k) or {"EvaluacionId": 1, "Aceptado": True, "MotivoRechazo": None},
    )

    class ModeloFalso:
        def predict_proba(self, x):
            return np.column_stack([1 - np.linspace(0.1, 0.9, len(x)), np.linspace(0.1, 0.9, len(x))])

    x_val = pd.DataFrame({"a": range(20)})
    y_val = pd.Series([0, 1] * 10)
    factor, offset = factor_offset()

    resultado = evaluacion.evaluar_modelo(
        version_modelo="v1", alternativa="REGRESION_LOGISTICA_WOE", es_modelo_servido=True,
        modelo=ModeloFalso(), x_val=x_val, y_val=y_val, df_cv=pd.DataFrame(), target_col="Target",
        cliente_col="ClienteId", n_entrenamiento=50, entrenador_cv=entrenar_regresion_logistica,
        factor=factor, offset=offset,
    )
    assert resultado["validacion_cruzada"] is None
    assert len(registros) == 1
    assert registros[0]["cv"] is None


# ---------- Verificación en vivo ----------


def test_obtener_umbral_operativo_contra_la_base_real():
    umbral = evaluacion.obtener_umbral_operativo()
    assert isinstance(umbral, int)
    assert 0 <= umbral <= 1000


VERSION_PRUEBA = "TEST_EVALUACION_8_12"


def _limpiar_version_prueba():
    ejecutar(
        "DELETE FROM dbo.EvaluacionMetrica WHERE EvaluacionId IN "
        "(SELECT EvaluacionId FROM dbo.EvaluacionModelo WHERE VersionModelo = ?)",
        (VERSION_PRUEBA,),
    )
    ejecutar("DELETE FROM dbo.EvaluacionModelo WHERE VersionModelo = ?", (VERSION_PRUEBA,))


@pytest.fixture
def evaluacion_descartable():
    _limpiar_version_prueba()
    yield
    _limpiar_version_prueba()


def test_registrar_evaluacion_contra_la_base_real(evaluacion_descartable):
    resultado = evaluacion.registrar_evaluacion(
        version_modelo=VERSION_PRUEBA,
        alternativa="REGRESION_LOGISTICA_WOE",
        es_modelo_servido=False,
        ks=0.35,
        auc=0.72,
        n_entrenamiento=120,
        n_validacion=40,
        n_positivos_validacion=12,
        conteos_validacion={"VP": 9, "FP": 4, "FN": 3, "VN": 24},
        cv={"conteos": {"VP": 45, "FP": 18, "FN": 15, "VN": 122}, "num_particiones": 5, "desviacion_f1": 0.06, "desviacion_sensibilidad": 0.04},
    )
    assert resultado["Aceptado"] is True
    assert resultado["MotivoRechazo"] is None

    fila = read_sql("SELECT * FROM dbo.EvaluacionModelo WHERE VersionModelo = ?", (VERSION_PRUEBA,))
    assert len(fila) == 1
    assert fila.iloc[0]["Alternativa"] == "REGRESION_LOGISTICA_WOE"

    metricas = read_sql("SELECT * FROM dbo.EvaluacionMetrica WHERE EvaluacionId = ?", (resultado["EvaluacionId"],))
    assert set(metricas["Origen"]) == {"VALIDACION", "CROSSVAL"}

    comparativo = read_sql("SELECT * FROM dbo.vw_ComparativoAlternativas WHERE VersionModelo = ?", (VERSION_PRUEBA,))
    assert len(comparativo) == 1


def test_registrar_evaluacion_rechazado_por_ks_bajo_queda_registrado(evaluacion_descartable):
    resultado = evaluacion.registrar_evaluacion(
        version_modelo=VERSION_PRUEBA,
        alternativa="REGRESION_LOGISTICA_WOE",
        es_modelo_servido=False,
        ks=0.10,  # inferior a 0.25: el propio procedimiento lo rechaza
        auc=0.60,
        n_entrenamiento=120,
        n_validacion=40,
        n_positivos_validacion=12,
        conteos_validacion={"VP": 2, "FP": 4, "FN": 10, "VN": 24},
    )
    assert resultado["Aceptado"] is False
    assert "KS" in resultado["MotivoRechazo"]
