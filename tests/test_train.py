import numpy as np
import pandas as pd
import pytest

from src import calidad, esquema, extract, train


def _veredicto_calidad_ok() -> dict:
    return {"EjecucionId": "prueba", "Criticos": 0, "Advertencias": 0, "Conformes": 21, "Veredicto": "APTO — el conjunto de datos supera todos los controles"}


def _veredicto_esquema_ok() -> dict:
    return {"Criticos": 0, "Advertencias": 0}


def _aislar_controles_previos(monkeypatch) -> None:
    """Aísla entrenar() de los controles de esquema/calidad/baseline
    (todos hablan con la base real) para las pruebas que solo quieren
    ejercitar la lógica de entrenamiento en sí sobre datos sintéticos."""
    monkeypatch.setattr(train, "validar_esquema", lambda: _veredicto_esquema_ok())
    monkeypatch.setattr(train, "validar_antes_de_entrenar", lambda: _veredicto_calidad_ok())
    monkeypatch.setattr(train, "registrar_baseline", lambda *a, **k: {"filas_categoricas": 0, "filas_numericas": 0})
    monkeypatch.setattr(
        train,
        "evaluar_modelo",
        lambda **k: {
            "ks": 0.0, "auc": 0.0, "umbral_operativo": 0,
            "conteos_validacion": {"VP": 0, "FP": 0, "FN": 0, "VN": 0},
            "validacion_cruzada": None, "EvaluacionId": 1, "Aceptado": True, "MotivoRechazo": None,
        },
    )


def _dataset_entrenamiento_sintetico(seed: int = 7) -> pd.DataFrame:
    """Point-in-time simulado con dos grupos de clientes DISJUNTOS por
    diseño: 15 clientes "antiguos" solo facturan en el período de
    entrenamiento (ene-sep 2024) y 10 clientes "nuevos" solo facturan en
    el período de validación (nov-dic 2024, los últimos ~2 meses). Así el
    split temporal + la purga por cliente no se pisan entre sí — a
    diferencia de un round-robin de los mismos clientes en todo el rango,
    que dejaría val_df vacío tras purgar (los 20 clientes ya estarían
    todos en entrenamiento)."""
    rng = np.random.default_rng(seed)

    fechas_train = pd.date_range("2024-01-01", "2024-09-28", freq="3D")
    clientes_train = np.tile(np.arange(1, 16), len(fechas_train) // 15 + 1)[: len(fechas_train)]

    fechas_val = pd.date_range("2024-11-01", "2024-12-30", freq="3D")
    clientes_val = np.tile(np.arange(101, 111), len(fechas_val) // 10 + 1)[: len(fechas_val)]

    fechas = list(fechas_train) + list(fechas_val)
    clientes = np.concatenate([clientes_train, clientes_val])
    n = len(fechas)

    dias_mora = rng.integers(0, 100, size=n).astype(float)
    prob_malo = 0.15 + 0.4 * (dias_mora / 100)
    target = (rng.random(n) < prob_malo).astype(int)

    return pd.DataFrame(
        {
            "FacturaId": range(1, n + 1),
            "ClienteId": clientes,
            "FechaEmision": fechas,
            "DiasMoraPromedioPrevio": dias_mora,
            "SinHistorialPago": 0,
            "Target": target,
        }
    ).set_index("FacturaId")


def _config_entrenamiento_prueba(**overrides) -> dict:
    base = {
        "columna_fecha": "FechaEmision",
        "columna_target": "Target",
        "columna_cliente": "ClienteId",
        "excluir_sin_historial": True,
        "meses_validacion": 2,
        "class_weight": "balanced",
        "entrenar_xgboost": False,
        "ks_minimo": 0.0,
        "auc_minimo": 0.0,
        "auc_maximo_sospechoso": 0.95,
        "score_base": 600,
        "odds_base": 50,
        "pdo": 20,
    }
    base.update(overrides)
    return base


def test_dividir_temporal_por_cliente_separa_por_fecha_no_al_azar():
    df = pd.DataFrame(
        {
            "ClienteId": [1, 2, 3],
            "FechaEmision": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-04-01"]),
        }
    )
    # max fecha = 2024-04-01; meses_validacion=1 -> corte = 2024-03-01.
    # Las dos primeras filas (anteriores al corte) van a entrenamiento; la
    # última (posterior) va a validación. Ningún cliente se repite, así
    # que no hay purga que aplicar en este caso.
    train_df, val_df = train.dividir_temporal_por_cliente(
        df, fecha_col="FechaEmision", cliente_col="ClienteId", meses_validacion=1
    )
    assert set(train_df["ClienteId"]) == {1, 2}
    assert set(val_df["ClienteId"]) == {3}


def test_dividir_temporal_por_cliente_purga_clientes_que_aparecen_en_ambos_lados():
    df = pd.DataFrame(
        {
            "ClienteId": ["A", "A", "B", "C", "D"],
            "FechaEmision": pd.to_datetime(
                ["2024-01-01", "2024-12-01", "2024-02-01", "2024-12-01", "2025-01-01"]
            ),
        }
    )
    # max fecha = 2025-01-01; meses_validacion=6 -> corte = 2024-07-01.
    # A tiene una factura antes del corte (entrenamiento) y otra después
    # (candidata a validación) -> esa segunda fila de A debe purgarse.
    train_df, val_df = train.dividir_temporal_por_cliente(
        df, fecha_col="FechaEmision", cliente_col="ClienteId", meses_validacion=6
    )
    assert set(train_df["ClienteId"]) == {"A", "B"}
    assert set(val_df["ClienteId"]) == {"C", "D"}  # A purgado, no llega a aparecer en val
    assert len(val_df) == 2


def test_calcular_ks_maximo_para_separacion_perfecta():
    y_true = [0] * 20 + [1] * 20
    y_score = list(np.linspace(0, 0.4, 20)) + list(np.linspace(0.6, 1.0, 20))
    assert train.calcular_ks(y_true, y_score) == 1.0


def test_calcular_ks_cero_para_distribuciones_identicas():
    valores = list(np.linspace(0, 1, 20))
    y_true = [0] * 20 + [1] * 20
    y_score = valores + valores
    assert train.calcular_ks(y_true, y_score) == 0.0


def test_calcular_ks_lanza_error_si_falta_una_clase():
    with pytest.raises(train.DatosInsuficientesError):
        train.calcular_ks([0, 0, 0], [0.1, 0.2, 0.3])


def test_entrenar_regresion_logistica_devuelve_modelo_ajustado(monkeypatch):
    monkeypatch.setattr(train, "_config_entrenamiento", lambda: _config_entrenamiento_prueba())
    x = pd.DataFrame({"VarA": [1.0, -1.0, 0.8, -0.9, 0.5, -0.6]})
    y = pd.Series([1, 0, 1, 0, 1, 0])
    modelo = train.entrenar_regresion_logistica(x, y)
    assert hasattr(modelo, "coef_")
    assert modelo.coef_.shape == (1, 1)


def test_generar_scorecard_incluye_base_y_todos_los_tramos(monkeypatch):
    monkeypatch.setattr(train, "_config_entrenamiento", lambda: _config_entrenamiento_prueba())
    x = pd.DataFrame({"VarA": [0.5, -0.3, 0.1, -0.2, 0.4, -0.4], "VarB": [1.0, -1.0, 0.5, -0.5, 0.2, -0.2]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    modelo = train.entrenar_regresion_logistica(x, y)

    mapeo = {
        "VarA": {"tramos": [{"tramo_id": 0, "etiqueta": "<= 0", "woe": -0.5}, {"tramo_id": 1, "etiqueta": "> 0", "woe": 0.5}]},
        "VarB": {"tramos": [{"tramo_id": 0, "etiqueta": "<= 0", "woe": -0.8}, {"tramo_id": 1, "etiqueta": "> 0", "woe": 0.8}]},
    }
    tabla = train.generar_scorecard(modelo, mapeo, ["VarA", "VarB"])

    assert (tabla["variable"] == "(base)").sum() == 1
    assert set(tabla.loc[tabla["variable"] == "VarA", "tramo"]) == {"<= 0", "> 0"}
    assert set(tabla.loc[tabla["variable"] == "VarB", "tramo"]) == {"<= 0", "> 0"}
    assert len(tabla) == 1 + 2 + 2


def test_guardar_modelo_versiona_con_fecha_y_actualiza_ruta_estable(tmp_path, monkeypatch):
    ruta_modelo = tmp_path / "modelo.pkl"
    monkeypatch.setattr(train, "_config_entrenamiento", lambda: _config_entrenamiento_prueba(ruta_modelo=str(ruta_modelo)))

    resultado = {"modelo": "objeto-de-prueba", "metricas": {"auc": 0.7, "ks": 0.3}}
    ruta_devuelta = train.guardar_modelo(resultado)

    assert ruta_devuelta == ruta_modelo
    assert ruta_modelo.exists()
    versionados = list(tmp_path.glob("modelo_*.pkl"))
    assert len(versionados) == 1


def test_entrenar_rechaza_por_ks_bajo_y_no_sobrescribe_modelo_anterior(monkeypatch, tmp_path):
    df = _dataset_entrenamiento_sintetico()
    monkeypatch.setattr(train, "extraer_dataset_entrenamiento", lambda: df)
    _aislar_controles_previos(monkeypatch)

    ruta_modelo = tmp_path / "modelo.pkl"
    ruta_modelo.write_bytes(b"MODELO ANTERIOR")
    ruta_scorecard = tmp_path / "scorecard.csv"

    config = _config_entrenamiento_prueba(
        ks_minimo=0.999,  # inalcanzable -> fuerza el rechazo
        ruta_modelo=str(ruta_modelo),
        ruta_scorecard=str(ruta_scorecard),
    )
    monkeypatch.setattr(train, "_config_entrenamiento", lambda: config)

    with pytest.raises(train.ModeloRechazadoError, match="KS"):
        train.entrenar()

    assert ruta_modelo.read_bytes() == b"MODELO ANTERIOR"  # no se sobrescribió
    assert not ruta_scorecard.exists()  # tampoco se llegó a generar el scorecard


def test_entrenar_rechaza_por_auc_sospechoso_de_fuga(monkeypatch, tmp_path):
    df = _dataset_entrenamiento_sintetico()
    monkeypatch.setattr(train, "extraer_dataset_entrenamiento", lambda: df)
    _aislar_controles_previos(monkeypatch)

    ruta_modelo = tmp_path / "modelo.pkl"
    config = _config_entrenamiento_prueba(
        ks_minimo=0.0,  # que no bloquee por KS
        auc_maximo_sospechoso=-1,  # inalcanzable de cumplir -> siempre "sospechoso"
        ruta_modelo=str(ruta_modelo),
        ruta_scorecard=str(tmp_path / "scorecard.csv"),
    )
    monkeypatch.setattr(train, "_config_entrenamiento", lambda: config)

    with pytest.raises(train.ModeloRechazadoError, match="AUC"):
        train.entrenar()

    assert not ruta_modelo.exists()


def test_entrenar_acepta_y_guarda_con_umbrales_razonables(monkeypatch, tmp_path):
    df = _dataset_entrenamiento_sintetico()
    monkeypatch.setattr(train, "extraer_dataset_entrenamiento", lambda: df)
    _aislar_controles_previos(monkeypatch)

    ruta_modelo = tmp_path / "modelo.pkl"
    ruta_scorecard = tmp_path / "scorecard.csv"
    config = _config_entrenamiento_prueba(ruta_modelo=str(ruta_modelo), ruta_scorecard=str(ruta_scorecard))
    monkeypatch.setattr(train, "_config_entrenamiento", lambda: config)

    resultado = train.entrenar()
    assert ruta_modelo.exists()
    assert ruta_scorecard.exists()
    assert 0.0 <= resultado["metricas"]["ks"] <= 1.0
    assert 0.0 <= resultado["metricas"]["auc"] <= 1.0


def test_entrenar_no_extrae_datos_si_la_calidad_tiene_criticos(monkeypatch):
    llamadas = []
    monkeypatch.setattr(train, "validar_esquema", lambda: _veredicto_esquema_ok())
    monkeypatch.setattr(
        train,
        "validar_antes_de_entrenar",
        lambda: (_ for _ in ()).throw(calidad.CalidadDatosError("NO APTO — corregir los hallazgos críticos antes de entrenar o scorear")),
    )
    monkeypatch.setattr(train, "extraer_dataset_entrenamiento", lambda: llamadas.append(1))

    with pytest.raises(calidad.CalidadDatosError, match="NO APTO"):
        train.entrenar()

    assert llamadas == []  # el gate detuvo el proceso antes de llegar a extraer datos


def test_entrenar_no_llama_a_calidad_si_el_esquema_tiene_criticos(monkeypatch):
    llamadas_calidad = []
    monkeypatch.setattr(
        train,
        "validar_esquema",
        lambda: (_ for _ in ()).throw(esquema.EsquemaInvalidoError("1 hallazgo(s) crítico(s) de esquema.")),
    )
    monkeypatch.setattr(train, "validar_antes_de_entrenar", lambda: llamadas_calidad.append(1))

    with pytest.raises(esquema.EsquemaInvalidoError, match="hallazgo"):
        train.entrenar()

    assert llamadas_calidad == []  # el esquema se valida antes que la calidad, y la detiene primero


def test_entrenar_con_datos_reales_produce_modelo_aceptado(tmp_path, monkeypatch):
    # Verificación en vivo del paso 8.5 completo contra
    # dbo.vw_DatasetEntrenamientoFactura: split temporal + purga por
    # cliente, entrenamiento, validación y controles de fuga.
    ruta_modelo = tmp_path / "modelo.pkl"
    ruta_scorecard = tmp_path / "scorecard.csv"
    config_real = train._config_entrenamiento()
    config_con_rutas_temp = {**config_real, "ruta_modelo": str(ruta_modelo), "ruta_scorecard": str(ruta_scorecard)}
    monkeypatch.setattr(train, "_config_entrenamiento", lambda: config_con_rutas_temp)

    # Las compuertas de esquema y calidad (agregadas tras 8.5) corren
    # contra los datos reales tal como están hoy: si tienen hallazgos
    # críticos genuinos, el entrenamiento debe detenerse ahí — es el
    # comportamiento correcto, no una falla de esta prueba.
    try:
        resultado = train.entrenar()
    except (calidad.CalidadDatosError, esquema.EsquemaInvalidoError) as exc:
        pytest.skip(f"Datos reales con hallazgos críticos de esquema/calidad (gate funcionando correctamente): {exc}")

    # Prueba de fuga (documento, sección 4.4): con el dataset point-in-time
    # el AUC debe quedar lejos del techo de sospecha (0.95), no cerca de 1.
    assert 0.25 <= resultado["metricas"]["ks"] <= 1.0
    assert 0.5 <= resultado["metricas"]["auc"] <= 0.95
    assert ruta_modelo.exists()
    assert ruta_scorecard.exists()
    assert "VendedorId" in resultado["columnas"] or "CiudadId" in resultado["columnas"]
