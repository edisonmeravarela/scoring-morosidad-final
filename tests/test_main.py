import logging

import pandas as pd
import pytest

import main
from src import calidad, drift, esquema, extract, monitor, rules, score, train, writeback


def _veredicto_calidad_ok() -> dict:
    return {"EjecucionId": "prueba", "Criticos": 0, "Advertencias": 0, "Conformes": 21, "Veredicto": "APTO"}


def _aislar_controles_scorear(monkeypatch) -> None:
    """comando_scorear ahora encadena esquema/calidad al inicio y deriva
    al final; las pruebas que solo quieren ejercitar el armado de
    resultados y la delegación a writeback aíslan los tres."""
    monkeypatch.setattr(esquema, "validar_esquema", lambda: {"Criticos": 0, "Advertencias": 0})
    monkeypatch.setattr(calidad, "validar_antes_de_entrenar", lambda: _veredicto_calidad_ok())
    monkeypatch.setattr(train, "cargar_modelo", lambda: {"mapeo": {}, "version_modelo": "v-prueba"})
    monkeypatch.setattr(drift, "medir_deriva", lambda *a, **k: pd.DataFrame())


# ---------- construir_parser ----------


def test_parser_entrenar():
    args = main.construir_parser().parse_args(["entrenar"])
    assert args.comando == "entrenar"


def test_parser_scorear_sin_dry_run():
    args = main.construir_parser().parse_args(["scorear"])
    assert args.comando == "scorear"
    assert args.dry_run is False


def test_parser_scorear_con_dry_run():
    args = main.construir_parser().parse_args(["scorear", "--dry-run"])
    assert args.dry_run is True


def test_parser_monitorear():
    args = main.construir_parser().parse_args(["monitorear"])
    assert args.comando == "monitorear"


def test_parser_sin_comando_falla():
    with pytest.raises(SystemExit):
        main.construir_parser().parse_args([])


# ---------- _ejecutar_con_bitacora: criterio de aceptación (inicio/fin/duración/errores) ----------


def test_ejecutar_con_bitacora_registra_inicio_fin_y_retorna_cero(caplog):
    with caplog.at_level(logging.INFO, logger="scoring_morosidad.main"):
        codigo = main._ejecutar_con_bitacora("prueba", lambda: 7)
    assert codigo == 0
    assert any("INICIO prueba" in r.message for r in caplog.records)
    assert any("FIN prueba" in r.message and "7 registros procesados" in r.message for r in caplog.records)


def test_ejecutar_con_bitacora_captura_excepcion_y_retorna_uno(caplog):
    def _falla():
        raise RuntimeError("fallo simulado")

    with caplog.at_level(logging.INFO, logger="scoring_morosidad.main"):
        codigo = main._ejecutar_con_bitacora("prueba", _falla)
    assert codigo == 1
    assert any("FIN prueba CON ERROR" in r.message and "fallo simulado" in r.message for r in caplog.records)


# ---------- comando_entrenar ----------


def test_comando_entrenar_devuelve_conteo_de_filas(monkeypatch):
    monkeypatch.setattr(
        train,
        "entrenar",
        lambda: {
            "metricas": {"auc": 0.7, "ks": 0.4},
            "n_entrenamiento": 200,
            "n_validacion": 50,
        },
    )
    assert main.comando_entrenar() == 250


# ---------- comando_scorear ----------


def _fila_cliente_prueba():
    return pd.Series(
        {
            "score": 415.0,
            "DiasMoraMaxima": 10,
            "NumFacturas": 7,
            "SaldoActualTotal": 100,
            "DiasDesdeUltimaFactura": 20,
            "ValorFacturaPromedio": 500,
        }
    )


def test_comando_scorear_arma_resultados_y_delega_en_writeback(monkeypatch):
    _aislar_controles_scorear(monkeypatch)
    df_clientes = pd.DataFrame([_fila_cliente_prueba().drop("score")], index=pd.Index([1], name="ClienteId"))
    monkeypatch.setattr(extract, "extraer_features", lambda: df_clientes)
    monkeypatch.setattr(score, "calcular_scores", lambda df: pd.DataFrame({"score": [415.0]}, index=df.index))

    monkeypatch.setattr(rules, "asignar_segmento", lambda s: 3)
    monkeypatch.setattr(rules, "aplicar_overrides", lambda cliente, seg_id: "ALTO")
    monkeypatch.setattr(rules, "resolver_segmento_final", lambda seg_id, codigo: 3)
    monkeypatch.setattr(rules, "recomendar_plazo", lambda codigo: 2)
    monkeypatch.setattr(rules, "clasificar_cliente", lambda cliente: "B")
    monkeypatch.setattr(rules, "resolver_clasificacion_id", lambda codigo: 2)
    monkeypatch.setattr(rules, "marcar_revision", lambda cliente, codigo: True)

    llamadas = {}

    def _guardar_resultados_falso(df, dry_run=False, **kwargs):
        llamadas["df"] = df
        llamadas["dry_run"] = dry_run
        return {"modo": "simulacion" if dry_run else "real", "total": len(df), "cambiaron_segmento": [1], "sin_cambios": [], "fallidos": []}

    monkeypatch.setattr(writeback, "guardar_resultados", _guardar_resultados_falso)

    resultado = main.comando_scorear(dry_run=True)

    assert resultado == 1
    assert llamadas["dry_run"] is True
    fila = llamadas["df"].iloc[0]
    assert fila["ClienteId"] == 1
    assert fila["ScoreRiesgo"] == 415.0
    assert fila["SegmentoRiesgoId"] == 3
    assert fila["ClasificacionId"] == 2
    assert fila["PlazoPagoRecomendadoId"] == 2
    assert "ALTO" in fila["Motivo"]


def test_comando_scorear_no_lanza_si_hay_fallidos(monkeypatch):
    _aislar_controles_scorear(monkeypatch)
    df_clientes = pd.DataFrame([_fila_cliente_prueba().drop("score")], index=pd.Index([1], name="ClienteId"))
    monkeypatch.setattr(extract, "extraer_features", lambda: df_clientes)
    monkeypatch.setattr(score, "calcular_scores", lambda df: pd.DataFrame({"score": [415.0]}, index=df.index))
    monkeypatch.setattr(rules, "asignar_segmento", lambda s: 3)
    monkeypatch.setattr(rules, "aplicar_overrides", lambda cliente, seg_id: "ALTO")
    monkeypatch.setattr(rules, "resolver_segmento_final", lambda seg_id, codigo: 3)
    monkeypatch.setattr(rules, "recomendar_plazo", lambda codigo: 2)
    monkeypatch.setattr(rules, "clasificar_cliente", lambda cliente: "B")
    monkeypatch.setattr(rules, "resolver_clasificacion_id", lambda codigo: 2)
    monkeypatch.setattr(rules, "marcar_revision", lambda cliente, codigo: False)
    monkeypatch.setattr(
        writeback,
        "guardar_resultados",
        lambda df, dry_run=False, **k: {"modo": "real", "total": 1, "cambiaron_segmento": [], "sin_cambios": [], "fallidos": [1]},
    )

    resultado = main.comando_scorear(dry_run=False)  # no debe lanzar

    assert resultado == 1


def test_comando_scorear_no_falla_si_la_deriva_no_se_puede_medir(monkeypatch):
    _aislar_controles_scorear(monkeypatch)
    monkeypatch.setattr(drift, "medir_deriva", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sin baseline")))
    df_clientes = pd.DataFrame([_fila_cliente_prueba().drop("score")], index=pd.Index([1], name="ClienteId"))
    monkeypatch.setattr(extract, "extraer_features", lambda: df_clientes)
    monkeypatch.setattr(score, "calcular_scores", lambda df: pd.DataFrame({"score": [415.0]}, index=df.index))
    monkeypatch.setattr(rules, "asignar_segmento", lambda s: 3)
    monkeypatch.setattr(rules, "aplicar_overrides", lambda cliente, seg_id: "ALTO")
    monkeypatch.setattr(rules, "resolver_segmento_final", lambda seg_id, codigo: 3)
    monkeypatch.setattr(rules, "recomendar_plazo", lambda codigo: 2)
    monkeypatch.setattr(rules, "clasificar_cliente", lambda cliente: "B")
    monkeypatch.setattr(rules, "resolver_clasificacion_id", lambda codigo: 2)
    monkeypatch.setattr(rules, "marcar_revision", lambda cliente, codigo: False)
    monkeypatch.setattr(
        writeback, "guardar_resultados", lambda df, dry_run=False, **k: {"modo": "real", "total": 1, "cambiaron_segmento": [], "sin_cambios": [], "fallidos": []}
    )

    resultado = main.comando_scorear(dry_run=False)  # no debe lanzar aunque la deriva falle

    assert resultado == 1


def test_comando_scorear_no_llega_a_scorear_si_el_esquema_tiene_criticos(monkeypatch):
    llamadas = []
    monkeypatch.setattr(
        esquema, "validar_esquema", lambda: (_ for _ in ()).throw(esquema.EsquemaInvalidoError("1 hallazgo(s) crítico(s)."))
    )
    monkeypatch.setattr(extract, "extraer_features", lambda: llamadas.append(1))

    with pytest.raises(esquema.EsquemaInvalidoError):
        main.comando_scorear(dry_run=True)

    assert llamadas == []


def test_comando_scorear_no_llega_a_scorear_si_la_calidad_tiene_criticos(monkeypatch):
    llamadas = []
    monkeypatch.setattr(esquema, "validar_esquema", lambda: {"Criticos": 0, "Advertencias": 0})
    monkeypatch.setattr(
        calidad, "validar_antes_de_entrenar", lambda: (_ for _ in ()).throw(calidad.CalidadDatosError("NO APTO"))
    )
    monkeypatch.setattr(extract, "extraer_features", lambda: llamadas.append(1))

    with pytest.raises(calidad.CalidadDatosError):
        main.comando_scorear(dry_run=True)

    assert llamadas == []


# ---------- comando_validar_esquema / comando_validar_calidad / comando_medir_deriva ----------


def test_parser_validar_esquema():
    assert main.construir_parser().parse_args(["validar-esquema"]).comando == "validar-esquema"


def test_parser_validar_calidad():
    assert main.construir_parser().parse_args(["validar-calidad"]).comando == "validar-calidad"


def test_parser_medir_deriva():
    assert main.construir_parser().parse_args(["medir-deriva"]).comando == "medir-deriva"


def test_comando_validar_esquema_devuelve_criticos_mas_advertencias(monkeypatch):
    monkeypatch.setattr(esquema, "validar_esquema", lambda: {"Criticos": 1, "Advertencias": 2})
    assert main.comando_validar_esquema() == 3


def test_comando_validar_calidad_devuelve_el_total_de_hallazgos(monkeypatch):
    monkeypatch.setattr(calidad, "validar_antes_de_entrenar", lambda: {"Criticos": 0, "Advertencias": 1, "Conformes": 20})
    assert main.comando_validar_calidad() == 21


def test_comando_medir_deriva_usa_la_version_del_modelo_vigente(monkeypatch):
    llamadas = {}
    monkeypatch.setattr(train, "cargar_modelo", lambda: {"mapeo": {"VarA": {}}, "version_modelo": "v-1"})

    def _medir_falso(version_modelo, mapeo):
        llamadas["version"] = version_modelo
        return pd.DataFrame({"Variable": ["VarA"], "PSI": [0.1]})

    monkeypatch.setattr(drift, "medir_deriva", _medir_falso)

    assert main.comando_medir_deriva() == 1
    assert llamadas["version"] == "v-1"


def test_comando_medir_deriva_usa_fecha_entrenamiento_si_no_hay_version_modelo(monkeypatch):
    # Compatibilidad con modelos guardados antes de que existiera esta clave.
    llamadas = {}
    monkeypatch.setattr(train, "cargar_modelo", lambda: {"mapeo": {}, "fecha_entrenamiento": "2026-01-01T00:00:00"})

    def _medir_falso(version_modelo, mapeo):
        llamadas["version"] = version_modelo
        return pd.DataFrame()

    monkeypatch.setattr(drift, "medir_deriva", _medir_falso)

    main.comando_medir_deriva()
    assert llamadas["version"] == "2026-01-01T00:00:00"


# ---------- comando_monitorear ----------


def test_comando_monitorear_maneja_datos_insuficientes(monkeypatch):
    def _falla(*a, **k):
        raise monitor.DatosInsuficientesError("no hay suficientes registros")

    monkeypatch.setattr(monitor, "evaluar_desempeno", _falla)
    monkeypatch.setattr(monitor, "detectar_drift", lambda: pd.DataFrame({"variable": ["A", "B"], "psi": [0.1, 0.3], "alerta": [False, True]}))

    resultado = main.comando_monitorear()  # no debe propagar la excepción

    assert resultado == 2  # 2 variables evaluadas por drift, evaluar_desempeno no sumó nada


def test_comando_monitorear_suma_observaciones_y_variables(monkeypatch):
    monkeypatch.setattr(
        monitor,
        "evaluar_desempeno",
        lambda: {"ks_actual": 0.3, "ks_validacion": 0.5, "umbral_alerta": 0.4, "alerta_reentrenamiento": True, "n_observaciones": 25},
    )
    monkeypatch.setattr(monitor, "detectar_drift", lambda: pd.DataFrame({"variable": ["A"], "psi": [0.1], "alerta": [False]}))

    assert main.comando_monitorear() == 26


# ---------- Verificación en vivo: pipeline completo ensamblado por primera vez ----------


def test_main_scorear_dry_run_pipeline_completo_contra_datos_reales():
    # Primera vez que se ejecutan encadenados extract -> score -> rules ->
    # writeback tal como haría una ejecución real, sobre los ~169 clientes
    # reales. --dry-run garantiza que no se escribe nada.
    codigo = main.main(["scorear", "--dry-run"])
    assert codigo == 0


def test_main_monitorear_no_falla_con_datos_reales():
    codigo = main.main(["monitorear"])
    assert codigo == 0


def test_main_validar_esquema_contra_datos_reales():
    # No se asume el estado del esquema real: ambos desenlaces (0 o
    # exit 1 con hallazgos críticos) son un resultado válido de la CLI.
    codigo = main.main(["validar-esquema"])
    assert codigo in (0, 1)


def test_main_validar_calidad_contra_datos_reales():
    codigo = main.main(["validar-calidad"])
    assert codigo in (0, 1)


def test_main_medir_deriva_contra_el_modelo_vigente():
    # Requiere que exista un baseline registrado para el modelo guardado
    # en models/modelo.pkl (train.entrenar() ya lo registra al terminar).
    codigo = main.main(["medir-deriva"])
    assert codigo == 0
