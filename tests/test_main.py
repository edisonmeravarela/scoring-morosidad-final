import logging

import pandas as pd
import pytest

import main
from src import extract, monitor, rules, score, train, writeback


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
