import csv
import logging
from pathlib import Path

import pandas as pd
import pytest

import main
from src import calidad, drift, esquema, extract, monitor, rules, score, train, writeback


def _veredicto_calidad_ok() -> dict:
    return {"EjecucionId": "prueba", "Criticos": 0, "Advertencias": 0, "Conformes": 21, "Veredicto": "APTO"}


def _aislar_controles_scorear(monkeypatch) -> None:
    """comando_scorear ahora encadena esquema/calidad al inicio y deriva
    al final; las pruebas que solo quieren ejercitar el armado de
    resultados y la delegación a writeback aíslan los tres. También aísla
    extract.agregar_historial_evaluable (de lo contrario haría una
    consulta real contra dbo.vw_HistorialClienteActual incluso en estas
    pruebas unitarias, que no necesitan ese dato porque además mockean
    rules.aplicar_overrides/clasificar_cliente directamente). También aísla
    score.construir_detalle_variables (de lo contrario cargaría el modelo
    real de disco en cada prueba unitaria de comando_scorear, igual motivo
    que agregar_historial_evaluable)."""
    monkeypatch.setattr(esquema, "validar_esquema", lambda: {"Criticos": 0, "Advertencias": 0})
    monkeypatch.setattr(calidad, "validar_antes_de_entrenar", lambda: _veredicto_calidad_ok())
    monkeypatch.setattr(train, "cargar_modelo", lambda: {"mapeo": {}, "version_modelo": "v-prueba"})
    monkeypatch.setattr(drift, "medir_deriva", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(extract, "agregar_historial_evaluable", lambda df: df)
    monkeypatch.setattr(score, "construir_detalle_variables", lambda df: {})


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


def test_parser_scorear_con_clientes():
    args = main.construir_parser().parse_args(["scorear", "--clientes", "101,205,330"])
    assert args.clientes == "101,205,330"


def test_parser_scorear_sin_clientes_por_defecto():
    args = main.construir_parser().parse_args(["scorear"])
    assert args.clientes is None


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


def test_comando_scorear_arma_resultados_y_delega_en_writeback(monkeypatch, tmp_path, capsys):
    _aislar_controles_scorear(monkeypatch)
    monkeypatch.setattr(main, "LOGS_DIR", tmp_path)  # no escribir en scoring-morosidad/logs/ real
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

    # Protocolo de progreso (barra de progreso del frontend, pantalla
    # "Scorear (simulación)"): 7 líneas PROGRESO en orden, a stdout, más
    # RESULTADO_CSV porque esta corrida es dry_run=True.
    salida = capsys.readouterr().out
    lineas_progreso = [l for l in salida.splitlines() if l.startswith("PROGRESO ")]
    assert lineas_progreso == [
        "PROGRESO 1/7 Validando esquema de datos",
        "PROGRESO 2/7 Validando calidad de datos",
        "PROGRESO 3/7 Extrayendo datos de clientes",
        "PROGRESO 4/7 Calculando scores del modelo",
        "PROGRESO 5/7 Aplicando reglas de negocio",
        "PROGRESO 6/7 Comparando segmentos propuestos",
        "PROGRESO 7/7 Midiendo deriva de datos",
    ]
    lineas_csv = [l for l in salida.splitlines() if l.startswith("RESULTADO_CSV ")]
    assert len(lineas_csv) == 1
    ruta_csv = Path(lineas_csv[0].split(" ", 1)[1])
    assert ruta_csv.exists()
    assert ruta_csv.parent == tmp_path


def test_comando_scorear_dry_run_escribe_el_csv_con_el_detalle_por_cliente(monkeypatch, tmp_path):
    """El CSV de la pantalla "Scorear (simulación)" (barra de progreso +
    descarga de Excel, frontend): una fila por cliente, con el nombre, el
    código de segmento/clasificación propuestos, el plazo recomendado
    (resuelto desde config.reglas.plazo_por_segmento, sin tocar la base) y
    si cambia de segmento (comparado contra reporte["cambiaron_segmento"],
    que ya devuelve writeback.guardar_resultados)."""
    _aislar_controles_scorear(monkeypatch)
    monkeypatch.setattr(main, "LOGS_DIR", tmp_path)
    df_clientes = pd.DataFrame(
        [
            {**_fila_cliente_prueba().drop("score"), "NombreCompleto": "Cliente Uno"},
            {**_fila_cliente_prueba().drop("score"), "NombreCompleto": "Cliente Dos"},
        ],
        index=pd.Index([1, 2], name="ClienteId"),
    )
    monkeypatch.setattr(extract, "extraer_features", lambda: df_clientes)
    monkeypatch.setattr(score, "calcular_scores", lambda df: pd.DataFrame({"score": [415.0, 900.0]}, index=df.index))
    monkeypatch.setattr(rules, "asignar_segmento", lambda s: 3 if s < 500 else 1)
    monkeypatch.setattr(rules, "aplicar_overrides", lambda cliente, seg_id: "ALTO" if seg_id == 3 else "BAJO")
    monkeypatch.setattr(rules, "resolver_segmento_final", lambda seg_id, codigo: seg_id)
    monkeypatch.setattr(rules, "recomendar_plazo", lambda codigo: 2)
    monkeypatch.setattr(rules, "clasificar_cliente", lambda cliente: "B")
    monkeypatch.setattr(rules, "resolver_clasificacion_id", lambda codigo: 2)
    monkeypatch.setattr(rules, "marcar_revision", lambda cliente, codigo: codigo == "ALTO")
    monkeypatch.setattr(
        main,
        "cargar_config",
        lambda: {"reglas": {"plazo_por_segmento": {"ALTO": "15 DÍAS", "BAJO": "60 DÍAS"}}},
    )
    monkeypatch.setattr(
        writeback,
        "guardar_resultados",
        lambda df, dry_run=False, **k: {
            "modo": "simulacion", "total": len(df),
            "cambiaron_segmento": [1], "sin_cambios": [2], "fallidos": [],
        },
    )

    main.comando_scorear(dry_run=True)

    (ruta_csv,) = list(tmp_path.glob("simulacion_*.csv"))
    with open(ruta_csv, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    assert [f["ClienteId"] for f in filas] == ["1", "2"]
    fila_1, fila_2 = filas
    assert fila_1["NombreCliente"] == "Cliente Uno"
    assert fila_1["SegmentoPropuesto"] == "ALTO"
    assert fila_1["PlazoRecomendado"] == "15 DÍAS"
    assert fila_1["CambiaSegmento"] == "Sí"
    assert fila_1["RevisionManual"] == "Sí"
    assert fila_2["NombreCliente"] == "Cliente Dos"
    assert fila_2["SegmentoPropuesto"] == "BAJO"
    assert fila_2["PlazoRecomendado"] == "60 DÍAS"
    assert fila_2["CambiaSegmento"] == "No"
    assert fila_2["RevisionManual"] == "No"


def test_comando_scorear_incorpora_el_historial_evaluable_antes_de_las_reglas(monkeypatch, capsys):
    """Confirma que comando_scorear llama a
    extract.agregar_historial_evaluable (criterio único de "historial
    evaluable" compartido con el entrenamiento y la cola de reevaluación
    — src/rules.py ya no cuenta NumFacturas por su cuenta)."""
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
    monkeypatch.setattr(writeback, "guardar_resultados", lambda df, dry_run=False, **k: {"modo": "real", "total": len(df), "cambiaron_segmento": [], "sin_cambios": [1], "fallidos": []})

    llamadas = []
    monkeypatch.setattr(extract, "agregar_historial_evaluable", lambda df: (llamadas.append(df) or df.assign(TieneHistorial=True)))

    main.comando_scorear(dry_run=False)

    assert len(llamadas) == 1
    assert list(llamadas[0].index) == [1]

    # La ejecución real nunca escribe el CSV de simulación (la base de
    # datos ya es la fuente de verdad ahí) ni imprime RESULTADO_CSV.
    salida = capsys.readouterr().out
    assert not any(l.startswith("RESULTADO_CSV ") for l in salida.splitlines())


def test_comando_scorear_calcula_y_pasa_el_detalle_de_variables_a_writeback(monkeypatch):
    """Explicabilidad del score (sección 4.4.3 del manual técnico):
    comando_scorear debe llamar a score.construir_detalle_variables con la
    misma población que llega a calcular_scores, y pasar el resultado a
    writeback.guardar_resultados — pero solo cuando no es dry_run (en
    simulación no se escribe nada, calcularlo sería trabajo descartado)."""
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

    llamadas_detalle = []

    def _detalle_falso(df):
        llamadas_detalle.append(df)
        return {1: [{"Variable": "(base)", "Etiqueta": "Puntaje base", "ValorOriginal": None, "WOE": None, "Puntos": 600.0, "Orden": 0}]}

    monkeypatch.setattr(score, "construir_detalle_variables", _detalle_falso)

    llamadas_writeback = {}

    def _guardar_resultados_falso(df, dry_run=False, detalle_variables=None, **kwargs):
        llamadas_writeback["detalle_variables"] = detalle_variables
        return {"modo": "real", "total": len(df), "cambiaron_segmento": [1], "sin_cambios": [], "fallidos": []}

    monkeypatch.setattr(writeback, "guardar_resultados", _guardar_resultados_falso)

    main.comando_scorear(dry_run=False)

    assert len(llamadas_detalle) == 1
    assert list(llamadas_detalle[0].index) == [1]
    assert llamadas_writeback["detalle_variables"] == {
        1: [{"Variable": "(base)", "Etiqueta": "Puntaje base", "ValorOriginal": None, "WOE": None, "Puntos": 600.0, "Orden": 0}]
    }


def test_comando_scorear_en_dry_run_no_calcula_el_detalle_de_variables(monkeypatch, tmp_path):
    """En simulación no se escribe nada: calcular el detalle sería trabajo
    descartado, así que comando_scorear no debe llamarlo."""
    _aislar_controles_scorear(monkeypatch)
    monkeypatch.setattr(main, "LOGS_DIR", tmp_path)  # no escribir en scoring-morosidad/logs/ real
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

    llamadas_detalle = []
    monkeypatch.setattr(score, "construir_detalle_variables", lambda df: (llamadas_detalle.append(df) or {}))

    llamadas_writeback = {}

    def _guardar_resultados_falso(df, dry_run=False, detalle_variables=None, **kwargs):
        llamadas_writeback["detalle_variables"] = detalle_variables
        return {"modo": "simulacion", "total": len(df), "cambiaron_segmento": [], "sin_cambios": [1], "fallidos": []}

    monkeypatch.setattr(writeback, "guardar_resultados", _guardar_resultados_falso)

    main.comando_scorear(dry_run=True)

    assert llamadas_detalle == []
    assert llamadas_writeback["detalle_variables"] is None


def test_comando_scorear_con_clientes_ids_filtra_la_poblacion(monkeypatch):
    """Reevaluación bajo demanda (cola dbo.ColaRescoreo): clientes_ids
    debe limitar qué filas de extraer_features() llegan a score/writeback,
    sin tocar la consulta SQL de la vista."""
    _aislar_controles_scorear(monkeypatch)
    df_clientes = pd.DataFrame(
        [_fila_cliente_prueba().drop("score"), _fila_cliente_prueba().drop("score")],
        index=pd.Index([1, 2], name="ClienteId"),
    )
    monkeypatch.setattr(extract, "extraer_features", lambda: df_clientes)
    monkeypatch.setattr(score, "calcular_scores", lambda df: pd.DataFrame({"score": [415.0] * len(df)}, index=df.index))

    monkeypatch.setattr(rules, "asignar_segmento", lambda s: 3)
    monkeypatch.setattr(rules, "aplicar_overrides", lambda cliente, seg_id: "ALTO")
    monkeypatch.setattr(rules, "resolver_segmento_final", lambda seg_id, codigo: 3)
    monkeypatch.setattr(rules, "recomendar_plazo", lambda codigo: 2)
    monkeypatch.setattr(rules, "clasificar_cliente", lambda cliente: "B")
    monkeypatch.setattr(rules, "resolver_clasificacion_id", lambda codigo: 2)
    monkeypatch.setattr(rules, "marcar_revision", lambda cliente, codigo: False)

    llamadas = {}

    def _guardar_resultados_falso(df, dry_run=False, **k):
        llamadas["df"] = df
        return {"modo": "real", "total": len(df), "cambiaron_segmento": [], "sin_cambios": list(df["ClienteId"]), "fallidos": []}

    monkeypatch.setattr(writeback, "guardar_resultados", _guardar_resultados_falso)

    resultado = main.comando_scorear(dry_run=False, clientes_ids=[2])

    assert resultado == 1
    assert list(llamadas["df"]["ClienteId"]) == [2]


def test_comando_scorear_con_clientes_ids_sin_coincidencias_no_llama_a_writeback(monkeypatch):
    _aislar_controles_scorear(monkeypatch)
    df_clientes = pd.DataFrame([_fila_cliente_prueba().drop("score")], index=pd.Index([1], name="ClienteId"))
    monkeypatch.setattr(extract, "extraer_features", lambda: df_clientes)

    def _writeback_no_debe_llamarse(*a, **k):
        raise AssertionError("writeback.guardar_resultados no debía llamarse sin clientes coincidentes")

    monkeypatch.setattr(writeback, "guardar_resultados", _writeback_no_debe_llamarse)

    resultado = main.comando_scorear(dry_run=False, clientes_ids=[999])

    assert resultado == 0


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
    antes = set(main.LOGS_DIR.glob("simulacion_*.csv"))
    codigo = main.main(["scorear", "--dry-run"])
    assert codigo == 0

    # CSV de la pantalla "Scorear (simulación)" (barra de progreso +
    # descarga de Excel, frontend): confirma que el pipeline real produce
    # el archivo, con una fila por cliente evaluado. Se borra al terminar
    # para no acumular artefactos de prueba en scoring-morosidad/logs/
    # (logs/*.csv ya está en .gitignore, pero no hay motivo para dejarlo).
    (ruta_csv,) = set(main.LOGS_DIR.glob("simulacion_*.csv")) - antes
    try:
        with open(ruta_csv, encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        assert len(filas) > 0
        assert set(filas[0]) == {
            "ClienteId", "NombreCliente", "ScorePropuesto", "SegmentoPropuesto",
            "ClasificacionPropuesta", "PlazoRecomendado", "CambiaSegmento", "RevisionManual",
        }
    finally:
        ruta_csv.unlink()


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
