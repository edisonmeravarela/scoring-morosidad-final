"""Parametrización de la fecha de corte del conjunto de entrenamiento
(FechaCorte_Parametrizacion.sql, guía Guia_Parametrizacion_FechaCorte.pdf).

Unitarias: sin fecha de corte todo sigue leyendo las vistas de siempre;
con fecha de corte se usan las funciones nuevas con la fecha como
parámetro. En vivo (solo lectura): las funciones reproducen las vistas
con la fecha actual y el corte de la versión servida 20261027_172037
reconstruye su conjunto (197 facturas, 105 incumplimientos)."""

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

import main
from src import extract, train
from src.db import consultar_escalar, read_sql

CORTE_VERSION_SERVIDA = datetime(2026, 10, 27, 17, 20, 35)  # logs/extract.log, extracción previa a 20261027_172037
EVALUACION_SERVIDA = 105


# ---------- Unitarias ----------


def test_origen_dataset_sin_fecha_usa_la_vista_configurada():
    sql, origen, params = extract._origen_dataset("dbo.vw_DatasetEntrenamientoFactura")
    assert sql == "SELECT * FROM dbo.vw_DatasetEntrenamientoFactura"
    assert origen == "dbo.vw_DatasetEntrenamientoFactura"
    assert params is None


def test_origen_dataset_con_fecha_usa_la_funcion_parametrizada():
    sql, origen, params = extract._origen_dataset("dbo.vw_DatasetEntrenamientoFactura", CORTE_VERSION_SERVIDA)
    assert sql == "SELECT * FROM dbo.fn_DatasetEntrenamientoFactura(?)"
    assert params == (CORTE_VERSION_SERVIDA,)
    assert "2026-10-27 17:20:35" in origen


def test_origen_historial_sin_y_con_fecha():
    sql, params = extract._origen_historial()
    assert "dbo.vw_HistorialClienteActual" in sql and params is None
    sql, params = extract._origen_historial(CORTE_VERSION_SERVIDA)
    assert "dbo.fn_HistorialClientesACorte(?)" in sql and params == (CORTE_VERSION_SERVIDA,)


def test_extraer_dataset_entrenamiento_con_fecha_pasa_la_fecha_como_parametro():
    df = pd.DataFrame({"FacturaId": range(1, 31), "SinHistorialPago": [0] * 30})
    with patch("src.extract.read_sql", return_value=df) as simulado:
        extract.extraer_dataset_entrenamiento(fecha_corte=CORTE_VERSION_SERVIDA)
    sql, params = simulado.call_args.args
    assert "fn_DatasetEntrenamientoFactura(?)" in sql
    assert params == (CORTE_VERSION_SERVIDA,)


def test_extraer_dataset_entrenamiento_sin_fecha_no_cambia_la_consulta():
    df = pd.DataFrame({"FacturaId": range(1, 31), "SinHistorialPago": [0] * 30})
    with patch("src.extract.read_sql", return_value=df) as simulado:
        extract.extraer_dataset_entrenamiento()
    assert simulado.call_args.args == ("SELECT * FROM dbo.vw_DatasetEntrenamientoFactura",)


def test_registrar_fecha_corte_llama_al_procedimiento_con_conteos(monkeypatch):
    llamadas = []
    monkeypatch.setattr(train, "exec_sp", lambda nombre, params: llamadas.append((nombre, params)))
    df = pd.DataFrame({"Target": [1, 0, 1, 1]})
    train.registrar_fecha_corte(7, CORTE_VERSION_SERVIDA, df, "Target")
    assert llamadas == [
        (
            "dbo.sp_RegistrarFechaCorte",
            {"EvaluacionId": 7, "FechaCorte": CORTE_VERSION_SERVIDA, "FilasDataset": 4, "Incumplimientos": 3, "Fuente": "ENTRENAMIENTO"},
        )
    ]


def test_registrar_fecha_corte_segura_no_interrumpe_si_falla(monkeypatch):
    def _falla(*a, **k):
        raise RuntimeError("sin conexión")

    monkeypatch.setattr(train, "registrar_fecha_corte", _falla)
    train._registrar_fecha_corte_segura({"EvaluacionId": 1}, CORTE_VERSION_SERVIDA, pd.DataFrame({"Target": [1]}), "Target", "v")


def test_parser_reconstruir_dataset_exige_fecha_corte():
    args = main.construir_parser().parse_args(["reconstruir-dataset", "--fecha-corte", "2026-10-27 17:20:35"])
    assert args.comando == "reconstruir-dataset"
    assert args.fecha_corte == "2026-10-27 17:20:35"
    with pytest.raises(SystemExit):
        main.construir_parser().parse_args(["reconstruir-dataset"])


def test_comando_reconstruir_dataset_aplica_el_filtro_del_entrenamiento(monkeypatch, tmp_path):
    dataset = pd.DataFrame(
        {"FacturaId": [1, 2, 3], "ClienteId": [10, 10, 20], "SinHistorialPago": [0, 0, 1], "Target": [1, 0, 1]}
    ).set_index("FacturaId")
    historial = pd.DataFrame({"ClienteId": [10, 20], "FacturasResueltas": [4, 0], "UmbralVigente": [2, 2], "TieneHistorial": [1, 0]})
    recibidas = {}

    def _extraer(fecha_corte=None):
        recibidas["dataset"] = fecha_corte
        return dataset

    monkeypatch.setattr(extract, "extraer_dataset_entrenamiento", _extraer)
    monkeypatch.setattr(extract, "read_sql", lambda sql, params=None: (recibidas.setdefault("historial", params), historial)[1])
    monkeypatch.setattr(main, "LOGS_DIR", tmp_path)

    filas = main.comando_reconstruir_dataset("2026-10-27 17:20:35")

    assert filas == 2  # la factura 3 (SinHistorialPago = 1) se excluye, como en train.entrenar()
    assert recibidas["dataset"] == CORTE_VERSION_SERVIDA
    assert recibidas["historial"] == (CORTE_VERSION_SERVIDA,)
    csv = pd.read_csv(tmp_path / "dataset_corte_20261027_172035.csv", index_col="FacturaId")
    assert list(csv.index) == [1, 2]
    assert list(csv["TieneHistorial"]) == [1, 1]


# ---------- En vivo (solo lectura) ----------


def test_funcion_con_la_fecha_actual_coincide_con_la_vista():
    ahora = datetime.now().replace(microsecond=0)
    for vista, funcion in (
        ("dbo.vw_DatasetEntrenamientoFactura", "dbo.fn_DatasetEntrenamientoFactura(?)"),
        ("dbo.vw_FeaturesFactura", "dbo.fn_FeaturesFacturaACorte(?)"),
    ):
        for a, b in ((vista, funcion), (funcion, vista)):
            diferencias = consultar_escalar(
                f"SELECT COUNT(*) FROM (SELECT * FROM {a} EXCEPT SELECT * FROM {b}) x", (ahora,)
            )
            assert diferencias == 0, f"{a} EXCEPT {b}: {diferencias} filas"

    cols = "ClienteId, FacturasResueltas, UmbralVigente, TieneHistorial"
    for a, b in (
        ("dbo.vw_HistorialClienteActual", "dbo.fn_HistorialClientesACorte(?)"),
        ("dbo.fn_HistorialClientesACorte(?)", "dbo.vw_HistorialClienteActual"),
    ):
        diferencias = consultar_escalar(f"SELECT COUNT(*) FROM (SELECT {cols} FROM {a} EXCEPT SELECT {cols} FROM {b}) x", (ahora,))
        assert diferencias == 0, f"{a} EXCEPT {b}: {diferencias} filas"


def test_corte_de_la_version_servida_reconstruye_197_filas_y_105_incumplimientos():
    # Fija la prueba de integración que con la vista varía según el día:
    # a esta fecha el conjunto ya no cambia (salvo cargas retroactivas).
    df = extract.excluir_sin_historial(extract.extraer_dataset_entrenamiento(fecha_corte=CORTE_VERSION_SERVIDA))
    assert len(df) == 197
    assert int(df["Target"].sum()) == 105


def test_bitacora_registra_la_fecha_de_corte_de_la_version_servida():
    fila = read_sql(
        "SELECT FechaCorte, FilasDataset, Incumplimientos, Fuente FROM dbo.EvaluacionFechaCorte WHERE EvaluacionId = ?",
        (EVALUACION_SERVIDA,),
    )
    assert len(fila) == 1
    assert pd.Timestamp(fila.loc[0, "FechaCorte"]) == pd.Timestamp(CORTE_VERSION_SERVIDA)
    assert (int(fila.loc[0, "FilasDataset"]), int(fila.loc[0, "Incumplimientos"])) == (197, 105)
    assert fila.loc[0, "Fuente"] == "RETROACTIVO"
