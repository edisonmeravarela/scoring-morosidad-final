import contextlib

import pandas as pd
import pytest

from src import extract, score, writeback
from src.db import get_engine, read_sql


class _FakeEngine:
    def __init__(self, registrar_apertura=None):
        self._registrar_apertura = registrar_apertura

    def begin(self):
        if self._registrar_apertura:
            self._registrar_apertura()
        return contextlib.nullcontext(object())


@pytest.fixture(autouse=True)
def _entorno_de_prueba(monkeypatch):
    monkeypatch.setattr(writeback, "get_engine", lambda: _FakeEngine())
    monkeypatch.setattr(writeback, "_config_writeback", lambda: {"tamano_lote": 50, "motivo_por_defecto": "Motivo de prueba"})


def _resultados_prueba() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ClienteId": [1, 2, 3],
            "ScoreRiesgo": [415.0, 900.0, 100.0],
            "SegmentoRiesgoId": [3, 1, 4],
            "ClasificacionId": [2, 1, 4],
            "PlazoPagoRecomendadoId": [2, 5, 1],
        }
    )


# ---------- Unidad: dry-run, ejecución real, fallos, lotes ----------


def test_guardar_resultados_dry_run_no_ejecuta_el_procedimiento(monkeypatch):
    llamadas = []
    monkeypatch.setattr(writeback, "_ejecutar_sp", lambda *a, **k: llamadas.append(a))
    monkeypatch.setattr(writeback, "_segmento_actual_por_cliente", lambda ids: {1: 3, 2: 2, 3: 4})

    resultado = writeback.guardar_resultados(_resultados_prueba(), dry_run=True)

    assert llamadas == []
    assert resultado["modo"] == "simulacion"
    assert resultado["total"] == 3
    assert set(resultado["cambiaron_segmento"]) == {2}  # cliente 2: vigente 2 -> propuesto 1
    assert set(resultado["sin_cambios"]) == {1, 3}
    assert resultado["fallidos"] == []


def test_guardar_resultados_real_ejecuta_el_procedimiento_por_cada_fila(monkeypatch):
    llamadas = []
    monkeypatch.setattr(writeback, "_ejecutar_sp", lambda conn, cliente_id, *resto: llamadas.append(cliente_id))
    monkeypatch.setattr(writeback, "_segmento_actual_por_cliente", lambda ids: {1: 3, 2: 2, 3: 4})

    resultado = writeback.guardar_resultados(_resultados_prueba(), dry_run=False)

    assert sorted(llamadas) == [1, 2, 3]
    assert resultado["modo"] == "real"
    assert set(resultado["cambiaron_segmento"]) == {2}
    assert set(resultado["sin_cambios"]) == {1, 3}


def test_guardar_resultados_registra_fallidos_sin_abortar_el_lote(monkeypatch):
    def _ejecutar_sp_falso(conn, cliente_id, *resto):
        if cliente_id == 2:
            raise RuntimeError("fallo simulado")

    monkeypatch.setattr(writeback, "_ejecutar_sp", _ejecutar_sp_falso)
    monkeypatch.setattr(writeback, "_segmento_actual_por_cliente", lambda ids: {1: 3, 2: 2, 3: 4})

    resultado = writeback.guardar_resultados(_resultados_prueba(), dry_run=False)

    assert resultado["fallidos"] == [2]
    assert set(resultado["sin_cambios"]) == {1, 3}


def test_guardar_resultados_respeta_indice_clienteid(monkeypatch):
    monkeypatch.setattr(writeback, "_ejecutar_sp", lambda *a, **k: None)
    monkeypatch.setattr(writeback, "_segmento_actual_por_cliente", lambda ids: {})
    df = _resultados_prueba().set_index("ClienteId")
    resultado = writeback.guardar_resultados(df, dry_run=False)
    assert resultado["total"] == 3


def test_guardar_resultados_procesa_en_lotes_separados(monkeypatch):
    aperturas = []
    monkeypatch.setattr(writeback, "get_engine", lambda: _FakeEngine(registrar_apertura=lambda: aperturas.append(1)))
    monkeypatch.setattr(writeback, "_ejecutar_sp", lambda *a, **k: None)
    monkeypatch.setattr(writeback, "_segmento_actual_por_cliente", lambda ids: {})

    writeback.guardar_resultados(_resultados_prueba(), dry_run=False, tamano_lote=1)

    assert len(aperturas) == 3  # 3 clientes, lotes de 1 -> 3 transacciones


def test_guardar_resultados_usa_motivo_por_defecto_si_no_se_provee(monkeypatch):
    motivos = []
    monkeypatch.setattr(
        writeback,
        "_ejecutar_sp",
        lambda conn, cliente_id, score, segmento_id, clasificacion_id, plazo_id, motivo, usuario_id: motivos.append(motivo),
    )
    monkeypatch.setattr(writeback, "_segmento_actual_por_cliente", lambda ids: {})

    writeback.guardar_resultados(_resultados_prueba(), dry_run=False)

    assert motivos == ["Motivo de prueba"] * 3


# ---------- Unidad: detalle_variables (explicabilidad del score) ----------


def test_guardar_resultados_inserta_detalle_cuando_hay_fila_nueva_de_historial(monkeypatch):
    """Cliente 1 "cambió" (su MAX(HistorialId) subió de 10 a 11 tras el
    EXEC): el detalle de sus variables debe insertarse contra el
    HistorialId nuevo (11), no el anterior."""
    monkeypatch.setattr(writeback, "_ejecutar_sp", lambda *a, **k: None)
    monkeypatch.setattr(writeback, "_segmento_actual_por_cliente", lambda ids: {1: 3, 2: 2, 3: 4})
    monkeypatch.setattr(writeback, "_ultimo_historial_por_cliente", lambda ids: {1: 10, 2: 20})
    monkeypatch.setattr(writeback, "_historial_actual", lambda conn, cliente_id: {1: 11, 2: 20}.get(cliente_id, 0))

    inserciones = []
    monkeypatch.setattr(writeback, "_insertar_detalle_variables", lambda conn, hid, filas: inserciones.append((hid, filas)))

    detalle_variables = {
        1: [{"Variable": "(base)", "Etiqueta": "Puntaje base", "ValorOriginal": None, "WOE": None, "Puntos": 600.0, "Orden": 0}],
        2: [{"Variable": "(base)", "Etiqueta": "Puntaje base", "ValorOriginal": None, "WOE": None, "Puntos": 600.0, "Orden": 0}],
    }

    writeback.guardar_resultados(_resultados_prueba(), dry_run=False, detalle_variables=detalle_variables)

    # cliente 1: historial cambió (10 -> 11) -> se inserta; cliente 2: no
    # cambió (20 -> 20) -> no se inserta; cliente 3: no está en
    # detalle_variables -> no se consulta ni se inserta.
    assert inserciones == [(11, detalle_variables[1])]


def test_guardar_resultados_no_inserta_detalle_si_no_cambio_el_historial(monkeypatch):
    monkeypatch.setattr(writeback, "_ejecutar_sp", lambda *a, **k: None)
    monkeypatch.setattr(writeback, "_segmento_actual_por_cliente", lambda ids: {})
    monkeypatch.setattr(writeback, "_ultimo_historial_por_cliente", lambda ids: {1: 10})
    monkeypatch.setattr(writeback, "_historial_actual", lambda conn, cliente_id: 10)  # sin cambio

    inserciones = []
    monkeypatch.setattr(writeback, "_insertar_detalle_variables", lambda conn, hid, filas: inserciones.append((hid, filas)))

    detalle_variables = {1: [{"Variable": "(base)", "Etiqueta": "Puntaje base", "ValorOriginal": None, "WOE": None, "Puntos": 600.0, "Orden": 0}]}
    writeback.guardar_resultados(_resultados_prueba(), dry_run=False, detalle_variables=detalle_variables)

    assert inserciones == []


def test_guardar_resultados_sin_detalle_variables_no_consulta_historial(monkeypatch):
    """Comportamiento actual intacto: si no se pasa detalle_variables (el
    valor por defecto, None), no se hace ninguna consulta/inserción nueva."""
    monkeypatch.setattr(writeback, "_ejecutar_sp", lambda *a, **k: None)
    monkeypatch.setattr(writeback, "_segmento_actual_por_cliente", lambda ids: {})

    llamadas_ultimo = []
    monkeypatch.setattr(writeback, "_ultimo_historial_por_cliente", lambda ids: (llamadas_ultimo.append(ids) or {}))
    llamadas_actual = []
    monkeypatch.setattr(writeback, "_historial_actual", lambda conn, cliente_id: (llamadas_actual.append(cliente_id) or 0))

    writeback.guardar_resultados(_resultados_prueba(), dry_run=False)

    assert llamadas_ultimo == []
    assert llamadas_actual == []


# ---------- Verificación en vivo contra dbo.sp_ActualizarScoringCliente ----------


def _crear_cliente_prueba(segmento_inicial: int = 1) -> int:
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO dbo.Cliente (NombreCompleto, SegmentoRiesgoId, ClasificacionId, PlazoPagoId) VALUES (?, ?, ?, ?)",
            ("__prueba_writeback_8_8__", segmento_inicial, 4, 3),
        )
        # SCOPE_IDENTITY() vuelve NULL aquí (dbo.Cliente tiene un trigger que
        # rompe el scope, igual que con OUTPUT INSERTED — ver memoria del
        # proyecto); IDENT_CURRENT no depende del scope de la sesión.
        cliente_id = conn.exec_driver_sql("SELECT IDENT_CURRENT('dbo.Cliente')").scalar()
    return int(cliente_id)


def _eliminar_cliente_prueba(cliente_id: int | None) -> None:
    if cliente_id is None:
        return
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "DELETE FROM dbo.ScoringDetalleVariable WHERE HistorialId IN "
            "(SELECT HistorialId FROM dbo.ClienteRiesgoHistorial WHERE ClienteId=?)",
            (cliente_id,),
        )
        conn.exec_driver_sql("DELETE FROM dbo.ClienteRiesgoHistorial WHERE ClienteId=?", (cliente_id,))
        conn.exec_driver_sql("DELETE FROM dbo.Cliente WHERE ClienteId=?", (cliente_id,))


@pytest.fixture(autouse=False)
def _sin_dobles(monkeypatch):
    monkeypatch.undo()


def test_guardar_resultados_real_actualiza_cliente_y_registra_historial(_sin_dobles):
    cliente_id = None
    try:
        cliente_id = _crear_cliente_prueba(segmento_inicial=1)
        historial_antes = int(
            read_sql(f"SELECT COUNT(*) AS n FROM dbo.ClienteRiesgoHistorial WHERE ClienteId={cliente_id}")["n"].iloc[0]
        )

        df = pd.DataFrame(
            {
                "ClienteId": [cliente_id],
                "ScoreRiesgo": [250.0],
                "SegmentoRiesgoId": [4],  # distinto del inicial (1)
                "ClasificacionId": [4],
                "PlazoPagoRecomendadoId": [1],
                "Motivo": ["Prueba automatizada del paso 8.8"],
            }
        )

        resultado = writeback.guardar_resultados(df, dry_run=False)

        assert resultado["cambiaron_segmento"] == [cliente_id]
        assert resultado["fallidos"] == []

        fila = read_sql(
            f"SELECT SegmentoRiesgoId, ClasificacionId, PlazoPagoId FROM dbo.Cliente WHERE ClienteId={cliente_id}"
        ).iloc[0]
        assert int(fila["SegmentoRiesgoId"]) == 4
        assert int(fila["ClasificacionId"]) == 4
        assert int(fila["PlazoPagoId"]) == 1

        historial_despues = int(
            read_sql(f"SELECT COUNT(*) AS n FROM dbo.ClienteRiesgoHistorial WHERE ClienteId={cliente_id}")["n"].iloc[0]
        )
        assert historial_despues == historial_antes + 1
    finally:
        _eliminar_cliente_prueba(cliente_id)


def test_guardar_resultados_real_con_detalle_variables_persiste_el_desglose(_sin_dobles):
    """Explicabilidad del score (sección 4.4.3): cuando sí cambia el
    segmento (y por tanto sp_ActualizarScoringCliente inserta una fila en
    ClienteRiesgoHistorial), guardar_resultados debe persistir el desglose
    de detalle_variables contra el HistorialId que acaba de crearse."""
    cliente_id = None
    try:
        cliente_id = _crear_cliente_prueba(segmento_inicial=1)

        df = pd.DataFrame(
            {
                "ClienteId": [cliente_id],
                "ScoreRiesgo": [250.0],
                "SegmentoRiesgoId": [4],  # distinto del inicial (1) -> genera fila de historial
                "ClasificacionId": [4],
                "PlazoPagoRecomendadoId": [1],
                "Motivo": ["Prueba automatizada del detalle de variables"],
            }
        )
        detalle_variables = {
            cliente_id: [
                {"Variable": "(base)", "Etiqueta": "Puntaje base", "ValorOriginal": None, "WOE": None, "Puntos": 487.1, "Orden": 0},
                {"Variable": "DiasMoraPromedioPrevio", "Etiqueta": "Mora promedio (días)", "ValorOriginal": 45, "WOE": -0.5, "Puntos": -12.345, "Orden": 1},
            ]
        }

        resultado = writeback.guardar_resultados(df, dry_run=False, detalle_variables=detalle_variables)
        assert resultado["cambiaron_segmento"] == [cliente_id]

        historial_id = int(
            read_sql(
                f"SELECT TOP 1 HistorialId FROM dbo.ClienteRiesgoHistorial WHERE ClienteId={cliente_id} ORDER BY HistorialId DESC"
            )["HistorialId"].iloc[0]
        )
        filas = read_sql(
            f"SELECT Variable, Etiqueta, ValorOriginal, WOE, Puntos, Orden FROM dbo.ScoringDetalleVariable "
            f"WHERE HistorialId={historial_id} ORDER BY Orden"
        )
        assert len(filas) == 2
        assert filas.iloc[0]["Variable"] == "(base)"
        assert pd.isna(filas.iloc[0]["ValorOriginal"])
        assert filas.iloc[1]["Variable"] == "DiasMoraPromedioPrevio"
        assert filas.iloc[1]["ValorOriginal"] == "45"
        assert float(filas.iloc[1]["Puntos"]) == -12.345
    finally:
        _eliminar_cliente_prueba(cliente_id)


def test_guardar_resultados_real_con_detalle_variables_del_modelo_real_no_falla_por_tipos_numpy(_sin_dobles):
    """Cierra el hueco que dejaba la prueba anterior (con floats de Python
    escritos a mano): aquí el detalle viene de score.construir_detalle_variables
    contra un cliente real (modelo.coef_/intercept_ son np.float64; sin el
    cast a tipos nativos en score.py, pyodbc podía no bindear algún valor
    igual que un float/int de Python)."""
    cliente_id = None
    try:
        cliente_id = _crear_cliente_prueba(segmento_inicial=1)

        # Un cliente real cualquiera, solo para obtener una fila con la
        # forma real de dbo.vw_FeaturesScoringCliente (no se escribe nada
        # sobre ese cliente; el detalle se liga al HistorialId del cliente
        # descartable, no a su identidad).
        df_real = extract.extraer_features().iloc[[0]]
        detalle_real = score.construir_detalle_variables(df_real)
        filas_detalle = next(iter(detalle_real.values()))

        df = pd.DataFrame(
            {
                "ClienteId": [cliente_id],
                "ScoreRiesgo": [250.0],
                "SegmentoRiesgoId": [4],
                "ClasificacionId": [4],
                "PlazoPagoRecomendadoId": [1],
                "Motivo": ["Prueba automatizada: detalle del modelo real"],
            }
        )

        resultado = writeback.guardar_resultados(
            df, dry_run=False, detalle_variables={cliente_id: filas_detalle}
        )
        assert resultado["fallidos"] == []

        historial_id = int(
            read_sql(
                f"SELECT TOP 1 HistorialId FROM dbo.ClienteRiesgoHistorial WHERE ClienteId={cliente_id} ORDER BY HistorialId DESC"
            )["HistorialId"].iloc[0]
        )
        n_filas = int(
            read_sql(f"SELECT COUNT(*) AS n FROM dbo.ScoringDetalleVariable WHERE HistorialId={historial_id}")["n"].iloc[0]
        )
        assert n_filas == len(filas_detalle)
    finally:
        _eliminar_cliente_prueba(cliente_id)


def test_guardar_resultados_dry_run_no_modifica_la_base_real(_sin_dobles):
    cliente_id = None
    try:
        cliente_id = _crear_cliente_prueba(segmento_inicial=1)
        df = pd.DataFrame(
            {
                "ClienteId": [cliente_id],
                "ScoreRiesgo": [900.0],
                "SegmentoRiesgoId": [4],  # propone un cambio real...
                "ClasificacionId": [1],
                "PlazoPagoRecomendadoId": [5],
            }
        )

        resultado = writeback.guardar_resultados(df, dry_run=True)
        assert resultado["cambiaron_segmento"] == [cliente_id]  # ...lo detecta...

        fila = read_sql(f"SELECT SegmentoRiesgoId FROM dbo.Cliente WHERE ClienteId={cliente_id}").iloc[0]
        assert int(fila["SegmentoRiesgoId"]) == 1  # ...pero no lo escribe
    finally:
        _eliminar_cliente_prueba(cliente_id)
