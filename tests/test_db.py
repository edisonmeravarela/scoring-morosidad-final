import pandas as pd
import pytest

from src import db


def test_get_engine_lee_credenciales_de_variables_de_entorno(monkeypatch):
    # Criterio de aceptación 8.2: las credenciales nunca están escritas en
    # el código; deben salir de las variables de entorno en tiempo de
    # ejecución. Lo verificamos capturando la cadena de conexión real que
    # arma get_engine() y confirmando que refleja el valor de entorno.
    monkeypatch.setenv("DB_SERVER", "servidor-de-prueba")
    monkeypatch.setenv("DB_NAME", "base-de-prueba")
    monkeypatch.setenv("DB_USER", "usuario-de-prueba")
    monkeypatch.setenv("DB_PASSWORD", "clave-de-prueba-xyz")

    engine = db.get_engine()
    assert engine is not None

    # SQLAlchemy no expone la cadena ODBC cruda vía str(url) de forma
    # legible; verificamos directamente la función que la construye para
    # confirmar que efectivamente usó las variables de entorno.
    connection_string = db._connection_string()
    assert "servidor-de-prueba" in connection_string
    assert "base-de-prueba" in connection_string
    assert "usuario-de-prueba" in connection_string
    assert "clave-de-prueba-xyz" in connection_string


def test_get_engine_falla_con_mensaje_claro_si_falta_una_variable(monkeypatch):
    monkeypatch.delenv("DB_SERVER", raising=False)
    monkeypatch.delenv("DB_NAME", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)

    with pytest.raises(db.ConfiguracionIncompleta, match="DB_SERVER"):
        db.get_engine()


def test_connection_string_no_tiene_credenciales_hardcodeadas_en_el_modulo():
    # El código fuente de db.py no debe contener la contraseña real usada
    # en desarrollo (ni ninguna otra): todo debe salir de _variable_entorno().
    with open(db.__file__, encoding="utf-8") as f:
        contenido = f.read()
    assert "Solticem" not in contenido


def test_read_sql_conecta_a_sql_server_real_y_devuelve_dataframe():
    # Verificación en vivo contra la base real (no un doble): confirma que
    # get_engine()/read_sql() arman una conexión que SQL Server acepta.
    df = db.read_sql("SELECT 1 AS valor")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert int(df.iloc[0]["valor"]) == 1


def test_read_sql_puede_consultar_una_tabla_real_del_esquema():
    df = db.read_sql("SELECT TOP 5 ClienteId, NombreCompleto FROM dbo.Cliente")
    assert isinstance(df, pd.DataFrame)
    assert "ClienteId" in df.columns
    assert "NombreCompleto" in df.columns
    assert len(df) <= 5


def test_exec_sp_ejecuta_un_procedimiento_real_con_parametros_nombrados():
    # Verificación en vivo de exec_sp() (paso 8.2), nunca antes ejercitada
    # contra un procedimiento real: usa dbo.sp_ActualizarScoringCliente
    # (paso 8.8) sobre un cliente descartable, para no depender de datos
    # reales ni dejar rastro. Un cliente descartable se crea y se limpia
    # en el propio test para no depender de otros módulos.
    engine = db.get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO dbo.Cliente (NombreCompleto, SegmentoRiesgoId) VALUES (?, ?)",
            ("__prueba_exec_sp_8_2__", 1),
        )
        cliente_id = int(conn.exec_driver_sql("SELECT IDENT_CURRENT('dbo.Cliente')").scalar())

    try:
        db.exec_sp(
            "dbo.sp_ActualizarScoringCliente",
            {
                "ClienteId": cliente_id,
                "ScoreRiesgo": 500.0,
                "SegmentoRiesgoId": 2,
                "ClasificacionId": None,
                "PlazoPagoRecomendadoId": None,
                "Motivo": "Prueba automatizada de exec_sp (paso 8.2)",
                "UsuarioId": None,
            },
        )
        fila = db.read_sql(f"SELECT SegmentoRiesgoId FROM dbo.Cliente WHERE ClienteId={cliente_id}").iloc[0]
        assert int(fila["SegmentoRiesgoId"]) == 2
    finally:
        with engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM dbo.ClienteRiesgoHistorial WHERE ClienteId=?", (cliente_id,))
            conn.exec_driver_sql("DELETE FROM dbo.Cliente WHERE ClienteId=?", (cliente_id,))
