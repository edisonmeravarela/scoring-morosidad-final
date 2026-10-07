from unittest.mock import patch

import pandas as pd
import pytest

from src import extract


def _df_de(n: int) -> pd.DataFrame:
    return pd.DataFrame({"ClienteId": range(1, n + 1), "NumFacturas": [1] * n})


def test_extraer_features_devuelve_dataframe_indexado_por_clienteid_real():
    # Verificación en vivo: dbo.vw_FeaturesScoringCliente ya existe en la
    # base real y tiene 169 filas (confirmado a mano contra el SQL Server).
    df = extract.extraer_features()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "ClienteId"
    assert len(df) >= 30
    assert "TuvoMoraHistorica" in df.columns
    assert "PorcFacturasEnMora" in df.columns


def test_extraer_features_registra_filas_y_fecha_en_el_log():
    log_path = extract.LOGS_DIR / "extract.log"
    # Longitud en caracteres, no en bytes: el log tiene tildes en UTF-8
    # (multibyte), así que un offset en bytes cortaría el texto a la mitad.
    contenido_antes = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    extract.extraer_features()

    contenido_despues = log_path.read_text(encoding="utf-8")
    contenido_nuevo = contenido_despues[len(contenido_antes) :]
    assert "filas obtenidas" in contenido_nuevo


def test_extraer_features_aborta_si_hay_menos_de_30_clientes():
    with patch("src.extract.read_sql", return_value=_df_de(10)):
        with pytest.raises(extract.DatosInsuficientesError, match="10"):
            extract.extraer_features()


def test_extraer_features_acepta_si_hay_exactamente_el_minimo():
    with patch("src.extract.read_sql", return_value=_df_de(30)):
        df = extract.extraer_features()
    assert len(df) == 30
    assert df.index.name == "ClienteId"


def test_extraer_features_usa_el_minimo_configurado(monkeypatch):
    # El umbral de 30 no está hardcodeado: viene de config.extract.minimo_clientes.
    with patch("src.extract.cargar_config", return_value={"extract": {"minimo_clientes": 5}}):
        with patch("src.extract.read_sql", return_value=_df_de(6)):
            df = extract.extraer_features()
        assert len(df) == 6

        with patch("src.extract.read_sql", return_value=_df_de(4)):
            with pytest.raises(extract.DatosInsuficientesError):
                extract.extraer_features()


def test_extraer_dataset_entrenamiento_devuelve_dataframe_indexado_por_facturaid_real():
    # Verificación en vivo: dbo.vw_DatasetEntrenamientoFactura (point-in-time,
    # ajuste post-8.4 contra leakage) ya existe y tiene ~413 facturas.
    df = extract.extraer_dataset_entrenamiento()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "FacturaId"
    assert len(df) >= 30
    assert "Target" in df.columns
    assert "SinHistorialPago" in df.columns
    assert "PorcMoraPrevia" in df.columns


def test_extraer_dataset_entrenamiento_aborta_si_hay_menos_de_30_facturas():
    df_chico = pd.DataFrame({"FacturaId": range(1, 11), "SinHistorialPago": [0] * 10})
    with patch("src.extract.read_sql", return_value=df_chico):
        with pytest.raises(extract.DatosInsuficientesError, match="10"):
            extract.extraer_dataset_entrenamiento()


def _df_entrenamiento(n_con_historial: int, n_sin_historial: int) -> pd.DataFrame:
    total = n_con_historial + n_sin_historial
    return pd.DataFrame(
        {
            "FacturaId": range(1, total + 1),
            "SinHistorialPago": [0] * n_con_historial + [1] * n_sin_historial,
            "Target": [0, 1] * (total // 2) + [0] * (total % 2),
        }
    )


def _df_historial(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ClienteId": range(1, n + 1),
            "FacturasResueltas": [5] * n,
            "UmbralVigente": [3] * n,
            "TieneHistorial": [1] * n,
        }
    )


def test_agregar_historial_evaluable_incorpora_las_tres_columnas():
    df = _df_de(3).set_index("ClienteId")
    with patch("src.extract.read_sql", return_value=_df_historial(3)):
        resultado = extract.agregar_historial_evaluable(df)
    assert list(resultado["TieneHistorial"]) == [1, 1, 1]
    assert list(resultado["FacturasResueltas"]) == [5, 5, 5]
    assert list(resultado["UmbralVigente"]) == [3, 3, 3]
    assert resultado.index.name == "ClienteId"


def test_agregar_historial_evaluable_no_descarta_clientes_sin_fila_de_historial():
    # left join: un cliente que por lo que sea no aparezca en
    # dbo.vw_HistorialClienteActual no debe perderse de df_clientes.
    df = _df_de(3).set_index("ClienteId")
    with patch("src.extract.read_sql", return_value=_df_historial(2)):
        resultado = extract.agregar_historial_evaluable(df)
    assert len(resultado) == 3
    assert pd.isna(resultado.loc[3, "TieneHistorial"])


def test_agregar_historial_evaluable_sobre_datos_reales():
    df = extract.extraer_features()
    resultado = extract.agregar_historial_evaluable(df)
    assert len(resultado) == len(df)
    assert "TieneHistorial" in resultado.columns
    assert resultado["TieneHistorial"].isin([0, 1]).all()


def test_excluir_sin_historial_descarta_solo_esas_filas():
    df = _df_entrenamiento(n_con_historial=7, n_sin_historial=3)
    resultado = extract.excluir_sin_historial(df)
    assert len(resultado) == 7
    assert (resultado["SinHistorialPago"] == 0).all()


def test_excluir_sin_historial_es_no_op_si_no_existe_la_columna():
    df = pd.DataFrame({"FacturaId": [1, 2], "Target": [0, 1]})
    resultado = extract.excluir_sin_historial(df)
    assert len(resultado) == 2


def test_excluir_sin_historial_sobre_datos_reales_reduce_el_dataset():
    df_completo = extract.extraer_dataset_entrenamiento()
    df_filtrado = extract.excluir_sin_historial(df_completo)
    assert len(df_filtrado) < len(df_completo)
    assert (df_filtrado["SinHistorialPago"] == 0).all()
