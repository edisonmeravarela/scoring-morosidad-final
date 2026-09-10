import pandas as pd
import pytest

from src import esquema


def _detalle_sintetico(criticos=0, advertencias=0):
    filas = []
    for i in range(criticos):
        filas.append(
            {
                "Objeto": "Factura",
                "Columna": f"ColCritica{i}",
                "Hallazgo": "COLUMNA AUSENTE",
                "TipoDatoEsperado": "int",
                "TipoDatoActual": None,
                "Severidad": "CRITICO",
            }
        )
    for i in range(advertencias):
        filas.append(
            {
                "Objeto": "Cliente",
                "Columna": f"ColNueva{i}",
                "Hallazgo": "COLUMNA NUEVA NO DECLARADA",
                "TipoDatoEsperado": None,
                "TipoDatoActual": "nvarchar",
                "Severidad": "ADVERTENCIA",
            }
        )
    filas.append(
        {
            "Objeto": "Cliente",
            "Columna": "ClienteId",
            "Hallazgo": "CONFORME",
            "TipoDatoEsperado": "int",
            "TipoDatoActual": "int",
            "Severidad": "OK",
        }
    )
    return pd.DataFrame(filas)


def test_validar_esquema_pasa_si_no_hay_criticos_ni_advertencias(monkeypatch):
    detalle = _detalle_sintetico()
    monkeypatch.setattr(esquema, "obtener_validacion", lambda version=esquema.VERSION_CONTRATO_DEFECTO: (detalle, {"Criticos": 0, "Advertencias": 0}))
    resultado = esquema.validar_esquema()
    assert resultado["Criticos"] == 0


def test_validar_esquema_lanza_error_si_hay_un_solo_critico(monkeypatch):
    detalle = _detalle_sintetico(criticos=1)
    monkeypatch.setattr(esquema, "obtener_validacion", lambda version=esquema.VERSION_CONTRATO_DEFECTO: (detalle, {"Criticos": 1, "Advertencias": 0}))
    with pytest.raises(esquema.EsquemaInvalidoError, match="1 hallazgo"):
        esquema.validar_esquema()


def test_validar_esquema_no_lanza_por_advertencias(monkeypatch):
    detalle = _detalle_sintetico(advertencias=3)
    monkeypatch.setattr(esquema, "obtener_validacion", lambda version=esquema.VERSION_CONTRATO_DEFECTO: (detalle, {"Criticos": 0, "Advertencias": 3}))
    resultado = esquema.validar_esquema()  # no debe lanzar
    assert resultado["Advertencias"] == 3


def test_validar_esquema_usa_la_version_indicada(monkeypatch):
    versiones_consultadas = []

    def _obtener(version=esquema.VERSION_CONTRATO_DEFECTO):
        versiones_consultadas.append(version)
        return _detalle_sintetico(), {"Criticos": 0, "Advertencias": 0}

    monkeypatch.setattr(esquema, "obtener_validacion", _obtener)
    esquema.validar_esquema(version="2.0")
    assert versiones_consultadas == ["2.0"]


# ---------- Verificación en vivo ----------


def test_obtener_validacion_contra_el_esquema_real():
    detalle, veredicto = esquema.obtener_validacion()
    assert {"Objeto", "Columna", "Hallazgo", "Severidad"} <= set(detalle.columns)
    assert veredicto["Criticos"] >= 0
    assert veredicto["Advertencias"] >= 0
    assert len(detalle) > 0


def test_validar_esquema_contra_datos_reales():
    # No se asume de antemano el estado del esquema real: ambos
    # desenlaces son correctos y se verifican explícitamente.
    try:
        veredicto = esquema.validar_esquema()
        assert veredicto["Criticos"] == 0
    except esquema.EsquemaInvalidoError as exc:
        assert "hallazgo" in str(exc)
