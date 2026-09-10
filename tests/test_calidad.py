import pandas as pd
import pytest

from src import calidad


def _veredicto(criticos=0, advertencias=0, conformes=21):
    return {
        "EjecucionId": "11111111-1111-1111-1111-111111111111",
        "Criticos": criticos,
        "Advertencias": advertencias,
        "Conformes": conformes,
        "Veredicto": (
            "NO APTO — corregir los hallazgos críticos antes de entrenar o scorear"
            if criticos > 0
            else "APTO CON OBSERVACIONES — documentar las advertencias en el informe de trazabilidad"
            if advertencias > 0
            else "APTO — el conjunto de datos supera todos los controles"
        ),
    }


def test_validar_antes_de_entrenar_pasa_si_todo_esta_conforme(monkeypatch):
    monkeypatch.setattr(calidad, "obtener_veredicto", lambda: _veredicto(criticos=0, advertencias=0))
    resultado = calidad.validar_antes_de_entrenar()
    assert resultado["Criticos"] == 0
    assert "APTO —" in resultado["Veredicto"]


def test_validar_antes_de_entrenar_lanza_error_con_el_mensaje_del_veredicto_si_hay_criticos(monkeypatch):
    veredicto = _veredicto(criticos=1, advertencias=0)
    monkeypatch.setattr(calidad, "obtener_veredicto", lambda: veredicto)

    with pytest.raises(calidad.CalidadDatosError) as exc_info:
        calidad.validar_antes_de_entrenar()

    assert str(exc_info.value) == veredicto["Veredicto"]
    assert "NO APTO" in str(exc_info.value)


def test_validar_antes_de_entrenar_no_lanza_por_advertencias_pero_las_registra(monkeypatch, caplog):
    import logging

    veredicto = _veredicto(criticos=0, advertencias=2)
    monkeypatch.setattr(calidad, "obtener_veredicto", lambda: veredicto)
    monkeypatch.setattr(
        calidad,
        "obtener_advertencias",
        lambda ejecucion_id: pd.DataFrame(
            [
                {
                    "Dimension": "CONSISTENCIA",
                    "Codigo": "B2_SOBREPAGO",
                    "Descripcion": "Facturas con abonos que superan su valor total",
                    "RegistrosAfectados": 43,
                    "ValorObservado": "43",
                    "UmbralEsperado": "0",
                },
            ]
        ),
    )

    with caplog.at_level(logging.WARNING, logger="scoring_morosidad.calidad"):
        resultado = calidad.validar_antes_de_entrenar()  # no debe lanzar

    assert resultado["Advertencias"] == 2
    assert any("B2_SOBREPAGO" in r.message for r in caplog.records)


def test_validar_antes_de_entrenar_no_consulta_advertencias_si_no_hay(monkeypatch):
    llamadas = []
    monkeypatch.setattr(calidad, "obtener_veredicto", lambda: _veredicto(criticos=0, advertencias=0))
    monkeypatch.setattr(calidad, "obtener_advertencias", lambda ejecucion_id: llamadas.append(1))

    calidad.validar_antes_de_entrenar()

    assert llamadas == []


# ---------- Verificación en vivo ----------


def test_obtener_veredicto_contra_datos_reales():
    veredicto = calidad.obtener_veredicto()
    assert set(veredicto) >= {"EjecucionId", "Criticos", "Advertencias", "Conformes", "Veredicto"}
    assert veredicto["Criticos"] >= 0
    assert isinstance(veredicto["Veredicto"], str) and veredicto["Veredicto"]


def test_validar_antes_de_entrenar_contra_datos_reales():
    # No se asume de antemano si los datos reales están limpios: ambos
    # desenlaces son correctos y se verifican explícitamente.
    try:
        veredicto = calidad.validar_antes_de_entrenar()
        assert veredicto["Criticos"] == 0
        assert veredicto["Veredicto"].startswith("APTO")
    except calidad.CalidadDatosError as exc:
        assert "NO APTO" in str(exc)
