"""
tests/test_reporte_calibracion.py
==================================
Pruebas de ejecutar_calibracion_definitiva() (reporte_calibracion.py): la
función que main.py:comando_calibrar() invoca para el botón "Calibración
Modelo" del frontend de GESTIONCARTERA. No toca la base real: se
monkeypatchean conjunto_ventana_temporal/conjunto_validacion_cruzada (las
dos únicas funciones de este módulo que consultan
dbo.vw_DatasetEntrenamientoFactura) y las escrituras de
persistir_estado/guardar_calibrador.

    pytest tests/test_reporte_calibracion.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

import reporte_calibracion as repcal
from src import calibracion as cal


def _datos_sinteticos(n=60, semilla=5):
    """(y, p, grupos, prevalencia_entrenamiento) con una relación conocida
    entre y y p — mismo criterio que tests/test_calibracion.py::_datos."""
    rng = np.random.default_rng(semilla)
    x = rng.normal(size=n)
    logit = 1.2 * x
    p_real = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p_real)
    grupos = np.arange(n)
    return y, p_real, grupos, float(y.mean())


@pytest.fixture(autouse=True)
def _sin_base_real(monkeypatch, tmp_path):
    """Aísla las dos únicas fuentes de datos reales de este módulo, y
    cualquier escritura (DB o disco) que no sea a tmp_path."""
    datos = _datos_sinteticos()
    monkeypatch.setattr(repcal, "conjunto_validacion_cruzada", lambda *a, **k: datos)
    monkeypatch.setattr(repcal, "conjunto_ventana_temporal", lambda *a, **k: datos)

    llamadas_exec_sp = []
    llamadas_ejecutar_muchos = []
    monkeypatch.setattr(repcal.db, "exec_sp", lambda *a, **k: llamadas_exec_sp.append((a, k)))
    monkeypatch.setattr(repcal.db, "ejecutar_muchos", lambda *a, **k: llamadas_ejecutar_muchos.append((a, k)))

    guardados = []
    monkeypatch.setattr(cal, "guardar_calibrador", lambda c, ruta: (guardados.append(c), ruta)[1])

    return {"exec_sp": llamadas_exec_sp, "ejecutar_muchos": llamadas_ejecutar_muchos, "guardados": guardados}


def test_devuelve_el_resumen_esperado(_sin_base_real, tmp_path):
    resultado = repcal.ejecutar_calibracion_definitiva(salida=tmp_path)

    assert set(resultado) == {
        "evaluacion_id", "resultados", "parrafo", "persistido",
        "calibrador_principal", "calibrador_guardado",
    }
    assert "crossval" in resultado["resultados"]
    assert "ventana" in resultado["resultados"]
    assert resultado["persistido"] is False
    assert resultado["calibrador_guardado"] is None  # aplicar=False por defecto
    assert resultado["parrafo"]


def test_persistir_false_no_escribe_en_la_base(_sin_base_real, tmp_path):
    repcal.ejecutar_calibracion_definitiva(persistir=False, salida=tmp_path)

    assert _sin_base_real["exec_sp"] == []
    assert _sin_base_real["ejecutar_muchos"] == []


def test_persistir_true_escribe_en_la_base(_sin_base_real, tmp_path):
    resultado = repcal.ejecutar_calibracion_definitiva(
        evaluacion_id=999, persistir=True, salida=tmp_path)

    assert resultado["persistido"] is True
    assert len(_sin_base_real["exec_sp"]) > 0


def test_persistir_sin_evaluacion_id_lanza(_sin_base_real, tmp_path):
    with pytest.raises(ValueError, match="evaluacion_id"):
        repcal.ejecutar_calibracion_definitiva(persistir=True, salida=tmp_path)


def test_aplicar_con_calibrador_ninguno_no_guarda_nada(_sin_base_real, tmp_path):
    # El criterio forzado a NINGUNO replica el caso real ya observado en
    # vivo (CROSSVAL de esta sesión): sin corrección, no debe tocar
    # models/calibrador.pkl aunque aplicar=True.
    resultado = repcal.ejecutar_calibracion_definitiva(
        forzado="NINGUNO", aplicar=True, salida=tmp_path)

    assert resultado["calibrador_principal"] == "NINGUNO"
    assert resultado["calibrador_guardado"] is None
    assert _sin_base_real["guardados"] == []


def test_aplicar_con_calibrador_real_guarda_el_archivo(_sin_base_real, tmp_path):
    resultado = repcal.ejecutar_calibracion_definitiva(
        forzado="PRIOR", aplicar=True, salida=tmp_path)

    assert resultado["calibrador_principal"] == "PRIOR"
    assert resultado["calibrador_guardado"] == "PRIOR"
    assert len(_sin_base_real["guardados"]) == 1
