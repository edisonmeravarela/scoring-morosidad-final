"""
tests/test_calibracion.py
==========================
Doce pruebas que fijan el comportamiento del modulo de calibracion. Siguen el
criterio del proyecto: las que tocan la base real llevan el sufijo _vivo, sin
marcadores de pytest.

    pytest tests/test_calibracion.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import calibracion as cal


# ===========================================================================
# Datos sinteticos de apoyo
# ===========================================================================
def _datos(n=600, prevalencia=0.5, ruido=1.0, semilla=7):
    """Genera (y, p) con una relacion conocida entre ambos."""
    rng = np.random.default_rng(semilla)
    x = rng.normal(size=n)
    logit = 1.5 * x + np.log(prevalencia / (1 - prevalencia))
    p_real = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p_real)
    p_decl = 1 / (1 + np.exp(-(1.5 * x / ruido + np.log(prevalencia / (1 - prevalencia)))))
    return y, p_decl


# ===========================================================================
# 1. Un modelo perfectamente calibrado produce las metricas de referencia
# ===========================================================================
def test_modelo_calibrado_da_pendiente_uno_e_intercepto_cero():
    rng = np.random.default_rng(11)
    p = rng.uniform(0.05, 0.95, size=4000)
    y = rng.binomial(1, p)

    m = cal.metricas_calibracion(y, p)

    assert 0.85 < m.pendiente < 1.15, f"pendiente={m.pendiente}"
    assert abs(m.intercepto) < 0.20, f"intercepto={m.intercepto}"
    assert m.ece < 0.05, f"ECE={m.ece}"
    assert abs(m.sesgo_prevalencia) < 0.03


# ===========================================================================
# 2. Un desplazamiento conocido se detecta como sesgo
# ===========================================================================
def test_sobreestimacion_se_refleja_en_el_sesgo_y_el_intercepto():
    rng = np.random.default_rng(13)
    p_real = rng.uniform(0.05, 0.95, size=3000)
    y = rng.binomial(1, p_real)

    # Se infla la probabilidad declarada en +1 logit
    z = np.log(p_real / (1 - p_real)) + 1.0
    p_inflada = 1 / (1 + np.exp(-z))

    m = cal.metricas_calibracion(y, p_inflada)

    assert m.sesgo_prevalencia > 0.10, "deberia detectar sobreestimacion"
    assert m.intercepto < -0.7, f"el intercepto deberia acercarse a -1: {m.intercepto}"


# ===========================================================================
# 3. La correccion de prevalencia revierte exactamente el desplazamiento
# ===========================================================================
def test_ajustar_prior_calcula_el_desplazamiento_correcto():
    c = cal.ajustar_prior(0.533, 0.225)
    assert abs(c.ajuste_logit - (-1.3690)) < 0.001, c.ajuste_logit

    # Caso trivial: misma prevalencia, ajuste nulo
    c0 = cal.ajustar_prior(0.4, 0.4)
    assert abs(c0.ajuste_logit) < 1e-12


def test_prior_rechaza_prevalencias_invalidas():
    for mala in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            cal.ajustar_prior(0.5, mala)


# ===========================================================================
# 4. LA PRUEBA CLAVE: la calibracion no altera el ordenamiento
# ===========================================================================
# Es la garantia de que AUC y KS ya reportados en la tesis siguen siendo validos.
def test_la_calibracion_no_cambia_el_area_bajo_la_curva():
    y, p = _datos(n=800, prevalencia=0.53)

    for c in (
        cal.ajustar_prior(0.533, 0.225),
        cal.ajustar_platt(y, p),
        cal.ajustar_isotonico(y, p),
    ):
        auc_antes = roc_auc_score(y, p)
        auc_despues = roc_auc_score(y, c.transformar(p))
        # La isotonica empata valores, lo que puede mover el AUC marginalmente
        tolerancia = 0.02 if c.tipo == "ISOTONICO" else 1e-9
        assert abs(auc_antes - auc_despues) < tolerancia, (
            f"{c.tipo}: AUC cambio de {auc_antes:.6f} a {auc_despues:.6f}"
        )


def test_verificar_monotonia_acepta_los_tres_calibradores():
    y, p = _datos(n=500)
    for c in (cal.ajustar_prior(0.5, 0.3), cal.ajustar_platt(y, p), cal.ajustar_isotonico(y, p)):
        assert cal.verificar_monotonia(p, c), f"{c.tipo} no es monotono"


def test_verificar_monotonia_rechaza_una_transformacion_no_monotona():
    """Control negativo: si el verificador no detectara esto, no serviria."""
    class CalibradorRoto:
        tipo = "ROTO"
        def transformar(self, p):
            return 1.0 - np.asarray(p)   # invierte el ordenamiento

    _, p = _datos(n=200)
    assert not cal.verificar_monotonia(p, CalibradorRoto())


# ===========================================================================
# 5. La correccion mejora efectivamente las metricas
# ===========================================================================
def test_la_correccion_reduce_el_sesgo_y_el_error_de_calibracion():
    rng = np.random.default_rng(23)
    n = 2000
    p_real = rng.uniform(0.02, 0.60, size=n)
    y = rng.binomial(1, p_real)

    prev_obs = y.mean()
    # Modelo entrenado sobre una poblacion mas morosa: sobreestima
    z = np.log(p_real / (1 - p_real)) + 1.2
    p_decl = 1 / (1 + np.exp(-z))

    c = cal.ajustar_prior(prevalencia_entrenamiento=float(1 / (1 + np.exp(-1.2)) * 0 + 0.55),
                          prevalencia_objetivo=float(prev_obs))
    comp = cal.comparar_antes_despues(y, p_decl, c)

    sesgo = comp.loc[comp.metrica == "Sesgo de prevalencia"].iloc[0]
    ece = comp.loc[comp.metrica == "ECE"].iloc[0]
    assert abs(sesgo.despues) < abs(sesgo.antes), "el sesgo deberia reducirse"
    assert ece.despues <= ece.antes + 1e-6, "el ECE no deberia empeorar"


# ===========================================================================
# 6. Tabla de fiabilidad
# ===========================================================================
def test_la_tabla_de_fiabilidad_cubre_todos_los_casos():
    y, p = _datos(n=700)
    t = cal.tabla_fiabilidad(y, p, n_tramos=10)

    assert len(t) == 10
    assert t["casos"].sum() == len(y), "todos los casos deben caer en algun tramo"
    assert t["positivos"].sum() == int(y.sum())
    assert (t["limite_inferior"] <= t["limite_superior"]).all()


def test_tramos_por_cuantiles_reparte_los_casos_de_forma_pareja():
    y, p = _datos(n=600)
    t = cal.tabla_fiabilidad(y, p, n_tramos=5, por_cuantiles=True)
    con_casos = t[t["casos"] > 0]
    # Ningun tramo debe acumular mas de la mitad de las observaciones
    assert con_casos["casos"].max() < 0.5 * len(y)


# ===========================================================================
# 7. Validacion de entradas
# ===========================================================================
def test_rechaza_entradas_mal_formadas():
    with pytest.raises(ValueError):
        cal.metricas_calibracion([0, 1, 0], [0.1, 0.2])          # longitudes distintas
    with pytest.raises(ValueError):
        cal.metricas_calibracion([0, 1, 2], [0.1, 0.2, 0.3])     # etiqueta invalida
    with pytest.raises(ValueError):
        cal.metricas_calibracion([0, 1, 0], [0.1, 1.4, 0.3])     # probabilidad fuera de rango


# ===========================================================================
# 8. Verificacion contra los datos reales
# ===========================================================================
def test_calibracion_sobre_el_conjunto_real_vivo(tmp_path, monkeypatch):
    """Reproduce el calculo del reporte contra la base real.

    Se omite si el entorno no tiene acceso a la base. Usa el mismo split que
    entrenar() usa de verdad (ajustes_evaluacion.dividir_particion_agrupada)
    — no train.dividir_temporal_por_cliente(), que ese mismo módulo deja sin
    usar (ver su propio docstring) — y el nombre real de la columna objetivo,
    leído de train._config_entrenamiento() (no existe una constante
    train.TARGET_COL)."""
    try:
        from src import extract, features, train
        from src.ajustes_evaluacion import dividir_particion_agrupada
        config = train._config_entrenamiento()
        target_col = config.get("columna_target", "Target")
        datos = extract.excluir_sin_historial(extract.extraer_dataset_entrenamiento())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"sin acceso a la base: {exc}")

    tr, va = dividir_particion_agrupada(
        datos,
        fecha_col=config.get("columna_fecha", "FechaEmision"),
        cliente_col=config.get("columna_cliente", "ClienteId"),
        target_col=target_col,
        prop_validacion=config.get("prop_validacion", 0.20),
        min_filas_validacion=config.get("min_filas_validacion", 30),
        min_positivos_validacion=config.get("min_positivos_validacion", 10),
        estricto=config.get("particion_estricta", False),
    )
    # preparar_features() guarda el mapeo en features.ruta_mapeo: a un
    # directorio temporal, no sobre models/mapeo_woe.pkl.
    config_features_real = features._config_features()
    monkeypatch.setattr(
        features, "_config_features",
        lambda: {**config_features_real, "ruta_mapeo": str(tmp_path / "mapeo_woe.pkl")},
    )
    X_tr, mapeo = features.preparar_features(tr, target_col)
    X_va = features.transformar_a_woe(va, mapeo)
    modelo = train.entrenar_regresion_logistica(X_tr, tr[target_col])

    p = modelo.predict_proba(X_va)[:, 1]
    y = va[target_col].to_numpy()

    m = cal.metricas_calibracion(y, p)
    print("\n" + m.resumen())

    assert 0.0 <= m.brier <= 1.0
    assert m.n == len(y)

    c = cal.ajustar_prior(float(tr[target_col].mean()), float(y.mean()))
    assert cal.verificar_monotonia(p, c), "la correccion no debe reordenar"


# ===========================================================================
# Parche 2026-10-28 — intercepto por máxima verosimilitud, evaluación honesta
# y predicciones fuera de muestra de la validación cruzada
# ===========================================================================
def _ventana_real(semilla=2026):
    """Reproduce la ventana temporal del proyecto: 40 obs, 9 incumplimientos."""
    rng = np.random.default_rng(semilla)
    z = np.concatenate([rng.normal(0.9, 1.0, 9), rng.normal(0.0, 1.0, 31)])
    z = z + np.log(0.533 / 0.467)
    y = np.concatenate([np.ones(9, int), np.zeros(31, int)])
    grupos = rng.integers(0, 12, size=40)
    return y, 1 / (1 + np.exp(-z)), grupos


# ===========================================================================
# 9. El calibrador de intercepto coincide con la metrica que lo reporta
# ===========================================================================
def test_el_ajuste_mv_coincide_con_el_intercepto_reportado():
    """metricas_calibracion().intercepto ES la corrección que hay que aplicar.

    Es la relación que permitió detectar, sobre los datos reales, que el prior
    shift de -1,3689 solo cubría el 65,9 % del ajuste necesario de -2,0778.
    """
    y, p, _ = _ventana_real()
    m = cal.metricas_calibracion(y, p)
    c = cal.ajustar_intercepto_mv(y, p)
    assert abs(c.ajuste_logit - m.intercepto) < 1e-6, (
        f"ajuste={c.ajuste_logit} vs intercepto={m.intercepto}"
    )


def test_el_ajuste_mv_elimina_el_sesgo_de_prevalencia():
    y, p, _ = _ventana_real()
    c = cal.ajustar_intercepto_mv(y, p)
    m = cal.metricas_calibracion(y, c.transformar(p))
    assert abs(m.sesgo_prevalencia) < 1e-3, f"sesgo residual={m.sesgo_prevalencia}"


def test_el_ajuste_mv_corrige_mas_que_el_prior_shift():
    """Sobre los datos reales, el prior shift queda corto. Esta prueba lo fija."""
    y, p, _ = _ventana_real()
    prior = cal.ajustar_prior(0.533, float(y.mean()))
    mv = cal.ajustar_intercepto_mv(y, p)

    sesgo_prior = abs(cal.metricas_calibracion(y, prior.transformar(p)).sesgo_prevalencia)
    sesgo_mv = abs(cal.metricas_calibracion(y, mv.transformar(p)).sesgo_prevalencia)
    assert sesgo_mv < sesgo_prior, "el ajuste por máxima verosimilitud debe ser al menos tan bueno"


def test_el_ajuste_mv_es_monotono_y_no_toca_el_area_bajo_la_curva():
    y, p, _ = _ventana_real()
    c = cal.ajustar_intercepto_mv(y, p)
    assert cal.verificar_monotonia(p, c)
    assert abs(roc_auc_score(y, p) - roc_auc_score(y, c.transformar(p))) < 1e-9


# ===========================================================================
# 10. La evaluacion honesta detecta el optimismo
# ===========================================================================
def test_la_evaluacion_honesta_penaliza_al_calibrador_sobreajustado():
    """El isotónico luce bien dentro de muestra y mal fuera. Si la evaluación
    honesta no lo detectara, no serviría para elegir."""
    y, p, grupos = _ventana_real()

    dentro = cal.metricas_calibracion(y, cal.ajustar_isotonico(y, p).transformar(p))
    fuera = cal.evaluar_calibrador_honesto(y, p, cal.ajustar_isotonico, grupos=grupos)

    assert fuera.get("brier_skill") is not None
    assert fuera["brier_skill"] < dentro.brier_skill, (
        f"dentro={dentro.brier_skill:.4f} fuera={fuera['brier_skill']:.4f}: "
        "la evaluación fuera de muestra debería ser menor"
    )


def test_la_evaluacion_honesta_informa_cuando_no_puede_calcular():
    """Con una sola clase no puede evaluar; debe decirlo en lugar de fallar."""
    y = np.zeros(30, dtype=int)
    p = np.full(30, 0.3)
    res = cal.evaluar_calibrador_honesto(y, p, cal.ajustar_intercepto_mv)
    assert "error" in res or np.isnan(res.get("brier_skill", np.nan))


# ===========================================================================
# 11. La recomendacion aplica el criterio documentado
# ===========================================================================
def test_con_pocos_positivos_se_restringe_a_un_parametro():
    """9 incumplimientos no sostienen dos parámetros libres."""
    y, p, grupos = _ventana_real()
    elegido, tabla, notas = cal.recomendar_calibrador(
        y, p, prevalencia_entrenamiento=0.533, grupos=grupos
    )
    assert elegido.tipo in ("NINGUNO", "PRIOR", "INTERCEPTO_MV")
    assert any("un parámetro" in n for n in notas), notas


def test_con_muchos_positivos_se_admite_platt():
    """Con 105 incumplimientos, como en la validación cruzada, Platt entra."""
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-x)))
    p = 1 / (1 + np.exp(-(0.4 * x + 1.0)))   # descalibrado en pendiente e intercepto
    grupos = rng.integers(0, 40, size=n)

    _, tabla, notas = cal.recomendar_calibrador(y, p, grupos=grupos)
    assert "PLATT" in set(tabla["calibrador"])
    assert not any("un parámetro" in n for n in notas), notas


def test_la_tabla_comparativa_marca_la_monotonia_de_todos():
    y, p, grupos = _ventana_real()
    t = cal.comparar_calibradores(y, p, prevalencia_entrenamiento=0.533, grupos=grupos)
    assert t["monotono"].all(), "los calibradores del módulo deben ser monótonos"
    assert {"NINGUNO", "PRIOR", "INTERCEPTO_MV"} <= set(t["calibrador"])


# ===========================================================================
# 12. Predicciones fuera de muestra
# ===========================================================================
def test_predicciones_fuera_de_muestra_cubre_todas_las_filas():
    """Cada observación debe aparecer exactamente una vez, en una sola partición."""
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame({
        "ClienteId": rng.integers(0, 20, size=n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    df["EsMorosa"] = rng.binomial(1, 1 / (1 + np.exp(-df.x1)))

    # Dobles sencillos: el modulo no debe depender de features.py ni train.py
    def ajustar(tr, target, candidatas=None):
        return [c for c in (candidatas or tr.columns) if c not in (target, "ClienteId")]

    def aplicar(d, art):
        return d[art]

    def entrenar(X, yy):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000).fit(X, yy)

    oof = cal.predicciones_fuera_de_muestra(
        df, "EsMorosa", grupo_col="ClienteId", n_particiones=5,
        ajustar_fn=ajustar, aplicar_fn=aplicar, entrenar_fn=entrenar,
    )

    assert len(oof) == n
    assert oof["indice"].nunique() == n, "ninguna fila debe repetirse"
    assert oof["particion"].nunique() <= 5
    assert oof["p"].between(0, 1).all()

    # Ningún cliente puede aparecer en dos particiones
    por_cliente = oof.groupby("grupo")["particion"].nunique()
    assert (por_cliente == 1).all(), "la partición debe agrupar por cliente"
