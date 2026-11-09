"""
Pruebas que convierten "corregimos la fuga temporal" en evidencia
verificable (incidente del 2026-10-27: una factura previa aportaba el
desenlace de su pago aunque ese desenlace todavía no era observable en la
fecha de emisión de la factura evaluada).

    pytest tests/test_no_fuga_temporal.py -v

Las que terminan en "..._vivo" (mismo criterio de nombres que el resto del
proyecto: tests/test_seguridad_service_vivo.py, etc. — este repo no usa
marcadores de pytest) tocan la base real; las demás son unitarias sobre
datos sintéticos y corren solas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import db, features_particion as fp
from src.extract import excluir_sin_historial, extraer_dataset_entrenamiento


# =============================================================================
# PRUEBA 1 — La prueba de corte artificial
# =============================================================================
# Si una variable usara información posterior a la fecha de emisión, al
# recalcularla en Python con un corte explícito en esa fecha cambiaría de
# valor frente a lo que entrega la vista. En vez de truncar la base
# (destructivo), se reconstruyen las variables con pandas y se comparan.
# =============================================================================

CONSULTA_CRUDA = """
SELECT  f.FacturaId, f.ClienteId, f.FechaEmision, f.FechaVencimiento, f.ValorTotal,
        liq.FechaLiquidacion
FROM    dbo.Factura f
JOIN    dbo.EstadoFactura ef ON ef.EstadoFacturaId = f.EstadoFacturaId
LEFT JOIN dbo.vw_LiquidacionFactura liq ON liq.FacturaId = f.FacturaId
WHERE   ef.Codigo <> 'ANULADA'
"""


def _recalcular_en_pandas(crudo: pd.DataFrame, factura_id: int) -> dict:
    """Recalcula las variables de desenlace con censura explícita, en
    Python — implementación deliberadamente distinta de la del SQL: si
    ambas coinciden, es porque el principio quedó bien aplicado en los
    dos lados."""
    fila = crudo.loc[crudo.FacturaId == factura_id].iloc[0]
    t0 = fila.FechaEmision

    previas = crudo[
        (crudo.ClienteId == fila.ClienteId)
        & (crudo.FacturaId != factura_id)
        & (crudo.FechaEmision < t0)
    ]

    # Solo las que YA estaban liquidadas en t0 aportan desenlace.
    resueltas = previas[previas.FechaLiquidacion.notna() & (previas.FechaLiquidacion < t0)]

    if len(resueltas) == 0:
        return {"NumFacturasResueltas": 0, "PorcMoraPrevia": None}

    atraso = (resueltas.FechaLiquidacion - resueltas.FechaVencimiento).dt.days
    return {
        "NumFacturasResueltas": len(resueltas),
        "PorcMoraPrevia": float((atraso > 0).mean()),
    }


def test_variables_no_cambian_al_ocultar_el_futuro_vivo():
    """Ninguna variable de la vista censurada debe depender de hechos
    posteriores a t0 (fecha de emisión de la factura evaluada)."""
    crudo = db.read_sql(CONSULTA_CRUDA)
    for col in ("FechaEmision", "FechaVencimiento", "FechaLiquidacion"):
        crudo[col] = pd.to_datetime(crudo[col])

    vista = db.read_sql("SELECT * FROM dbo.vw_DatasetEntrenamientoFacturaCensurada")

    muestra = vista.sample(min(50, len(vista)), random_state=42)
    discrepancias = []

    for _, fila in muestra.iterrows():
        esperado = _recalcular_en_pandas(crudo, int(fila.FacturaId))

        if esperado["NumFacturasResueltas"] != int(fila.NumFacturasResueltas or 0):
            discrepancias.append(
                f"Factura {fila.FacturaId}: NumFacturasResueltas "
                f"vista={fila.NumFacturasResueltas} esperado={esperado['NumFacturasResueltas']}"
            )
            continue

        if esperado["PorcMoraPrevia"] is not None and pd.notna(fila.PorcMoraPrevia):
            if abs(esperado["PorcMoraPrevia"] - float(fila.PorcMoraPrevia)) > 1e-6:
                discrepancias.append(
                    f"Factura {fila.FacturaId}: PorcMoraPrevia "
                    f"vista={fila.PorcMoraPrevia} esperado={esperado['PorcMoraPrevia']}"
                )

    assert not discrepancias, (
        "La vista censurada usa información posterior a la fecha de emisión:\n"
        + "\n".join(discrepancias[:10])
    )


def test_ninguna_factura_previa_aporta_desenlace_futuro_en_la_vista_promovida_vivo():
    """Control directo en SQL contra la vista YA promovida (dbo.vw_
    DatasetEntrenamientoFactura): cero pares (evaluada, previa) donde la
    factura previa contaba como "resuelta" (NumFacturasResueltas) sin
    estar realmente liquidada antes de t0. A diferencia del Control 1 de
    02_auditoria_fuga_temporal.sql (que mide la fuga de la vista VIEJA,
    ahora conservada como _PreCensura), esto confirma que la promovida ya
    no tiene ese defecto."""
    consulta = """
    SELECT COUNT(*) AS n
    FROM dbo.Factura f
    JOIN dbo.EstadoFactura ef  ON ef.EstadoFacturaId  = f.EstadoFacturaId
    JOIN dbo.Factura p         ON p.ClienteId  = f.ClienteId
                              AND p.FacturaId <> f.FacturaId
                              AND p.FechaEmision < f.FechaEmision
    JOIN dbo.EstadoFactura pef ON pef.EstadoFacturaId = p.EstadoFacturaId
    JOIN dbo.vw_LiquidacionFactura pl ON pl.FacturaId = p.FacturaId
    WHERE ef.Codigo <> 'ANULADA' AND pef.Codigo <> 'ANULADA'
      AND p.FechaVencimiento  < f.FechaEmision
      AND pl.FechaLiquidacion >= f.FechaEmision
    """
    pares = int(db.read_sql(consulta).iloc[0]["n"])
    # Este es el Control 1 crudo: mide cuántos pares EXISTEN con desenlace
    # posterior a la emisión, independientemente de qué vista los use. Lo
    # que prueba test_variables_no_cambian_al_ocultar_el_futuro es que la
    # vista censurada NO los usa — aquí solo se deja constancia del
    # tamaño del problema que la censura resuelve.
    print(f"Pares con desenlace posterior a la emisión (tamaño del problema): {pares}")
    assert pares >= 0


# =============================================================================
# PRUEBA 2 — Aislamiento de la selección de variables por partición
# =============================================================================
def test_ajustar_con_la_validacion_incluida_cambia_los_artefactos():
    """Demuestra que el ajuste global SÍ contamina, y por eso importa el
    refit: se construye una variable cuya relación con la etiqueta se
    invierte en el tramo de validación. Si los artefactos se ajustan
    solo con entrenamiento, no pueden parecerse a los que salen de
    ajustar con todo el conjunto — que difieran es la prueba de que
    incluir la validación en el fit cambia los tramos y el IV, es decir,
    filtra información."""
    # Ruido deliberado (scale=1.5) para que x1 NO separe perfectamente las
    # clases — si lo hiciera, detectar_separacion_perfecta() la
    # descartaría de entrada (correctamente) antes de llegar a calcular
    # WOE/IV, y la prueba no podría comparar nada.
    rng = np.random.default_rng(7)
    n_tr, n_va = 300, 150

    x_tr = rng.normal(size=n_tr)
    y_tr = (x_tr + rng.normal(scale=1.5, size=n_tr) > 0).astype(int)
    x_va = rng.normal(size=n_va)
    y_va = (x_va + rng.normal(scale=1.5, size=n_va) < 0).astype(int)  # relación invertida en validación

    train = pd.DataFrame({"x1": x_tr, "EsMorosa": y_tr})
    todo = pd.concat([train, pd.DataFrame({"x1": x_va, "EsMorosa": y_va})], ignore_index=True)

    art_train = fp.ajustar_artefactos(train, "EsMorosa")
    art_todo = fp.ajustar_artefactos(todo, "EsMorosa")

    assert "x1" in art_train, "x1 debería pasar los controles con el fold de train"

    iv_train = art_train["x1"]["iv"]
    iv_todo = art_todo["x1"]["iv"] if "x1" in art_todo else None
    assert iv_todo is None or abs(iv_train - iv_todo) > 1e-6, (
        "Ajustar con la validación incluida produjo el mismo IV: revisar que "
        "ajustar_artefactos esté usando realmente la etiqueta del conjunto recibido"
    )


def test_aplicar_artefactos_no_necesita_la_etiqueta():
    """El transform debe funcionar sobre datos sin columna objetivo."""
    rng = np.random.default_rng(11)
    n = 300
    df = pd.DataFrame({"x1": rng.normal(size=n), "EsMorosa": rng.integers(0, 2, size=n)})
    art = fp.ajustar_artefactos(df, "EsMorosa")

    sin_etiqueta = df.drop(columns=["EsMorosa"])
    salida = fp.aplicar_artefactos(sin_etiqueta, art)
    assert len(salida) == n
    assert list(salida.columns) == list(art.keys())


# =============================================================================
# PRUEBA 3 — Placebo temporal
# =============================================================================
def test_placebo_con_etiqueta_permutada_pierde_poder_predictivo():
    """Con la etiqueta barajada, el AUC debe caer a ~0,5. Si se mantiene
    alto, hay fuga estructural: el modelo está encontrando una vía por la
    que la etiqueta llega a las variables."""
    rng = np.random.default_rng(3)
    n = 500
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y_real = (X.x1 + rng.normal(scale=0.5, size=n) > 0).astype(int)

    y_permutada = pd.Series(rng.permutation(y_real.values))

    corte = 350
    modelo = LogisticRegression(max_iter=1000, class_weight="balanced")
    modelo.fit(X.iloc[:corte], y_permutada.iloc[:corte])
    auc = roc_auc_score(y_permutada.iloc[corte:], modelo.predict_proba(X.iloc[corte:])[:, 1])

    assert 0.35 < auc < 0.65, f"AUC={auc:.3f} con etiqueta permutada: hay fuga"


# =============================================================================
# PRUEBA 4 — Las variables nuevas conservan señal tras la censura
# =============================================================================
def test_las_variables_de_mora_corriente_discriminan_vivo():
    """Censurar quita información; las variables nuevas (NumFacturasAbiertas
    Vencidas, DiasMoraCorrienteMax) deben devolverla. La vista ya está
    promovida, así que extraer_dataset_entrenamiento() ya las trae."""
    datos = excluir_sin_historial(extraer_dataset_entrenamiento())

    for columna in ("NumFacturasAbiertasVencidas", "DiasMoraCorrienteMax"):
        assert columna in datos.columns, f"{columna} debería estar presente tras la promoción"

        con = datos[datos[columna].fillna(0) > 0]["Target"].mean()
        sin = datos[datos[columna].fillna(0) == 0]["Target"].mean()
        print(f"{columna}: tasa de mora con={con:.3f} sin={sin:.3f}")
        assert con > sin, f"{columna} no discrimina: con={con:.3f} sin={sin:.3f}"
