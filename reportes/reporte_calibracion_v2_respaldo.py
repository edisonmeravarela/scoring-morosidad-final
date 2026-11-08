"""
reporte_calibracion.py
=======================
Script ejecutable que produce todo lo que la observacion 4 reclama. Version 2
(2026-10-28), reemplaza a la primera:

CAMBIOS RESPECTO DE LA PRIMERA VERSIÓN
--------------------------------------
La corrida real del 28/10 mostró dos límites de aquella:

  1. Aplicaba el prior shift sin comprobar si bastaba. Sobre los datos reales
     cubría solo el 65,9 % del ajuste necesario (-1,3689 de -2,0778), dejando un
     sesgo residual de +0,1064.

  2. Calibraba sobre la ventana temporal de 40 observaciones con 9
     incumplimientos, por debajo del mínimo de 10 que el propio pipeline exige.

Esta versión:

  - Calibra sobre las predicciones fuera de muestra de la validación cruzada
    agrupada por cliente: ~197 observaciones con ~105 incumplimientos.
  - Compara las cuatro opciones con evaluación FUERA de muestra, de modo que un
    calibrador de dos parámetros no parezca mejor de lo que es.
  - Elige con un criterio explícito y lo documenta en el informe.
  - Reporta ambos conjuntos, porque la ventana temporal sigue siendo la que
    describe el período reciente.

USO
---
    python reporte_calibracion.py                          # informe completo
    python reporte_calibracion.py --conjunto ventana       # solo la ventana temporal
    python reporte_calibracion.py --persistir --evaluacion-id <real>
    python reporte_calibracion.py --aplicar

ADAPTADO (2026-10-28) — mismos dos puntos que la primera versión:
  - `train.dividir_temporal_por_cliente` existe pero `entrenar()` ya no lo usa
    (su propio docstring dice "se deja sin usar, solo para comparar"); el
    split real es `ajustes_evaluacion.dividir_particion_agrupada`, con los
    mismos parámetros que `entrenar()` le pasa de verdad — se reproduce aquí
    igual que en la v1.
  - No existe `train.TARGET_COL`; el nombre real de la columna objetivo sale
    de `train._config_entrenamiento()["columna_target"]` ("Target" por
    defecto) — la misma función privada que tests/test_train.py ya
    monkeypatchea directamente.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src import calibracion as cal            # noqa: E402
from src import db, extract, features, train  # noqa: E402
from src.ajustes_evaluacion import dividir_particion_agrupada  # noqa: E402


# ---------------------------------------------------------------------------
def _config_y_target() -> tuple[dict, str]:
    config = train._config_entrenamiento()
    target_col = config.get("columna_target", "Target")
    return config, target_col


# ---------------------------------------------------------------------------
# Obtención de los dos conjuntos
# ---------------------------------------------------------------------------
def conjunto_ventana_temporal():
    """Corte temporal único, el mismo split que usa entrenar() de verdad
    (ajustes_evaluacion.dividir_particion_agrupada, no
    train.dividir_temporal_por_cliente)."""
    config, target_col = _config_y_target()
    datos = extract.excluir_sin_historial(extract.extraer_dataset_entrenamiento())

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

    X_tr, mapeo = features.preparar_features(tr, target_col)
    X_va = features.transformar_a_woe(va, mapeo)
    modelo = train.entrenar_regresion_logistica(X_tr, tr[target_col])

    p = modelo.predict_proba(X_va)[:, 1]
    y = va[target_col].to_numpy()
    grupos = va["ClienteId"].to_numpy() if "ClienteId" in va.columns else va.index.to_numpy()
    prev_entrenamiento = float(tr[target_col].mean())
    return y, p, grupos, prev_entrenamiento


def conjunto_validacion_cruzada(n_particiones: int = 5):
    """Predicciones fuera de muestra de la validación cruzada agrupada.

    Es el conjunto grande: muchas más observaciones e incumplimientos que la
    ventana temporal. Calibrar aquí es sustancialmente más robusto.
    """
    _, target_col = _config_y_target()
    datos = extract.excluir_sin_historial(extract.extraer_dataset_entrenamiento())
    oof = cal.predicciones_fuera_de_muestra(
        datos, target_col, grupo_col="ClienteId", n_particiones=n_particiones
    )
    prev = float(datos[target_col].mean())
    return oof["y"].to_numpy(), oof["p"].to_numpy(), oof["grupo"].to_numpy(), prev, oof


# ---------------------------------------------------------------------------
def informe_conjunto(nombre, y, p, grupos, prev_entrenamiento, salida: Path,
                     tramos: int, cuantiles: bool) -> dict:
    """Produce el informe completo de un conjunto y devuelve el resumen."""
    print(f"\n{'=' * 74}\n{nombre.upper()}\n{'=' * 74}")
    print(f"observaciones={len(y)}  incumplimientos={int(y.sum())}  "
          f"prevalencia={y.mean():.4f}  clientes={len(np.unique(grupos))}")

    m0 = cal.metricas_calibracion(y, p, n_tramos=tramos)
    print("\nSin calibrar:")
    print("  " + m0.resumen().replace("\n", "\n  "))
    for linea in m0.interpretacion():
        print("  - " + linea)

    print("\nComparación de calibradores "
          "(la columna decisiva es brier_skill_oof, fuera de muestra):")
    elegido, tabla, notas = cal.recomendar_calibrador(
        y, p, prevalencia_entrenamiento=prev_entrenamiento, grupos=grupos
    )
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nCriterio de selección:")
    for n in notas:
        print("  * " + n)

    if not cal.verificar_monotonia(p, elegido):
        print("\nERROR: el calibrador elegido altera el ordenamiento. No se aplica.")
        return {}

    p_cal = np.asarray(elegido.transformar(p)).ravel()
    m1 = cal.metricas_calibracion(y, p_cal, n_tramos=tramos, calibrador=elegido.tipo)

    print(f"\nCalibrado con {elegido.tipo}:")
    print("  " + m1.resumen().replace("\n", "\n  "))

    comp = cal.comparar_antes_despues(y, p, elegido, n_tramos=tramos)
    print("\nAntes y después:")
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    t0 = cal.tabla_fiabilidad(y, p, n_tramos=tramos, por_cuantiles=cuantiles)
    t1 = cal.tabla_fiabilidad(y, p_cal, n_tramos=tramos, por_cuantiles=cuantiles)

    pref = nombre.lower().replace(" ", "_")
    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / f"{pref}_calibradores.csv", index=False)
    comp.to_csv(salida / f"{pref}_comparativa.csv", index=False)
    t0.to_csv(salida / f"{pref}_fiabilidad_sin_calibrar.csv", index=False)
    t1.to_csv(salida / f"{pref}_fiabilidad_calibrado.csv", index=False)

    png = dibujar_diagrama(t0, t1, salida / f"{pref}_diagrama_fiabilidad.png",
                           f"Diagrama de fiabilidad — {nombre}")
    if png:
        print(f"\nDiagrama: {png}")

    return {"nombre": nombre, "antes": m0, "despues": m1,
            "calibrador": elegido, "notas": notas, "tabla": tabla,
            "fiabilidad_antes": t0, "fiabilidad_despues": t1}


# ---------------------------------------------------------------------------
def dibujar_diagrama(t0, t1, ruta: Path, titulo: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no disponible: se omite el diagrama.")
        return None

    fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=150)
    ax.plot([0, 1], [0, 1], "--", color="#888780", linewidth=1, label="Calibración perfecta")
    for t, color, etiqueta, marcador in (
        (t0, "#D85A30", "Sin calibrar", "o"),
        (t1, "#1D9E75", "Calibrado", "s"),
    ):
        con = t[t["casos"] > 0]
        ax.plot(con["prob_media"], con["frec_observada"], marcador + "-",
                color=color, linewidth=1.6, markersize=6, label=etiqueta)

    ax.set_xlabel("Probabilidad declarada por el modelo")
    ax.set_ylabel("Frecuencia de incumplimiento observada")
    ax.set_title(titulo, fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_aspect("equal")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
def parrafo_para_tesis(res: dict) -> str:
    a, d, c = res["antes"], res["despues"], res["calibrador"]
    ajuste = getattr(c, "ajuste_logit", None)
    # CalibradorPrior y CalibradorIntercepto comparten el atributo
    # ajuste_logit pero NO el método: Prior es un desplazamiento en forma
    # cerrada a partir de las dos prevalencias (no es un ajuste por máxima
    # verosimilitud); Intercepto_MV sí lo es (Newton-Raphson sobre la
    # verosimilitud, ver _intercepto_con_offset). Describirlos igual sería
    # inexacto en el texto de la tesis.
    if c.tipo == "INTERCEPTO_MV":
        texto_ajuste = (f"mediante un desplazamiento del intercepto de {ajuste:.4f} en escala "
                        f"logarítmica, ajustado por máxima verosimilitud")
    elif c.tipo == "PRIOR":
        texto_ajuste = (f"por cambio de prevalencia, con un ajuste de {ajuste:.4f} "
                        f"en escala logarítmica")
    elif ajuste is not None:
        texto_ajuste = f"mediante un desplazamiento del intercepto de {ajuste:.4f} en escala logarítmica"
    else:
        texto_ajuste = f"mediante el procedimiento {c.tipo}"

    veredicto = (
        "Las probabilidades resultantes quedan en condiciones de sustentar decisiones "
        "que ponderen costos."
        if d.brier_skill > 0 else
        "La puntuación de Brier no supera aún la de predecir siempre la prevalencia, "
        "de modo que las probabilidades sirven para ordenar a los clientes pero no "
        "para decidir mediante umbrales de costo, lo que es coherente con el volumen "
        "de datos disponible."
    )

    return (
        f"Sobre el conjunto de {res['nombre'].lower()}, de {a.n} observaciones con una "
        f"frecuencia de incumplimiento del {a.prevalencia_observada:.1%}, el modelo declara "
        f"una probabilidad media del {a.prevalencia_predicha:.1%}, lo que supone una "
        f"sobreestimación de {a.sesgo_prevalencia:+.1%}. La puntuación de Brier asciende a "
        f"{a.brier:.4f}, la pendiente de calibración a {a.pendiente:.3f} y el intercepto a "
        f"{a.intercepto:+.3f}, frente a los valores de referencia de uno y cero "
        f"respectivamente, con un error de calibración esperado de {a.ece:.4f}. "
        f"Tras la corrección {texto_ajuste}, la probabilidad media declarada se sitúa en "
        f"{d.prevalencia_predicha:.1%}, la puntuación de Brier en {d.brier:.4f} y el error "
        f"de calibración esperado en {d.ece:.4f}. La corrección es monótona y, por tanto, no "
        f"altera el ordenamiento de los clientes ni los valores del área bajo la curva y del "
        f"estadístico de Kolmogorov-Smirnov ya reportados. {veredicto}"
    )


# ---------------------------------------------------------------------------
def persistir(evaluacion_id: int, origen: str, m, tabla: pd.DataFrame) -> None:
    db.exec_sp("dbo.sp_RegistrarCalibracion", {
        "EvaluacionId": evaluacion_id, "Origen": origen,
        "Brier": round(m.brier, 6), "BrierSkill": round(m.brier_skill, 6),
        "PendienteCalibracion": round(m.pendiente, 6),
        "InterceptoCalibracion": round(m.intercepto, 6),
        "ECE": round(m.ece, 6), "MCE": round(m.mce, 6),
        "HosmerLemeshowChi2": None if m.hl_chi2 is None else round(m.hl_chi2, 6),
        "HosmerLemeshowP": None if m.hl_p is None else round(m.hl_p, 6),
        "PrevalenciaObservada": round(m.prevalencia_observada, 6),
        "PrevalenciaPredicha": round(m.prevalencia_predicha, 6),
        "CalibradorAplicado": m.calibrador,
    })
    filas = [
        (evaluacion_id, origen, m.calibrador, int(r.tramo),
         float(r.limite_inferior), float(r.limite_superior),
         int(r.casos), int(r.positivos), float(r.prob_media), float(r.frec_observada))
        for r in tabla.itertuples()
    ]
    db.ejecutar_muchos(
        """INSERT INTO dbo.CalibracionTramo
           (EvaluacionId, Origen, CalibradorAplicado, Tramo, LimiteInferior,
            LimiteSuperior, Casos, Positivos, ProbMedia, FrecObservada)
           VALUES (?,?,?,?,?,?,?,?,?,?)""", filas)
    print(f"Persistido: EvaluacionId={evaluacion_id} Origen={origen} ({len(filas)} tramos).")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Reporte de calibración probabilística")
    ap.add_argument("--conjunto", choices=["ambos", "ventana", "crossval"], default="ambos")
    ap.add_argument("--tramos", type=int, default=5,
                    help="5 por defecto: con pocas observaciones, 10 tramos dejan "
                         "celdas de uno o dos casos y el MCE deja de ser informativo")
    ap.add_argument("--cuantiles", action="store_true", default=True)
    ap.add_argument("--particiones", type=int, default=5)
    ap.add_argument("--persistir", action="store_true")
    ap.add_argument("--evaluacion-id", type=int, default=None)
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--salida", default="reportes")
    args = ap.parse_args()

    salida = Path(args.salida)
    resultados = {}

    if args.conjunto in ("ambos", "crossval"):
        print("Reconstruyendo las predicciones fuera de muestra de la validación cruzada...")
        y, p, g, prev, _ = conjunto_validacion_cruzada(args.particiones)
        resultados["crossval"] = informe_conjunto(
            "Validación cruzada", y, p, g, prev, salida, args.tramos, args.cuantiles)

    if args.conjunto in ("ambos", "ventana"):
        print("\nReconstruyendo las predicciones de la ventana temporal...")
        y, p, g, prev = conjunto_ventana_temporal()
        resultados["ventana"] = informe_conjunto(
            "Ventana temporal", y, p, g, prev, salida, args.tramos, args.cuantiles)

    # --- Recomendación final -------------------------------------------------
    print(f"\n{'=' * 74}\nRECOMENDACIÓN\n{'=' * 74}")
    principal = resultados.get("crossval") or resultados.get("ventana")
    if not principal:
        print("Sin resultados.")
        return 1

    c = principal["calibrador"]
    print(f"Calibrador: {c.tipo}  (ajustado sobre {principal['nombre'].lower()})")
    if hasattr(c, "ajuste_logit"):
        print(f"Ajuste del intercepto: {c.ajuste_logit:+.4f} en escala logarítmica")

    if "crossval" in resultados and "ventana" in resultados:
        print("\nSe recomienda ajustar sobre la validación cruzada "
              f"({resultados['crossval']['antes'].n} observaciones, "
              f"{int(resultados['crossval']['antes'].prevalencia_observada * resultados['crossval']['antes'].n)} "
              f"incumplimientos) y no sobre la ventana temporal "
              f"({resultados['ventana']['antes'].n} observaciones, "
              f"{int(resultados['ventana']['antes'].prevalencia_observada * resultados['ventana']['antes'].n)} "
              "incumplimientos).")

    texto = parrafo_para_tesis(principal)
    (salida / "parrafo_tesis.txt").write_text(texto, encoding="utf-8")
    print(f"\n{'=' * 74}\nPÁRRAFO PARA LA SECCIÓN 7.3\n{'=' * 74}")
    print(texto)

    # --- Salidas opcionales --------------------------------------------------
    if args.persistir:
        if args.evaluacion_id is None:
            print("\n--persistir requiere --evaluacion-id")
            return 1
        for clave, origen in (("ventana", "VALIDACION"), ("crossval", "CROSSVAL")):
            if clave in resultados:
                r = resultados[clave]
                persistir(args.evaluacion_id, origen, r["antes"], r["fiabilidad_antes"])
                persistir(args.evaluacion_id, origen, r["despues"], r["fiabilidad_despues"])

    if args.aplicar:
        ruta = cal.guardar_calibrador(c, Path("models") / "calibrador.pkl")
        print(f"\nCalibrador guardado en {ruta}")
        print("score.py lo aplicará en la próxima ejecución.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
