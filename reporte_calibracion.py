"""
10_reporte_calibracion_v3.py
============================
Destino: scoring-morosidad/reporte_calibracion.py  (reemplaza a la versión 2)

CAMBIOS RESPECTO DE LA VERSIÓN 2
--------------------------------
La corrida del 28/10 dejó dos defectos verificables en la base:

  EvaluacionId=105  VALIDACION  NINGUNO   5 tramos    (correcto)
  EvaluacionId=105  VALIDACION  PRIOR     5 tramos    (correcto)
  EvaluacionId=105  CROSSVAL    NINGUNO  10 tramos    (DUPLICADO)

El duplicado se produce cuando el calibrador elegido es NINGUNO: el script
persistía el estado "antes" y el "después" con la misma etiqueta. Y en
dbo.EvaluacionMetrica la segunda escritura pisaba la primera, de modo que se
perdía el "antes".

Esta versión:

  - Persiste en dbo.CalibracionModelo, con clave que incluye el calibrador.
  - No escribe dos veces cuando el calibrador seleccionado es NINGUNO.
  - Limpia los tramos previos del mismo trío antes de insertar.
  - Marca cuál es el calibrador seleccionado (EsSeleccionado).
  - Advierte cuando la pendiente de calibración está lejos de 1, que es lo que
    los datos reales revelaron: 0,1354 en la ventana temporal y 0,5166 en la
    validación cruzada.
  - Permite forzar un calibrador con --calibrador.

Requiere haber ejecutado 09_calibracion_modelo.sql (ya ejecutado en base).

ADAPTADO (2026-10-28) — el borrador pegado traía de vuelta los dos mismos
puntos ya corregidos en la v2, porque se redactó sobre una copia anterior a
esa corrección:
  - `train.dividir_temporal_por_cliente` existe pero `entrenar()` ya no lo usa
    (su propio docstring dice "se deja sin usar, solo para comparar"); el
    split real es `ajustes_evaluacion.dividir_particion_agrupada`, con los
    mismos parámetros que `entrenar()` le pasa de verdad.
  - No existe `train.TARGET_COL`; el nombre real de la columna objetivo sale
    de `train._config_entrenamiento()["columna_target"]` ("Target" por
    defecto) — la misma función privada que tests/test_train.py ya
    monkeypatchea directamente.
  Se reaplica aquí la misma corrección ya probada en reporte_calibracion.py
  (v2): `_config_y_target()` + `dividir_particion_agrupada(...)`. El resto
  del script (persistencia en CalibracionModelo, aviso de pendiente,
  --calibrador) es nuevo de esta versión y no tenía ese defecto.

USO
---
    python reporte_calibracion.py
    python reporte_calibracion.py --persistir --evaluacion-id 105
    python reporte_calibracion.py --calibrador PLATT --conjunto crossval
    python reporte_calibracion.py --persistir --evaluacion-id 105 --aplicar
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

# Rango dentro del cual la pendiente se considera aceptable
PENDIENTE_MIN, PENDIENTE_MAX = 0.80, 1.25


# ---------------------------------------------------------------------------
def _config_y_target() -> tuple[dict, str]:
    config = train._config_entrenamiento()
    target_col = config.get("columna_target", "Target")
    return config, target_col


# ---------------------------------------------------------------------------
# Conjuntos
# ---------------------------------------------------------------------------
def conjunto_ventana_temporal():
    """Mismo split que usa entrenar() de verdad: dividir_particion_agrupada,
    no train.dividir_temporal_por_cliente (ver nota ADAPTADO más arriba)."""
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
    return y, p, grupos, float(tr[target_col].mean())


def conjunto_validacion_cruzada(n_particiones: int = 5):
    _, target_col = _config_y_target()
    datos = extract.excluir_sin_historial(extract.extraer_dataset_entrenamiento())
    oof = cal.predicciones_fuera_de_muestra(
        datos, target_col, grupo_col="ClienteId", n_particiones=n_particiones)
    return (oof["y"].to_numpy(), oof["p"].to_numpy(), oof["grupo"].to_numpy(),
            float(datos[target_col].mean()))


# ---------------------------------------------------------------------------
def construir_calibrador(nombre: str, y, p, prev_entrenamiento: float | None):
    """Instancia el calibrador indicado por nombre."""
    prev_obs = float(np.mean(y))
    if nombre == "NINGUNO":
        return cal._CalibradorIdentidad()
    if nombre == "PRIOR":
        if prev_entrenamiento is None:
            raise ValueError("PRIOR requiere la prevalencia de entrenamiento")
        return cal.ajustar_prior(prev_entrenamiento, prev_obs)
    if nombre == "INTERCEPTO_MV":
        return cal.ajustar_intercepto_mv(y, p)
    if nombre == "PLATT":
        return cal.ajustar_platt(y, p)
    if nombre == "ISOTONICO":
        return cal.ajustar_isotonico(y, p)
    raise ValueError(f"Calibrador desconocido: {nombre}")


def informe_conjunto(nombre, y, p, grupos, prev_entrenamiento, salida: Path,
                     tramos: int, cuantiles: bool, forzado: str | None) -> dict:
    print(f"\n{'=' * 76}\n{nombre.upper()}\n{'=' * 76}")
    n_pos = int(y.sum())
    print(f"observaciones={len(y)}  incumplimientos={n_pos}  "
          f"prevalencia={y.mean():.4f}  clientes={len(np.unique(grupos))}")

    m0 = cal.metricas_calibracion(y, p, n_tramos=tramos)
    print("\nSin calibrar:")
    print("  " + m0.resumen().replace("\n", "\n  "))
    for linea in m0.interpretacion():
        print("  - " + linea)

    # --- Advertencia de pendiente -------------------------------------------
    if not (PENDIENTE_MIN <= m0.pendiente <= PENDIENTE_MAX):
        factor = 1 / m0.pendiente if m0.pendiente > 0 else float("inf")
        print(f"\n  AVISO: la pendiente de {m0.pendiente:.4f} queda fuera del rango "
              f"[{PENDIENTE_MIN}, {PENDIENTE_MAX}].")
        print(f"  El logit declarado se mueve {factor:.1f} veces mas que el real.")
        print("  Corregir solo el intercepto no resuelve esto: hacen falta PLATT o")
        print("  ISOTONICO, que ajustan tambien la pendiente. Con pocos incumplimientos")
        print("  esos calibradores sobreajustan, de modo que conviene calibrar sobre el")
        print("  conjunto de validacion cruzada.")

    # --- Comparación y selección --------------------------------------------
    print("\nComparacion de calibradores (decide la columna brier_skill_oof):")
    elegido, tabla, notas = cal.recomendar_calibrador(
        y, p, prevalencia_entrenamiento=prev_entrenamiento, grupos=grupos)
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if forzado:
        print(f"\nCalibrador forzado por parametro: {forzado}")
        elegido = construir_calibrador(forzado, y, p, prev_entrenamiento)
        notas = [f"Calibrador impuesto manualmente: {forzado}."]
    else:
        print("\nCriterio de seleccion:")
        for n in notas:
            print("  * " + n)

    if not cal.verificar_monotonia(p, elegido):
        print("\nERROR: el calibrador altera el ordenamiento. Se aborta.")
        return {}

    p_cal = np.asarray(elegido.transformar(p)).ravel()
    m1 = cal.metricas_calibracion(y, p_cal, n_tramos=tramos, calibrador=elegido.tipo)

    if elegido.tipo != "NINGUNO":
        print(f"\nCalibrado con {elegido.tipo}:")
        print("  " + m1.resumen().replace("\n", "\n  "))
        print("\nAntes y despues:")
        print(cal.comparar_antes_despues(y, p, elegido, n_tramos=tramos)
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        print("\nNo se aplica correccion: ningun calibrador supera al estado actual "
              "fuera de muestra.")

    t0 = cal.tabla_fiabilidad(y, p, n_tramos=tramos, por_cuantiles=cuantiles)
    t1 = cal.tabla_fiabilidad(y, p_cal, n_tramos=tramos, por_cuantiles=cuantiles)

    pref = nombre.lower().replace(" ", "_")
    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / f"{pref}_calibradores.csv", index=False)
    t0.to_csv(salida / f"{pref}_fiabilidad_sin_calibrar.csv", index=False)
    if elegido.tipo != "NINGUNO":
        t1.to_csv(salida / f"{pref}_fiabilidad_calibrado.csv", index=False)

    png = dibujar_diagrama(t0, t1 if elegido.tipo != "NINGUNO" else None,
                           salida / f"{pref}_diagrama_fiabilidad.png",
                           f"Diagrama de fiabilidad — {nombre}")
    if png:
        print(f"\nDiagrama: {png}")

    return {"nombre": nombre, "antes": m0, "despues": m1, "calibrador": elegido,
            "notas": notas, "tabla": tabla, "fiabilidad_antes": t0,
            "fiabilidad_despues": t1, "n": len(y), "n_pos": n_pos}


# ---------------------------------------------------------------------------
def dibujar_diagrama(t0, t1, ruta: Path, titulo: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=150)
    ax.plot([0, 1], [0, 1], "--", color="#888780", linewidth=1, label="Calibración perfecta")
    series = [(t0, "#D85A30", "Sin calibrar", "o")]
    if t1 is not None:
        series.append((t1, "#1D9E75", "Calibrado", "s"))
    for t, color, etiqueta, marcador in series:
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
# Persistencia corregida
# ---------------------------------------------------------------------------
def persistir_estado(evaluacion_id: int, origen: str, m, tabla: pd.DataFrame,
                     n: int, n_pos: int, ajuste_logit: float | None,
                     seleccionado: bool) -> None:
    """Escribe un estado (un calibrador) en dbo.CalibracionModelo y sus tramos."""
    db.exec_sp("dbo.sp_RegistrarCalibracionModelo", {
        "EvaluacionId": evaluacion_id,
        "Origen": origen,
        "CalibradorAplicado": m.calibrador,
        "Observaciones": int(n),
        "Incumplimientos": int(n_pos),
        "Brier": round(m.brier, 6),
        "BrierSkill": round(m.brier_skill, 6),
        "PendienteCalibracion": round(m.pendiente, 6),
        "InterceptoCalibracion": round(m.intercepto, 6),
        "ECE": round(m.ece, 6),
        "MCE": round(m.mce, 6),
        "HosmerLemeshowChi2": None if m.hl_chi2 is None else round(m.hl_chi2, 6),
        "HosmerLemeshowP": None if m.hl_p is None else round(m.hl_p, 6),
        "PrevalenciaObservada": round(m.prevalencia_observada, 6),
        "PrevalenciaPredicha": round(m.prevalencia_predicha, 6),
        "AjusteLogit": None if ajuste_logit is None else round(float(ajuste_logit), 6),
        "EsSeleccionado": 1 if seleccionado else 0,
    })

    # Los tramos se reemplazan, nunca se acumulan
    db.exec_sp("dbo.sp_LimpiarTramosCalibracion", {
        "EvaluacionId": evaluacion_id, "Origen": origen,
        "CalibradorAplicado": m.calibrador})

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
    print(f"  {origen:<11} {m.calibrador:<14} {len(filas)} tramos"
          f"{'  [seleccionado]' if seleccionado else ''}")


def persistir_conjunto(evaluacion_id: int, origen: str, res: dict) -> None:
    """Escribe el estado sin calibrar y, si difiere, el calibrado.

    Es el punto que corrige el duplicado: cuando el calibrador seleccionado es
    NINGUNO, ambos estados son el mismo y se escribe una sola vez.
    """
    c = res["calibrador"]
    sin_correccion = c.tipo == "NINGUNO"

    persistir_estado(evaluacion_id, origen, res["antes"], res["fiabilidad_antes"],
                     res["n"], res["n_pos"], None, seleccionado=sin_correccion)

    if not sin_correccion:
        persistir_estado(evaluacion_id, origen, res["despues"], res["fiabilidad_despues"],
                         res["n"], res["n_pos"],
                         getattr(c, "ajuste_logit", None), seleccionado=True)


# ---------------------------------------------------------------------------
def parrafo_para_tesis(res: dict) -> str:
    a, d, c = res["antes"], res["despues"], res["calibrador"]
    if c.tipo == "NINGUNO":
        return (
            f"Sobre el conjunto de {res['nombre'].lower()}, de {a.n} observaciones con "
            f"{res['n_pos']} incumplimientos, la puntuación de Brier asciende a {a.brier:.4f} "
            f"y el error de calibración esperado a {a.ece:.4f}, con una pendiente de "
            f"calibración de {a.pendiente:.3f} y un intercepto de {a.intercepto:+.3f}. "
            f"La evaluación fuera de muestra de los procedimientos de recalibración "
            f"disponibles no mejoró estos valores, por lo que no se aplicó corrección "
            f"alguna: la distribución de probabilidades declarada ya es la mejor que el "
            f"volumen de datos disponible permite sostener."
        )

    ajuste = getattr(c, "ajuste_logit", None)
    texto_ajuste = (f"mediante un desplazamiento del intercepto de {ajuste:.4f} en escala "
                    f"logarítmica" if ajuste is not None
                    else f"mediante el procedimiento {c.tipo}")
    veredicto = (
        "Las probabilidades resultantes quedan en condiciones de sustentar decisiones "
        "que ponderen costos."
        if d.brier_skill > 0 else
        "La puntuación de Brier no supera aún la de predecir siempre la prevalencia, de "
        "modo que las probabilidades sirven para ordenar a los clientes pero no para "
        "decidir mediante umbrales de costo. Esta limitación no afecta a la operación "
        "del sistema, que asigna los segmentos de riesgo mediante cortes sobre la "
        "puntuación y no sobre la probabilidad declarada."
    )
    return (
        f"Sobre el conjunto de {res['nombre'].lower()}, de {a.n} observaciones con una "
        f"frecuencia de incumplimiento del {a.prevalencia_observada:.1%}, el modelo declara "
        f"una probabilidad media del {a.prevalencia_predicha:.1%}, lo que supone una "
        f"desviación de {a.sesgo_prevalencia:+.1%}. La puntuación de Brier asciende a "
        f"{a.brier:.4f}, la pendiente de calibración a {a.pendiente:.3f} y el intercepto a "
        f"{a.intercepto:+.3f}, frente a los valores de referencia de uno y cero "
        f"respectivamente, con un error de calibración esperado de {a.ece:.4f}. Tras la "
        f"corrección {texto_ajuste}, la probabilidad media declarada se sitúa en "
        f"{d.prevalencia_predicha:.1%}, la puntuación de Brier en {d.brier:.4f} y el error "
        f"de calibración esperado en {d.ece:.4f}. La corrección es monótona y, por tanto, "
        f"no altera el ordenamiento de los clientes ni los valores del área bajo la curva "
        f"y del estadístico de Kolmogorov-Smirnov ya reportados. {veredicto}"
    )


# ---------------------------------------------------------------------------
def ejecutar_calibracion_definitiva(
    evaluacion_id: int | None = None, *, conjunto: str = "ambos",
    tramos: int = 5, cuantiles: bool = True, particiones: int = 5,
    forzado: str | None = None, persistir: bool = False, aplicar: bool = False,
    salida: Path = Path("reportes"),
) -> dict:
    """Misma orquestación que la CLI de este script (main()): corre el/los
    conjunto(s), decide y compara calibradores (informe_conjunto, que ya
    imprime en stdout), genera el párrafo de tesis, y si se pide, persiste
    en dbo.CalibracionModelo (persistir_conjunto) y aplica el calibrador
    principal a models/calibrador.pkl (guardar_calibrador).

    Reutilizada por main() (CLI de este archivo) y por
    scoring-morosidad/main.py:comando_calibrar() (botón "Calibración
    Modelo" del frontend de GESTIONCARTERA)."""
    if persistir and evaluacion_id is None:
        raise ValueError("persistir=True requiere evaluacion_id")

    resultados: dict = {}

    if conjunto in ("ambos", "crossval"):
        print("Reconstruyendo las predicciones fuera de muestra de la validacion cruzada...")
        y, p, g, prev = conjunto_validacion_cruzada(particiones)
        resultados["crossval"] = informe_conjunto(
            "Validación cruzada", y, p, g, prev, salida, tramos, cuantiles, forzado)

    if conjunto in ("ambos", "ventana"):
        print("\nReconstruyendo las predicciones de la ventana temporal...")
        y, p, g, prev = conjunto_ventana_temporal()
        resultados["ventana"] = informe_conjunto(
            "Ventana temporal", y, p, g, prev, salida, tramos, cuantiles, forzado)

    print(f"\n{'=' * 76}\nRESUMEN\n{'=' * 76}")
    for clave, origen in (("crossval", "CROSSVAL"), ("ventana", "VALIDACION")):
        if clave in resultados:
            r = resultados[clave]
            print(f"{origen:<11} n={r['n']:<4} positivos={r['n_pos']:<4} "
                  f"calibrador={r['calibrador'].tipo:<14} "
                  f"Brier {r['antes'].brier:.4f} -> {r['despues'].brier:.4f}  "
                  f"skill {r['antes'].brier_skill:+.4f} -> {r['despues'].brier_skill:+.4f}")

    principal = resultados.get("crossval") or resultados.get("ventana")
    if not principal:
        raise RuntimeError("No se pudo calcular ningún conjunto de calibración.")

    texto = parrafo_para_tesis(principal)
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "parrafo_tesis.txt").write_text(texto, encoding="utf-8")
    print(f"\n{'=' * 76}\nPARRAFO PARA LA SECCION 7.3\n{'=' * 76}\n{texto}")

    persistido = False
    if persistir:
        print(f"\nPersistiendo en EvaluacionId={evaluacion_id}:")
        for clave, origen in (("ventana", "VALIDACION"), ("crossval", "CROSSVAL")):
            if clave in resultados:
                persistir_conjunto(evaluacion_id, origen, resultados[clave])
        print("\nVerifique con:  SELECT * FROM dbo.vw_CalibracionVigente;")
        persistido = True

    calibrador_guardado = None
    if aplicar:
        c = principal["calibrador"]
        if c.tipo == "NINGUNO":
            print("\nNo se guarda calibrador: el criterio determinó no aplicar corrección.")
        else:
            ruta = cal.guardar_calibrador(c, Path("models") / "calibrador.pkl")
            print(f"\nCalibrador guardado en {ruta}")
            calibrador_guardado = c.tipo

    return {
        "evaluacion_id": evaluacion_id,
        "resultados": resultados,
        "parrafo": texto,
        "persistido": persistido,
        "calibrador_principal": principal["calibrador"].tipo,
        "calibrador_guardado": calibrador_guardado,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Reporte de calibración probabilística")
    ap.add_argument("--conjunto", choices=["ambos", "ventana", "crossval"], default="ambos")
    ap.add_argument("--tramos", type=int, default=5)
    ap.add_argument("--cuantiles", action="store_true", default=True)
    ap.add_argument("--particiones", type=int, default=5)
    ap.add_argument("--calibrador", default=None,
                    choices=["NINGUNO", "PRIOR", "INTERCEPTO_MV", "PLATT", "ISOTONICO"],
                    help="fuerza un calibrador en lugar de dejar que el criterio decida")
    ap.add_argument("--persistir", action="store_true")
    ap.add_argument("--evaluacion-id", type=int, default=None)
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--salida", default="reportes")
    args = ap.parse_args()

    if args.persistir and args.evaluacion_id is None:
        print("\n--persistir requiere --evaluacion-id")
        return 1

    try:
        ejecutar_calibracion_definitiva(
            args.evaluacion_id, conjunto=args.conjunto, tramos=args.tramos,
            cuantiles=args.cuantiles, particiones=args.particiones,
            forzado=args.calibrador, persistir=args.persistir, aplicar=args.aplicar,
            salida=Path(args.salida))
    except RuntimeError as exc:
        print(f"\n{exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
