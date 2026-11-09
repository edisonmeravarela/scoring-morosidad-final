"""Orquestador de línea de comandos del pipeline de scoring de morosidad
(Paso 8.10; ampliado con los controles de esquema y deriva del
documento "Controles de Esquema, Calidad y Deriva de Datos").

    python main.py validar-esquema       # Verificación estructural (siempre bloquea)
    python main.py validar-calidad       # Cuatro dimensiones de calidad (bloquea ante críticos)
    python main.py medir-deriva          # Índice de estabilidad poblacional (nunca bloquea)
    python main.py entrenar              # Encadena esquema, calidad y entrenamiento
    python main.py scorear --dry-run     # Encadena esquema, calidad, scoring y deriva, en simulación
    python main.py scorear               # Ídem, ejecución real con escritura en base de datos
    python main.py scorear --clientes 101,205   # Ídem, acotado a esos ClienteId (reevaluación bajo demanda)
    python main.py monitorear            # Desempeño en producción y desviación de datos (paso 8.9)
    python main.py calibrar              # Recalibra las probabilidades del modelo vigente (ver docs/11_README_corrida_definitiva.md)

Cada subcomando corre envuelto en _ejecutar_con_bitacora(), que registra
inicio, fin, duración, número de registros procesados y errores en
logs/main.log con marca de tiempo — criterio de aceptación de este paso,
para poder auditar qué proceso modificó a qué cliente y cuándo (los
ClienteId concretos quedan, además, en logs/writeback.log y en
dbo.ClienteRiesgoHistorial, paso 8.8).

Orden de ejecución de entrenar/scorear: esquema (siempre bloquea) →
calidad (bloquea ante críticos) → entrenamiento o scoring → [scorear
además mide la deriva al final, que nunca bloquea]. No tiene sentido
validar reglas de negocio sobre una estructura que cambió, ni medir la
deriva de un conjunto que no superó los controles de calidad.
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import calidad, db, drift, esquema, extract, monitor, rules, score, train, writeback
from src.config import RAIZ_PROYECTO, cargar_config

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.main")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "main.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())


TOTAL_ETAPAS_SCOREAR = 7


def _reportar_progreso(indice: int, total: int, etapa: str) -> None:
    """Protocolo mínimo de progreso para quien invoque este script como
    subproceso (backend/src/services/scoring_runner.py): una línea por
    etapa a STDOUT (nunca al logger, que va a stderr y mezclaría texto
    humano con el marcador). `comando_scorear()` es el único comando
    instrumentado hoy — es el único con una barra de progreso en el
    frontend (pantalla "Scorear (simulación)")."""
    print(f"PROGRESO {indice}/{total} {etapa}", flush=True)


def _ejecutar_con_bitacora(nombre_comando: str, funcion) -> int:
    """Envuelve cualquier subcomando con el registro de inicio/fin/duración/
    registros procesados que exige el criterio de aceptación del paso 8.10,
    sin importar cuál subcomando corra. Devuelve el código de salida."""
    inicio = datetime.now()
    logger.info("INICIO %s (%s)", nombre_comando, inicio.isoformat())
    try:
        registros_procesados = funcion()
        fin = datetime.now()
        logger.info(
            "FIN %s (%s, duración %.1fs): %d registros procesados.",
            nombre_comando,
            fin.isoformat(),
            (fin - inicio).total_seconds(),
            registros_procesados,
        )
        return 0
    except Exception as exc:
        fin = datetime.now()
        logger.error(
            "FIN %s CON ERROR (%s, duración %.1fs): %s",
            nombre_comando,
            fin.isoformat(),
            (fin - inicio).total_seconds(),
            exc,
        )
        return 1


def comando_validar_esquema() -> int:
    """Verificación estructural aislada (dbo.sp_ValidarEsquema), sin
    encadenar nada más — útil para revisar el contrato tras un cambio de
    esquema, sin tocar datos de negocio."""
    veredicto = esquema.validar_esquema()
    return veredicto["Criticos"] + veredicto["Advertencias"]


def comando_validar_calidad() -> int:
    """Las cuatro dimensiones de calidad aisladas (dbo.sp_ValidarCalidadDatos),
    sin encadenar el entrenamiento — para revisar el estado de los datos
    en cualquier momento."""
    veredicto = calidad.validar_antes_de_entrenar()
    return veredicto["Criticos"] + veredicto["Advertencias"] + veredicto["Conformes"]


def comando_medir_deriva() -> int:
    """Mide la deriva contra el baseline del modelo vigente
    (models/modelo.pkl), sin volver a scorear ni escribir nada — nunca
    bloquea, es puramente informativo."""
    bundle = train.cargar_modelo()
    version_modelo = bundle.get("version_modelo") or bundle.get("fecha_entrenamiento", "desconocida")
    resultados = drift.medir_deriva(version_modelo, bundle["mapeo"])
    con_alerta = int((resultados["PSI"] > 0.25).sum()) if not resultados.empty else 0
    logger.info("Deriva (versión %s): %d variables evaluadas, %d con PSI > 0.25.", version_modelo, len(resultados), con_alerta)
    return len(resultados)


def comando_entrenar() -> int:
    """Pasos 2, 3 y 4: extraer_dataset_entrenamiento + preparar_features +
    entrenar, todo orquestado ya dentro de train.entrenar()."""
    resultado = train.entrenar()
    logger.info(
        "Modelo aceptado: AUC=%.4f, KS=%.4f (entrenamiento: %d filas, validación: %d filas).",
        resultado["metricas"]["auc"],
        resultado["metricas"]["ks"],
        resultado["n_entrenamiento"],
        resultado["n_validacion"],
    )
    return resultado["n_entrenamiento"] + resultado["n_validacion"]


def _construir_resultados_scoring(df_clientes: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Pasos 5 y 6: aplica la Capa 1 (score.calcular_scores, ya incluido
    en df_clientes) y la Capa 2 (src/rules.py) a cada cliente, armando el
    DataFrame que espera writeback.guardar_resultados (paso 7)."""
    filas = []
    clientes_en_revision = []

    for cliente_id, fila in df_clientes.iterrows():
        segmento_id_asignado = rules.asignar_segmento(fila["score"])
        segmento_codigo = rules.aplicar_overrides(fila, segmento_id_asignado)
        segmento_final_id = rules.resolver_segmento_final(segmento_id_asignado, segmento_codigo)
        plazo_id = rules.recomendar_plazo(segmento_codigo)
        clasificacion_codigo = rules.clasificar_cliente(fila)
        clasificacion_id = rules.resolver_clasificacion_id(clasificacion_codigo)

        if rules.marcar_revision(fila, segmento_codigo):
            clientes_en_revision.append(cliente_id)

        filas.append(
            {
                "ClienteId": cliente_id,
                "ScoreRiesgo": fila["score"],
                "SegmentoRiesgoId": segmento_final_id,
                "ClasificacionId": clasificacion_id,
                "PlazoPagoRecomendadoId": plazo_id,
                "Motivo": f"Actualización automática por modelo de scoring (segmento {segmento_codigo}, clasificación {clasificacion_codigo})",
                # Códigos legibles (no solo los ids resueltos arriba): los usa
                # comando_scorear() para el CSV de resultados de la
                # simulación (barra de progreso + Excel, frontend), sin
                # necesidad de otra consulta — writeback.guardar_resultados
                # ignora cualquier columna que no use.
                "SegmentoCodigo": segmento_codigo,
                "ClasificacionCodigo": clasificacion_codigo,
            }
        )

    return pd.DataFrame(filas), clientes_en_revision


def _guardar_csv_simulacion(
    df_resultados: pd.DataFrame, df_clientes: pd.DataFrame, clientes_en_revision: list, reporte: dict
) -> Path:
    """Detalle por cliente de una simulación (scorear --dry-run): un
    archivo nuevo por corrida (nombre con marca de tiempo, para que dos
    simulaciones concurrentes no se pisen), pensado para la pantalla
    "Scorear (simulación)" del frontend (barra de progreso + descarga de
    Excel — ver RESULTADO_CSV en _reportar_progreso). Nunca se escribe
    para la ejecución real: ahí dbo.ClienteRiesgoHistorial ya es la
    fuente de verdad."""
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = LOGS_DIR / f"simulacion_{marca_tiempo}.csv"
    mapa_plazo_por_segmento = cargar_config().get("reglas", {}).get("plazo_por_segmento", {})
    cambiaron = set(reporte["cambiaron_segmento"])
    revision = set(clientes_en_revision)

    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(
            [
                "ClienteId", "NombreCliente", "ScorePropuesto", "SegmentoPropuesto",
                "ClasificacionPropuesta", "PlazoRecomendado", "CambiaSegmento", "RevisionManual",
            ]
        )
        for _, fila in df_resultados.iterrows():
            cliente_id = int(fila["ClienteId"])
            nombre = df_clientes.at[cliente_id, "NombreCompleto"] if "NombreCompleto" in df_clientes.columns else ""
            segmento_codigo = fila["SegmentoCodigo"]
            escritor.writerow(
                [
                    cliente_id,
                    nombre,
                    fila["ScoreRiesgo"],
                    segmento_codigo,
                    fila["ClasificacionCodigo"],
                    mapa_plazo_por_segmento.get(segmento_codigo, ""),
                    "Sí" if cliente_id in cambiaron else "No",
                    "Sí" if cliente_id in revision else "No",
                ]
            )
    return ruta


def comando_scorear(dry_run: bool, clientes_ids: list[int] | None = None) -> int:
    """Pasos 2, 3, 5, 6 y 7: extraer_features (inferencia) + calcular_scores
    + reglas de negocio + guardar_resultados (real o en simulación).

    Encadena, en orden: esquema (siempre bloquea) → calidad (bloquea
    ante críticos) → scoring/reglas/escritura → deriva (nunca bloquea,
    se mide al final contra el baseline del modelo vigente; un fallo al
    medirla no invalida un scoring ya escrito).

    `clientes_ids`, si se pasa, acota el scoring a esos ClienteId (uso:
    reevaluación bajo demanda de la cola dbo.ColaRescoreo, en vez del
    scoring mensual de toda la cartera). El filtro se aplica DESPUÉS de
    extraer_features() — nunca sobre la consulta SQL de la vista — para
    que extract.py siga validando el mínimo de clientes configurado
    (config.extract.minimo_clientes) contra la población completa, no
    contra el subconjunto.

    extract.agregar_historial_evaluable() incorpora TieneHistorial (y
    FacturasResueltas/UmbralVigente) a cada cliente antes de llamar a las
    reglas de negocio: src/rules.py ya no cuenta NumFacturas por su
    cuenta para decidir el override NUEVO/la clasificación RFM, consulta
    el mismo criterio que usan el entrenamiento y la cola de
    reevaluación (dbo.vw_HistorialClienteActual)."""
    esquema.validar_esquema()
    _reportar_progreso(1, TOTAL_ETAPAS_SCOREAR, "Validando esquema de datos")
    calidad.validar_antes_de_entrenar()
    _reportar_progreso(2, TOTAL_ETAPAS_SCOREAR, "Validando calidad de datos")

    df_clientes = extract.extraer_features()
    df_clientes = extract.agregar_historial_evaluable(df_clientes)
    if clientes_ids is not None:
        df_clientes = df_clientes.loc[df_clientes.index.isin(clientes_ids)]
        if df_clientes.empty:
            logger.info("Reevaluación bajo demanda: ninguno de los %d ClienteId pedidos es evaluable ahora mismo.", len(clientes_ids))
            return 0
    _reportar_progreso(3, TOTAL_ETAPAS_SCOREAR, "Extrayendo datos de clientes")
    scores = score.calcular_scores(df_clientes)
    combinado = df_clientes.join(scores)
    _reportar_progreso(4, TOTAL_ETAPAS_SCOREAR, "Calculando scores del modelo")

    df_resultados, clientes_en_revision = _construir_resultados_scoring(combinado)
    _reportar_progreso(5, TOTAL_ETAPAS_SCOREAR, "Aplicando reglas de negocio")
    # Detalle por variable (explicabilidad del score, sección 4.4.3 del
    # manual técnico): solo tiene sentido calcularlo si de verdad se va a
    # escribir — en dry_run no se guarda nada, sería trabajo descartado.
    detalle_variables = score.construir_detalle_variables(df_clientes) if not dry_run else None
    reporte = writeback.guardar_resultados(df_resultados, dry_run=dry_run, detalle_variables=detalle_variables)
    _reportar_progreso(6, TOTAL_ETAPAS_SCOREAR, "Guardando resultados" if not dry_run else "Comparando segmentos propuestos")

    if dry_run:
        ruta_csv = _guardar_csv_simulacion(df_resultados, df_clientes, clientes_en_revision, reporte)
        print(f"RESULTADO_CSV {ruta_csv.resolve()}", flush=True)

    logger.info(
        "Scoring (%s): %d cambiaron de segmento, %d sin cambios, %d fallidos, %d marcados para revisión manual.",
        reporte["modo"],
        len(reporte["cambiaron_segmento"]),
        len(reporte["sin_cambios"]),
        len(reporte["fallidos"]),
        len(clientes_en_revision),
    )
    if clientes_en_revision:
        logger.warning("Clientes marcados para revisión manual: %s", ", ".join(str(c) for c in clientes_en_revision))
    if reporte["fallidos"]:
        logger.error("Clientes que fallaron al escribir: %s", ", ".join(str(c) for c in reporte["fallidos"]))

    if dry_run:
        # medir_deriva() persiste cada medición en dbo.DriftMedicion: una
        # simulación ("sin escribir en la base de datos") no debe hacerlo.
        logger.info("Simulación: la deriva no se mide (se registraría en dbo.DriftMedicion).")
        _reportar_progreso(7, TOTAL_ETAPAS_SCOREAR, "Deriva no medida (simulación)")
        return reporte["total"]

    try:
        bundle = train.cargar_modelo()
        version_modelo = bundle.get("version_modelo") or bundle.get("fecha_entrenamiento", "desconocida")
        resultados_deriva = drift.medir_deriva(version_modelo, bundle["mapeo"], df_clientes)
        con_alerta = int((resultados_deriva["PSI"] > 0.25).sum()) if not resultados_deriva.empty else 0
        logger.info(
            "Deriva medida tras el scoring (versión %s): %d variables evaluadas, %d con PSI > 0.25.",
            version_modelo,
            len(resultados_deriva),
            con_alerta,
        )
    except Exception as exc:
        # La deriva nunca bloquea el pipeline: un fallo al medirla no
        # invalida un scoring que ya se ejecutó correctamente.
        logger.warning("No se pudo medir la deriva tras el scoring: %s", exc)

    _reportar_progreso(7, TOTAL_ETAPAS_SCOREAR, "Midiendo deriva de datos")
    return reporte["total"]


def comando_monitorear() -> int:
    """Paso 8: desempeño en producción (KS actual vs. validación) y
    desviación de datos (PSI) — src/monitor.py."""
    registros_procesados = 0

    try:
        desempeno = monitor.evaluar_desempeno()
        registros_procesados += desempeno["n_observaciones"]
        logger.info(
            "Desempeño en producción: KS=%.4f (validación %.4f, umbral %.4f). Alerta de reentrenamiento: %s.",
            desempeno["ks_actual"],
            desempeno["ks_validacion"],
            desempeno["umbral_alerta"],
            desempeno["alerta_reentrenamiento"],
        )
    except monitor.DatosInsuficientesError as exc:
        logger.warning("No se pudo evaluar el desempeño en producción: %s", exc)

    drift = monitor.detectar_drift()
    registros_procesados += len(drift)
    variables_con_alerta = drift.loc[drift["alerta"], "variable"].tolist() if not drift.empty else []
    logger.info("Desviación de datos: %d variables evaluadas, %d con alerta.", len(drift), len(variables_con_alerta))

    return registros_procesados


def _evaluacion_id_servida() -> int:
    """Resuelve el EvaluacionId del modelo actualmente servido
    (dbo.EvaluacionModelo.EsModeloServido = 1) — mismo criterio usado para
    distinguir el modelo real de corridas de prueba (ver
    reporte_calibracion.py)."""
    fila = db.consultar_escalar(
        "SELECT EvaluacionId FROM dbo.EvaluacionModelo WHERE EsModeloServido = 1"
    )
    if fila is None:
        raise RuntimeError(
            "No hay un modelo marcado como EsModeloServido=1 en "
            "dbo.EvaluacionModelo; no se puede calibrar."
        )
    return int(fila)


def comando_calibrar() -> int:
    """Calibración probabilística del modelo vigente
    (reporte_calibracion.ejecutar_calibracion_definitiva): reconstruye las
    predicciones fuera de muestra de validación cruzada y ventana
    temporal, decide si hace falta corrección (criterio en
    src/calibracion.py:recomendar_calibrador), persiste en
    dbo.CalibracionModelo y, si corresponde, guarda el calibrador elegido
    en models/calibrador.pkl."""
    import reporte_calibracion as repcal

    evaluacion_id = _evaluacion_id_servida()
    resultado = repcal.ejecutar_calibracion_definitiva(
        evaluacion_id, persistir=True, aplicar=True
    )
    logger.info(
        "Calibración completa (EvaluacionId=%d): calibrador principal=%s, guardado=%s.",
        evaluacion_id, resultado["calibrador_principal"], resultado["calibrador_guardado"],
    )
    return sum(r["n"] for r in resultado["resultados"].values())


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="Orquestador del pipeline de scoring de morosidad.")
    subparsers = parser.add_subparsers(dest="comando", required=True)

    subparsers.add_parser("validar-esquema", help="Verifica el esquema real contra dbo.ContratoEsquema (siempre bloquea ante críticos).")
    subparsers.add_parser("validar-calidad", help="Corre las 4 dimensiones de calidad de datos (bloquea ante críticos).")
    subparsers.add_parser("medir-deriva", help="Índice de estabilidad poblacional (PSI) contra el baseline del modelo vigente (nunca bloquea).")

    subparsers.add_parser("entrenar", help="Extrae, prepara variables y entrena el modelo (pasos 2, 3 y 4).")

    parser_scorear = subparsers.add_parser(
        "scorear", help="Calcula scores, aplica reglas de negocio y escribe en base de datos (pasos 2, 3, 5, 6 y 7)."
    )
    parser_scorear.add_argument("--dry-run", action="store_true", help="Simula sin escribir en base de datos.")
    parser_scorear.add_argument(
        "--clientes",
        type=str,
        default=None,
        help="Lista de ClienteId separados por coma; acota el scoring a esos clientes en vez de toda la cartera (reevaluación bajo demanda).",
    )

    subparsers.add_parser("monitorear", help="Evalúa el desempeño en producción y detecta desviación de datos (paso 8).")

    subparsers.add_parser("calibrar", help="Recalibra las probabilidades del modelo vigente y, si corresponde, guarda un calibrador nuevo.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    if args.comando == "validar-esquema":
        return _ejecutar_con_bitacora("validar-esquema", comando_validar_esquema)
    if args.comando == "validar-calidad":
        return _ejecutar_con_bitacora("validar-calidad", comando_validar_calidad)
    if args.comando == "medir-deriva":
        return _ejecutar_con_bitacora("medir-deriva", comando_medir_deriva)
    if args.comando == "entrenar":
        return _ejecutar_con_bitacora("entrenar", comando_entrenar)
    if args.comando == "scorear":
        clientes_ids = [int(x) for x in args.clientes.split(",")] if args.clientes else None
        nombre = "scorear --dry-run" if args.dry_run else "scorear"
        if clientes_ids is not None:
            nombre += f" --clientes ({len(clientes_ids)})"
        return _ejecutar_con_bitacora(nombre, lambda: comando_scorear(args.dry_run, clientes_ids))
    if args.comando == "monitorear":
        return _ejecutar_con_bitacora("monitorear", comando_monitorear)
    if args.comando == "calibrar":
        return _ejecutar_con_bitacora("calibrar", comando_calibrar)

    parser.error(f"Comando no reconocido: {args.comando}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
