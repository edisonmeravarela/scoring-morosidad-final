"""Orquestador de línea de comandos del pipeline de scoring de morosidad
(Paso 8.10; ampliado con los controles de esquema y deriva del
documento "Controles de Esquema, Calidad y Deriva de Datos").

    python main.py validar-esquema       # Verificación estructural (siempre bloquea)
    python main.py validar-calidad       # Cuatro dimensiones de calidad (bloquea ante críticos)
    python main.py medir-deriva          # Índice de estabilidad poblacional (nunca bloquea)
    python main.py entrenar              # Encadena esquema, calidad y entrenamiento
    python main.py scorear --dry-run     # Encadena esquema, calidad, scoring y deriva, en simulación
    python main.py scorear               # Ídem, ejecución real con escritura en base de datos
    python main.py monitorear            # Desempeño en producción y desviación de datos (paso 8.9)

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
import logging
import sys
from datetime import datetime

import pandas as pd

from src import calidad, drift, esquema, extract, monitor, rules, score, train, writeback
from src.config import RAIZ_PROYECTO

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.main")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "main.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())


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
            }
        )

    return pd.DataFrame(filas), clientes_en_revision


def comando_scorear(dry_run: bool) -> int:
    """Pasos 2, 3, 5, 6 y 7: extraer_features (inferencia) + calcular_scores
    + reglas de negocio + guardar_resultados (real o en simulación).

    Encadena, en orden: esquema (siempre bloquea) → calidad (bloquea
    ante críticos) → scoring/reglas/escritura → deriva (nunca bloquea,
    se mide al final contra el baseline del modelo vigente; un fallo al
    medirla no invalida un scoring ya escrito)."""
    esquema.validar_esquema()
    calidad.validar_antes_de_entrenar()

    df_clientes = extract.extraer_features()
    scores = score.calcular_scores(df_clientes)
    combinado = df_clientes.join(scores)

    df_resultados, clientes_en_revision = _construir_resultados_scoring(combinado)
    reporte = writeback.guardar_resultados(df_resultados, dry_run=dry_run)

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
        # invalida un scoring que ya se ejecutó (o simuló) correctamente.
        logger.warning("No se pudo medir la deriva tras el scoring: %s", exc)

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

    subparsers.add_parser("monitorear", help="Evalúa el desempeño en producción y detecta desviación de datos (paso 8).")

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
        nombre = "scorear --dry-run" if args.dry_run else "scorear"
        return _ejecutar_con_bitacora(nombre, lambda: comando_scorear(args.dry_run))
    if args.comando == "monitorear":
        return _ejecutar_con_bitacora("monitorear", comando_monitorear)

    parser.error(f"Comando no reconocido: {args.comando}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
