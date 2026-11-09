"""Escritura en base de datos del resultado del scoring (Paso 8.8).

Ejecuta dbo.sp_ActualizarScoringCliente por cada cliente de un DataFrame
de resultados ya resuelto (columnas ClienteId, ScoreRiesgo,
SegmentoRiesgoId, ClasificacionId y PlazoPagoRecomendadoId — ids reales
de catálogo, ver src/rules.py: resolver_segmento_final() y
resolver_clasificacion_id()), en lotes dentro de una transacción, y
reporta cuántos clientes cambiaron de segmento, cuántos quedaron igual y
cuántos fallaron.

Modo simulación (dry_run=True): genera el mismo reporte sin ejecutar
ningún UPDATE — criterio de aceptación de este paso, indispensable para
que el área de crédito valide los resultados antes de la primera
ejecución real (documento, sección 8.11).
"""

import logging
import re

import pandas as pd

from src.config import RAIZ_PROYECTO, cargar_config
from src.db import get_engine, read_sql

LOGS_DIR = RAIZ_PROYECTO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("scoring_morosidad.writeback")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOGS_DIR / "writeback.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())


def _config_writeback() -> dict:
    return cargar_config().get("writeback", {})


def _segmento_actual_por_cliente(cliente_ids: list[int]) -> dict[int, int | None]:
    """Segmento vigente ANTES de escribir, para poder reportar cuántos
    clientes efectivamente cambian de segmento sin depender de que el
    procedimiento almacenado devuelva esa información."""
    if not cliente_ids:
        return {}
    marcadores = ", ".join(str(int(c)) for c in cliente_ids)
    df = read_sql(f"SELECT ClienteId, SegmentoRiesgoId FROM dbo.Cliente WHERE ClienteId IN ({marcadores})")
    return dict(zip(df["ClienteId"], df["SegmentoRiesgoId"]))


def _ejecutar_sp(conn, cliente_id: int, score: float, segmento_id: int, clasificacion_id, plazo_id, motivo: str, usuario_id) -> None:
    conn.exec_driver_sql(
        "EXEC dbo.sp_ActualizarScoringCliente "
        "@ClienteId=?, @ScoreRiesgo=?, @SegmentoRiesgoId=?, @ClasificacionId=?, "
        "@PlazoPagoRecomendadoId=?, @Motivo=?, @UsuarioId=?",
        (cliente_id, score, segmento_id, clasificacion_id, plazo_id, motivo, usuario_id),
    )


def _ultimo_historial_por_cliente(cliente_ids: list[int]) -> dict[int, int]:
    """HistorialId más reciente de dbo.ClienteRiesgoHistorial por cliente,
    ANTES de ejecutar el lote — permite detectar, por comparación, cuál
    cliente recibió una fila nueva (sp_ActualizarScoringCliente inserta en
    ClienteRiesgoHistorial solo si cambia segmento o clasificación, y no
    expone el HistorialId generado; no se usa OUTPUT porque el INSERT
    ocurre dentro del procedimiento, no aquí)."""
    if not cliente_ids:
        return {}
    marcadores = ", ".join(str(int(c)) for c in cliente_ids)
    df = read_sql(
        f"SELECT ClienteId, MAX(HistorialId) AS UltimoId FROM dbo.ClienteRiesgoHistorial "
        f"WHERE ClienteId IN ({marcadores}) GROUP BY ClienteId"
    )
    return dict(zip(df["ClienteId"], df["UltimoId"]))


def _historial_actual(conn, cliente_id: int) -> int:
    """HistorialId más reciente del cliente DESPUÉS del EXEC, en la misma
    transacción — comparado contra _ultimo_historial_por_cliente() para
    saber si sp_ActualizarScoringCliente insertó una fila nueva."""
    return conn.exec_driver_sql(
        "SELECT ISNULL(MAX(HistorialId), 0) FROM dbo.ClienteRiesgoHistorial WHERE ClienteId = ?",
        (cliente_id,),
    ).scalar()


def _insertar_detalle_variables(conn, historial_id: int, filas: list[dict]) -> None:
    """Persiste el desglose por variable (score.construir_detalle_variables)
    de una fila de ClienteRiesgoHistorial recién creada — ver
    dbo.ScoringDetalleVariable en database/schema_dbo.sql."""
    for f in filas:
        conn.exec_driver_sql(
            "INSERT INTO dbo.ScoringDetalleVariable "
            "(HistorialId, Variable, Etiqueta, ValorOriginal, WOE, Puntos, Orden) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                historial_id,
                f["Variable"],
                f["Etiqueta"],
                None if f["ValorOriginal"] is None else str(f["ValorOriginal"]),
                f["WOE"],
                f["Puntos"],
                f["Orden"],
            ),
        )


_PATRON_SCORE_MOTIVO = re.compile(r"\(score: -?[\d.]+\)$")


def _corregir_ultimo_historial(conn, historial_id: int, score: float, filas: list[dict]) -> bool:
    """Cuando sp_ActualizarScoringCliente no inserta fila (no cambió el
    segmento ni la clasificación), la última fila del historial sigue
    mostrando el score y el detalle de un cálculo anterior. Aquí se
    reescriben el "(score: …)" de su Motivo y su detalle con el cálculo
    actual, para que el total que muestra el frontend coincida con la suma
    del detalle. Solo se tocan filas generadas por el scoring (Motivo con
    "(score: …)"); una recalificación manual no se modifica. Devuelve si
    corrigió."""
    motivo = conn.exec_driver_sql(
        "SELECT Motivo FROM dbo.ClienteRiesgoHistorial WHERE HistorialId = ?", (historial_id,)
    ).scalar()
    if not motivo or not _PATRON_SCORE_MOTIVO.search(motivo):
        return False
    nuevo_motivo = _PATRON_SCORE_MOTIVO.sub(f"(score: {float(score):.2f})", motivo)
    conn.exec_driver_sql(
        "UPDATE dbo.ClienteRiesgoHistorial SET Motivo = ? WHERE HistorialId = ?", (nuevo_motivo, historial_id)
    )
    conn.exec_driver_sql("DELETE FROM dbo.ScoringDetalleVariable WHERE HistorialId = ?", (historial_id,))
    _insertar_detalle_variables(conn, historial_id, filas)
    return True


def _valor_opcional(fila: pd.Series, columna: str):
    valor = fila.get(columna)
    return None if pd.isna(valor) else valor


def guardar_resultados(
    df: pd.DataFrame,
    dry_run: bool = False,
    usuario_id: int | None = None,
    tamano_lote: int | None = None,
    detalle_variables: dict[int, list[dict]] | None = None,
) -> dict:
    """Recorre `df` (columna o índice ClienteId; columnas ScoreRiesgo,
    SegmentoRiesgoId, ClasificacionId, PlazoPagoRecomendadoId y,
    opcionalmente, Motivo) y ejecuta dbo.sp_ActualizarScoringCliente por
    cada fila, en lotes de `tamano_lote` dentro de una transacción cada
    uno. Un fallo en una fila no aborta el lote: se registra en el log y
    en el reporte devuelto, y se sigue con el resto.

    En modo simulación (dry_run=True) no ejecuta ningún UPDATE: solo
    compara el SegmentoRiesgoId propuesto contra el vigente en base de
    datos para construir el mismo reporte de cambios.

    `detalle_variables`, si se indica (score.construir_detalle_variables()),
    persiste en dbo.ScoringDetalleVariable el desglose por variable de cada
    cliente: si recibió una fila nueva en dbo.ClienteRiesgoHistorial
    (detectado comparando el HistorialId antes/después de cada EXEC, en la
    misma transacción), el detalle va con esa fila; si no, se corrigen el
    score y el detalle de su última fila de scoring
    (_corregir_ultimo_historial), para que el total mostrado y la suma del
    detalle coincidan tras cada recálculo. Nunca en dry_run."""
    if df.index.name == "ClienteId":
        df = df.reset_index()

    config = _config_writeback()
    tamano_lote = tamano_lote or config.get("tamano_lote", 50)
    motivo_por_defecto = config.get("motivo_por_defecto", "Actualización automática por modelo de scoring")

    segmento_vigente = _segmento_actual_por_cliente(df["ClienteId"].tolist())
    historial_vigente = _ultimo_historial_por_cliente(df["ClienteId"].tolist()) if detalle_variables else {}

    cambiaron, sin_cambios, fallidos, historial_corregido = [], [], [], []
    engine = get_engine()

    for inicio in range(0, len(df), tamano_lote):
        lote = df.iloc[inicio : inicio + tamano_lote]

        if dry_run:
            for _, fila in lote.iterrows():
                cliente_id = int(fila["ClienteId"])
                cambia = segmento_vigente.get(cliente_id) != int(fila["SegmentoRiesgoId"])
                (cambiaron if cambia else sin_cambios).append(cliente_id)
            continue

        with engine.begin() as conn:
            for _, fila in lote.iterrows():
                cliente_id = int(fila["ClienteId"])
                try:
                    _ejecutar_sp(
                        conn,
                        cliente_id,
                        float(fila["ScoreRiesgo"]),
                        int(fila["SegmentoRiesgoId"]),
                        _valor_opcional(fila, "ClasificacionId"),
                        _valor_opcional(fila, "PlazoPagoRecomendadoId"),
                        fila.get("Motivo", motivo_por_defecto),
                        usuario_id,
                    )
                    cambia = segmento_vigente.get(cliente_id) != int(fila["SegmentoRiesgoId"])
                    (cambiaron if cambia else sin_cambios).append(cliente_id)
                    if detalle_variables and cliente_id in detalle_variables:
                        nuevo_historial_id = _historial_actual(conn, cliente_id)
                        if nuevo_historial_id > historial_vigente.get(cliente_id, 0):
                            _insertar_detalle_variables(conn, nuevo_historial_id, detalle_variables[cliente_id])
                        elif nuevo_historial_id and _corregir_ultimo_historial(
                            conn, nuevo_historial_id, float(fila["ScoreRiesgo"]), detalle_variables[cliente_id]
                        ):
                            historial_corregido.append(cliente_id)
                except Exception as exc:
                    logger.error("Falló la actualización del cliente %d: %s", cliente_id, exc)
                    fallidos.append(cliente_id)

    resultado = {
        "modo": "simulacion" if dry_run else "real",
        "total": len(df),
        "cambiaron_segmento": cambiaron,
        "sin_cambios": sin_cambios,
        "fallidos": fallidos,
        "historial_corregido": historial_corregido,
    }
    logger.info(
        "Writeback (%s): %d clientes procesados, %d cambiaron de segmento, %d sin cambios, %d fallidos, "
        "%d con la última fila del historial corregida.",
        resultado["modo"],
        resultado["total"],
        len(cambiaron),
        len(sin_cambios),
        len(fallidos),
        len(historial_corregido),
    )
    return resultado
