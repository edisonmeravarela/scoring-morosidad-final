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


def _valor_opcional(fila: pd.Series, columna: str):
    valor = fila.get(columna)
    return None if pd.isna(valor) else valor


def guardar_resultados(df: pd.DataFrame, dry_run: bool = False, usuario_id: int | None = None, tamano_lote: int | None = None) -> dict:
    """Recorre `df` (columna o índice ClienteId; columnas ScoreRiesgo,
    SegmentoRiesgoId, ClasificacionId, PlazoPagoRecomendadoId y,
    opcionalmente, Motivo) y ejecuta dbo.sp_ActualizarScoringCliente por
    cada fila, en lotes de `tamano_lote` dentro de una transacción cada
    uno. Un fallo en una fila no aborta el lote: se registra en el log y
    en el reporte devuelto, y se sigue con el resto.

    En modo simulación (dry_run=True) no ejecuta ningún UPDATE: solo
    compara el SegmentoRiesgoId propuesto contra el vigente en base de
    datos para construir el mismo reporte de cambios."""
    if df.index.name == "ClienteId":
        df = df.reset_index()

    config = _config_writeback()
    tamano_lote = tamano_lote or config.get("tamano_lote", 50)
    motivo_por_defecto = config.get("motivo_por_defecto", "Actualización automática por modelo de scoring")

    segmento_vigente = _segmento_actual_por_cliente(df["ClienteId"].tolist())

    cambiaron, sin_cambios, fallidos = [], [], []
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
                except Exception as exc:
                    logger.error("Falló la actualización del cliente %d: %s", cliente_id, exc)
                    fallidos.append(cliente_id)

    resultado = {
        "modo": "simulacion" if dry_run else "real",
        "total": len(df),
        "cambiaron_segmento": cambiaron,
        "sin_cambios": sin_cambios,
        "fallidos": fallidos,
    }
    logger.info(
        "Writeback (%s): %d clientes procesados, %d cambiaron de segmento, %d sin cambios, %d fallidos.",
        resultado["modo"],
        resultado["total"],
        len(cambiaron),
        len(sin_cambios),
        len(fallidos),
    )
    return resultado
