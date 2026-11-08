/* =============================================================================
   02_auditoria_fuga_temporal.sql
   Objetivo : demostrar, con numeros, que la vista vigente usa informacion que
              no era observable en la fecha de emision, y cuantificar el efecto
              de la correccion.

   ADAPTADO (2026-10-27) contra el esquema real: los Controles 2 y 4 del
   paquete original referenciaban columnas que no existen con esos nombres en
   la base real (BrechaPlazo/DiasMaxMora/EsMorosa) — ver el encabezado de
   01_vistas_censura_temporal.sql para el detalle completo de las diferencias
   encontradas. Aqui se comparan las columnas que SI comparten nombre entre
   ambas vistas (PorcMoraPrevia, DiasMoraMaximaPrevia, NumFacturasResueltas,
   SinHistorialPago) y se usa Target (no EsMorosa) para la etiqueta.

   Ejecutar DESPUES de 01_vistas_censura_temporal.sql y ANTES de promover.
   ============================================================================= */

SET NOCOUNT ON;

/* -----------------------------------------------------------------------------
   CONTROL 1 — Facturas cuyo desenlace ocurrio DESPUES de la emision de una
   factura posterior del mismo cliente. Cada fila aqui es un caso concreto en
   que la vista vigente (que solo exige "vencimiento transcurrido") esta
   leyendo el futuro.
   ----------------------------------------------------------------------------- */
PRINT '=== CONTROL 1: pares (factura evaluada, factura previa) con desenlace futuro ===';

WITH Pares AS (
    SELECT
        f.FacturaId          AS FacturaEvaluada,
        f.FechaEmision       AS T0,
        p.FacturaId          AS FacturaPrevia,
        p.FechaVencimiento   AS VtoPrevia,
        pl.FechaLiquidacion  AS LiquidacionPrevia,
        DATEDIFF(day, f.FechaEmision, pl.FechaLiquidacion) AS DiasDeFuturoFiltrado
    FROM dbo.Factura AS f
    JOIN dbo.EstadoFactura AS ef  ON ef.EstadoFacturaId  = f.EstadoFacturaId
    JOIN dbo.Factura AS p         ON p.ClienteId         = f.ClienteId
                                 AND p.FacturaId        <> f.FacturaId
                                 AND p.FechaEmision      < f.FechaEmision
    JOIN dbo.EstadoFactura AS pef ON pef.EstadoFacturaId = p.EstadoFacturaId
    LEFT JOIN dbo.vw_LiquidacionFactura AS pl ON pl.FacturaId = p.FacturaId
    WHERE ef.Codigo  <> 'ANULADA'
      AND pef.Codigo <> 'ANULADA'
      AND p.FechaVencimiento < f.FechaEmision          -- la vista vigente SI la cuenta
      AND (pl.FechaLiquidacion IS NULL
           OR pl.FechaLiquidacion >= f.FechaEmision)   -- pero su pago no se conocia
)
SELECT
    COUNT(*)                                  AS ParesConFuga,
    COUNT(DISTINCT FacturaEvaluada)           AS FacturasEvaluadasAfectadas,
    AVG(CAST(DiasDeFuturoFiltrado AS FLOAT))  AS DiasPromedioDeFuturo,
    MAX(DiasDeFuturoFiltrado)                 AS DiasMaximoDeFuturo
FROM Pares;

/* Detalle, para incluir dos o tres ejemplos en el Anexo 7 */
PRINT '--- 10 ejemplos concretos ---';
WITH Pares AS (
    SELECT TOP 10
        f.FacturaId AS FacturaEvaluada, f.FechaEmision AS T0,
        p.FacturaId AS FacturaPrevia,   p.FechaVencimiento AS VtoPrevia,
        pl.FechaLiquidacion AS LiquidacionPrevia,
        DATEDIFF(day, f.FechaEmision, pl.FechaLiquidacion) AS DiasDeFuturo
    FROM dbo.Factura AS f
    JOIN dbo.EstadoFactura AS ef  ON ef.EstadoFacturaId  = f.EstadoFacturaId
    JOIN dbo.Factura AS p         ON p.ClienteId         = f.ClienteId
                                 AND p.FacturaId        <> f.FacturaId
                                 AND p.FechaEmision      < f.FechaEmision
    JOIN dbo.EstadoFactura AS pef ON pef.EstadoFacturaId = p.EstadoFacturaId
    LEFT JOIN dbo.vw_LiquidacionFactura AS pl ON pl.FacturaId = p.FacturaId
    WHERE ef.Codigo <> 'ANULADA' AND pef.Codigo <> 'ANULADA'
      AND p.FechaVencimiento < f.FechaEmision
      AND pl.FechaLiquidacion >= f.FechaEmision
    ORDER BY DATEDIFF(day, f.FechaEmision, pl.FechaLiquidacion) DESC
)
SELECT * FROM Pares;


/* -----------------------------------------------------------------------------
   CONTROL 2 — Diferencia fila a fila entre la vista vigente y la censurada,
   sobre las columnas que comparten nombre y significado comparable.
   PorcMoraPrevia y NumFacturasResueltas son las que mas deberian diferir: la
   vigente cuenta "vencidas" como resueltas, la censurada solo "liquidadas".
   ----------------------------------------------------------------------------- */
PRINT '=== CONTROL 2: diferencias entre vista vigente y vista censurada ===';

SELECT
    COUNT(*)                                                      AS FilasComparadas,
    SUM(CASE WHEN ISNULL(v.NumFacturasResueltas,-1)
                 <> ISNULL(n.NumFacturasResueltas,-1)
             THEN 1 ELSE 0 END)                                   AS DifiereNumFacturasResueltas,
    SUM(CASE WHEN ABS(ISNULL(v.PorcMoraPrevia,-1)
                    - ISNULL(n.PorcMoraPrevia,-1)) > 0.0001
             THEN 1 ELSE 0 END)                                   AS DifierePorcMoraPrevia,
    SUM(CASE WHEN ABS(ISNULL(v.DiasMoraMaximaPrevia,-1)
                    - ISNULL(n.DiasMoraMaximaPrevia,-1)) > 0.01
             THEN 1 ELSE 0 END)                                   AS DifiereDiasMoraMaximaPrevia,
    SUM(CASE WHEN ISNULL(v.SinHistorialPago,-1)
                 <> ISNULL(n.SinHistorialPago,-1)
             THEN 1 ELSE 0 END)                                   AS DifiereSinHistorial
FROM dbo.vw_DatasetEntrenamientoFactura           AS v
JOIN dbo.vw_DatasetEntrenamientoFacturaCensurada  AS n
      ON n.FacturaId = v.FacturaId;


/* -----------------------------------------------------------------------------
   CONTROL 3 — Cuanto cambia el conjunto entrenable.
   Importa porque "sin historial" pasa de contarse por VENCIDAS a contarse por
   LIQUIDADAS: es esperable perder filas, y hay que poder decir cuantas.
   ----------------------------------------------------------------------------- */
PRINT '=== CONTROL 3: tamano del conjunto entrenable antes y despues ===';

SELECT 'Vigente' AS Version,
       COUNT(*) AS FilasTotales,
       SUM(CASE WHEN SinHistorialPago = 0 THEN 1 ELSE 0 END) AS FilasEntrenables
FROM dbo.vw_DatasetEntrenamientoFactura
UNION ALL
SELECT 'Censurada',
       COUNT(*),
       SUM(CASE WHEN SinHistorialPago = 0 THEN 1 ELSE 0 END)
FROM dbo.vw_DatasetEntrenamientoFacturaCensurada;


/* -----------------------------------------------------------------------------
   CONTROL 4 — Las dos variables nuevas tienen senal.
   Si NumFacturasAbiertasVencidas y DiasMoraCorrienteMax no se correlacionan con
   la etiqueta (Target), la censura habria restado informacion sin devolver nada.
   ----------------------------------------------------------------------------- */
PRINT '=== CONTROL 4: tasa de mora por tramo de las variables nuevas ===';

SELECT
    CASE WHEN ISNULL(NumFacturasAbiertasVencidas,0) = 0 THEN '0 abiertas'
         WHEN NumFacturasAbiertasVencidas = 1          THEN '1 abierta'
         WHEN NumFacturasAbiertasVencidas BETWEEN 2 AND 3 THEN '2-3 abiertas'
         ELSE '4+ abiertas' END                        AS Tramo,
    COUNT(*)                                           AS Casos,
    AVG(CAST(Target AS FLOAT))                         AS TasaMora
FROM dbo.vw_DatasetEntrenamientoFacturaCensurada
WHERE SinHistorialPago = 0
GROUP BY CASE WHEN ISNULL(NumFacturasAbiertasVencidas,0) = 0 THEN '0 abiertas'
              WHEN NumFacturasAbiertasVencidas = 1          THEN '1 abierta'
              WHEN NumFacturasAbiertasVencidas BETWEEN 2 AND 3 THEN '2-3 abiertas'
              ELSE '4+ abiertas' END
ORDER BY Tramo;

SELECT
    CASE WHEN ISNULL(DiasMoraCorrienteMax,0) = 0  THEN 'sin mora corriente'
         WHEN DiasMoraCorrienteMax <= 30          THEN '1-30 dias'
         WHEN DiasMoraCorrienteMax <= 90          THEN '31-90 dias'
         ELSE '90+ dias' END                           AS Tramo,
    COUNT(*)                                           AS Casos,
    AVG(CAST(Target AS FLOAT))                         AS TasaMora
FROM dbo.vw_DatasetEntrenamientoFacturaCensurada
WHERE SinHistorialPago = 0
GROUP BY CASE WHEN ISNULL(DiasMoraCorrienteMax,0) = 0 THEN 'sin mora corriente'
              WHEN DiasMoraCorrienteMax <= 30         THEN '1-30 dias'
              WHEN DiasMoraCorrienteMax <= 90         THEN '31-90 dias'
              ELSE '90+ dias' END
ORDER BY Tramo;
