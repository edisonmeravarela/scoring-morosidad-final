/* =============================================================================
   01_vistas_censura_temporal.sql
   Proyecto : GestionCartera / scoring-morosidad  (Distrilife)
   Objetivo : Corregir la fuga temporal senalada por el tutor —
              "que una factura anterior haya vencido no significa que su pago
               definitivo ya fuera conocido".

   ADAPTADO CONTRA EL ESQUEMA REAL (2026-10-27): la version original de este
   archivo (paquete 00_README_orden_de_ejecucion.md) asumia nombres de columna
   y una estructura de vistas que no coinciden con la base real. Diferencias
   encontradas, verificadas con OBJECT_DEFINITION():

     1. dbo.vw_DatasetEntrenamientoFactura NO lee directamente de dbo.Factura:
        se construye sobre dbo.vw_FeaturesFactura (que ya resuelve ValorTotal,
        PlazoDias via dbo.PlazoPago, y la etiqueta EsMorosa->Target) y
        dbo.vw_ParametroHistorial (MIN_FACTURAS_HISTORIAL ya resuelto).
     2. Los nombres de salida reales son MontoFactura (no ValorTotal) y Target
        (no EsMorosa), entre otros — preservados tal cual abajo para no romper
        src/extract.py / src/features.py, que dependen de ellos por nombre.
     3. LA FUGA REAL no esta en "vencida != resuelta" de forma simple: esta en
        que dbo.vw_FeaturesFactura.DiasPagoReal/BrechaPlazo/DiasMora/EsMorosa/
        NumAbonos de una factura PREVIA se calculan con su FechaUltimoPago y
        SaldoPendiente ACTUALES (incluso con GETDATE() en el caso de DiasMora/
        EsMorosa) — es decir, con el desenlace FINAL de esa factura previa,
        sin importar cuando ocurrio. El filtro "p.FechaVencimiento < t0" de
        dbo.vw_DatasetEntrenamientoFactura solo exige que la factura previa YA
        estuviera vencida en t0, no que su pago ya se conociera en t0. Por eso
        la correccion real es reemplazar esa fuente (vw_FeaturesFactura) por un
        calculo propio, censurado a dbo.vw_LiquidacionFactura, para las
        variables de desenlace de las facturas PREVIAS — nunca para la propia
        factura evaluada, cuya etiqueta (Target) sigue viniendo de
        vw_FeaturesFactura.EsMorosa tal cual: ahi no hay fuga, es la etiqueta
        que el modelo predice, no una variable de entrada.
     4. dbo.vw_DatasetEntrenamientoFactura.prev / .expo (volumen, recencia y
        exposicion) YA estan correctamente censuradas a t0 en la vista vigente
        (prev solo usa ValorTotal/FechaEmision; expo trunca los abonos con
        "a.FechaAbono < f.FechaEmision") — no se tocan.

   Estrategia de despliegue (NO sobrescribe nada en este paso):
     1. Crea dbo.vw_LiquidacionFactura         (auxiliar, nueva)
     2. Crea dbo.vw_DatasetEntrenamientoFacturaCensurada  (nueva, en paralelo,
        MISMAS columnas que la vista vigente + 3 nuevas al final)
     3. Se compara contra la vista vigente con 02_auditoria_fuga_temporal.sql
     4. Solo despues se promueve (ver el bloque PROMOCION al final)
   ============================================================================= */

SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

/* -----------------------------------------------------------------------------
   PASO 1. Fecha en que cada factura quedo efectivamente LIQUIDADA.
   -----------------------------------------------------------------------------
   No se usa Factura.FechaUltimoPago (lo que ya usa vw_FeaturesFactura.DiasMora/
   EsMorosa): ese campo es MAX(FechaAbono), que para una factura con saldo
   pendiente es un abono parcial, no una liquidacion. Aqui se acumula abono a
   abono y se toma la primera fecha en que el acumulado alcanza el ValorTotal.
   Una factura nunca liquidada no aparece en esta vista.
   ----------------------------------------------------------------------------- */
CREATE OR ALTER VIEW dbo.vw_LiquidacionFactura
AS
SELECT
    acum.FacturaId,
    MIN(acum.FechaAbono) AS FechaLiquidacion
FROM (
    SELECT
        a.FacturaId,
        a.FechaAbono,
        SUM(a.ValorAbono + a.ValorRetencion) OVER (
            PARTITION BY a.FacturaId
            ORDER BY a.FechaAbono, a.AbonoId
            ROWS UNBOUNDED PRECEDING
        ) AS Acumulado
    FROM dbo.Abono AS a
) AS acum
JOIN dbo.Factura AS f
      ON f.FacturaId = acum.FacturaId
WHERE acum.Acumulado >= f.ValorTotal
GROUP BY acum.FacturaId;
GO


/* -----------------------------------------------------------------------------
   PASO 2. Dataset de entrenamiento con censura a la fecha de emision (t0).
   -----------------------------------------------------------------------------
   Mismas columnas de salida que dbo.vw_DatasetEntrenamientoFactura, en el
   mismo orden, mas 3 nuevas al final. prev/expo se copian sin cambios de la
   vista vigente (ya censuradas). hist/hist6 se recalculan censurados contra
   dbo.vw_LiquidacionFactura en vez de leer el desenlace final de
   vw_FeaturesFactura. abi es nueva: mora corriente de facturas vencidas pero
   AUN NO liquidadas en t0 (el hueco que antes no se reportaba en absoluto).
   ----------------------------------------------------------------------------- */
CREATE OR ALTER VIEW dbo.vw_DatasetEntrenamientoFacturaCensurada
AS
SELECT
    /* ---------- Identificacion ---------- */
    f.FacturaId,
    f.ClienteId,
    f.NumeroFactura,
    f.FechaEmision,
    f.FechaVencimiento,

    /* ---------- Propias de la factura (legitimas, se conocen al emitir) ---------- */
    f.ValorTotal                                        AS MontoFactura,
    f.PlazoDias                                          AS PlazoOtorgadoDias,
    f.VendedorId,
    c.CiudadId,
    DATEDIFF(DAY, c.FechaRegistro, f.FechaEmision)       AS DiasAntiguedadCliente,

    /* ---------- Volumen y recencia — SIN CAMBIOS, ya censuradas en la vista vigente ---------- */
    ISNULL(prev.NumFacturasPrevias, 0)                   AS NumFacturasPrevias,
    prev.MontoPromedioPrevio,
    prev.MontoAcumuladoPrevio,
    DATEDIFF(DAY, prev.FechaFacturaAnterior, f.FechaEmision) AS DiasDesdeFacturaAnterior,
    CAST(ISNULL(prev.NumFacturasPrevias, 0) AS DECIMAL(10,4))
        / NULLIF(DATEDIFF(MONTH, c.FechaRegistro, f.FechaEmision), 0) AS FacturasPorMes,

    /* ---------- Comportamiento de pago — SOLO facturas LIQUIDADAS antes de t0 ---------- */
    ISNULL(hist.NumFacturasResueltas, 0)                 AS NumFacturasResueltas,
    hist.DiasPagoPromedioPrevio,
    hist.DiasPagoDesviacionPrevia,
    hist.BrechaPlazoPromediaPrevia,
    hist.BrechaPlazoMaximaPrevia,
    hist.DiasMoraPromedioPrevio,
    hist.DiasMoraMaximaPrevia,
    hist.AbonosPorFacturaPrevio,
    CAST(hist.FacturasMorosasPrevias AS DECIMAL(10,4))
        / NULLIF(hist.NumFacturasResueltas, 0)           AS PorcMoraPrevia,
    CAST(hist6.MorosasUlt6M AS DECIMAL(10,4))
        / NULLIF(hist6.ResueltasUlt6M, 0)                AS PorcMoraPrevia6M,

    /* ---------- Exposicion — SIN CAMBIOS, ya censurada en la vista vigente ---------- */
    ISNULL(expo.SaldoVigenteEnT, 0)                      AS SaldoVigenteEmision,
    ISNULL(expo.FacturasAbiertasEnT, 0)                  AS FacturasAbiertasEmision,
    ISNULL(expo.SaldoVencidoEnT, 0)                      AS SaldoVencidoEmision,

    /* ---------- Historial evaluable — ahora por LIQUIDADAS, no por "vencidas" ---------- */
    CASE WHEN ISNULL(hist.NumFacturasResueltas, 0) < par.MinFacturasHistorial
         THEN 1 ELSE 0 END                               AS SinHistorialPago,
    par.MinFacturasHistorial                             AS UmbralHistorial,

    /* ---------- Etiqueta objetivo: SIN CAMBIOS — es el desenlace de ESTA
                  factura, no una variable de entrada; no hay fuga en usarlo. ---------- */
    f.EsMorosa                                           AS Target,

    /* ---------- NUEVAS: mora corriente de facturas AUN abiertas en t0 ---------- */
    abi.NumFacturasAbiertasVencidas,
    abi.MaxMoraCorriente                                 AS DiasMoraCorrienteMax,
    abi.MontoVencidoImpago

FROM dbo.vw_FeaturesFactura AS f
INNER JOIN dbo.Cliente AS c ON c.ClienteId = f.ClienteId
CROSS JOIN dbo.vw_ParametroHistorial AS par

/* --- A) Facturas previas emitidas: volumen y recencia (copiado, sin cambios) --- */
OUTER APPLY (
    SELECT COUNT(*)                     AS NumFacturasPrevias,
           AVG(p.ValorTotal)            AS MontoPromedioPrevio,
           SUM(p.ValorTotal)            AS MontoAcumuladoPrevio,
           MAX(p.FechaEmision)          AS FechaFacturaAnterior
    FROM dbo.vw_FeaturesFactura AS p
    WHERE p.ClienteId = f.ClienteId
      AND p.FechaEmision < f.FechaEmision
) AS prev

/* --- B) Desenlace: SOLO facturas LIQUIDADAS antes de t0 (LA CENSURA) --- */
OUTER APPLY (
    SELECT
        COUNT(*)                                                               AS NumFacturasResueltas,
        SUM(CASE WHEN pl.FechaLiquidacion > p.FechaVencimiento THEN 1 ELSE 0 END) AS FacturasMorosasPrevias,
        AVG(CAST(DATEDIFF(DAY, p.FechaEmision, pl.FechaLiquidacion) AS DECIMAL(10,2)))   AS DiasPagoPromedioPrevio,
        STDEV(CAST(DATEDIFF(DAY, p.FechaEmision, pl.FechaLiquidacion) AS DECIMAL(10,2))) AS DiasPagoDesviacionPrevia,
        AVG(CAST(DATEDIFF(DAY, p.FechaEmision, pl.FechaLiquidacion) - pp.Dias AS DECIMAL(10,2))) AS BrechaPlazoPromediaPrevia,
        MAX(DATEDIFF(DAY, p.FechaEmision, pl.FechaLiquidacion) - pp.Dias)                AS BrechaPlazoMaximaPrevia,
        AVG(CAST(CASE WHEN pl.FechaLiquidacion > p.FechaVencimiento
                      THEN DATEDIFF(DAY, p.FechaVencimiento, pl.FechaLiquidacion) ELSE 0 END AS DECIMAL(10,2))) AS DiasMoraPromedioPrevio,
        MAX(CASE WHEN pl.FechaLiquidacion > p.FechaVencimiento
                 THEN DATEDIFF(DAY, p.FechaVencimiento, pl.FechaLiquidacion) ELSE 0 END) AS DiasMoraMaximaPrevia,
        -- Abonos contados SOLO hasta t0 (uno posterior no era observable en t0)
        AVG(CAST(ab.NumAbonosHastaT0 AS DECIMAL(10,2)))                                  AS AbonosPorFacturaPrevio
    FROM dbo.Factura AS p
    JOIN dbo.EstadoFactura AS pef ON pef.EstadoFacturaId = p.EstadoFacturaId
    LEFT JOIN dbo.PlazoPago AS pp ON pp.PlazoPagoId = p.PlazoPagoId
    JOIN dbo.vw_LiquidacionFactura AS pl ON pl.FacturaId = p.FacturaId
    CROSS APPLY (
        SELECT COUNT(*) AS NumAbonosHastaT0
        FROM dbo.Abono AS a
        WHERE a.FacturaId = p.FacturaId AND a.FechaAbono < f.FechaEmision
    ) AS ab
    WHERE p.ClienteId         = f.ClienteId
      AND p.FacturaId        <> f.FacturaId
      AND p.FechaEmision      < f.FechaEmision
      AND pef.Codigo         <> 'ANULADA'
      AND pl.FechaLiquidacion < f.FechaEmision   /* <<<<<< LA CENSURA <<<<<< */
) AS hist

/* --- C) Igual que B, pero limitado a los 6 meses previos a t0 --- */
OUTER APPLY (
    SELECT
        COUNT(*)                                                     AS ResueltasUlt6M,
        SUM(CASE WHEN pl.FechaLiquidacion > p.FechaVencimiento THEN 1 ELSE 0 END) AS MorosasUlt6M
    FROM dbo.Factura AS p
    JOIN dbo.EstadoFactura AS pef ON pef.EstadoFacturaId = p.EstadoFacturaId
    JOIN dbo.vw_LiquidacionFactura AS pl ON pl.FacturaId = p.FacturaId
    WHERE p.ClienteId         = f.ClienteId
      AND p.FacturaId        <> f.FacturaId
      AND p.FechaEmision      < f.FechaEmision
      AND p.FechaEmision     >= DATEADD(MONTH, -6, f.FechaEmision)
      AND pef.Codigo         <> 'ANULADA'
      AND pl.FechaLiquidacion < f.FechaEmision
) AS hist6

/* --- D) Exposicion a t0 (copiado, sin cambios — ya truncaba abonos a t0) --- */
OUTER APPLY (
    SELECT SUM(saldo.SaldoEnT)                                       AS SaldoVigenteEnT,
           SUM(CASE WHEN saldo.SaldoEnT > 0 THEN 1 ELSE 0 END)        AS FacturasAbiertasEnT,
           SUM(CASE WHEN saldo.SaldoEnT > 0 AND saldo.YaVencidaEnT = 1
                    THEN saldo.SaldoEnT ELSE 0 END)                   AS SaldoVencidoEnT
    FROM (
        SELECT p.FacturaId,
               CASE WHEN p.FechaVencimiento < f.FechaEmision THEN 1 ELSE 0 END AS YaVencidaEnT,
               p.ValorTotal - ISNULL(
                   (SELECT SUM(a.ValorAbono + a.ValorRetencion)
                    FROM dbo.Abono a
                    WHERE a.FacturaId = p.FacturaId
                      AND a.FechaAbono < f.FechaEmision), 0)          AS SaldoEnT
        FROM dbo.vw_FeaturesFactura AS p
        WHERE p.ClienteId = f.ClienteId
          AND p.FechaEmision < f.FechaEmision
    ) AS saldo
) AS expo

/* --- E) NUEVA: mora corriente — vencidas en t0 pero AUN NO liquidadas ---
   El DATEDIFF contra f.FechaEmision (referencia externa) se precalcula en
   la subconsulta interna (mismo motivo que ya documenta el bloque "expo"
   de la vista vigente): un agregado no admite combinar una referencia
   externa con columnas internas en la misma expresion (error 8124). --- */
OUTER APPLY (
    SELECT
        COUNT(*)                    AS NumFacturasAbiertasVencidas,
        MAX(mora.DiasMoraCorriente) AS MaxMoraCorriente,
        SUM(mora.ValorTotal)        AS MontoVencidoImpago
    FROM (
        SELECT
            p.ValorTotal,
            DATEDIFF(DAY, p.FechaVencimiento, f.FechaEmision) AS DiasMoraCorriente
        FROM dbo.Factura AS p
        JOIN dbo.EstadoFactura AS pef ON pef.EstadoFacturaId = p.EstadoFacturaId
        LEFT JOIN dbo.vw_LiquidacionFactura AS pl ON pl.FacturaId = p.FacturaId
        WHERE p.ClienteId         = f.ClienteId
          AND p.FacturaId        <> f.FacturaId
          AND p.FechaEmision      < f.FechaEmision
          AND pef.Codigo         <> 'ANULADA'
          AND p.FechaVencimiento  < f.FechaEmision      -- ya vencida en t0
          AND (pl.FechaLiquidacion IS NULL
               OR pl.FechaLiquidacion >= f.FechaEmision) -- pero AUN NO liquidada en t0
    ) AS mora
) AS abi

WHERE f.FechaVencimiento IS NOT NULL
  AND f.FechaVencimiento < CAST(GETDATE() AS DATE);
GO


/* =============================================================================
   PASO 4 — PROMOCION  (ejecutar SOLO despues de correr 02_auditoria y de que
   la comparacion sea la esperada, y con la base respaldada)
   =============================================================================

   EXEC sp_rename 'dbo.vw_DatasetEntrenamientoFactura',
                  'vw_DatasetEntrenamientoFactura_PreCensura';

   CREATE OR ALTER VIEW dbo.vw_DatasetEntrenamientoFactura AS
   SELECT * FROM dbo.vw_DatasetEntrenamientoFacturaCensurada;

   NOTA sobre dbo.fn_HistorialCliente: igual que el original de este paquete
   senalaba, "cuenta las facturas con vencimiento ya transcurrido" y llama
   FacturasResueltas a ese conteo — vencida no es lo mismo que liquidada. Si
   se decide alinear tambien ese criterio, afecta a vw_HistorialClienteActual,
   vw_RescoreoPendiente, TR_Abono_EncolarRescoreo y a rules.py: cambio aparte,
   fuera de alcance de esta correccion puntual.
   ============================================================================= */
