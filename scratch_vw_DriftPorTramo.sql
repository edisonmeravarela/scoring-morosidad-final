
/* ============================================================================
   4. VISTA DE DESGLOSE PARA REVISIÓN
   ============================================================================ */

CREATE   VIEW dbo.vw_DriftPorTramo
AS
SELECT
    m.Variable,
    m.VersionModelo,
    CAST(m.Fecha AS DATE)                                   AS Fecha,
    d.Tramo,
    CAST(100.0 * d.ProporcionBase   AS DECIMAL(6,2))        AS PctBase,
    CAST(100.0 * d.ProporcionActual AS DECIMAL(6,2))        AS PctActual,
    CAST(100.0 * (d.ProporcionActual - d.ProporcionBase) AS DECIMAL(6,2)) AS Variacion,
    d.AportePSI,
    CAST(100.0 * d.AportePSI / NULLIF(m.PSI, 0) AS DECIMAL(6,2)) AS PctDelPSI,
    CASE WHEN d.EnBaseline = 1 THEN 'CONOCIDO' ELSE 'NO VISTO EN ENTRENAMIENTO' END AS Origen
FROM dbo.DriftDetalleTramo d
JOIN dbo.DriftMedicion m ON m.MedicionId = d.MedicionId;
