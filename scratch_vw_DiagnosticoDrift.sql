
/* ============================================================================
   3. VISTA DE DIAGNÓSTICO
      Aplica los dos criterios que separan deriva real de falta de cobertura.
   ============================================================================ */

CREATE   VIEW dbo.vw_DiagnosticoDrift
AS
SELECT
    m.MedicionId,
    m.Fecha,
    m.VersionModelo,
    m.Variable,
    m.PSI,
    m.PctFueraDeRango,
    m.Interpretacion,
    t.TramoDominante,
    CAST(100.0 * t.AporteMaximo / NULLIF(m.PSI, 0) AS DECIMAL(6,2)) AS PctPSI_TramoDominante,
    CASE
        WHEN m.PSI < 0.25
             THEN 'ESTABLE — sin acción'
        WHEN m.PctFueraDeRango > 20
             THEN 'COBERTURA INSUFICIENTE — agrupar categorías o ampliar el entrenamiento; reentrenar no resuelve'
        WHEN 100.0 * t.AporteMaximo / NULLIF(m.PSI, 0) > 50
             THEN 'CONCENTRADO EN UN TRAMO — revisar la representación de esa categoría antes de reentrenar'
        ELSE 'DERIVA REAL — la población cambió; evaluar reentrenamiento'
    END                                                             AS Diagnostico
FROM dbo.DriftMedicion m
OUTER APPLY
(
    SELECT TOP 1 d.Tramo AS TramoDominante, d.AportePSI AS AporteMaximo
    FROM dbo.DriftDetalleTramo d
    WHERE d.MedicionId = m.MedicionId
    ORDER BY d.AportePSI DESC
) t;
