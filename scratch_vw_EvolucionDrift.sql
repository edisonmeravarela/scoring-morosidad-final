
/* ============================================================================
   5. VISTA DE SEGUIMIENTO
      Evolución del drift por variable entre ejecuciones: permite distinguir
      una desviación puntual de una tendencia sostenida que justifica
      reentrenar.
   ============================================================================ */

CREATE   VIEW dbo.vw_EvolucionDrift
AS
SELECT
    d.Variable,
    d.VersionModelo,
    CAST(d.Fecha AS DATE)                       AS Fecha,
    d.PSI,
    d.PctFueraDeRango,
    d.Interpretacion,
    d.Origen,
    LAG(d.PSI) OVER (PARTITION BY d.Variable ORDER BY d.Fecha) AS PSI_Anterior,
    d.PSI - LAG(d.PSI) OVER (PARTITION BY d.Variable ORDER BY d.Fecha) AS Variacion
FROM dbo.DriftMedicion d;
