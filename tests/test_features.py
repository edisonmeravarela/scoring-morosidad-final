import logging

import numpy as np
import pandas as pd
import pytest

from src import extract, features


def _dataset_sintetico(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Dataset controlado: DiasMoraPromedio tiene una relación real pero
    moderada con el target (IV esperable en el rango "sano" 0.02–1.0, como
    una variable predictiva genuina, no una fuga); RuidoAleatorio es puro
    ruido (debe salir con IV bajo, cercano a 0); VariableConNulos simula
    clientes sin abonos (20% nulos) pero conserva la misma señal moderada."""
    rng = np.random.default_rng(seed)
    dias_mora = rng.integers(0, 100, size=n).astype(float)
    prob_malo = 0.15 + 0.4 * (dias_mora / 100)  # va de 15% a 55%, no determinístico
    target = (rng.random(n) < prob_malo).astype(int)
    ruido = rng.random(n)

    con_nulos = dias_mora.copy()
    con_nulos[::5] = np.nan  # 20% nulos, patrón determinístico

    return pd.DataFrame(
        {
            "ClienteId": range(1, n + 1),
            "DiasMoraPromedio": dias_mora,
            "RuidoAleatorio": ruido,
            "VariableConNulos": con_nulos,
            "TuvoMoraHistorica": target,
        }
    ).set_index("ClienteId")


def test_identificar_nulos_cuenta_correctamente():
    df = _dataset_sintetico(n=100)
    variables = ["DiasMoraPromedio", "RuidoAleatorio", "VariableConNulos"]
    conteo = features.identificar_nulos(df, variables)
    assert conteo["DiasMoraPromedio"] == 0
    assert conteo["RuidoAleatorio"] == 0
    assert conteo["VariableConNulos"] == 20  # 1 de cada 5 en 100 filas


def test_variables_candidatas_excluye_columnas_configuradas():
    df = pd.DataFrame(
        {
            "NombreCompleto": ["A", "B"],
            "CiudadId": [1, 2],
            "SegmentoActualId": [1, 2],
            "DiasMoraPromedio": [10.0, 20.0],
            "TuvoMoraHistorica": [0, 1],
        }
    )
    candidatas = features.variables_candidatas(df, target_col="TuvoMoraHistorica")
    assert "DiasMoraPromedio" in candidatas
    assert "TuvoMoraHistorica" not in candidatas
    assert "CiudadId" not in candidatas
    assert "SegmentoActualId" not in candidatas
    assert "NombreCompleto" not in candidatas  # ni siquiera es numérica


def test_variables_candidatas_excluye_las_configuradas_como_categoricas():
    # Ajuste post-8.4 (sección 5, "Propias de la factura"): VendedorId y
    # CiudadId no se descartan, pero tampoco se discretizan por cuantiles
    # como si fueran continuas — van por variables_categoricas_candidatas.
    df = pd.DataFrame(
        {
            "VendedorId": [1, 2],
            "CiudadId": [1, 2],
            "MontoFactura": [100.0, 200.0],
            "TuvoMoraHistorica": [0, 1],
        }
    )
    candidatas = features.variables_candidatas(df, target_col="TuvoMoraHistorica")
    assert "MontoFactura" in candidatas
    assert "VendedorId" not in candidatas
    assert "CiudadId" not in candidatas


def test_variables_categoricas_candidatas_solo_devuelve_las_presentes():
    # VendedorId no existe en la vista de inferencia por cliente; no debe
    # reventar, simplemente no aparece en la lista.
    df = pd.DataFrame({"CiudadId": [1, 2], "MontoFactura": [100.0, 200.0]})
    candidatas = features.variables_categoricas_candidatas(df)
    assert candidatas == ["CiudadId"]


def _dataset_con_categoria(n: int = 300, seed: int = 5) -> pd.DataFrame:
    """VendedorId tiene 3 categorías frecuentes con distinta tasa de mora
    real (predictiva pero no determinística) y 2 categorías raras (menos
    del min_conteo configurado) que deben agruparse en OTRAS."""
    rng = np.random.default_rng(seed)
    vendedores_frecuentes = rng.choice([1, 2, 3], size=n - 4, p=[0.5, 0.3, 0.2])
    vendedores = np.concatenate([vendedores_frecuentes, [99, 99, 100, 100]])  # 2+2 casos raros
    rng.shuffle(vendedores)

    prob_malo_por_vendedor = {1: 0.15, 2: 0.35, 3: 0.55, 99: 0.5, 100: 0.5}
    prob_malo = np.array([prob_malo_por_vendedor[v] for v in vendedores])
    target = (rng.random(n) < prob_malo).astype(int)

    return pd.DataFrame(
        {
            "ClienteId": range(1, n + 1),
            "VendedorId": vendedores,
            "TuvoMoraHistorica": target,
        }
    ).set_index("ClienteId")


def test_calcular_woe_categoria_agrupa_categorias_infrecuentes_en_otras():
    df = _dataset_con_categoria()
    resultado = features.calcular_woe_categoria(df["VendedorId"], df["TuvoMoraHistorica"], min_conteo=5)

    assert resultado["tipo"] == "categorica"
    tramo_ids = {t["tramo_id"] for t in resultado["tramos"]}
    assert {1, 2, 3} <= tramo_ids
    assert features.CATEGORIA_OTRAS in tramo_ids
    assert 99 not in tramo_ids  # se agrupó dentro de OTRAS, no es su propio tramo
    assert 100 not in tramo_ids

    tramo_otras = next(t for t in resultado["tramos"] if t["tramo_id"] == features.CATEGORIA_OTRAS)
    assert tramo_otras["n"] == 4  # los 2+2 casos raros


def test_calcular_woe_categoria_da_iv_positivo_para_categoria_predictiva():
    df = _dataset_con_categoria()
    resultado = features.calcular_woe_categoria(df["VendedorId"], df["TuvoMoraHistorica"], min_conteo=5)
    assert resultado["iv"] > 0.02

    # el vendedor con menor tasa de mora real (1) debe tener mejor WOE que
    # el de mayor tasa de mora real (3).
    woe_por_vendedor = {t["tramo_id"]: t["woe"] for t in resultado["tramos"]}
    assert woe_por_vendedor[1] > woe_por_vendedor[3]


def test_calcular_woe_categoria_trata_nulos_como_tramo_propio():
    df = _dataset_con_categoria()
    df.loc[df.index[:10], "VendedorId"] = np.nan
    resultado = features.calcular_woe_categoria(df["VendedorId"], df["TuvoMoraHistorica"], min_conteo=5)

    tramo_missing = next(t for t in resultado["tramos"] if t["tramo_id"] == features.TRAMO_FALTANTE)
    assert tramo_missing["n"] == 10


def test_transformar_a_woe_categorica_asigna_categoria_no_vista_a_otras(caplog):
    df = _dataset_con_categoria()
    mapeo = {"VendedorId": features.calcular_woe_categoria(df["VendedorId"], df["TuvoMoraHistorica"], min_conteo=5)}

    cliente_nuevo = pd.DataFrame({"VendedorId": [999]}, index=pd.Index([12345], name="ClienteId"))
    with caplog.at_level(logging.WARNING, logger="scoring_morosidad.features"):
        resultado = features.transformar_a_woe(cliente_nuevo, mapeo)

    assert "categoría no vista en entrenamiento" in caplog.text
    assert "12345" in caplog.text
    woe_otras = next(
        t["woe"] for t in mapeo["VendedorId"]["tramos"] if t["tramo_id"] == features.CATEGORIA_OTRAS
    )
    assert resultado["VendedorId"].iloc[0] == woe_otras


def test_transformar_a_woe_categorica_es_reproducible_con_el_mismo_mapeo():
    df = _dataset_con_categoria()
    mapeo = {"VendedorId": features.calcular_woe_categoria(df["VendedorId"], df["TuvoMoraHistorica"], min_conteo=5)}

    x = features.transformar_a_woe(df, mapeo)
    cliente_referencia = df.iloc[[0]]
    x_nuevo = features.transformar_a_woe(cliente_referencia, mapeo)
    assert x_nuevo["VendedorId"].iloc[0] == x["VendedorId"].iloc[0]


def test_preparar_features_incluye_variables_categoricas(tmp_path, monkeypatch):
    df_numerico = _dataset_sintetico(n=300)
    df_categorico = _dataset_con_categoria(n=300)
    df = df_numerico.join(df_categorico[["VendedorId"]])

    ruta_mapeo = tmp_path / "mapeo_categorico.pkl"
    config_con_ruta_temporal = {**features._config_features(), "ruta_mapeo": str(ruta_mapeo)}
    monkeypatch.setattr(features, "_config_features", lambda: config_con_ruta_temporal)

    x_woe, mapeo = features.preparar_features(df, target_col="TuvoMoraHistorica")
    assert "VendedorId" in mapeo
    assert mapeo["VendedorId"]["tipo"] == "categorica"
    assert "VendedorId" in x_woe.columns


def test_calcular_woe_variable_da_mas_iv_a_la_variable_predictiva():
    df = _dataset_sintetico(n=300)
    resultado_predictiva = features.calcular_woe_variable(df["DiasMoraPromedio"], df["TuvoMoraHistorica"])
    resultado_ruido = features.calcular_woe_variable(df["RuidoAleatorio"], df["TuvoMoraHistorica"])

    assert resultado_predictiva["iv"] > resultado_ruido["iv"]
    assert resultado_predictiva["iv"] > 0.02  # debería sobrevivir el filtro
    # el tramo de mayor mora debe tener peor (más negativo) WOE que el de menor mora
    tramos = sorted(resultado_predictiva["tramos"], key=lambda t: t["tramo_id"])
    tramos_reales = [t for t in tramos if t["tramo_id"] >= 0]
    assert tramos_reales[0]["woe"] > tramos_reales[-1]["woe"]


def test_calcular_woe_variable_trata_nulos_como_tramo_propio():
    df = _dataset_sintetico(n=100)
    resultado = features.calcular_woe_variable(df["VariableConNulos"], df["TuvoMoraHistorica"])

    tramo_missing = next(t for t in resultado["tramos"] if t["tramo_id"] == -1)
    assert tramo_missing["etiqueta"] == features.TRAMO_FALTANTE
    assert tramo_missing["n"] == 20
    # el WOE del tramo faltante es un valor calculado, no cero por relleno
    assert isinstance(tramo_missing["woe"], float)


def test_filtrar_por_iv_descarta_variables_de_bajo_iv():
    df = _dataset_sintetico(n=300)
    mapeo = features.calcular_mapeo_woe(df, target_col="TuvoMoraHistorica")
    assert "DiasMoraPromedio" in mapeo
    assert "RuidoAleatorio" in mapeo  # calcular_mapeo_woe no filtra, solo calcula

    # Con datos sintéticos y pocos tramos, el ruido puro rara vez da IV
    # exactamente 0 (es una variable aleatoria); un umbral de 0.1 separa con
    # margen el ruido (IV ~0.04 con esta semilla) de la señal real (~0.3-0.4).
    filtrado = features.filtrar_por_iv(mapeo, iv_minimo=0.1)
    assert "DiasMoraPromedio" in filtrado
    assert "RuidoAleatorio" not in filtrado


def test_filtrar_por_iv_descarta_variables_con_iv_sospechosamente_alto():
    # Prueba de fuga (ajuste post-8.4): una variable que es prácticamente
    # una copia de la etiqueta debe descartarse aunque su IV sea "bueno" en
    # apariencia — un IV > 1.0 es indicio de leakage, no de señal genuina.
    df = _dataset_sintetico(n=200)
    target = df["TuvoMoraHistorica"]
    # Variable derivada casi 1:1 del target, con un poco de ruido para no
    # ser perfectamente determinística (peor caso: filtración parcial).
    rng = np.random.default_rng(42)
    df["VariableFiltrada"] = target * 100 + rng.normal(0, 1, size=len(df))

    mapeo = features.calcular_mapeo_woe(df, target_col="TuvoMoraHistorica")
    assert mapeo["VariableFiltrada"]["iv"] > 1.0

    filtrado = features.filtrar_por_iv(mapeo, iv_minimo=0.02, iv_maximo=1.0)
    assert "VariableFiltrada" not in filtrado
    assert "DiasMoraPromedio" in filtrado  # la señal genuina sí sobrevive


def test_filtrar_por_iv_usa_el_iv_maximo_configurado():
    df = _dataset_sintetico(n=200)
    mapeo = features.calcular_mapeo_woe(df, target_col="TuvoMoraHistorica")
    iv_real = mapeo["DiasMoraPromedio"]["iv"]

    # Con un techo más bajo que el IV real de una variable "buena", también
    # debe descartarse: el umbral es configurable, no un caso especial.
    filtrado = features.filtrar_por_iv(mapeo, iv_minimo=0.02, iv_maximo=iv_real / 2)
    assert "DiasMoraPromedio" not in filtrado


def test_detectar_separacion_perfecta_identifica_variable_leaky():
    # Control obligatorio del documento V1 (sección 4.4): comparar el
    # mínimo de una clase contra el máximo de la otra, independientemente
    # del IV. Construimos una variable donde TODOS los malos (target=1)
    # quedan por encima de TODOS los buenos (target=0): separación perfecta.
    df = _dataset_sintetico(n=100)
    target = df["TuvoMoraHistorica"]
    df["VariableSeparadora"] = np.where(target == 1, 1000.0, 0.0)

    sospechosas = features.detectar_separacion_perfecta(df, "TuvoMoraHistorica", ["VariableSeparadora"])
    assert sospechosas == ["VariableSeparadora"]


def test_detectar_separacion_perfecta_no_marca_variables_con_distribuciones_solapadas():
    df = _dataset_sintetico(n=200)
    # DiasMoraPromedio y RuidoAleatorio tienen el mismo rango de valores
    # posibles para buenos y malos (se solapan): no deben marcarse.
    sospechosas = features.detectar_separacion_perfecta(
        df, "TuvoMoraHistorica", ["DiasMoraPromedio", "RuidoAleatorio"]
    )
    assert sospechosas == []


def test_detectar_separacion_perfecta_ignora_variable_sin_datos_en_una_clase():
    df = pd.DataFrame(
        {
            "TuvoMoraHistorica": [0, 0, 1, 1],
            "SoloTieneBuenos": [10.0, 20.0, np.nan, np.nan],
        }
    )
    # target=1 no tiene ningún valor no nulo en esta variable: no hay con
    # qué comparar, así que no puede evaluarse (no debe lanzar error).
    sospechosas = features.detectar_separacion_perfecta(df, "TuvoMoraHistorica", ["SoloTieneBuenos"])
    assert sospechosas == []


def test_preparar_features_excluye_variables_con_separacion_perfecta(tmp_path, monkeypatch):
    df = _dataset_sintetico(n=200)
    target = df["TuvoMoraHistorica"]
    df["VariableSeparadora"] = np.where(target == 1, 1000.0, 0.0)

    ruta_mapeo = tmp_path / "mapeo_separacion.pkl"
    config_con_ruta_temporal = {**features._config_features(), "ruta_mapeo": str(ruta_mapeo)}
    monkeypatch.setattr(features, "_config_features", lambda: config_con_ruta_temporal)

    x_woe, mapeo = features.preparar_features(df, target_col="TuvoMoraHistorica")
    assert "VariableSeparadora" not in mapeo
    assert "VariableSeparadora" not in x_woe.columns
    assert "DiasMoraPromedio" in mapeo  # la señal genuina no se ve afectada


def test_transformar_a_woe_es_reproducible_con_el_mismo_mapeo():
    # Criterio de aceptación 8.4: el mismo mapeo de tramos usado en
    # entrenamiento debe aplicarse en el scoring y dar el mismo resultado
    # para el mismo valor de entrada.
    df_entrenamiento = _dataset_sintetico(n=300, seed=1)
    mapeo = features.filtrar_por_iv(features.calcular_mapeo_woe(df_entrenamiento, "TuvoMoraHistorica"))

    x_entrenamiento = features.transformar_a_woe(df_entrenamiento, mapeo)

    # "Scoring" de un cliente nuevo con exactamente el mismo valor que uno
    # ya visto en entrenamiento: debe dar exactamente el mismo WOE.
    cliente_referencia = df_entrenamiento.iloc[[0]]
    x_nuevo = features.transformar_a_woe(cliente_referencia, mapeo)

    assert x_nuevo["DiasMoraPromedio"].iloc[0] == x_entrenamiento["DiasMoraPromedio"].iloc[0]


def test_transformar_a_woe_asigna_tramo_mas_cercano_fuera_de_rango(caplog):
    df_entrenamiento = _dataset_sintetico(n=300, seed=2)
    mapeo = features.filtrar_por_iv(features.calcular_mapeo_woe(df_entrenamiento, "TuvoMoraHistorica"))
    info = mapeo["DiasMoraPromedio"]
    valor_maximo_visto = info["max_observado"]

    cliente_nuevo = pd.DataFrame(
        {"DiasMoraPromedio": [valor_maximo_visto + 500]}, index=pd.Index([9999], name="ClienteId")
    )

    with caplog.at_level(logging.WARNING, logger="scoring_morosidad.features"):
        resultado = features.transformar_a_woe(cliente_nuevo, {"DiasMoraPromedio": info})

    assert "fuera del rango de entrenamiento" in caplog.text
    assert "9999" in caplog.text
    # debe caer en el último tramo real (el más cercano a un valor extremo alto)
    ultimo_tramo_id = max(t["tramo_id"] for t in info["tramos"] if t["tramo_id"] >= 0)
    woe_esperado = next(t["woe"] for t in info["tramos"] if t["tramo_id"] == ultimo_tramo_id)
    assert resultado["DiasMoraPromedio"].iloc[0] == woe_esperado


def test_guardar_y_cargar_mapeo_son_compatibles(tmp_path):
    df = _dataset_sintetico(n=150, seed=3)
    mapeo = features.filtrar_por_iv(features.calcular_mapeo_woe(df, "TuvoMoraHistorica"))

    ruta = tmp_path / "mapeo_prueba.pkl"
    features.guardar_mapeo(mapeo, ruta)
    mapeo_cargado = features.cargar_mapeo(ruta)

    assert set(mapeo_cargado.keys()) == set(mapeo.keys())
    for variable in mapeo:
        assert mapeo_cargado[variable]["iv"] == mapeo[variable]["iv"]


def test_preparar_features_con_el_dataset_de_entrenamiento_real(tmp_path, monkeypatch):
    # Verificación en vivo del ajuste post-8.4 contra leakage: usa
    # dbo.vw_DatasetEntrenamientoFactura (point-in-time, una fila por
    # factura), no dbo.vw_FeaturesScoringCliente (esa es solo para
    # inferencia — ver docstring de src/extract.py). Corre todo el paso
    # 8.4 sin mockear nada del pipeline de features.
    ruta_mapeo = tmp_path / "mapeo_real.pkl"
    config_con_ruta_temporal = {**features._config_features(), "ruta_mapeo": str(ruta_mapeo)}
    monkeypatch.setattr(features, "_config_features", lambda: config_con_ruta_temporal)

    df_real = extract.excluir_sin_historial(extract.extraer_dataset_entrenamiento())
    x_woe, mapeo = features.preparar_features(df_real, target_col="Target")

    assert len(x_woe) == len(df_real)
    assert set(x_woe.columns) == set(mapeo.keys())
    # Prueba de fuga (ajuste post-8.4): con el dataset point-in-time, ya
    # ninguna variable debería quedar en el rango sospechoso de leakage.
    assert all(0.02 <= info["iv"] <= 1.0 for info in mapeo.values())
    assert ruta_mapeo.exists()

    mapeo_recargado = features.cargar_mapeo(ruta_mapeo)
    assert set(mapeo_recargado.keys()) == set(mapeo.keys())


def test_dataset_entrenamiento_real_no_tiene_variables_con_separacion_perfecta():
    # Control obligatorio del documento V1 (sección 4.4), verificado contra
    # los datos reales: ninguna variable point-in-time debería separar
    # perfectamente las clases (eso sería indicio de una fuga no detectada
    # por el umbral de IV).
    df_real = extract.excluir_sin_historial(extract.extraer_dataset_entrenamiento())
    sospechosas = features.detectar_separacion_perfecta(df_real, target_col="Target")
    assert sospechosas == []


def test_dataset_entrenamiento_ya_no_tiene_variables_con_iv_de_fuga():
    # Antes del ajuste (vw_FeaturesScoringCliente, nivel cliente), variables
    # como PorcFacturasEnMora/DiasMoraPromedio salían con IV > 3 porque el
    # target se derivaba de la misma ventana. Con el dataset point-in-time
    # a nivel de factura, sus equivalentes deben quedar en un rango sano.
    df_real = extract.excluir_sin_historial(extract.extraer_dataset_entrenamiento())
    mapeo_completo = features.calcular_mapeo_woe(df_real, target_col="Target")

    variables_de_mora = ["PorcMoraPrevia", "PorcMoraPrevia6M", "DiasMoraPromedioPrevio", "DiasMoraMaximaPrevia"]
    for variable in variables_de_mora:
        assert variable in mapeo_completo
        assert mapeo_completo[variable]["iv"] <= 1.0, (
            f"{variable} tiene IV={mapeo_completo[variable]['iv']:.2f}, "
            f"sospechosamente alto para una variable point-in-time."
        )
