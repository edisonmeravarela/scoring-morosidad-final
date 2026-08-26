import pandas as pd
import pytest

from src import rules


def _catalogo_segmentos_prueba() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SegmentoRiesgoId": [1, 2, 3, 4],
            "Codigo": ["BAJO", "MEDIO", "ALTO", "CRITICO"],
            "LimiteCredito": [5000, 2500, 1000, 0],
            "DiasMaxMora": [15, 30, 45, 0],
            "BloqueaVentas": [False, False, False, True],
        }
    ).set_index("Codigo")


def _catalogo_plazos_prueba() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PlazoPagoId": [1, 2, 3, 4, 5, 6, 7],
            "Nombre": ["CONTADO", "15 DÍAS", "30 DÍAS", "45 DÍAS", "60 DÍAS", "90 DÍAS", "120 DÍAS"],
        }
    ).set_index("Nombre")


def _config_reglas_prueba(**overrides) -> dict:
    base = {
        "cortes_score": [
            {"hasta": 300, "segmento": "CRITICO"},
            {"hasta": 500, "segmento": "ALTO"},
            {"hasta": 700, "segmento": "MEDIO"},
            {"hasta": None, "segmento": "BAJO"},
        ],
        "dias_mora_critico": 90,
        "minimo_facturas_para_evaluar": 3,
        "plazo_por_segmento": {
            "BAJO": "60 DÍAS",
            "MEDIO": "30 DÍAS",
            "ALTO": "15 DÍAS",
            "CRITICO": "CONTADO",
            "NUEVO": "CONTADO",
        },
        "rfm": {
            "recencia_dias_buena": 30,
            "recencia_dias_regular": 90,
            "frecuencia_alta": 10,
            "frecuencia_regular": 4,
            "monto_alto": 2000,
            "monto_regular": 500,
        },
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _catalogos_de_prueba(monkeypatch):
    monkeypatch.setattr(rules, "_catalogo_segmentos", _catalogo_segmentos_prueba)
    monkeypatch.setattr(rules, "_catalogo_plazos", _catalogo_plazos_prueba)
    monkeypatch.setattr(rules, "_config_reglas", _config_reglas_prueba)


# ---------- asignar_segmento: cortes de score ----------


@pytest.mark.parametrize(
    "score,segmento_id_esperado",
    [
        (0, 4),  # CRITICO
        (300, 4),  # límite inferior exacto -> aún CRITICO
        (301, 3),  # ALTO
        (415, 3),  # ejemplo del documento (sección 3.3): 415 -> ALTO
        (500, 3),  # límite exacto -> aún ALTO
        (501, 2),  # MEDIO
        (700, 2),  # límite exacto -> aún MEDIO
        (701, 1),  # BAJO
        (1000, 1),  # BAJO, sin techo
    ],
)
def test_asignar_segmento_cortes_exactos(score, segmento_id_esperado):
    assert rules.asignar_segmento(score) == segmento_id_esperado


# ---------- aplicar_overrides ----------


def test_aplicar_overrides_sin_condiciones_mantiene_el_segmento():
    cliente = {"DiasMoraMaxima": 48, "NumFacturas": 7}
    assert rules.aplicar_overrides(cliente, segmento_id=3) == "ALTO"  # ejemplo del documento


def test_aplicar_overrides_mora_en_el_limite_no_fuerza_critico():
    cliente = {"DiasMoraMaxima": 90, "NumFacturas": 7}
    assert rules.aplicar_overrides(cliente, segmento_id=2) == "MEDIO"


def test_aplicar_overrides_mora_supera_el_limite_fuerza_critico():
    cliente = {"DiasMoraMaxima": 91, "NumFacturas": 7}
    assert rules.aplicar_overrides(cliente, segmento_id=1) == "CRITICO"


def test_aplicar_overrides_facturas_en_el_minimo_no_fuerza_nuevo():
    cliente = {"DiasMoraMaxima": 0, "NumFacturas": 3}
    assert rules.aplicar_overrides(cliente, segmento_id=1) == "BAJO"


def test_aplicar_overrides_menos_del_minimo_de_facturas_fuerza_nuevo():
    cliente = {"DiasMoraMaxima": 0, "NumFacturas": 2}
    assert rules.aplicar_overrides(cliente, segmento_id=1) == rules.NUEVO


def test_aplicar_overrides_mora_tiene_prioridad_sobre_sin_historial():
    cliente = {"DiasMoraMaxima": 91, "NumFacturas": 1}
    assert rules.aplicar_overrides(cliente, segmento_id=1) == "CRITICO"


# ---------- recomendar_plazo ----------


@pytest.mark.parametrize(
    "segmento,plazo_id_esperado",
    [("BAJO", 5), ("MEDIO", 3), ("ALTO", 2), ("CRITICO", 1), (rules.NUEVO, 1)],
)
def test_recomendar_plazo_por_segmento(segmento, plazo_id_esperado):
    assert rules.recomendar_plazo(segmento) == plazo_id_esperado


def test_recomendar_plazo_segmento_desconocido_lanza_error():
    with pytest.raises(ValueError):
        rules.recomendar_plazo("INEXISTENTE")


# ---------- clasificar_cliente (RFM) ----------


def test_clasificar_cliente_sin_historial_es_nuevo():
    cliente = {"NumFacturas": 2, "DiasDesdeUltimaFactura": 1, "ValorFacturaPromedio": 5000}
    assert rules.clasificar_cliente(cliente) == rules.NUEVO


def test_clasificar_cliente_rfm_alto_es_a():
    cliente = {"NumFacturas": 12, "DiasDesdeUltimaFactura": 10, "ValorFacturaPromedio": 3000}
    assert rules.clasificar_cliente(cliente) == "A"


def test_clasificar_cliente_rfm_medio_es_b():
    # Recencia regular (2) + frecuencia regular (2) + monto regular (2) = 6 -> B.
    # Mismo caso que el ejemplo del documento (7 facturas, monto medio) -> "B".
    cliente = {"NumFacturas": 7, "DiasDesdeUltimaFactura": 60, "ValorFacturaPromedio": 1000}
    assert rules.clasificar_cliente(cliente) == "B"


def test_clasificar_cliente_rfm_bajo_es_c():
    cliente = {"NumFacturas": 4, "DiasDesdeUltimaFactura": 200, "ValorFacturaPromedio": 100}
    assert rules.clasificar_cliente(cliente) == "C"


def test_clasificar_cliente_recencia_ausente_no_falla():
    cliente = {"NumFacturas": 5, "ValorFacturaPromedio": 1000}
    assert rules.clasificar_cliente(cliente) in {"A", "B", "C"}


# ---------- marcar_revision ----------


def test_marcar_revision_dentro_de_limites_no_marca():
    cliente = {"SaldoActualTotal": 1000, "DiasMoraMaxima": 45}
    assert rules.marcar_revision(cliente, "ALTO") is False  # límite exacto, no lo supera


def test_marcar_revision_saldo_excede_limite():
    cliente = {"SaldoActualTotal": 1430, "DiasMoraMaxima": 10}
    assert rules.marcar_revision(cliente, "ALTO") is True


def test_marcar_revision_mora_excede_tolerancia():
    cliente = {"SaldoActualTotal": 0, "DiasMoraMaxima": 46}
    assert rules.marcar_revision(cliente, "ALTO") is True


def test_marcar_revision_reproduce_ejemplo_del_documento():
    # Sección 3.3: saldo 1.430 > límite 1.000 de ALTO, y mora 48 > tolerancia 45.
    cliente = {"SaldoActualTotal": 1430, "DiasMoraMaxima": 48}
    assert rules.marcar_revision(cliente, "ALTO") is True


def test_marcar_revision_nuevo_usa_el_tope_de_critico():
    cliente = {"SaldoActualTotal": 1, "DiasMoraMaxima": 0}
    assert rules.marcar_revision(cliente, rules.NUEVO) is True  # CRITICO: límite 0


# ---------- resolver_segmento_final / resolver_clasificacion_id (paso 8.8) ----------


@pytest.mark.parametrize("codigo,id_esperado", [("BAJO", 1), ("MEDIO", 2), ("ALTO", 3), ("CRITICO", 4)])
def test_resolver_segmento_final_segmentos_reales(codigo, id_esperado):
    assert rules.resolver_segmento_final(segmento_id_asignado=99, segmento_codigo=codigo) == id_esperado


def test_resolver_segmento_final_nuevo_conserva_el_segmento_original_por_score():
    # NUEVO no tiene fila en el catálogo; a diferencia de marcar_revision,
    # aquí NO cae en CRITICO (eso implicaría BloqueaVentas=1) sino que se
    # conserva el segmento que asignar_segmento() calculó con el score.
    assert rules.resolver_segmento_final(segmento_id_asignado=1, segmento_codigo=rules.NUEVO) == 1


@pytest.fixture
def catalogo_clasificaciones_prueba(monkeypatch):
    df = pd.DataFrame({"ClasificacionId": [1, 2, 3, 4], "Codigo": ["A", "B", "C", "NUEVO"]}).set_index("Codigo")
    monkeypatch.setattr(rules, "_catalogo_clasificaciones", lambda: df)


@pytest.mark.parametrize("codigo,id_esperado", [("A", 1), ("B", 2), ("C", 3), ("NUEVO", 4)])
def test_resolver_clasificacion_id(catalogo_clasificaciones_prueba, codigo, id_esperado):
    assert rules.resolver_clasificacion_id(codigo) == id_esperado


# ---------- Verificación en vivo: reproduce el ejemplo completo del documento ----------


def test_flujo_completo_reproduce_ejemplo_del_documento_contra_catalogos_reales(monkeypatch):
    monkeypatch.undo()  # usar los catálogos y la configuración reales, no los de prueba

    cliente = {
        "DiasMoraMaxima": 48,
        "NumFacturas": 7,
        "SaldoActualTotal": 1430,
        "DiasDesdeUltimaFactura": 60,
        "ValorFacturaPromedio": 1000,
    }
    score = 415

    segmento_id = rules.asignar_segmento(score)
    segmento_codigo = rules.aplicar_overrides(cliente, segmento_id)
    plazo_id = rules.recomendar_plazo(segmento_codigo)
    clasificacion = rules.clasificar_cliente(cliente)
    revisar = rules.marcar_revision(cliente, segmento_codigo)

    assert segmento_codigo == "ALTO"
    assert clasificacion == "B"
    assert revisar is True  # saldo y mora superan los topes de ALTO, aunque el segmento no llegó a CRÍTICO

    catalogo_plazos = rules._catalogo_plazos()
    nombre_plazo = catalogo_plazos.index[catalogo_plazos["PlazoPagoId"] == plazo_id][0]
    assert nombre_plazo == "15 DÍAS"  # el documento: "el plazo del cliente baja de 30 a 15 días"
