"""
calibracion.py
==============
Responde a la observacion 4 del tutor: "Incorporen métricas de calibración
probabilística".

QUE RESUELVE
------------
El modelo se ajusta sobre un conjunto con 53,3 % de incumplimiento y se aplica
sobre un periodo reciente con 22,5 %. A eso se suma que train.py entrena con
LogisticRegression(class_weight="balanced"), que desplaza el intercepto. El
resultado es que las probabilidades declaradas sobreestiman el riesgo, lo que
explica que la tasa de falsos positivos pase del 56,5 % (validacion cruzada) al
71,0 % (ventana temporal).

Este modulo mide ese desajuste y lo corrige.

PROPIEDAD IMPORTANTE
--------------------
Las tres correcciones son monotonas crecientes, de modo que NO alteran el
ordenamiento de los clientes. AUC y KS quedan intactos: lo unico que cambia es
la escala de la probabilidad y, con ella, cuantos clientes caen a cada lado del
corte. `verificar_monotonia()` lo comprueba de forma explicita.

NO reimplementa nada de train.py ni de score.py: opera sobre el vector de
probabilidades que `modelo.predict_proba(X)[:, 1]` ya produce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-9


# ===========================================================================
# Utilidades
# ===========================================================================
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoide(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _validar(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y y p deben tener la misma longitud: {y.shape} vs {p.shape}")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("y debe contener solo 0 y 1")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p debe estar en [0, 1]")
    return y, p


# ===========================================================================
# Metricas
# ===========================================================================
@dataclass
class MetricasCalibracion:
    """Las seis metricas que el Capstone debe reportar, mas el contexto."""

    n: int
    prevalencia_observada: float
    prevalencia_predicha: float
    brier: float
    brier_skill: float
    pendiente: float
    intercepto: float
    ece: float
    mce: float
    hl_chi2: float | None = None
    hl_p: float | None = None
    calibrador: str = "NINGUNO"

    @property
    def sesgo_prevalencia(self) -> float:
        """Positivo = el modelo sobreestima el riesgo."""
        return self.prevalencia_predicha - self.prevalencia_observada

    def resumen(self) -> str:
        return (
            f"n={self.n}  prev_obs={self.prevalencia_observada:.4f}  "
            f"prev_pred={self.prevalencia_predicha:.4f}  "
            f"sesgo={self.sesgo_prevalencia:+.4f}\n"
            f"Brier={self.brier:.4f}  BrierSkill={self.brier_skill:+.4f}  "
            f"pendiente={self.pendiente:.4f}  intercepto={self.intercepto:+.4f}\n"
            f"ECE={self.ece:.4f}  MCE={self.mce:.4f}  calibrador={self.calibrador}"
        )

    def interpretacion(self) -> list[str]:
        """Lectura en lenguaje llano, lista para la tesis."""
        out = []
        if abs(self.sesgo_prevalencia) < 0.02:
            out.append("La probabilidad media declarada coincide con la frecuencia observada.")
        elif self.sesgo_prevalencia > 0:
            out.append(
                f"El modelo sobreestima el riesgo en {self.sesgo_prevalencia:.2%}: "
                "declara mas incumplimientos de los que ocurren, lo que incrementa "
                "los falsos positivos."
            )
        else:
            out.append(
                f"El modelo subestima el riesgo en {abs(self.sesgo_prevalencia):.2%}: "
                "declara menos incumplimientos de los que ocurren."
            )

        if self.pendiente < 0.8:
            out.append(
                f"La pendiente de {self.pendiente:.3f} indica probabilidades demasiado "
                "extremas: el modelo esta mas seguro de lo que los datos justifican."
            )
        elif self.pendiente > 1.2:
            out.append(
                f"La pendiente de {self.pendiente:.3f} indica probabilidades demasiado "
                "conservadoras: el modelo discrimina mas de lo que su escala refleja."
            )
        else:
            out.append(f"La pendiente de {self.pendiente:.3f} esta dentro del rango aceptable.")

        if self.brier_skill > 0:
            out.append(
                f"La puntuacion de Brier mejora en {self.brier_skill:.1%} respecto de "
                "predecir siempre la prevalencia."
            )
        else:
            out.append(
                "La puntuacion de Brier no mejora respecto de predecir siempre la "
                "prevalencia: revisar antes de usar las probabilidades para decidir."
            )
        return out


def metricas_calibracion(y, p, n_tramos: int = 10, calibrador: str = "NINGUNO") -> MetricasCalibracion:
    """Calcula las metricas de calibracion sobre un conjunto de validacion."""
    y, p = _validar(y, p)
    n = len(y)
    prev_obs = float(y.mean())
    prev_pred = float(p.mean())

    # --- Brier y Brier skill ---
    brier = float(np.mean((p - y) ** 2))
    brier_ref = float(np.mean((prev_obs - y) ** 2))  # predecir siempre la prevalencia
    brier_skill = float(1 - brier / brier_ref) if brier_ref > EPS else 0.0

    # --- Pendiente e intercepto (Cox, 1958) ---
    pendiente, intercepto = _pendiente_intercepto(y, p)

    # --- ECE y MCE ---
    tabla = tabla_fiabilidad(y, p, n_tramos=n_tramos)
    con_casos = tabla[tabla["casos"] > 0]
    pesos = con_casos["casos"] / n
    brecha = (con_casos["prob_media"] - con_casos["frec_observada"]).abs()
    ece = float((pesos * brecha).sum())
    mce = float(brecha.max()) if len(brecha) else 0.0

    # --- Hosmer-Lemeshow ---
    hl_chi2, hl_p = hosmer_lemeshow(y, p, n_grupos=min(10, max(2, n // 10)))

    return MetricasCalibracion(
        n=n,
        prevalencia_observada=prev_obs,
        prevalencia_predicha=prev_pred,
        brier=brier,
        brier_skill=brier_skill,
        pendiente=pendiente,
        intercepto=intercepto,
        ece=ece,
        mce=mce,
        hl_chi2=hl_chi2,
        hl_p=hl_p,
        calibrador=calibrador,
    )


def _pendiente_intercepto(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Regresion logistica de y sobre logit(p).

    Pendiente 1 e intercepto 0 equivalen a calibracion perfecta.
    El intercepto se calcula con la pendiente fijada en 1, que es la convencion
    de 'calibration-in-the-large': mide el desplazamiento puro.
    """
    from sklearn.linear_model import LogisticRegression

    z = _logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    # Pendiente: ambos parametros libres
    mod = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    mod.fit(z, y)
    pendiente = float(mod.coef_[0][0])

    # Intercepto: pendiente fijada en 1 (offset)
    intercepto = _intercepto_con_offset(y, z.ravel())
    return pendiente, intercepto


def _intercepto_con_offset(y: np.ndarray, z: np.ndarray) -> float:
    """Ajusta solo el intercepto manteniendo el logit como offset.

    Newton-Raphson de una dimension: converge en pocas iteraciones.
    """
    a = 0.0
    for _ in range(100):
        pr = _sigmoide(z + a)
        grad = float(np.sum(y - pr))
        hess = float(-np.sum(pr * (1 - pr)))
        if abs(hess) < EPS:
            break
        paso = grad / hess
        a -= paso
        if abs(paso) < 1e-10:
            break
    return float(a)


def tabla_fiabilidad(y, p, n_tramos: int = 10, por_cuantiles: bool = False) -> pd.DataFrame:
    """Tabla que sustenta el diagrama de fiabilidad.

    por_cuantiles=False -> tramos de igual anchura (lectura estandar)
    por_cuantiles=True  -> tramos de igual numero de casos (util con n pequeno)
    """
    y, p = _validar(y, p)

    if por_cuantiles:
        bordes = np.unique(np.quantile(p, np.linspace(0, 1, n_tramos + 1)))
        bordes[0], bordes[-1] = 0.0, 1.0
    else:
        bordes = np.linspace(0.0, 1.0, n_tramos + 1)

    idx = np.clip(np.digitize(p, bordes[1:-1], right=False), 0, len(bordes) - 2)

    filas = []
    for k in range(len(bordes) - 1):
        sel = idx == k
        casos = int(sel.sum())
        filas.append(
            {
                "tramo": k + 1,
                "limite_inferior": float(bordes[k]),
                "limite_superior": float(bordes[k + 1]),
                "casos": casos,
                "positivos": int(y[sel].sum()) if casos else 0,
                "prob_media": float(p[sel].mean()) if casos else 0.0,
                "frec_observada": float(y[sel].mean()) if casos else 0.0,
            }
        )
    tabla = pd.DataFrame(filas)
    tabla["brecha"] = tabla["prob_media"] - tabla["frec_observada"]
    return tabla


def hosmer_lemeshow(y, p, n_grupos: int = 10) -> tuple[float | None, float | None]:
    """Prueba de bondad de ajuste sobre grupos de igual tamano.

    Un valor p alto indica que no se detecta falta de ajuste. Con n pequeno la
    prueba tiene poca potencia, de modo que debe leerse junto al ECE.
    """
    try:
        from scipy.stats import chi2
    except ImportError:
        return None, None

    y, p = _validar(y, p)
    if len(y) < n_grupos * 2:
        return None, None

    orden = np.argsort(p)
    grupos = np.array_split(orden, n_grupos)

    chi_cuadrado = 0.0
    for g in grupos:
        if len(g) == 0:
            continue
        obs1 = float(y[g].sum())
        esp1 = float(p[g].sum())
        obs0 = len(g) - obs1
        esp0 = len(g) - esp1
        if esp1 > EPS:
            chi_cuadrado += (obs1 - esp1) ** 2 / esp1
        if esp0 > EPS:
            chi_cuadrado += (obs0 - esp0) ** 2 / esp0

    gl = max(1, n_grupos - 2)
    return float(chi_cuadrado), float(1 - chi2.cdf(chi_cuadrado, gl))


# ===========================================================================
# Calibradores
# ===========================================================================
@dataclass
class CalibradorPrior:
    """Corrige el desplazamiento por cambio de prevalencia.

    Es el calibrador indicado para este proyecto: el modelo se entrena sobre una
    poblacion con una proporcion de incumplimiento y se aplica sobre otra. No
    necesita un conjunto de calibracion, solo ambas prevalencias.
    """

    prevalencia_entrenamiento: float
    prevalencia_objetivo: float
    ajuste_logit: float
    tipo: str = "PRIOR"

    def transformar(self, p):
        return _sigmoide(_logit(p) + self.ajuste_logit)


def ajustar_prior(prevalencia_entrenamiento: float, prevalencia_objetivo: float) -> CalibradorPrior:
    """Calcula el desplazamiento del intercepto entre dos prevalencias.

    Para los valores reales del proyecto (0,533 y 0,225) el ajuste es -1,3690.
    """
    for nombre, v in (("entrenamiento", prevalencia_entrenamiento), ("objetivo", prevalencia_objetivo)):
        if not 0 < v < 1:
            raise ValueError(f"La prevalencia de {nombre} debe estar en (0, 1): {v}")

    odds_ent = prevalencia_entrenamiento / (1 - prevalencia_entrenamiento)
    odds_obj = prevalencia_objetivo / (1 - prevalencia_objetivo)
    return CalibradorPrior(
        prevalencia_entrenamiento=float(prevalencia_entrenamiento),
        prevalencia_objetivo=float(prevalencia_objetivo),
        ajuste_logit=float(np.log(odds_obj / odds_ent)),
    )


@dataclass
class CalibradorPlatt:
    """Regresion logistica sobre el logit. Corrige pendiente e intercepto."""

    a: float
    b: float
    tipo: str = "PLATT"

    def transformar(self, p):
        return _sigmoide(self.a * _logit(p) + self.b)


def ajustar_platt(y_cal, p_cal) -> CalibradorPlatt:
    """Ajusta sobre un conjunto de calibracion INDEPENDIENTE del de validacion."""
    from sklearn.linear_model import LogisticRegression

    y_cal, p_cal = _validar(y_cal, p_cal)
    if len(np.unique(y_cal)) < 2:
        raise ValueError("El conjunto de calibracion necesita ambas clases")

    z = _logit(p_cal).reshape(-1, 1)
    mod = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    mod.fit(z, y_cal)
    return CalibradorPlatt(a=float(mod.coef_[0][0]), b=float(mod.intercept_[0]))


@dataclass
class CalibradorIsotonico:
    """Regresion isotonica. Mas flexible, pero exige mas datos.

    Con 40 observaciones de validacion es probable que sobreajuste; se incluye
    por completitud y para cuando el historial crezca.
    """

    modelo: object
    tipo: str = "ISOTONICO"

    def transformar(self, p):
        return np.asarray(self.modelo.predict(np.asarray(p, dtype=float).ravel()))


def ajustar_isotonico(y_cal, p_cal) -> CalibradorIsotonico:
    from sklearn.isotonic import IsotonicRegression

    y_cal, p_cal = _validar(y_cal, p_cal)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    return CalibradorIsotonico(modelo=iso)


# ===========================================================================
# Verificacion
# ===========================================================================
def verificar_monotonia(p, calibrador, tol: float = 1e-9) -> bool:
    """Comprueba que la correccion no altera el ordenamiento.

    Es la garantia de que AUC y KS no cambian. Debe ejecutarse siempre antes de
    promover un calibrador: una correccion que reordena a los clientes
    invalidaria las metricas ya reportadas en la tesis.
    """
    p = np.asarray(p, dtype=float).ravel()
    if len(p) < 2:
        return True
    orden = np.argsort(p)
    p_cal = np.asarray(calibrador.transformar(p)).ravel()[orden]
    return bool(np.all(np.diff(p_cal) >= -tol))


def comparar_antes_despues(y, p, calibrador, n_tramos: int = 10) -> pd.DataFrame:
    """Tabla de una sola lectura: metricas antes y despues de calibrar."""
    y, p = _validar(y, p)
    p_cal = np.asarray(calibrador.transformar(p)).ravel()

    antes = metricas_calibracion(y, p, n_tramos=n_tramos, calibrador="NINGUNO")
    despues = metricas_calibracion(y, p_cal, n_tramos=n_tramos, calibrador=calibrador.tipo)

    campos = [
        ("Prevalencia predicha", "prevalencia_predicha"),
        ("Sesgo de prevalencia", "sesgo_prevalencia"),
        ("Brier", "brier"),
        ("Brier skill", "brier_skill"),
        ("Pendiente", "pendiente"),
        ("Intercepto", "intercepto"),
        ("ECE", "ece"),
        ("MCE", "mce"),
    ]
    filas = []
    for etiqueta, campo in campos:
        va, vd = getattr(antes, campo), getattr(despues, campo)
        filas.append({"metrica": etiqueta, "antes": va, "despues": vd, "variacion": vd - va})
    return pd.DataFrame(filas)


def guardar_calibrador(calibrador, ruta: str | Path) -> Path:
    """Persiste el calibrador junto al modelo, para que score.py lo aplique."""
    import joblib

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrador, ruta)

    # Copia legible, util para la auditoria y para el anexo de la tesis
    meta = {k: v for k, v in asdict(calibrador).items() if k != "modelo"} \
        if hasattr(calibrador, "__dataclass_fields__") else {"tipo": calibrador.tipo}
    ruta.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def cargar_calibrador(ruta: str | Path):
    import joblib

    return joblib.load(Path(ruta))


# ===========================================================================
# Parche 2026-10-28 — corrección de intercepto completa + evaluación honesta
# ===========================================================================
# La corrida real del 28/10 mostró que el prior shift (-1,3689) solo cubría
# el 65,9 % del ajuste necesario (-2,0778): el desajuste no es solo el cambio
# de prevalencia, también el efecto de class_weight="balanced". Y que
# calibrar sobre los 40 casos de la ventana temporal (9 incumplimientos, por
# debajo del mínimo de 10 que el propio pipeline exige) es una base chica —
# la validación cruzada agrupada por cliente da 197 con 105.


# ---------------------------------------------------------------------------
# 1. Calibrador de intercepto por maxima verosimilitud
# ---------------------------------------------------------------------------
@dataclass
class CalibradorIntercepto:
    """Desplaza el logit para maximizar la verosimilitud del conjunto.

    A diferencia de CalibradorPrior, no supone que el unico desajuste provenga
    del cambio de prevalencia: captura tambien el efecto de la ponderacion
    equilibrada de clases y cualquier otro desplazamiento sistematico.

    Un solo parametro, de modo que es la opcion mas segura cuando el conjunto
    de calibracion es pequeno.
    """

    ajuste_logit: float
    n_ajuste: int = 0
    tipo: str = "INTERCEPTO_MV"

    def transformar(self, p):
        return _sigmoide(_logit(p) + self.ajuste_logit)


def ajustar_intercepto_mv(y, p) -> CalibradorIntercepto:
    """Calcula el desplazamiento optimo del intercepto.

    Reutiliza _intercepto_con_offset, que ya resuelve la misma ecuacion para
    la metrica de calibracion: el valor que devuelve metricas_calibracion()
    como `intercepto` es exactamente la correccion que hay que aplicar.
    """
    y, p = _validar(y, p)
    if len(np.unique(y)) < 2:
        raise ValueError("Se necesitan ambas clases para ajustar el intercepto")
    ajuste = _intercepto_con_offset(y, _logit(p))
    return CalibradorIntercepto(ajuste_logit=float(ajuste), n_ajuste=len(y))


# ---------------------------------------------------------------------------
# 2. Predicciones fuera de muestra de la validacion cruzada
# ---------------------------------------------------------------------------
def predicciones_fuera_de_muestra(
    df: pd.DataFrame,
    target_col: str,
    grupo_col: str = "ClienteId",
    n_particiones: int = 5,
    ajustar_fn=None,
    aplicar_fn=None,
    entrenar_fn=None,
    semilla: int = 42,
) -> pd.DataFrame:
    """Reproduce la validacion cruzada agrupada devolviendo cada prediccion.

    Mismo criterio que evaluacion.evaluar_validacion_cruzada: particion por
    cliente, nunca aleatoria por fila, con reajuste completo de los artefactos
    dentro de cada particion.

    Parametros
    ----------
    ajustar_fn, aplicar_fn, entrenar_fn
        Inyectados para no acoplar este modulo a features.py ni a train.py.
        Por defecto usa features_particion.ajustar_artefactos /
        aplicar_artefactos y train.entrenar_regresion_logistica.

    Devuelve
    --------
    DataFrame con columnas: indice original, grupo, particion, y, p
    """
    from sklearn.model_selection import GroupKFold

    if ajustar_fn is None or aplicar_fn is None:
        from . import features_particion as fp
        ajustar_fn = ajustar_fn or fp.ajustar_artefactos
        aplicar_fn = aplicar_fn or fp.aplicar_artefactos
    if entrenar_fn is None:
        from . import train
        entrenar_fn = train.entrenar_regresion_logistica

    if grupo_col not in df.columns:
        if df.index.name == grupo_col:
            grupos = df.index.to_numpy()
        else:
            raise ValueError(f"No se encuentra la columna de agrupacion '{grupo_col}'")
    else:
        grupos = df[grupo_col].to_numpy()

    n_grupos = len(np.unique(grupos))
    n_particiones = int(min(n_particiones, n_grupos))
    if n_particiones < 2:
        raise ValueError(f"Se necesitan al menos 2 grupos distintos, hay {n_grupos}")

    excluidas = {target_col, grupo_col}
    candidatas = [c for c in df.columns if c not in excluidas]

    filas = []
    gkf = GroupKFold(n_splits=n_particiones)
    for k, (i_tr, i_va) in enumerate(gkf.split(df, df[target_col], groups=grupos), start=1):
        tr, va = df.iloc[i_tr], df.iloc[i_va]
        if tr[target_col].nunique() < 2:
            continue

        art = ajustar_fn(tr, target_col, candidatas=candidatas) \
            if _acepta_candidatas(ajustar_fn) else ajustar_fn(tr, target_col)
        X_tr, X_va = aplicar_fn(tr, art), aplicar_fn(va, art)

        modelo = entrenar_fn(X_tr, tr[target_col])
        p = modelo.predict_proba(X_va)[:, 1]

        for pos, idx in enumerate(va.index):
            filas.append({
                "indice": idx,
                "grupo": grupos[i_va[pos]],
                "particion": k,
                "y": int(va[target_col].iloc[pos]),
                "p": float(p[pos]),
            })

    if not filas:
        raise RuntimeError("Ninguna particion produjo predicciones utilizables")
    return pd.DataFrame(filas)


def _acepta_candidatas(fn) -> bool:
    import inspect

    try:
        return "candidatas" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 3. Evaluacion honesta del calibrador
# ---------------------------------------------------------------------------
def evaluar_calibrador_honesto(
    y,
    p,
    constructor,
    grupos=None,
    n_particiones: int = 5,
    semilla: int = 42,
) -> dict:
    """Estima el desempeno del calibrador FUERA del conjunto donde se ajusta.

    Es imprescindible para comparar con justicia: un calibrador de dos
    parametros ajustado y evaluado sobre las mismas observaciones siempre
    parece mejor de lo que es.

    `constructor` recibe (y_train, p_train) y devuelve un calibrador.
    """
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    y, p = _validar(y, p)
    n = len(y)

    if grupos is not None:
        grupos = np.asarray(grupos).ravel()
        k = int(min(n_particiones, len(np.unique(grupos))))
        if k < 2:
            return {"error": "grupos insuficientes", "brier": np.nan, "brier_skill": np.nan}
        particiones = GroupKFold(n_splits=k).split(np.zeros(n), y, groups=grupos)
    else:
        k = int(min(n_particiones, np.bincount(y).min()))
        if k < 2:
            return {"error": "clase minoritaria insuficiente", "brier": np.nan, "brier_skill": np.nan}
        particiones = StratifiedKFold(n_splits=k, shuffle=True,
                                      random_state=semilla).split(np.zeros(n), y)

    p_oof = np.full(n, np.nan)
    fallos = 0
    for i_tr, i_va in particiones:
        if len(np.unique(y[i_tr])) < 2:
            fallos += 1
            continue
        try:
            cal_k = constructor(y[i_tr], p[i_tr])
            p_oof[i_va] = np.asarray(cal_k.transformar(p[i_va])).ravel()
        except Exception:  # noqa: BLE001
            fallos += 1

    validos = ~np.isnan(p_oof)
    if validos.sum() < 10:
        return {"error": "muy pocas predicciones validas", "brier": np.nan, "brier_skill": np.nan}

    yv, pv = y[validos], p_oof[validos]
    prev = float(yv.mean())
    brier = float(np.mean((pv - yv) ** 2))
    brier_ref = float(prev * (1 - prev))
    m = metricas_calibracion(yv, pv)

    return {
        "n_evaluadas": int(validos.sum()),
        "particiones_fallidas": fallos,
        "brier": brier,
        "brier_skill": float(1 - brier / brier_ref) if brier_ref > EPS else np.nan,
        "ece": m.ece,
        "sesgo": m.sesgo_prevalencia,
        "pendiente": m.pendiente,
    }


# ---------------------------------------------------------------------------
# 4. Comparacion y recomendacion
# ---------------------------------------------------------------------------
def comparar_calibradores(
    y,
    p,
    prevalencia_entrenamiento: float | None = None,
    grupos=None,
) -> pd.DataFrame:
    """Compara las cuatro opciones con evaluacion dentro y fuera de muestra.

    La columna `brier_skill_oof` es la que decide: las demas son optimistas
    para los calibradores con mas de un parametro.
    """
    y, p = _validar(y, p)
    prev_obs = float(y.mean())

    opciones: list[tuple[str, object]] = [
        ("NINGUNO", _CalibradorIdentidad()),
        ("INTERCEPTO_MV", ajustar_intercepto_mv(y, p)),
    ]
    if prevalencia_entrenamiento is not None:
        opciones.insert(1, ("PRIOR", ajustar_prior(prevalencia_entrenamiento, prev_obs)))
    try:
        opciones.append(("PLATT", ajustar_platt(y, p)))
    except Exception:  # noqa: BLE001
        pass
    try:
        opciones.append(("ISOTONICO", ajustar_isotonico(y, p)))
    except Exception:  # noqa: BLE001
        pass

    constructores = {
        "NINGUNO": lambda yy, pp: _CalibradorIdentidad(),
        "PRIOR": (lambda yy, pp: ajustar_prior(prevalencia_entrenamiento, float(yy.mean())))
        if prevalencia_entrenamiento is not None else None,
        "INTERCEPTO_MV": ajustar_intercepto_mv,
        "PLATT": ajustar_platt,
        "ISOTONICO": ajustar_isotonico,
    }

    filas = []
    for nombre, c in opciones:
        p_cal = np.asarray(c.transformar(p)).ravel()
        m = metricas_calibracion(y, p_cal, calibrador=nombre)
        monotono = verificar_monotonia(p, c)

        ctor = constructores.get(nombre)
        oof = evaluar_calibrador_honesto(y, p, ctor, grupos=grupos) if ctor else {}

        filas.append({
            "calibrador": nombre,
            "parametros": _n_parametros(nombre),
            "monotono": monotono,
            "sesgo": m.sesgo_prevalencia,
            "brier": m.brier,
            "brier_skill": m.brier_skill,
            "ece": m.ece,
            "pendiente": m.pendiente,
            "brier_skill_oof": oof.get("brier_skill", np.nan),
            "ece_oof": oof.get("ece", np.nan),
        })
    return pd.DataFrame(filas)


def _n_parametros(nombre: str) -> int | str:
    return {"NINGUNO": 0, "PRIOR": 0, "INTERCEPTO_MV": 1,
            "PLATT": 2, "ISOTONICO": "variable"}.get(nombre, "?")


class _CalibradorIdentidad:
    tipo = "NINGUNO"

    def transformar(self, p):
        return np.asarray(p, dtype=float).ravel()


def recomendar_calibrador(
    y,
    p,
    prevalencia_entrenamiento: float | None = None,
    grupos=None,
    min_positivos_para_platt: int = 30,
) -> tuple[object, pd.DataFrame, list[str]]:
    """Elige el calibrador con un criterio explicito y documentado.

    Criterio, en orden:
      1. Debe ser monotono. Si no, se descarta sin mas.
      2. Con menos de `min_positivos_para_platt` incumplimientos, se limita a
         calibradores de un parametro: dos parametros sobre pocos positivos
         sobreajustan.
      3. Entre los admisibles, gana el mayor Brier skill FUERA de muestra.
      4. A igualdad practica (diferencia menor a 0,01), gana el mas simple.

    Devuelve (calibrador, tabla_comparativa, justificacion)
    """
    y, p = _validar(y, p)
    n_pos = int(y.sum())
    tabla = comparar_calibradores(y, p, prevalencia_entrenamiento, grupos)

    notas = [f"Conjunto de calibración: {len(y)} observaciones, {n_pos} incumplimientos."]

    admisibles = tabla[tabla["monotono"]].copy()
    if len(admisibles) < len(tabla):
        descartados = ", ".join(tabla.loc[~tabla["monotono"], "calibrador"])
        notas.append(f"Descartados por no ser monótonos: {descartados}.")

    if n_pos < min_positivos_para_platt:
        antes = set(admisibles["calibrador"])
        admisibles = admisibles[admisibles["parametros"].isin([0, 1])]
        quitados = antes - set(admisibles["calibrador"])
        if quitados:
            notas.append(
                f"Con {n_pos} incumplimientos (mínimo {min_positivos_para_platt}) se "
                f"restringe a calibradores de un parámetro. Excluidos: {', '.join(sorted(quitados))}."
            )

    con_oof = admisibles.dropna(subset=["brier_skill_oof"])
    if len(con_oof):
        orden = con_oof.sort_values("brier_skill_oof", ascending=False)
        criterio = "Brier skill fuera de muestra"
    else:
        orden = admisibles.sort_values("brier_skill", ascending=False)
        criterio = "Brier skill dentro de muestra (sin estimación fuera de muestra disponible)"

    mejor = orden.iloc[0]
    col = "brier_skill_oof" if criterio.startswith("Brier skill fuera") else "brier_skill"

    empates = orden[orden[col] >= mejor[col] - 0.01]
    if len(empates) > 1:
        simples = empates[empates["parametros"] != "variable"]
        if len(simples):
            mejor = simples.sort_values("parametros").iloc[0]
            notas.append("Varias opciones empatan; se elige la de menor número de parámetros.")

    nombre = mejor["calibrador"]
    notas.append(f"Criterio de selección: {criterio}.")
    notas.append(
        f"Seleccionado: {nombre} "
        f"(sesgo {mejor['sesgo']:+.4f}, Brier skill {mejor[col]:+.4f}, ECE {mejor['ece']:.4f})."
    )
    if mejor[col] < 0:
        notas.append(
            "ADVERTENCIA: el Brier skill sigue siendo negativo. Las probabilidades "
            "ordenan correctamente pero no deben emplearse para decidir con umbrales "
            "de costo. Declararlo así en la memoria."
        )

    prev_obs = float(y.mean())
    construir = {
        "NINGUNO": lambda: _CalibradorIdentidad(),
        "PRIOR": lambda: ajustar_prior(prevalencia_entrenamiento, prev_obs),
        "INTERCEPTO_MV": lambda: ajustar_intercepto_mv(y, p),
        "PLATT": lambda: ajustar_platt(y, p),
        "ISOTONICO": lambda: ajustar_isotonico(y, p),
    }
    return construir[nombre](), tabla, notas
