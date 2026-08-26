"""Carga de config/config.yaml (parámetros no sensibles del pipeline).

Las credenciales de base de datos NUNCA viven aquí ni en el YAML: se leen
de variables de entorno en src/db.py (ver .env.example).
"""

from pathlib import Path

import yaml

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
_CONFIG_PATH = RAIZ_PROYECTO / "config" / "config.yaml"


def cargar_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
