from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalConfig:
    """Configurações globais do projeto."""

    RANDOM_STATE: int = 42
    SLA_HOURS: float = 12.0

    # Caminhos padrão (ajuste conforme seu projeto)
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    MODELS_DIR: str = "data/models"


CONFIG = GlobalConfig()
