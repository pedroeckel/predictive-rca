from __future__ import annotations

import argparse
from pathlib import Path

from src.backend.pipeline.pipeline_builder import PipelineBuilder
from src.backend.models.lightgbm_model import LightGBMModel
from src.backend.models.random_forest import RandomForestModel
from src.backend.models.xgboost_model import XGBoostModel
from src.backend.models.catboost_model import CatBoostModel
from src.backend.models.logistic_regression import LogisticRegressionModel

# Registro de modelos disponíveis
AVAILABLE_MODELS = {
    "lightgbm": LightGBMModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
    "logistic_regression": LogisticRegressionModel,
}


def ask_input(prompt: str, default: str = None) -> str:
    """Prompt interativo com valor padrão."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


def select_models_interactive() -> list[str]:
    """Permite selecionar modelos via prompt interativo."""
    print("\nModelos disponíveis:")
    for name in AVAILABLE_MODELS:
        print(f" - {name}")

    models_raw = ask_input(
        "\nInforme os modelos desejados (separados por vírgula)",
        "lightgbm"
    )

    selected = [m.strip().lower() for m in models_raw.split(",")]
    valid = [m for m in selected if m in AVAILABLE_MODELS]

    if not valid:
        print("Nenhum modelo válido selecionado. Usando modelo padrão: lightgbm.")
        return ["lightgbm"]

    return valid


def main():
    parser = argparse.ArgumentParser(
        description="Executa o pipeline de Predictive RCA com modelos configuráveis."
    )

    parser.add_argument(
        "--log",
        type=str,
        help="Caminho do arquivo de log (CSV). Se omitido, será solicitado no prompt."
    )

    parser.add_argument(
        "--models",
        type=str,
        help="Lista de modelos separados por vírgula. Ex: lightgbm,xgboost"
    )

    parser.add_argument(
        "--sla_hours",
        type=float,
        default=None,
        help="Valor de SLA em horas."
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Ativa otimização bayesiana (apenas para LightGBM)."
    )

    args = parser.parse_args()

    # -----------------------------------------
    # Seleção interativa do arquivo do log
    # -----------------------------------------
    if args.log:
        log_path = Path(args.log)
    else:
        log_path = Path(ask_input("Informe o caminho do arquivo CSV do log", "data/raw/event_log.csv"))

    if not log_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {log_path}")

    # -----------------------------------------
    # Seleção interativa dos modelos
    # -----------------------------------------
    if args.models:
        models = [m.strip().lower() for m in args.models.split(",")]
        models = [m for m in models if m in AVAILABLE_MODELS]
        if not models:
            print("\nNenhum modelo válido informado no argumento.")
            models = select_models_interactive()
    else:
        models = select_models_interactive()

    # -----------------------------------------
    # SLA via prompt
    # -----------------------------------------
    if args.sla_hours is not None:
        sla = args.sla_hours
    else:
        sla = float(ask_input("Informe o SLA em horas", "48"))

    optimize = args.optimize

    print("\n======================================")
    print("      EXECUTANDO PIPELINE RCA         ")
    print("======================================")
    print(f"Log: {log_path}")
    print(f"Modelos: {models}")
    print(f"SLA: {sla} horas")
    print(f"Otimização bayesiana: {optimize}")
    print("======================================\n")

    # -----------------------------------------------------
    # EXECUTAR MODELOS
    # -----------------------------------------------------
    for model_name in models:
        model_class = AVAILABLE_MODELS[model_name]

        print(f"\n\n--------------------------------------------")
        print(f" Treinando modelo: {model_name.upper()}")
        print("--------------------------------------------\n")

        pipeline = PipelineBuilder(
            model_class=model_class,
            optimize_hyperparams=optimize if model_name == "lightgbm" else False
        )

        pipeline.run_from_event_log(str(log_path), sla_hours=sla)


if __name__ == "__main__":
    main()
