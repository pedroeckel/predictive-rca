from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from bayes_opt import BayesianOptimization


def optimize_bayesian(
    evaluate_fn: Callable[..., float],
    pbounds: Dict[str, Tuple[float, float]],
    init_points: int = 8,
    n_iter: int = 20,
    random_state: int = 42,
    verbose: int = 2,
) -> Dict[str, float]:
    """
    Executa otimização Bayesiana genérica.

    Parameters
    ----------
    evaluate_fn : Callable[..., float]
        Função objetivo a ser maximizada.
    pbounds : Dict[str, Tuple[float, float]]
        Limites dos hiperparâmetros contínuos.
    init_points : int
        Número de pontos aleatórios iniciais.
    n_iter : int
        Número de iterações da otimização.
    random_state : int
        Semente.
    verbose : int
        Nível de verbosidade do BayesianOptimization.

    Returns
    -------
    Dict[str, float]
        Dicionário com os melhores hiperparâmetros encontrados.
    """
    optimizer = BayesianOptimization(
        f=evaluate_fn,
        pbounds=pbounds,
        random_state=random_state,
        verbose=verbose,
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_params: Dict[str, float] = optimizer.max["params"]
    return best_params
