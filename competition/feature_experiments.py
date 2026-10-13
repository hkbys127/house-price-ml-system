from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from competition.feature_engineering import (
    add_house_age,
    add_total_sf,
)
from competition.log_target_oof import (
    run_catboost_log_target_oof,
)
from competition.oof import (
    calculate_metrics,
)
from src.split import (
    RANDOM_STATE,
    load_training_data,
    make_holdout_split,
)

BASE_DIR = Path(__file__).resolve().parents[1]

RESULT_PATH = BASE_DIR / "reports" / "feature_experiment_results.csv"
N_SPLITS = 5


def evaluate_experiment(
    experiment_name: str,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float | str]:

    predictions, _ = run_catboost_log_target_oof(
        X_dev=X_dev,
        y_dev=y_dev,
        splits=splits,
    )

    metrics = calculate_metrics(
        y_dev,
        predictions,
    )

    return {
        "experiment": experiment_name,
        **metrics,
    }


def main() -> None:
    X, y = load_training_data()

    X_dev, _, y_dev, _ = make_holdout_split(
        X,
        y,
    )

    cross_validation = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    splits = list(cross_validation.split(X_dev))

    experiments = {
        "baseline": (X_dev.copy()),
        "add_total_sf": (add_total_sf(X_dev)),
        "add_house_age": (add_house_age(X_dev)),
        "add_total_sf_house_age": (add_house_age(add_total_sf(X_dev))),
    }

    results = []

    for (
        experiment_name,
        experiment_dataframe,
    ) in experiments.items():
        print()
        print("=" * 60)

        print(f"Experiment: {experiment_name}")

        print("=" * 60)

        result = evaluate_experiment(
            experiment_name,
            experiment_dataframe,
            y_dev,
            splits,
        )

        results.append(result)

        print()
        print(f"GLOBAL Log RMSE: {result['log_rmse']:.5f}")

    results_dataframe = pd.DataFrame(results).sort_values("log_rmse")

    print()
    print("===== Feature Experiment Comparison =====")

    print(
        results_dataframe[
            [
                "experiment",
                "log_rmse",
                "rmse",
                "mae",
                "r2",
            ]
        ].to_string(index=False)
    )
    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        RESULT_PATH,
        index=False,
    )


if __name__ == "__main__":
    main()
