from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold

from competition.feature_engineering import (
    add_house_age,
    add_total_sf,
)
from competition.log_target_oof import (
    run_catboost_log_target_oof,
)
from competition.oof import calculate_metrics
from src.split import (
    load_training_data,
    make_holdout_split,
)

CV_SEEDS = [
    42,
    52,
    62,
]

N_SPLITS = 5


def add_total_sf_house_age(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = add_total_sf(dataframe)

    dataframe = add_house_age(dataframe)

    return dataframe


def evaluate(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    seed: int,
    experiment_name: str,
) -> dict[str, float | int | str]:

    cross_validation = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    splits = list(cross_validation.split(X_dev))

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
        "seed": seed,
        "experiment": experiment_name,
        **metrics,
    }


def main() -> None:
    X, y = load_training_data()

    X_dev, _, y_dev, _ = make_holdout_split(
        X,
        y,
    )

    experiments = {
        "baseline": (X_dev.copy()),
        "total_sf_house_age": (add_total_sf_house_age(X_dev)),
    }

    results = []

    for seed in CV_SEEDS:
        print()
        print("=" * 60)

        print(f"CV seed: {seed}")

        print("=" * 60)

        for (
            experiment_name,
            experiment_dataframe,
        ) in experiments.items():
            print()
            print(f"Experiment: {experiment_name}")

            result = evaluate(
                X_dev=experiment_dataframe,
                y_dev=y_dev,
                seed=seed,
                experiment_name=experiment_name,
            )

            results.append(result)

            print(f"GLOBAL Log RMSE: {result['log_rmse']:.6f}")

    results_dataframe = pd.DataFrame(results)

    print()
    print("===== Results by Seed =====")

    print(
        results_dataframe[
            [
                "seed",
                "experiment",
                "log_rmse",
            ]
        ].to_string(index=False)
    )

    summary = (
        results_dataframe.groupby("experiment")
        .agg(
            mean_log_rmse=(
                "log_rmse",
                "mean",
            ),
            std_log_rmse=(
                "log_rmse",
                "std",
            ),
            best_log_rmse=(
                "log_rmse",
                "min",
            ),
            worst_log_rmse=(
                "log_rmse",
                "max",
            ),
        )
        .reset_index()
        .sort_values("mean_log_rmse")
    )

    print()
    print("===== Stability Summary =====")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
