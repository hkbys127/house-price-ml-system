from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluate import evaluate_model
from src.models import get_models
from src.split import (
    load_training_data,
    make_holdout_split,
)

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_PATH = BASE_DIR / "reports" / "model_comparison.csv"


def main() -> None:
    X, y = load_training_data()

    X_dev, X_holdout, y_dev, y_holdout = make_holdout_split(
        X,
        y,
    )

    print(f"Development rows: {len(X_dev)}")

    print(f"Holdout rows:     {len(X_holdout)}")

    print("Holdout is reserved and will not be used for model comparison.")

    print()

    models = get_models()

    experiment_results = []

    for model_name, model in models.items():
        print(f"Evaluating: {model_name}")

        result = evaluate_model(
            model_name=model_name,
            model=model,
            X_dev=X_dev,
            y_dev=y_dev,
        )

        experiment_results.append(result)

        print(f"RMSE: {result['cv_rmse_mean']:,.2f}")

        print(f"MAE:  {result['cv_mae_mean']:,.2f}")

        print(f"R2:   {result['cv_r2_mean']:.4f}")

        print()

    results_dataframe = pd.DataFrame(experiment_results)

    results_dataframe = results_dataframe.sort_values(
        by="cv_rmse_mean",
        ascending=True,
    ).reset_index(drop=True)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        REPORT_PATH,
        index=False,
    )

    print("===== Model Comparison =====")

    print(results_dataframe.to_string(index=False))

    print()

    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
