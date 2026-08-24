from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor
from src.split import (
    RANDOM_STATE,
    load_training_data,
    make_holdout_split,
)


def build_baseline_pipeline(X_dev):
    preprocessor = build_preprocessor(X_dev)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                DummyRegressor(
                    strategy="median",
                ),
            ),
        ]
    )

    return pipeline


def main() -> None:
    X, y = load_training_data()

    X_dev, X_holdout, y_dev, y_holdout = make_holdout_split(
        X,
        y,
    )

    baseline = build_baseline_pipeline(X_dev)

    cross_validation = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    cv_results = cross_validate(
        baseline,
        X_dev,
        y_dev,
        cv=cross_validation,
        scoring=scoring,
        return_train_score=False,
    )

    cv_rmse = -cv_results["test_rmse"]
    cv_mae = -cv_results["test_mae"]
    cv_r2 = cv_results["test_r2"]

    print("===== 5-Fold Cross-Validation =====")
    print(f"RMSE: {cv_rmse.mean():,.2f} +/- {cv_rmse.std():,.2f}")
    print(f"MAE:  {cv_mae.mean():,.2f} +/- {cv_mae.std():,.2f}")
    print(f"R2:   {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")

    print()

    baseline.fit(
        X_dev,
        y_dev,
    )

    predictions = baseline.predict(X_holdout)

    holdout_rmse = root_mean_squared_error(
        y_holdout,
        predictions,
    )

    holdout_mae = mean_absolute_error(
        y_holdout,
        predictions,
    )

    holdout_r2 = r2_score(
        y_holdout,
        predictions,
    )

    print("===== Final Holdout =====")
    print(f"RMSE: {holdout_rmse:,.2f}")
    print(f"MAE:  {holdout_mae:,.2f}")
    print(f"R2:   {holdout_r2:.4f}")

    print()
    print(
        "Baseline prediction:",
        f"{np.median(y_dev):,.2f}",
    )


if __name__ == "__main__":
    main()
