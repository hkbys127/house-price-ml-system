from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from competition.oof import calculate_metrics
from src.catboost_utils import prepare_catboost_data
from src.evaluate import build_model_pipeline
from src.features import get_feature_columns
from src.models import (
    get_catboost_model,
    get_models,
)
from src.split import (
    RANDOM_STATE,
    load_training_data,
    make_holdout_split,
)

BASE_DIR = Path(__file__).resolve().parents[1]

OOF_PATH = BASE_DIR / "reports" / "oof_log_target_predictions.csv"

FOLD_SCORE_PATH = BASE_DIR / "reports" / "oof_log_target_fold_scores.csv"

N_SPLITS = 5


def run_pipeline_log_target_oof(
    model_name: str,
    model,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[
    np.ndarray,
    list[dict[str, float | int | str]],
]:

    oof_predictions = np.zeros(
        len(X_dev),
        dtype=float,
    )

    fold_results = []

    for fold_number, (
        train_index,
        validation_index,
    ) in enumerate(
        splits,
        start=1,
    ):
        print(f"{model_name} - Fold {fold_number}/{len(splits)}")

        X_train = X_dev.iloc[train_index]

        X_validation = X_dev.iloc[validation_index]

        y_train = y_dev.iloc[train_index]

        y_validation = y_dev.iloc[validation_index]

        # -------------------------
        # 核心变化
        # -------------------------
        y_train_log = np.log(y_train.to_numpy(dtype=float))

        fold_model = clone(model)

        pipeline = build_model_pipeline(
            model=fold_model,
            X_dev=X_train,
        )

        # 模型现在学习的是 log(price)
        pipeline.fit(
            X_train,
            y_train_log,
        )

        log_predictions = pipeline.predict(X_validation)

        # 转回真实美元价格
        validation_predictions = np.exp(log_predictions)

        oof_predictions[validation_index] = validation_predictions

        metrics = calculate_metrics(
            y_validation,
            validation_predictions,
        )

        fold_results.append(
            {
                "model": model_name,
                "fold": fold_number,
                **metrics,
            }
        )

        print(f"  RMSE:     {metrics['rmse']:,.2f}")

        print(f"  Log RMSE: {metrics['log_rmse']:.5f}")

    return (
        oof_predictions,
        fold_results,
    )


def run_catboost_log_target_oof(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[
    np.ndarray,
    list[dict[str, float | int | str]],
]:

    _, categorical_columns = get_feature_columns(X_dev)

    X_catboost = prepare_catboost_data(
        X_dev,
        categorical_columns,
    )

    oof_predictions = np.zeros(
        len(X_dev),
        dtype=float,
    )

    fold_results = []

    for fold_number, (
        train_index,
        validation_index,
    ) in enumerate(
        splits,
        start=1,
    ):
        print(f"catboost_log_target - Fold {fold_number}/{len(splits)}")

        X_train = X_catboost.iloc[train_index]

        X_validation = X_catboost.iloc[validation_index]

        y_train = y_dev.iloc[train_index]

        y_validation = y_dev.iloc[validation_index]

        y_train_log = np.log(y_train.to_numpy(dtype=float))

        model = get_catboost_model()

        model.fit(
            X_train,
            y_train_log,
            cat_features=categorical_columns,
        )

        log_predictions = model.predict(X_validation)

        validation_predictions = np.exp(log_predictions)

        oof_predictions[validation_index] = validation_predictions

        metrics = calculate_metrics(
            y_validation,
            validation_predictions,
        )

        fold_results.append(
            {
                "model": "catboost_log_target",
                "fold": fold_number,
                **metrics,
            }
        )

        print(f"  RMSE:     {metrics['rmse']:,.2f}")

        print(f"  Log RMSE: {metrics['log_rmse']:.5f}")

    return (
        oof_predictions,
        fold_results,
    )


def print_global_metrics(
    model_name: str,
    y_true: pd.Series,
    predictions: np.ndarray,
) -> None:

    metrics = calculate_metrics(
        y_true,
        predictions,
    )

    print()
    print(f"{model_name} GLOBAL OOF")

    print(f"RMSE:     {metrics['rmse']:,.2f}")

    print(f"MAE:      {metrics['mae']:,.2f}")

    print(f"R2:       {metrics['r2']:.4f}")

    print(f"Log RMSE: {metrics['log_rmse']:.5f}")


def main() -> None:
    X, y = load_training_data()

    X_dev, _, y_dev, _ = make_holdout_split(
        X,
        y,
    )

    print(f"Development rows: {len(X_dev)}")

    print("Holdout remains locked.")

    cross_validation = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    splits = list(cross_validation.split(X_dev))

    fold_assignment = np.full(
        len(X_dev),
        -1,
        dtype=int,
    )

    for fold_number, (
        _,
        validation_index,
    ) in enumerate(
        splits,
        start=1,
    ):
        fold_assignment[validation_index] = fold_number

    oof_dataframe = pd.DataFrame(
        {
            "row_index": (X_dev.index.to_numpy()),
            "fold": (fold_assignment),
            "y_true": (y_dev.to_numpy()),
        }
    )

    all_fold_results = []

    models = get_models()

    candidate_names = [
        "gradient_boosting",
        "xgboost",
    ]

    for model_name in candidate_names:
        print()
        print("=" * 60)

        experiment_name = f"{model_name}_log_target"

        print(f"OOF: {experiment_name}")

        print("=" * 60)

        (
            predictions,
            fold_results,
        ) = run_pipeline_log_target_oof(
            model_name=experiment_name,
            model=models[model_name],
            X_dev=X_dev,
            y_dev=y_dev,
            splits=splits,
        )

        oof_dataframe[f"pred_{experiment_name}"] = predictions

        all_fold_results.extend(fold_results)

        print_global_metrics(
            experiment_name,
            y_dev,
            predictions,
        )

    print()
    print("=" * 60)

    print("OOF: catboost_log_target")

    print("=" * 60)

    (
        catboost_predictions,
        catboost_fold_results,
    ) = run_catboost_log_target_oof(
        X_dev=X_dev,
        y_dev=y_dev,
        splits=splits,
    )

    oof_dataframe["pred_catboost_log_target"] = catboost_predictions

    all_fold_results.extend(catboost_fold_results)

    print_global_metrics(
        "catboost_log_target",
        y_dev,
        catboost_predictions,
    )

    OOF_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    oof_dataframe.to_csv(
        OOF_PATH,
        index=False,
    )

    pd.DataFrame(all_fold_results).to_csv(
        FOLD_SCORE_PATH,
        index=False,
    )

    print()
    print("=" * 60)

    print("Log-target OOF completed.")

    print(f"Saved predictions: {OOF_PATH}")

    print(f"Saved fold scores: {FOLD_SCORE_PATH}")


if __name__ == "__main__":
    main()
