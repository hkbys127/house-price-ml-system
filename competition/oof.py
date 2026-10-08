from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold

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

OOF_PATH = BASE_DIR / "reports" / "oof_predictions.csv"

FOLD_SCORE_PATH = BASE_DIR / "reports" / "oof_fold_scores.csv"


N_SPLITS = 5


def competition_log_rmse(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    House Prices style RMSE on logarithmic prices.

    Predictions are clipped to a positive value because
    log(x) is undefined for x <= 0.
    """

    true_values = np.asarray(
        y_true,
        dtype=float,
    )

    predicted_values = np.asarray(
        y_pred,
        dtype=float,
    )

    predicted_values = np.clip(
        predicted_values,
        1.0,
        None,
    )

    return float(
        np.sqrt(np.mean((np.log(true_values) - np.log(predicted_values)) ** 2))
    )


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "rmse": root_mean_squared_error(
            y_true,
            y_pred,
        ),
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "r2": r2_score(
            y_true,
            y_pred,
        ),
        "log_rmse": competition_log_rmse(
            y_true,
            y_pred,
        ),
    }


def run_pipeline_oof(
    model_name: str,
    model,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:

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

        # 每个 Fold 都必须得到一套全新的模型。
        fold_model = clone(model)

        pipeline = build_model_pipeline(
            model=fold_model,
            X_dev=X_train,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        validation_predictions = pipeline.predict(X_validation)
        print("模型类型：", type(fold_model))
        print("预测结果类型：", type(validation_predictions))
        if not isinstance(validation_predictions, np.ndarray):
            raise TypeError("预测结果应为 NumPy 数组")

        # 把 validation prediction 放回
        # 它在整个 development set 中原本的位置。
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


def prepare_catboost_data(
    dataframe: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    for column in categorical_columns:
        dataframe[column] = (
            dataframe[column].astype("string").fillna("__MISSING__").astype(str)
        )

    return dataframe


def run_catboost_oof(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:

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
        print(f"catboost_native - Fold {fold_number}/{len(splits)}")

        X_train = X_catboost.iloc[train_index]

        X_validation = X_catboost.iloc[validation_index]

        y_train = y_dev.iloc[train_index]

        y_validation = y_dev.iloc[validation_index]

        # CatBoost 每个 Fold 重新创建。
        model = get_catboost_model()

        model.fit(
            X_train,
            y_train,
            cat_features=categorical_columns,
        )

        validation_predictions = model.predict(X_validation)

        oof_predictions[validation_index] = validation_predictions

        metrics = calculate_metrics(
            y_validation,
            validation_predictions,
        )

        fold_results.append(
            {
                "model": "catboost_native",
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


def main() -> None:
    X, y = load_training_data()

    X_dev, X_holdout, y_dev, y_holdout = make_holdout_split(
        X,
        y,
    )

    print(f"Development rows: {len(X_dev)}")

    print(f"Holdout rows:     {len(X_holdout)}")

    print("Holdout remains locked.")

    print()

    # 所有模型必须用完全相同的 folds。
    cross_validation = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    splits = list(cross_validation.split(X_dev))

    # 保存每一行属于哪个 validation fold。
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

    models = get_models()

    candidate_names = [
        "gradient_boosting",
        "xgboost",
    ]

    all_fold_results = []

    oof_dataframe = pd.DataFrame(
        {
            "row_index": (X_dev.index.to_numpy()),
            "fold": fold_assignment,
            "y_true": (y_dev.to_numpy()),
        }
    )

    # -----------------------------
    # GradientBoosting + XGBoost
    # -----------------------------
    for model_name in candidate_names:
        print()
        print("=" * 60)

        print(f"OOF: {model_name}")

        print("=" * 60)

        predictions, fold_results = run_pipeline_oof(
            model_name=model_name,
            model=models[model_name],
            X_dev=X_dev,
            y_dev=y_dev,
            splits=splits,
        )

        oof_dataframe[f"pred_{model_name}"] = predictions

        all_fold_results.extend(fold_results)

        global_metrics = calculate_metrics(
            y_dev,
            predictions,
        )

        print()
        print(f"{model_name} GLOBAL OOF")

        print(f"RMSE:     {global_metrics['rmse']:,.2f}")

        print(f"MAE:      {global_metrics['mae']:,.2f}")

        print(f"R2:       {global_metrics['r2']:.4f}")

        print(f"Log RMSE: {global_metrics['log_rmse']:.5f}")

    # -----------------------------
    # CatBoost
    # -----------------------------
    print()
    print("=" * 60)

    print("OOF: catboost_native")

    print("=" * 60)

    (
        catboost_predictions,
        catboost_fold_results,
    ) = run_catboost_oof(
        X_dev=X_dev,
        y_dev=y_dev,
        splits=splits,
    )

    oof_dataframe["pred_catboost_native"] = catboost_predictions

    all_fold_results.extend(catboost_fold_results)

    catboost_metrics = calculate_metrics(
        y_dev,
        catboost_predictions,
    )

    print()
    print("catboost_native GLOBAL OOF")

    print(f"RMSE:     {catboost_metrics['rmse']:,.2f}")

    print(f"MAE:      {catboost_metrics['mae']:,.2f}")

    print(f"R2:       {catboost_metrics['r2']:.4f}")

    print(f"Log RMSE: {catboost_metrics['log_rmse']:.5f}")

    # -----------------------------
    # Save artifacts
    # -----------------------------
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

    print("OOF generation completed.")

    print(f"Saved predictions: {OOF_PATH}")

    print(f"Saved fold scores: {FOLD_SCORE_PATH}")


if __name__ == "__main__":
    main()
