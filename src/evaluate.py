from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, get_feature_columns
from src.models import get_catboost_model
from src.split import RANDOM_STATE


def build_model_pipeline(
    model: BaseEstimator,
    X_dev: pd.DataFrame,
) -> Pipeline:
    """
    Combine preprocessing and a regression model
    into one complete machine-learning pipeline.
    """

    preprocessor = build_preprocessor(X_dev)

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


def evaluate_model(
    model_name: str,
    model: BaseEstimator,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
) -> dict[str, float | str]:
    """
    Evaluate one model with the same 5-fold
    cross-validation procedure used for all models.
    """

    pipeline = build_model_pipeline(
        model=model,
        X_dev=X_dev,
    )

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

    results = cross_validate(
        pipeline,
        X_dev,
        y_dev,
        cv=cross_validation,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    train_rmse = -results["train_rmse"]
    rmse = -results["test_rmse"]
    mae = -results["test_mae"]
    r2 = results["test_r2"]

    return {
        "model": model_name,
        "train_rmse_mean": np.mean(train_rmse),
        "cv_rmse_mean": np.mean(rmse),
        "rmse_gap": (np.mean(rmse) - np.mean(train_rmse)),
        "cv_rmse_std": np.std(rmse),
        "cv_mae_mean": np.mean(mae),
        "cv_mae_std": np.std(mae),
        "cv_r2_mean": np.mean(r2),
        "cv_r2_std": np.std(r2),
        "fit_time_mean": np.mean(results["fit_time"]),
    }


def prepare_catboost_data(
    dataframe: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    """
    Prepare categorical columns for CatBoost.

    CatBoost can handle numeric missing values directly,
    but categorical missing values are converted to an
    explicit string category.
    """

    dataframe = dataframe.copy()

    for column in categorical_columns:
        dataframe[column] = (
            dataframe[column].astype("string").fillna("__MISSING__").astype(str)
        )

    return dataframe


def evaluate_catboost(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
) -> dict[str, float | str]:

    _, categorical_columns = get_feature_columns(X_dev)

    X_catboost = prepare_catboost_data(
        X_dev,
        categorical_columns,
    )

    cross_validation = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_rmse_scores = []
    cv_rmse_scores = []
    cv_mae_scores = []
    cv_r2_scores = []
    fit_times = []

    import time

    for train_index, validation_index in cross_validation.split(X_catboost):
        X_train = X_catboost.iloc[train_index]
        X_validation = X_catboost.iloc[validation_index]

        y_train = y_dev.iloc[train_index]
        y_validation = y_dev.iloc[validation_index]

        model = get_catboost_model()

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
            cat_features=categorical_columns,
        )

        fit_time = time.perf_counter() - start_time

        train_predictions = model.predict(X_train)

        validation_predictions = model.predict(X_validation)

        train_rmse_scores.append(
            root_mean_squared_error(
                y_train,
                train_predictions,
            )
        )

        cv_rmse_scores.append(
            root_mean_squared_error(
                y_validation,
                validation_predictions,
            )
        )

        cv_mae_scores.append(
            mean_absolute_error(
                y_validation,
                validation_predictions,
            )
        )

        cv_r2_scores.append(
            r2_score(
                y_validation,
                validation_predictions,
            )
        )

        fit_times.append(fit_time)

    train_rmse_mean = np.mean(train_rmse_scores)

    cv_rmse_mean = np.mean(cv_rmse_scores)

    return {
        "model": "catboost_native",
        "train_rmse_mean": train_rmse_mean,
        "cv_rmse_mean": cv_rmse_mean,
        "rmse_gap": (cv_rmse_mean - train_rmse_mean),
        "cv_rmse_std": np.std(cv_rmse_scores),
        "cv_mae_mean": np.mean(cv_mae_scores),
        "cv_mae_std": np.std(cv_mae_scores),
        "cv_r2_mean": np.mean(cv_r2_scores),
        "cv_r2_std": np.std(cv_r2_scores),
        "fit_time_mean": np.mean(fit_times),
    }
