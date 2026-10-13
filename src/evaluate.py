"""Log-target evaluation helpers; use src.train for paired model selection."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.catboost_utils import prepare_catboost_data
from src.features import build_preprocessor, get_feature_columns
from src.models import get_catboost_model


def build_model_pipeline(model, X_dev):
    return Pipeline([("preprocessor", build_preprocessor(X_dev)), ("model", model)])


def evaluate_model(model_name, model, X_dev, y_dev, *, native_catboost=False):
    from src.competition_model import sklearn_frame
    from src.repeated_cv import SEEDS, metrics

    _, categories = get_feature_columns(X_dev)
    X = (
        prepare_catboost_data(X_dev, categories)
        if native_catboost
        else sklearn_frame(X_dev, categories)
    )
    rows = []
    for seed in SEEDS:
        for fit, valid in KFold(5, shuffle=True, random_state=seed).split(X):
            estimator = clone(model)
            start = time.perf_counter()
            if native_catboost:
                estimator.fit(
                    X.iloc[fit], np.log(y_dev.iloc[fit]), cat_features=categories
                )
            else:
                estimator = build_model_pipeline(estimator, X.iloc[fit])
                estimator.fit(X.iloc[fit], np.log(y_dev.iloc[fit]))
            elapsed = time.perf_counter() - start
            rows.append(
                dict(
                    **metrics(
                        y_dev.iloc[valid], np.exp(estimator.predict(X.iloc[valid]))
                    ),
                    fit_time=elapsed,
                )
            )
    scores = pd.DataFrame(rows)
    return {
        "model": model_name,
        "cv_log_rmse_mean": scores.log_rmse.mean(),
        "cv_log_rmse_std": scores.log_rmse.std(),
        "cv_rmse_mean": scores.rmse.mean(),
        "cv_rmse_std": scores.rmse.std(),
        "cv_mae_mean": scores.mae.mean(),
        "fit_time_mean": scores.fit_time.mean(),
    }


def evaluate_catboost(X_dev, y_dev):
    return evaluate_model(
        "catboost_log_target", get_catboost_model(), X_dev, y_dev, native_catboost=True
    )
