from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge


def get_models() -> dict[str, BaseEstimator]:
    """
    Return the models used in the current experiment.
    """

    return {
        "dummy_median": DummyRegressor(
            strategy="median",
        ),
        "linear_regression": LinearRegression(),
        "ridge": Ridge(
            alpha=10.0,
        ),
        "lasso": Lasso(
            alpha=100.0,
            max_iter=20_000,
        ),
    }


