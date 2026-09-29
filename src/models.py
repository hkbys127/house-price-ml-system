from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


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
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            max_features=0.7,
            random_state=42,
            n_jobs=1,
        ),
        "decision_tree_full": DecisionTreeRegressor(
            random_state=42,
        ),
        "decision_tree_depth5": DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
        ),
    }
