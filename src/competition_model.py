"""Versioned log-target artifacts shared by validation and submission."""

from importlib.metadata import version

import numpy as np

from src.catboost_utils import prepare_catboost_data
from src.evaluate import build_model_pipeline
from src.feature_engineering import add_model_features
from src.features import get_feature_columns
from src.models import get_catboost_model, get_models


def transform_features(frame, variant):
    if variant == "baseline":
        return frame.copy()
    if variant == "size_age":
        return add_model_features(frame)
    if variant != "sale_interactions":
        raise ValueError(f"Unknown feature variant: {variant}")
    result = add_model_features(frame)
    condition = result["sale_condition"].astype("string").fillna("__MISSING__")
    for value in ("Normal", "Partial", "Abnorml"):
        flag = condition.eq(value).astype(int)
        result[f"sale_{value.lower()}"] = flag
        for column in ("overall_qual", "gr_liv_area", "total_sf", "house_age"):
            result[f"{value.lower()}_{column}"] = flag * result[column]
    result["condition_quality"] = condition + "_" + result["overall_qual"].astype(str)
    result["quality_living_area"] = result["overall_qual"] * result["gr_liv_area"]
    result["quality_total_sf"] = result["overall_qual"] * result["total_sf"]
    result["remodel_age"] = result["yr_sold"] - result["year_remod_add"]
    return result


def fit_artifact(X, y, family="catboost", variant="baseline"):
    target = np.asarray(y, dtype=float)
    if (
        target.ndim != 1
        or len(target) != len(X)
        or not np.isfinite(target).all()
        or (target <= 0).any()
    ):
        raise ValueError(
            "Training targets must align with rows and be finite and positive"
        )
    frame = transform_features(X, variant)
    _, categories = get_feature_columns(frame)
    if family == "catboost":
        model = get_catboost_model()
        model.set_params(allow_writing_files=False, thread_count=4)
        model.fit(
            prepare_catboost_data(frame, categories), np.log(y), cat_features=categories
        )
    else:
        # Convert pandas nullable strings for sklearn imputers.
        frame = sklearn_frame(frame, categories)
        model = build_model_pipeline(get_models()[family], frame)
        model.fit(frame, np.log(y))
    return {
        "model": model,
        "family": family,
        "feature_variant": variant,
        "raw_feature_columns": X.columns.tolist(),
        "feature_columns": frame.columns.tolist(),
        "categorical_columns": categories,
        "target_transform": "log",
        "artifact_version": 2,
        "training_rows": len(X),
        "model_parameters": (
            model if family == "catboost" else model.named_steps["model"]
        ).get_params(),
        "library_versions": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "catboost", "xgboost")
        },
    }


def sklearn_frame(frame, categories):
    frame = frame.copy()
    for column in categories:
        frame[column] = (
            frame[column].astype(object).where(frame[column].notna(), np.nan)
        )
    return frame


def predict_artifact(artifact, X):
    if (
        artifact.get("artifact_version") != 2
        or artifact.get("target_transform") != "log"
    ):
        raise ValueError(
            "Expected a version 2 log-target artifact; retrain with src.submit."
        )
    missing = set(artifact["raw_feature_columns"]) - set(X.columns)
    if missing:
        raise ValueError(f"Missing input features: {sorted(missing)}")
    frame = transform_features(
        X[artifact["raw_feature_columns"]], artifact["feature_variant"]
    )
    frame = frame[artifact["feature_columns"]]
    if artifact["family"] == "catboost":
        frame = prepare_catboost_data(frame, artifact["categorical_columns"])
    else:
        frame = sklearn_frame(frame, artifact["categorical_columns"])
    prices = np.exp(artifact["model"].predict(frame))
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Predictions must be finite and strictly positive")
    return prices
