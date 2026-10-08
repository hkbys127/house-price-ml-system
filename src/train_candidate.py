from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.catboost_utils import prepare_catboost_data
from src.features import get_feature_columns
from src.models import get_catboost_model
from src.split import (
    RANDOM_STATE,
    load_training_data,
    make_holdout_split,
)

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "house_price_catboost_dev.joblib"

METADATA_PATH = BASE_DIR / "models" / "house_price_catboost_dev_metadata.json"


def main() -> None:
    X, y = load_training_data()

    X_dev, _, y_dev, _ = make_holdout_split(
        X,
        y,
    )

    numeric_columns, categorical_columns = get_feature_columns(X_dev)

    X_catboost = prepare_catboost_data(
        X_dev,
        categorical_columns,
    )

    model = get_catboost_model()

    print("Training development CatBoost model...")
    print(f"Training rows: {len(X_dev)}")

    model.fit(
        X_catboost,
        y_dev,
        cat_features=categorical_columns,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "feature_columns": (X_dev.columns.tolist()),
        "numeric_columns": (numeric_columns),
        "categorical_columns": (categorical_columns),
        "artifact_stage": "development",
        "random_state": RANDOM_STATE,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    metadata = {
        "model_type": "CatBoostRegressor",
        "artifact_stage": "development",
        "training_rows": len(X_dev),
        "feature_count": X_dev.shape[1],
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "random_state": RANDOM_STATE,
        "note": ("Development artifact only. Final holdout has not been used."),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Model saved:")
    print(MODEL_PATH)

    print()
    print("Metadata saved:")
    print(METADATA_PATH)


if __name__ == "__main__":
    main()
