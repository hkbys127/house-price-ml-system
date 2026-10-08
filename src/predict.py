from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.catboost_utils import prepare_catboost_data

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "house_price_catboost_dev.joblib"


def load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact was not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def predict_price(
    house_features: dict[str, Any],
) -> float:

    artifact = load_artifact()

    model = artifact["model"]

    feature_columns = artifact["feature_columns"]

    categorical_columns = artifact["categorical_columns"]

    missing_columns = [
        column for column in feature_columns if column not in house_features
    ]

    if missing_columns:
        raise ValueError(f"Missing required features: {missing_columns}")

    input_dataframe = pd.DataFrame([house_features])

    # 保证与训练阶段完全相同的列顺序。
    input_dataframe = input_dataframe[feature_columns]

    input_dataframe = prepare_catboost_data(
        input_dataframe,
        categorical_columns,
    )

    prediction = model.predict(input_dataframe)

    return float(prediction[0])


def main() -> None:
    from src.split import (
        load_training_data,
        make_holdout_split,
    )

    X, y = load_training_data()

    X_dev, _, _, _ = make_holdout_split(
        X,
        y,
    )

    sample = X_dev.iloc[0].to_dict()

    prediction = predict_price(sample)

    print(f"Predicted SalePrice: {prediction:,.2f}")


if __name__ == "__main__":
    main()
