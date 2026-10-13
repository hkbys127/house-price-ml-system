from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.competition_model import predict_artifact

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "house_price_final.joblib"


def load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact was not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def predict_price(
    house_features: dict[str, Any],
) -> float:

    artifact = load_artifact()

    feature_columns = artifact["raw_feature_columns"]

    missing_columns = [
        column for column in feature_columns if column not in house_features
    ]

    if missing_columns:
        raise ValueError(f"Missing required features: {missing_columns}")

    input_dataframe = pd.DataFrame([house_features])

    # 保证与训练阶段完全相同的列顺序。
    input_dataframe = input_dataframe[feature_columns]

    prediction = predict_artifact(artifact, input_dataframe)

    return float(prediction[0])


def main() -> None:
    from src.competition_data import load_competition_data

    _, test = load_competition_data()
    sample = test.drop(columns="id").iloc[0].to_dict()

    prediction = predict_price(sample)

    print(f"Predicted SalePrice: {prediction:,.2f}")


if __name__ == "__main__":
    main()
