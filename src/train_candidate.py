"""Train a log-target development artifact; final models use src.finalize."""

import json

import joblib

from src.competition_data import ROOT, load_competition_data
from src.competition_model import fit_artifact
from src.split import make_holdout_split


def main():
    train, _ = load_competition_data()
    X_dev, _, y_dev, _ = make_holdout_split(
        train.drop(columns=["id", "sale_price"]), train.sale_price
    )
    artifact = fit_artifact(X_dev, y_dev)
    artifact["artifact_stage"] = "development"
    directory = ROOT / "models"
    directory.mkdir(exist_ok=True)
    joblib.dump(artifact, directory / "house_price_catboost_dev.joblib")
    metadata = {key: value for key, value in artifact.items() if key != "model"}
    (directory / "house_price_catboost_dev_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Saved development log-target artifact; holdout untouched.")


if __name__ == "__main__":
    main()
