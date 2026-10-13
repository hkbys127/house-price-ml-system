"""Evaluate frozen selection once on holdout, then train all labeled data."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from src.competition_data import ROOT, load_competition_data
from src.competition_model import fit_artifact, predict_artifact
from src.repeated_cv import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path, default=ROOT / "reports/repeated_cv/selection.json"
    )
    path = parser.parse_args().selection
    selection = json.loads(path.read_text())
    fingerprint = hashlib.sha256((ROOT / "data/raw/train.csv").read_bytes()).hexdigest()
    if fingerprint != selection["train_sha256"]:
        raise ValueError("Training data changed after selection")
    result_path = ROOT / "reports/final_holdout.json"
    if result_path.exists():
        raise FileExistsError(
            "Holdout already evaluated. Read final_holdout.json; do not tune against it."
        )
    train, _ = load_competition_data()
    split = pd.read_csv(path.parent / "split_assignment.csv")
    merged = train.merge(split, left_on="id", right_on="Id", validate="one_to_one")
    if len(merged) != len(train) or set(merged.partition) != {"development", "holdout"}:
        raise ValueError("Invalid frozen split assignments")
    X = merged.drop(columns=["id", "Id", "sale_price", "partition"])
    dev = merged.partition.eq("development")
    artifact = fit_artifact(
        X.loc[dev],
        merged.loc[dev, "sale_price"],
        selection["family"],
        selection["variant"],
    )
    prediction = predict_artifact(artifact, X.loc[~dev])
    report = dict(
        selection=selection,
        holdout_rows=int((~dev).sum()),
        **metrics(merged.loc[~dev, "sale_price"], prediction),
    )
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "Id": merged.loc[~dev, "id"],
            "y_true": merged.loc[~dev, "sale_price"],
            "prediction": prediction,
        }
    ).to_csv(ROOT / "reports/holdout_predictions.csv", index=False)
    artifact = fit_artifact(
        X, merged.sale_price, selection["family"], selection["variant"]
    )
    artifact["artifact_stage"] = "full_training"
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(artifact, ROOT / "models/house_price_final.joblib")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
