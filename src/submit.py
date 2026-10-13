"""Train on all labeled rows, validate a submission, optionally upload to Kaggle."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.competition_data import ROOT, load_competition_data
from src.competition_model import fit_artifact, predict_artifact

COMPETITION = "house-prices-advanced-regression-techniques"


def validate_submission(submission, sample):
    if list(submission.columns) != ["Id", "SalePrice"]:
        raise ValueError("Submission columns must be Id, SalePrice")
    if not submission["Id"].equals(sample["Id"]) or submission["Id"].duplicated().any():
        raise ValueError(
            "Submission IDs/order must exactly match sample_submission.csv"
        )
    prices = submission["SalePrice"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("SalePrice must be finite and positive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--artifact", type=Path)
    source.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/submission.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--memory-auth",
        action="store_true",
        help="Refresh existing OAuth login without writing credentials",
    )
    parser.add_argument("--message", default="log-target baseline")
    args = parser.parse_args()
    train, test = load_competition_data(args.data_dir)
    if args.artifact:
        artifact = joblib.load(args.artifact)
    else:
        selection = json.loads(args.selection.read_text()) if args.selection else {}
        artifact = fit_artifact(
            train.drop(columns=["id", "sale_price"]),
            train.sale_price,
            selection.get("family", "catboost"),
            selection.get("variant", "baseline"),
        )
        artifact["artifact_stage"] = "full_training"
        path = ROOT / "models" / (args.output.stem + ".joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
    predictions = pd.Series(
        predict_artifact(artifact, test.drop(columns="id")), index=test.id
    )
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    submission = pd.DataFrame(
        {"Id": sample.Id, "SalePrice": sample.Id.map(predictions)}
    )
    validate_submission(submission, sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    metadata = {key: value for key, value in artifact.items() if key != "model"}
    metadata.update(
        created_at=datetime.now(UTC).isoformat(),
        rows=len(submission),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        leaderboard_status="not_submitted",
    )
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Validated {len(submission)} rows: {args.output}", flush=True)
    if args.upload:
        from src.leaderboard import authenticate

        try:
            api = authenticate(args.memory_auth)
            response = api.competition_submit(
                str(args.output), args.message, COMPETITION
            )
            metadata["leaderboard_status"] = "submitted_score_pending"
            metadata["submission_response"] = str(response)
            print(response)
        except Exception as error:
            metadata["leaderboard_status"] = "submission_failed"
            metadata["failure_type"] = type(error).__name__
            raise
        finally:
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
