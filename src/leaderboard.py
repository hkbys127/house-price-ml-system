"""Upload an existing validated CSV or fetch actual Kaggle submission scores."""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.competition_data import ROOT
from src.submit import COMPETITION, validate_submission


def authenticate(memory_auth=False):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    if not memory_auth:
        api.authenticate()
        return api
    # The SDK normally saves refreshed credentials. For read-only credential stores,
    # use its public refresh API and a process-local token; never copy/persist secrets.
    from kagglesdk.kaggle_creds import KaggleCredentials

    with api.build_kaggle_client() as client:
        credentials = KaggleCredentials.load(client)
        if credentials is None:
            raise RuntimeError("No existing Kaggle OAuth login found")
        response = credentials.generate_access_token()
    previous = os.environ.get("KAGGLE_API_TOKEN")
    try:
        os.environ["KAGGLE_API_TOKEN"] = response.token
        api.authenticate()
    finally:
        if previous is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = previous
    return api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--memory-auth", action="store_true")
    args = parser.parse_args()
    digest = hashlib.sha256(args.file.read_bytes()).hexdigest()
    message = f"house-price-ml sha256:{digest}"
    validate_submission(
        pd.read_csv(args.file), pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    )
    api = authenticate(args.memory_auth)
    submissions = api.competition_submissions(COMPETITION)
    matched = [s for s in submissions or [] if s.description == message]
    if args.upload and not matched:
        response = api.competition_submit(
            str(args.file.resolve()), message, COMPETITION
        )
        print(str(response))
        submissions = api.competition_submissions(COMPETITION)
        matched = [s for s in submissions or [] if s.description == message]
    rows = [
        {
            "ref": s.ref,
            "date": str(s.date),
            "status": str(s.status),
            "public_score": s.public_score,
            "private_score": s.private_score,
            "description": s.description,
        }
        for s in matched
    ]
    report = {
        "competition": COMPETITION,
        "file": args.file.name,
        "sha256": digest,
        "checked_at": datetime.now(UTC).isoformat(),
        "submissions": rows,
    }
    metadata_path = args.file.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("sha256") != digest:
            raise ValueError("Submission metadata hash does not match CSV")
        if rows:
            latest = rows[0]
            metadata["leaderboard_status"] = latest["status"]
            metadata["public_score"] = latest["public_score"]
            metadata["submission_ref"] = latest["ref"]
            metadata.pop("failure_type", None)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    args.file.with_suffix(".leaderboard.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
