"""Local CSV entry point; applies the same structural cleaning as SQL ingestion."""

from pathlib import Path

from src.clean import clean_dataframe, validate_matching_features
from src.ingest import load_csv

ROOT = Path(__file__).resolve().parents[1]


def load_competition_data(data_dir=None):
    data_dir = Path(data_dir or ROOT / "data/raw")
    train = clean_dataframe(
        load_csv(data_dir / "train.csv"), is_training_data=True, table_name="train"
    )
    test = clean_dataframe(
        load_csv(data_dir / "test.csv"), is_training_data=False, table_name="test"
    )
    validate_matching_features(train, test)
    return train, test
