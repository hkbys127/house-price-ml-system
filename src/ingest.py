from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.database import get_engine

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_CSV_PATH = BASE_DIR / "data" / "raw" / "train.csv"
TEST_CSV_PATH = BASE_DIR / "data" / "raw" / "test.csv"


def to_snake_case(column_name: str) -> str:
    """
    Convert Kaggle column names to SQL-friendly snake_case.

    Examples:
        SalePrice   -> sale_price
        OverallQual -> overall_qual
        1stFlrSF    -> 1st_flr_sf
    """
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", column_name)
    return name.lower()


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file was not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)

    dataframe.columns = [to_snake_case(column) for column in dataframe.columns]

    return dataframe


def create_indexes(connection) -> None:
    connection.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_house_train_raw_id
            ON house_train_raw (id)
            """
        )
    )

    connection.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_house_test_raw_id
            ON house_test_raw (id)
            """
        )
    )


def main() -> None:
    train_df = load_csv(TRAIN_CSV_PATH)
    test_df = load_csv(TEST_CSV_PATH)

    print(f"train shape: {train_df.shape}")
    print(f"test shape:  {test_df.shape}")

    engine = get_engine()

    with engine.begin() as connection:
        train_df.to_sql(
            name="house_train_raw",
            con=connection,
            if_exists="replace",
            index=False,
            chunksize=200,
            method="multi",
        )

        test_df.to_sql(
            name="house_test_raw",
            con=connection,
            if_exists="replace",
            index=False,
            chunksize=200,
            method="multi",
        )

        create_indexes(connection)

    print("CSV files were imported successfully.")
    print("Created table: house_train_raw")
    print("Created table: house_test_raw")


if __name__ == "__main__":
    main()
