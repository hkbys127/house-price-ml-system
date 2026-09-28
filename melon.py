from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.database import get_engine

TRAIN_RAW_TABLE = "house_train_raw"
TEST_RAW_TABLE = "house_test_raw"
TRAIN_CLEAN_TABLE = "house_train_clean"
TEST_CLEAN_TABLE = "house_test_clean"

ID_COLUMN = "id"
TARGET_COLUMN = "sale_price"

ABSENCE_CATEGORY_COLUMNS = ["MIE", "MIE2"]


def normalize_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:

    dataframe = dataframe.copy()

    text_columns = dataframe.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        dataframe[column] = (
            dataframe[column].astype("string").str.strip().replace("", pd.NA)
        )

    return dataframe


def replace_infinite_values(dataframe: pd.DataFrame) -> pd.DataFrame:

    dataframe = dataframe.copy()

    numeric_columns = dataframe.select_dtypes(include=["int64", "float64"]).columns

    dataframe[numeric_columns] = dataframe[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    return dataframe


def fill_semantic_absence(dataframe: pd.DataFrame) -> pd.DataFrame:

    dataframe = dataframe.copy()

    for column in ABSENCE_CATEGORY_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].fillna("None")

    return dataframe


def validate_id_column(dataframe: pd.DataFrame, table_name: str) -> None:

    if ID_COLUMN not in dataframe.columns:
        raise ValueError(f"{table_name}does not contain the '{ID_COLUMN}' column.")
    if dataframe[ID_COLUMN].isna().any():
        raise ValueError(f"{table_name}contains missing IDs")
    duplicated_count = dataframe[ID_COLUMN].duplicated().sum()
    if duplicated_count > 0:
        raise ValueError(f"{table_name}contains {duplicated_count}dupicated IDs.")


def validate_target(dataframe: pd.DataFrame, table_name: str) -> None:

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"{table_name}does not contain the '{TARGET_COLUMN}' column.")
    dataframe[TARGET_COLUMN] = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    )
    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(f"{table_name}contains missing or invalid targets")
    if (dataframe[TARGET_COLUMN] <= 0).any():
        raise ValueError(f"{table_name}contains targets less than or equal to zero")
    dataframe[TARGET_COLUMN]


def clean_dataframe(
    dataframe: pd.DataFrame, *, is_training_data: bool, table_name: str
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe = normalize_text_columns(dataframe)
    dataframe = replace_infinite_values(dataframe)
    dataframe = fill_semantic_absence(dataframe)

    validate_id_column(dataframe, table_name)

    if is_training_data:
        validate_target(dataframe, table_name)
    elif TARGET_COLUMN in dataframe.columns:
        raise ValueError(
            f"There was already containing {TARGET_COLUMN} in {table_name}"
        )

    dataframe = dataframe.sort_values(ID_COLUMN).reset_index(drop=True)

    return dataframe


def validate_matching_features(
    train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame
) -> None:

    train_feature_columns = set(train_dataframe.columns) - {TARGET_COLUMN}
    test_feature_columns = set(test_dataframe.columns)

    missing_from_test = train_feature_columns - test_feature_columns
    extra_in_test = test_feature_columns - train_feature_columns

    if missing_from_test or extra_in_test:
        raise ValueError(
            f"{missing_from_test}was missed, and {extra_in_test} was extra column"
        )


def create_indexes(connection) -> None:

    connection.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_house_train_clean_id
            ON house_train_clean (id)
            """
        )
    )

    connection.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_house_test_clean_id
            ON house_test_clean (id)
            """
        )
    )


def main() -> None:
    engine = get_engine()

    print("Reading raw tables from PostgreSQL...")

    train_raw = pd.read_sql_table(
        TRAIN_RAW_TABLE,
        con=engine,
    )

    test_raw = pd.read_sql_table(
        TEST_RAW_TABLE,
        con=engine,
    )
    print(f"raw train shape: {train_raw.shape}")
    print(f"raw test shape:  {test_raw.shape}")

    train_clean = clean_dataframe(
        train_raw,
        is_training_data=True,
        table_name=TRAIN_RAW_TABLE,
    )

    test_clean = clean_dataframe(
        test_raw,
        is_training_data=False,
        table_name=TEST_RAW_TABLE,
    )

    validate_matching_features(
        train_clean,
        test_clean,
    )
    with engine.begin() as connection:
        train_clean.to_sql(
            name=TRAIN_CLEAN_TABLE,
            con=connection,
            if_exists="replace",
            index=False,
            chunksize=200,
            method="multi",
        )

        test_clean.to_sql(
            name=TEST_CLEAN_TABLE,
            con=connection,
            if_exists="replace",
            index=False,
            chunksize=200,
            method="multi",
        )
    create_indexes(connection)

    print(f"clean train shape: {train_clean.shape}")
    print(f"clean test shape:  {test_clean.shape}")
    print("Cleaning completed successfully.")
    print(f"Created table: {TRAIN_CLEAN_TABLE}")
    print(f"Created table: {TEST_CLEAN_TABLE}")


if __name__ == "__main__":
    main()
