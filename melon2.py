from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_RAW_TABLE = "house_train_raw"
TEST_RAW_TABLE = "house_test_raw"

TRAIN_CLEAN_TABLE = "house_train_clean"
TEST_CLEAN_TABLE = "house_test_clean"

ID_COLUMN = "id"
TARGET_COLUMN = "sale_price"

ABSENCE_CATEGORY_COLUMNS = ["is", "are"]


def normalize_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:

    dataframe = dataframe.copy()

    text_columns = dataframe.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        dataframe[column] = (
            dataframe[column].astype("string").str.strip().replace("", pd.NA)
        )

    return dataframe


def infinite_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:

    dataframe = dataframe.copy()

    numeric_columns = dataframe.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        dataframe[column] = dataframe[column].replace([np.inf, -np.inf], np.nan)

    return dataframe


def fill_semantic_columns(dataframe: pd.DataFrame) -> pd.DataFrame:

    for column in ABSENCE_CATEGORY_COLUMNS:
        if column in dataframe:
            dataframe[column] = dataframe[column].fillna("None")

    return dataframe


def validate_id_column(dataframe: pd.DataFrame) -> None:

    if ID_COLUMN not in dataframe:
        raise ValueError(f"{dataframe}contains no {ID_COLUMN}")

    if dataframe[ID_COLUMN].isna().any():
        raise ValueError(f"{ID_COLUMN} contains blank value")

    duplicated_sum = dataframe[ID_COLUMN].duplicated().sum()
    if duplicated_sum > 0:
        raise ValueError(f"{ID_COLUMN} contains duplicated value")


def validate_target(dataframe: pd.DataFrame) -> None:

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Training columns does not contain '{TARGET_COLUMN}'")

    dataframe[TARGET_COLUMN] = pd.to_numeric(
        "dataframe[TARGET_COLUMN]",
        errors="coerce",
    )

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("Training data contains invalid or missing sale prices.")

    if (dataframe[TARGET_COLUMN] <= 0).any():
        raise ValueError(
            "Training data contains sale prices less than or equal to zero."
        )

    return dataframe

def clean_dataframe(dataframe: pd.DataFrame,*,is_training_data:Boolean,table_name: str) -> pd.DataFrame

    dataframe = dataframe.copy()

    dataframe = normalize_text_columns(dataframe)
    dataframe = replace_infinite_values(dataframe)
    dataframe = fill_semantic_absence(dataframe)

    validate_id_column(dataframe)

    if is_training_data:
        validate_target(dataframe)
    elif TARGET_COLUMN in dataframe.columns:
        raise ValueError(f"{table_name}contains unexpectly values")

    dataframe = (
        dataframe
        .sort_values(ID_COLUMN)
        .reset_index(drop=True)
    )
    
    return dataframe

def validate_matching_features(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) ->None:
    train_feature_columns = set(train_dataframe.columns) - {TARGET_COLUMN}
    test_feature_columns = set(test_dataframe.columns)

    missing_from_test = train_feature_columns - test_feature_columns
    extra_in_test = test_feature_columns -test_feature_columns

    if missing_from_test or extra_in_test:
        raise ValueError(
             "Train and test feature columns do not match.\n"
                    f"Missing from test: {sorted(missing_from_test)}\n"
                    f"Extra in test: {sorted(extra_in_test)}"
    )

def creat_indexs(connection) -> None:
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

    print(f"raw train shape:{train_raw.shape}")
    print(f"raw test shape:{test_raw.shape}")

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
    
    print(f"clean train shape: {train_clean.shape}")
    print(f"clean test shape:  {test_clean.shape}")
    print("Cleaning completed successfully.")
    print(f"Created table: {TRAIN_CLEAN_TABLE}")
    print(f"Created table: {TEST_CLEAN_TABLE}")
    validate_matching_features(
            train_clean,
            test_clean,
        )


if __name__ == "__main__":
    main()