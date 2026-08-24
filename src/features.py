from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "sale_price"
ID_COLUMN = "id"


# Kaggle 中这个字段虽然存的是数字，
# 但数字本身实际上是住宅类别代码。
CATEGORICAL_CODE_COLUMNS = [
    "ms_sub_class",
]


def get_feature_columns(
    dataframe: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Separate feature columns into numeric and categorical groups.
    """

    numeric_columns = (
        dataframe
        .select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_columns = (
        dataframe
        .select_dtypes(exclude=["number"])
        .columns
        .tolist()
    )

    for column in CATEGORICAL_CODE_COLUMNS:
        if column in numeric_columns:
            numeric_columns.remove(column)

        if column in dataframe.columns and column not in categorical_columns:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


def build_preprocessor(
    dataframe: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build preprocessing pipelines for numeric and categorical features.
    """

    numeric_columns, categorical_columns = get_feature_columns(
        dataframe
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor