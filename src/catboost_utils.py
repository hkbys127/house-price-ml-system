from __future__ import annotations

import pandas as pd

MISSING_CATEGORY_TOKEN = "__MISSING__"


def prepare_catboost_data(
    dataframe: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    """
    Prepare categorical columns for CatBoost.

    Numeric missing values are left as numeric NaN.
    Categorical missing values become an explicit string category.
    """

    dataframe = dataframe.copy()

    for column in categorical_columns:
        dataframe[column] = (
            dataframe[column]
            .astype("string")
            .fillna(MISSING_CATEGORY_TOKEN)
            .astype(str)
        )

    return dataframe
