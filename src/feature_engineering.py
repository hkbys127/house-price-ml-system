from __future__ import annotations

import pandas as pd


def add_total_sf(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    columns = [
        "total_bsmt_s_f",
        "1st_flr_s_f",
        "2nd_flr_s_f",
    ]

    missing_columns = [column for column in columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(
            f"Cannot build total_sf. Missing source columns: {missing_columns}"
        )

    dataframe["total_sf"] = dataframe[columns].fillna(0).sum(axis=1)

    return dataframe


def add_house_age(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    required_columns = [
        "yr_sold",
        "year_built",
    ]

    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Cannot build house_age. Missing source columns: {missing_columns}"
        )

    dataframe["house_age"] = dataframe["yr_sold"] - dataframe["year_built"]

    return dataframe


def add_model_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = add_total_sf(dataframe)

    dataframe = add_house_age(dataframe)

    return dataframe
