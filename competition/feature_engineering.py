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

    dataframe["total_sf"] = dataframe[columns].fillna(0).sum(axis=1)

    return dataframe


def add_house_age(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["house_age"] = dataframe["yr_sold"] - dataframe["year_built"]

    return dataframe


def add_remodel_age(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["remodel_age"] = dataframe["yr_sold"] - dataframe["year_remod_add"]

    return dataframe


def add_total_bathrooms(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    full_bath = dataframe["full_bath"].fillna(0)

    half_bath = dataframe["half_bath"].fillna(0)

    bsmt_full_bath = dataframe["bsmt_full_bath"].fillna(0)

    bsmt_half_bath = dataframe["bsmt_half_bath"].fillna(0)

    dataframe["total_bathrooms"] = (
        full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath
    )

    return dataframe
