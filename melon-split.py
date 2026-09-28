from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import data

from src.database import get_engine
from src.database import ID_COLUMN, TARGET_COlUMN


TRAIN_TABLE = "house_train_clean"

HOLDOUT_SIZE = 0.20
RANDOM_STATE = 42

def load_training_data() -> tuple[pd.DataFrame, pd.Series]
    engine = get_engine()

    dataframe = pd.read_sql_table(
        TRAIN_TABLE,
        con=engine,
    )
    dataframe.columns = dataframe.columns.map(str)

    X = dataframe.drop(
        columns=[
            ID_COLUMN,
            TARGET_COlUMN,
        ]
    )

    y = dataframe[TARGET_COlUMN].astype(float)

    return X,y

def make_holdout_split(
        X: pd.DataFrame,
        y: pd.Series,
):
    return train_test_split(
        X,
        y,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
