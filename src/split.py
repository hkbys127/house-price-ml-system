from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.database import get_engine
from src.features import ID_COLUMN, TARGET_COLUMN

TRAIN_TABLE = "house_train_clean"

HOLDOUT_SIZE = 0.20
RANDOM_STATE = 42


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    engine = get_engine()

    dataframe = pd.read_sql_table(
        TRAIN_TABLE,
        con=engine,
    )
    dataframe.columns = dataframe.columns.map(str)

    X = dataframe.drop(
        columns=[
            ID_COLUMN,
            TARGET_COLUMN,
        ]
    )

    y = dataframe[TARGET_COLUMN].astype(float)

    return X, y


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


def main() -> None:
    X, y = load_training_data()

    X_dev, X_holdout, y_dev, y_holdout = make_holdout_split(
        X,
        y,
    )

    print(f"X shape:         {X.shape}")
    print(f"y shape:         {y.shape}")
    print()
    print(f"X_dev shape:     {X_dev.shape}")
    print(f"y_dev shape:     {y_dev.shape}")
    print()
    print(f"X_holdout shape: {X_holdout.shape}")
    print(f"y_holdout shape: {y_holdout.shape}")


if __name__ == "__main__":
    main()
