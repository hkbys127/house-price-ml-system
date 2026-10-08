from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.split import (
    load_training_data,
    make_holdout_split,
)

BASE_DIR = Path(__file__).resolve().parents[1]

OOF_PATH = BASE_DIR / "reports" / "oof_log_target_predictions.csv"

ERROR_REPORT_PATH = BASE_DIR / "reports" / "catboost_oof_error_analysis.csv"

WORST_PATH = BASE_DIR / "reports" / "catboost_worst_20_predictions.csv"

NEIGHBORHOOD_PATH = BASE_DIR / "reports" / "catboost_error_by_neighborhood.csv"


PREDICTION_COLUMN = "pred_catboost_log_target"


def build_error_dataframe() -> pd.DataFrame:
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"OOF file not found: {OOF_PATH}")

    oof = pd.read_csv(OOF_PATH)

    required_columns = {
        "row_index",
        "fold",
        "y_true",
        PREDICTION_COLUMN,
    }

    missing_columns = required_columns - set(oof.columns)

    if missing_columns:
        raise ValueError(f"OOF file is missing columns: {sorted(missing_columns)}")

    prediction = oof[PREDICTION_COLUMN].astype(float)

    y_true = oof["y_true"].astype(float)

    oof["residual"] = prediction - y_true

    oof["absolute_error"] = oof["residual"].abs()

    oof["percentage_error"] = oof["absolute_error"] / y_true

    oof["log_error"] = np.log(prediction.clip(lower=1.0)) - np.log(y_true)

    oof["absolute_log_error"] = oof["log_error"].abs()

    oof["is_over_prediction"] = (oof["residual"] > 0).astype(int)

    return oof


def attach_features(
    error_dataframe: pd.DataFrame,
) -> pd.DataFrame:

    X, y = load_training_data()

    X_dev, _, y_dev, _ = make_holdout_split(
        X,
        y,
    )

    features = X_dev.copy()

    features["row_index"] = features.index

    # 只先带一些容易解释的字段，
    # 避免报告一下变成 100 多列。
    interesting_columns = [
        "row_index",
        "neighborhood",
        "overall_qual",
        "overall_cond",
        "gr_liv_area",
        "total_bsmt_sf",
        "1st_flr_sf",
        "2nd_flr_sf",
        "year_built",
        "year_remod_add",
        "garage_cars",
        "garage_area",
        "full_bath",
        "half_bath",
        "bedroom_abv_gr",
        "kitchen_abv_gr",
        "tot_rms_abv_grd",
        "house_style",
        "bldg_type",
        "sale_condition",
    ]

    existing_columns = [
        column for column in interesting_columns if column in features.columns
    ]

    features = features[existing_columns]

    merged = error_dataframe.merge(
        features,
        on="row_index",
        how="left",
        validate="one_to_one",
    )

    return merged


def build_neighborhood_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    if "neighborhood" not in dataframe.columns:
        return pd.DataFrame()

    report = (
        dataframe.groupby(
            "neighborhood",
            dropna=False,
        )
        .agg(
            house_count=(
                "y_true",
                "size",
            ),
            mean_sale_price=(
                "y_true",
                "mean",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            mean_percentage_error=(
                "percentage_error",
                "mean",
            ),
            mean_absolute_log_error=(
                "absolute_log_error",
                "mean",
            ),
            mean_prediction=(
                PREDICTION_COLUMN,
                "mean",
            ),
            mean_residual=(
                "residual",
                "mean",
            ),
            over_prediction_rate=(
                "is_over_prediction",
                "mean",
            ),
        )
        .reset_index()
    )

    report = report.sort_values(
        by="mean_absolute_log_error",
        ascending=False,
    )

    return report


def build_price_band_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe["price_band"] = pd.qcut(
        dataframe["y_true"],
        q=5,
        duplicates="drop",
    )

    report = (
        dataframe.groupby(
            "price_band",
            observed=True,
        )
        .agg(
            house_count=(
                "y_true",
                "size",
            ),
            mean_sale_price=(
                "y_true",
                "mean",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            mean_percentage_error=(
                "percentage_error",
                "mean",
            ),
            mean_absolute_log_error=(
                "absolute_log_error",
                "mean",
            ),
        )
        .reset_index()
    )

    return report


def build_quality_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    if "overall_qual" not in dataframe.columns:
        return pd.DataFrame()

    return (
        dataframe.groupby(
            "overall_qual",
            dropna=False,
        )
        .agg(
            house_count=(
                "y_true",
                "size",
            ),
            mean_sale_price=(
                "y_true",
                "mean",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            mean_percentage_error=(
                "percentage_error",
                "mean",
            ),
            mean_absolute_log_error=(
                "absolute_log_error",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("overall_qual")
    )


def main() -> None:
    error_dataframe = build_error_dataframe()

    error_dataframe = attach_features(error_dataframe)

    ERROR_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    error_dataframe.to_csv(
        ERROR_REPORT_PATH,
        index=False,
    )

    worst_predictions = error_dataframe.sort_values(
        by="absolute_log_error",
        ascending=False,
    ).head(20)

    worst_predictions.to_csv(
        WORST_PATH,
        index=False,
    )

    neighborhood_report = build_neighborhood_report(error_dataframe)

    if not neighborhood_report.empty:
        neighborhood_report.to_csv(
            NEIGHBORHOOD_PATH,
            index=False,
        )

    price_band_report = build_price_band_report(error_dataframe)

    quality_report = build_quality_report(error_dataframe)

    print()
    print("===== Error by Price Band =====")

    print(price_band_report.to_string(index=False))

    print()

    print("===== Error by Overall Quality =====")

    print(quality_report.to_string(index=False))

    print("===== Overall Error =====")

    print(
        "Mean absolute error:",
        f"{error_dataframe['absolute_error'].mean():,.2f}",
    )

    print(
        "Median absolute error:",
        f"{error_dataframe['absolute_error'].median():,.2f}",
    )

    print(
        "Mean percentage error:",
        f"{error_dataframe['percentage_error'].mean():.2%}",
    )

    print()

    print("===== Worst 20 OOF Predictions =====")

    display_columns = [
        "row_index",
        "y_true",
        PREDICTION_COLUMN,
        "absolute_error",
        "percentage_error",
        "absolute_log_error",
        "neighborhood",
        "overall_qual",
        "gr_liv_area",
        "year_built",
    ]

    display_columns = [
        column for column in display_columns if column in worst_predictions.columns
    ]

    print(worst_predictions[display_columns].to_string(index=False))

    print()

    if not neighborhood_report.empty:
        print("===== Highest Error Neighborhoods =====")

        print(neighborhood_report.head(10).to_string(index=False))

    print()
    print(f"Saved full report: {ERROR_REPORT_PATH}")

    print(f"Saved worst predictions: {WORST_PATH}")

    if not neighborhood_report.empty:
        print(f"Saved neighborhood report: {NEIGHBORHOOD_PATH}")


if __name__ == "__main__":
    main()
