"""Shared repeated folds, corrected paired tests and conservative selection."""

import argparse
import hashlib
import itertools
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold

from src.competition_data import ROOT, load_competition_data
from src.competition_model import fit_artifact, predict_artifact
from src.split import make_holdout_split

CANDIDATES = {
    "catboost_baseline": ("catboost", "baseline"),
    "catboost_size_age": ("catboost", "size_age"),
    "catboost_sale_interactions": ("catboost", "sale_interactions"),
    "xgboost_baseline": ("xgboost", "baseline"),
    "gradient_boosting_baseline": ("gradient_boosting", "baseline"),
}
SEEDS = (42, 52, 62)


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if not (
        np.isfinite(y).all()
        and np.isfinite(prediction).all()
        and (y > 0).all()
        and (prediction > 0).all()
    ):
        raise ValueError("Log RMSE requires finite, positive targets and predictions")
    return {
        "log_rmse": float(root_mean_squared_error(np.log(y), np.log(prediction))),
        "rmse": float(root_mean_squared_error(y, prediction)),
        "mae": float(mean_absolute_error(y, prediction)),
        "mean_residual": float(np.mean(prediction - y)),
    }


def corrected_paired_test(deltas, test_train_ratio):
    """Corrected resampled t approximation: account for overlapping training folds."""
    deltas = np.asarray(deltas, dtype=float)
    mean = float(deltas.mean())
    se = float(np.sqrt((1 / len(deltas) + test_train_ratio) * deltas.var(ddof=1)))
    p = (
        (1.0 if mean == 0 else 0.0)
        if se == 0
        else float(2 * t.sf(abs(mean / se), len(deltas) - 1))
    )
    radius = float(t.ppf(0.975, len(deltas) - 1) * se)
    return {
        "delta_log_rmse": mean,
        "ci_low": mean - radius,
        "ci_high": mean + radius,
        "p_value": p,
        "corrected_se": se,
    }


def paired_comparisons(folds):
    pivot = folds.pivot(index=["seed", "fold"], columns="model", values="log_rmse")
    if pivot.isna().any().any():
        raise ValueError("Paired comparisons require identical complete folds")
    ratio = float((folds.validation_rows / folds.training_rows).mean())
    result = pd.DataFrame(
        [
            dict(
                model_a=a,
                model_b=b,
                **corrected_paired_test(pivot[a] - pivot[b], ratio),
            )
            for a, b in itertools.combinations(pivot.columns, 2)
        ]
    )
    order = np.argsort(result.p_value.to_numpy())
    adjusted = np.maximum.accumulate(
        result.p_value.to_numpy()[order] * np.arange(len(order), 0, -1)
    )
    result.loc[order, "p_holm"] = np.minimum(1, adjusted)
    return result


def select_candidate(summary, pairs):
    baseline = "catboost_baseline"
    supported = [baseline]
    for row in pairs.itertuples():
        if row.p_holm >= 0.05:
            continue
        if row.model_a == baseline and row.delta_log_rmse > 0:
            supported.append(row.model_b)
        if row.model_b == baseline and row.delta_log_rmse < 0:
            supported.append(row.model_a)
    return (
        summary[summary.model.isin(supported)]
        .sort_values("cv_log_rmse_mean")
        .iloc[0]
        .model
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/repeated_cv")
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    train, _ = load_competition_data()
    X = train.drop(columns=["id", "sale_price"])
    X_dev, X_holdout, y_dev, _ = make_holdout_split(X, train.sale_price)
    pd.DataFrame(
        {
            "Id": train.id,
            "partition": np.where(
                train.index.isin(X_dev.index), "development", "holdout"
            ),
        }
    ).to_csv(output / "split_assignment.csv", index=False)
    fold_rows, prediction_frames = [], []
    for seed in SEEDS:
        for fold, (fit, valid) in enumerate(
            KFold(5, shuffle=True, random_state=seed).split(X_dev), 1
        ):
            high_cutoff = float(y_dev.iloc[fit].quantile(0.9))
            for name, (family, variant) in CANDIDATES.items():
                start = time.perf_counter()
                artifact = fit_artifact(
                    X_dev.iloc[fit], y_dev.iloc[fit], family, variant
                )
                fit_time = time.perf_counter() - start
                prediction = predict_artifact(artifact, X_dev.iloc[valid])
                score = metrics(y_dev.iloc[valid], prediction)
                fold_rows.append(
                    dict(
                        model=name,
                        seed=seed,
                        fold=fold,
                        **score,
                        fit_seconds=fit_time,
                        training_rows=len(fit),
                        validation_rows=len(valid),
                    )
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "Id": train.loc[X_dev.iloc[valid].index, "id"].to_numpy(),
                            "model": name,
                            "seed": seed,
                            "fold": fold,
                            "y_true": y_dev.iloc[valid].to_numpy(),
                            "prediction": prediction,
                            "sale_condition": X_dev.iloc[
                                valid
                            ].sale_condition.to_numpy(),
                            "high_price": y_dev.iloc[valid].to_numpy() >= high_cutoff,
                        }
                    )
                )
                pd.DataFrame(fold_rows).to_csv(output / "fold_scores.csv", index=False)
                print(f"{seed}/{fold} {name}: {score['log_rmse']:.6f}", flush=True)
    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    summary = (
        folds.groupby("model")
        .agg(
            cv_log_rmse_mean=("log_rmse", "mean"),
            cv_log_rmse_std=("log_rmse", "std"),
            cv_rmse_mean=("rmse", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
        )
        .reset_index()
    )
    pooled = predictions.groupby("model").apply(
        lambda g: metrics(g.y_true, g.prediction)["log_rmse"], include_groups=False
    )
    summary["pooled_repeated_log_rmse"] = summary.model.map(pooled)
    summary = summary.sort_values("cv_log_rmse_mean")
    summary.to_csv(output / "model_comparison.csv", index=False)
    summary.to_csv(ROOT / "reports/model_comparison.csv", index=False)
    pairs = paired_comparisons(folds)
    pairs.to_csv(output / "paired_tests.csv", index=False)
    segments = []
    for name, group in predictions.groupby("model"):
        selections = {
            "all": group,
            "high_price_train_q90": group[group.high_price],
            "over_400k": group[group.y_true > 400000],
        }
        selections.update(
            {
                f"sale_{condition}": part
                for condition, part in group.groupby("sale_condition")
            }
        )
        for segment, part in selections.items():
            if len(part):
                segments.append(
                    dict(
                        model=name,
                        segment=segment,
                        rows=len(part),
                        unique_houses=part.Id.nunique(),
                        **metrics(part.y_true, part.prediction),
                    )
                )
    pd.DataFrame(segments).to_csv(output / "segment_metrics.csv", index=False)
    selected = select_candidate(summary, pairs)
    family, variant = CANDIDATES[selected]
    selection = {
        "model": selected,
        "family": family,
        "variant": variant,
        "seeds": SEEDS,
        "folds": 5,
        "development_rows": len(X_dev),
        "holdout_rows": len(X_holdout),
        "train_sha256": hashlib.sha256(
            (ROOT / "data/raw/train.csv").read_bytes()
        ).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
        "rule": "Keep baseline unless Holm-adjusted corrected paired test p < 0.05",
        "inference": "Approximate corrected resampled t; marginal 95% CIs; Holm over all pairs",
    }
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Frozen selection: {selected}")


if __name__ == "__main__":
    main()
