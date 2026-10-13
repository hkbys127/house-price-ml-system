import numpy as np
import pandas as pd
import pytest

from src.competition_data import load_competition_data
from src.competition_model import fit_artifact, predict_artifact, transform_features
from src.repeated_cv import (
    corrected_paired_test,
    metrics,
    paired_comparisons,
    select_candidate,
)
from src.submit import validate_submission


def test_log_metric_matches_competition():
    assert metrics([10, 100], [20, 200])["log_rmse"] == pytest.approx(np.log(2))


@pytest.mark.parametrize("price", [0, -1, np.inf, np.nan])
def test_invalid_predictions_rejected(price):
    with pytest.raises(ValueError):
        metrics([10], [price])
    with pytest.raises(ValueError):
        validate_submission(
            pd.DataFrame({"Id": [1], "SalePrice": [price]}), pd.DataFrame({"Id": [1]})
        )


def test_submission_rejects_wrong_id_order():
    with pytest.raises(ValueError):
        validate_submission(
            pd.DataFrame({"Id": [2, 1], "SalePrice": [10, 20]}),
            pd.DataFrame({"Id": [1, 2]}),
        )


def test_features_are_row_local_and_do_not_mutate():
    train, _ = load_competition_data()
    X = train.drop(columns=["id", "sale_price"])
    original = X.copy(deep=True)
    full = transform_features(X, "sale_interactions")
    pd.testing.assert_frame_equal(
        full.iloc[:3], transform_features(X.iloc[:3], "sale_interactions")
    )
    pd.testing.assert_frame_equal(X, original)
    assert "sale_price" not in full
    assert full.loc[X.sale_condition.eq("Partial"), "sale_partial"].eq(1).all()


def test_artifact_roundtrip_and_column_order(tmp_path):
    import joblib

    train, _ = load_competition_data()
    X = train.drop(columns=["id", "sale_price"]).iloc[:40]
    artifact = fit_artifact(
        X, train.sale_price.iloc[:40], "gradient_boosting", "sale_interactions"
    )
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    restored = joblib.load(path)
    prediction = predict_artifact(restored, X[X.columns[::-1]])
    np.testing.assert_allclose(prediction, predict_artifact(artifact, X))
    assert (prediction > 1000).all()  # Must invert the log target.
    with pytest.raises(ValueError):
        predict_artifact(restored, X.drop(columns="sale_condition"))


def test_corrected_test_handles_dependence_and_ties():
    differences = np.array([0.01, 0.02, 0.03, 0.02, 0.01])
    test = corrected_paired_test(differences, 0.25)
    assert test["corrected_se"] > differences.std(ddof=1) / np.sqrt(len(differences))
    assert corrected_paired_test(np.zeros(15), 0.25)["p_value"] == 1


def test_pairing_and_conservative_selection():
    folds = pd.DataFrame(
        [
            {
                "seed": seed,
                "fold": fold,
                "model": model,
                "log_rmse": 0.12 + (0.001 if model == "catboost_baseline" else 0),
                "training_rows": 80,
                "validation_rows": 20,
            }
            for seed in [42, 52, 62]
            for fold in range(5)
            for model in ["catboost_baseline", "catboost_size_age"]
        ]
    )
    result = paired_comparisons(folds)
    assert (result.p_holm >= result.p_value).all()
    with pytest.raises(ValueError):
        paired_comparisons(folds.iloc[1:])
    summary = pd.DataFrame(
        {
            "model": ["catboost_baseline", "catboost_size_age"],
            "cv_log_rmse_mean": [0.121, 0.120],
        }
    )
    assert select_candidate(summary, result) == "catboost_size_age"
    result["p_holm"] = 0.3
    assert select_candidate(summary, result) == "catboost_baseline"


def test_native_catboost_missing_unseen_categories_and_log_inverse(monkeypatch):
    from catboost import CatBoostRegressor

    import src.competition_model as module

    monkeypatch.setattr(
        module,
        "get_catboost_model",
        lambda: CatBoostRegressor(
            iterations=5, verbose=False, allow_writing_files=False
        ),
    )
    train, _ = load_competition_data()
    X = train.drop(columns=["id", "sale_price"]).iloc[:30]
    artifact = fit_artifact(X, train.sale_price.iloc[:30])
    sample = X.iloc[:2].copy()
    sample.loc[sample.index[0], "sale_condition"] = "new_category"
    sample.loc[sample.index[1], "sale_condition"] = None
    prices = predict_artifact(artifact, sample)
    assert np.isfinite(prices).all() and (prices > 1000).all()
    artifact["target_transform"] = "identity"
    with pytest.raises(ValueError):
        predict_artifact(artifact, sample)
