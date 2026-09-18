import numpy as np
import pandas as pd
import pytest

from robust_apex_qd.research.models import (
    evaluate_predictions,
    fold_masks,
    paired_interval,
    ridge_fold,
    select_candidate,
)


def test_candidate_selection_includes_registered_ensembles_and_resolves_numeric_ties() -> None:
    candidates = [
        {"model": "APEX", "macro_top20_active": 0.8, "mae_log2_exact": 4.4},
        {"model": "mlp", "macro_top20_active": 0.6888888888888889, "mae_log2_exact": 2.2},
        {"model": "tuned", "macro_top20_active": 0.6888888888888888, "mae_log2_exact": 1.9},
    ]
    assert select_candidate(candidates)["model"] == "tuned"
    candidates.append(
        {"model": "tuned_apex_mean", "macro_top20_active": 1.0, "mae_log2_exact": 2.8}
    )
    assert select_candidate(candidates)["model"] == "tuned_apex_mean"


def test_validation_labels_and_features_do_not_fit_training_scaler() -> None:
    features = np.asarray([[0.0], [2.0], [1000.0]])
    train = np.asarray([True, True, False])
    valid = ~train
    strains = np.zeros(3, dtype=int)
    first, scaler, _ = ridge_fold(features, strains, np.array([3.0, 5.0, 10.0]), train, valid, 10.0)
    second, _, _ = ridge_fold(features, strains, np.array([3.0, 5.0, -100.0]), train, valid, 10.0)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(scaler.mean_, [1.0])


def test_one_group_has_no_uncertainty_interval() -> None:
    result = paired_interval(pd.DataFrame({"group": ["a"]}), "prediction", ["strain"], 42, 30)
    assert result["valid_replicates"] == 0
    assert result["delta_ci_low"] is None


def test_fold_masks_exclude_censored_training_and_whole_validation_group() -> None:
    rows = pd.DataFrame(
        {
            "validation_fold": [0, 0, 1, 2],
            "group": ["a", "a", "b", "c"],
            "exact_mic": [True, True, False, True],
            "mic_um": [8.0, 16.0, 64.0, 32.0],
        }
    )
    train, valid = fold_masks(rows, 0)
    assert train.tolist() == [False, False, False, True]
    assert valid.tolist() == [True, True, False, False]
    rows.loc[3, "group"] = "a"
    with pytest.raises(ValueError, match="group"):
        fold_masks(rows, 0)


def test_metrics_rank_low_mic_and_separate_censoring_from_regression() -> None:
    rows = pd.DataFrame(
        {
            "sequence": list("abcde"),
            "apex_pathogen": ["strain"] * 5,
            "group": list("abcde"),
            "active16": [1, 0, 0, 0, 0],
            "exact_mic": [True, True, True, True, False],
            "mic_um": [8.0, 32.0, 64.0, 128.0, 128.0],
            "prediction": [3.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    result = evaluate_predictions(rows, "prediction", ["strain"])
    assert result["macro_top20_active"] == 1.0
    assert result["mae_log2_exact"] == 0.0
    assert result["spearman_exact"] == 1.0
    rows.loc[0, "prediction"] = np.nan
    assert evaluate_predictions(rows, "prediction", ["strain"])["macro_top20_active"] is None


def test_paired_bootstrap_identical_predictors_have_zero_difference() -> None:
    rows = pd.DataFrame(
        {
            "sequence": list("abcde"),
            "apex_pathogen": ["strain"] * 5,
            "group": list("abcde"),
            "active16": [1, 1, 0, 0, 0],
            "exact_mic": [True] * 5,
            "mic_um": [8.0, 16.0, 32.0, 64.0, 128.0],
            "prediction": [3.0, 4.0, 5.0, 6.0, 7.0],
            "apex_log2": [3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    result = paired_interval(rows, "prediction", ["strain"], 42, 30)
    assert result["valid_replicates"] == 30
    assert result["delta_ci_low"] == result["delta_ci_high"] == 0.0
