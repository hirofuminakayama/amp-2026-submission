import numpy as np
import pytest

from robust_apex_qd.research.biomodels import (
    HC50Bundle,
    fit_hc50,
    fit_interval_hc50,
    hc50_signal,
    predict_hc50,
)


def test_hc50_scaler_and_weights_ignore_validation_labels() -> None:
    x = np.array([[0.0], [1.0], [10000.0]])
    y = np.array([1.0, 2.0, -10000.0])
    groups = np.array(["a", "b", "c"])
    train, valid = np.array([0, 1]), np.array([2])
    model = fit_hc50(x, y, groups, train, valid, alpha=1, feature_sha256="fixture")
    y[2] = 10000
    again = fit_hc50(x, y, groups, train, valid, alpha=1, feature_sha256="fixture")
    assert model == again
    assert model.mean == [0.5]
    loaded = HC50Bundle.model_validate_json(model.model_dump_json())
    np.testing.assert_equal(predict_hc50(model, x, "fixture"), predict_hc50(loaded, x, "fixture"))
    with pytest.raises(ValueError, match="feature"):
        predict_hc50(model, x, "changed-feature-version")


def test_hc50_rejects_leaking_groups_and_nonfinite_training_labels() -> None:
    x = np.array([[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="group"):
        fit_hc50(
            x,
            np.ones(3),
            np.array(["a", "b", "a"]),
            np.array([0, 1]),
            np.array([2]),
            alpha=1,
            feature_sha256="fixture",
        )


def test_positive_ordinal_signal_is_not_rejected_by_a_new_mae_gate() -> None:
    result = hc50_signal(mae=1.92, spearman=0.17, median_mae=1.80)
    assert result["ranking_signal"]
    assert not result["quantitative_signal"]
    assert result["scenario_candidate"]
    assert not hc50_signal(mae=2.0, spearman=-0.1, median_mae=1.8)["scenario_candidate"]


def test_interval_hc50_preserves_right_censoring() -> None:
    x = np.zeros((3, 1))
    model = fit_interval_hc50(
        x,
        np.array([7.0, 8.0, 0.0]),
        np.array([np.inf, np.inf, 0.0]),
        np.array(["a", "b", "c"]),
        np.array([0, 1]),
        np.array([2]),
        alpha=0.1,
        feature_sha256="fixture",
        epochs=200,
    )
    prediction = predict_hc50(model, x, "fixture")
    assert prediction[2] > 7.9
    assert model.endpoint == "measured_hc50"
    with pytest.raises(ValueError, match=r"[Ff]inite"):
        fit_hc50(
            x,
            np.array([np.nan, 1, 2]),
            np.array(["a", "b", "c"]),
            np.array([0, 1]),
            np.array([2]),
            alpha=1,
            feature_sha256="fixture",
        )
