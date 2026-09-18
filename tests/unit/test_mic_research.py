from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from robust_apex_qd.research.mic_data import fold_assignments, global_identity, measured_bounds
from robust_apex_qd.research.mic_models import censored_normal_nll, mic_metrics


def test_fold_fit_is_training_only_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    from robust_apex_qd.research.mic_data import MICConfig

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    runner = importlib.import_module("train_mic_models")
    config = MICConfig.model_validate_json(
        Path("configs/mic_research.json").read_text()
    ).model_copy(update={"epochs": 2, "cpu_threads": 1})
    rows = pd.DataFrame(
        dict(
            observation_id=["a", "b", "c"],
            sequence=["A", "B", "C"],
            homology_group=["a", "b", "c"],
            sequence_index=[0, 1, 2],
            species=["s"] * 3,
            species_index=[0] * 3,
            strain_index=[-1] * 3,
            objective=["measured_mic"] * 3,
            lower_um=[2.0, 4.0, 8.0],
            upper_um=[2.0, 4.0, 8.0],
            exact_regression=[True] * 3,
            mic_um=[2.0, 4.0, 8.0],
            active16=[True] * 3,
            medium=["m1", None, "held_out_medium"],
            cfu=[None] * 3,
        )
    )
    torch.set_num_threads(1)
    x = np.array([[0.0], [1.0], [10000.0]], dtype=np.float32)
    output = tmp_path / "fit"
    runner.run_fit(
        config,
        rows,
        x,
        np.array([0, 1]),
        np.array([2]),
        "esm8-assay-normal",
        2,
        42,
        output,
        "protocol",
    )
    bundle = torch.load(output / "weights.pt", weights_only=True)
    assert bundle["feature_mean"][0].item() == 0.5
    assert "held_out_medium" not in bundle["assay_categories"][0]
    assert bundle["supported_heads"] == [0]
    with pytest.raises(ValueError, match="cross"):
        runner.run_fit(
            config,
            rows,
            x,
            np.array([0, 1]),
            np.array([1]),
            "esm8-normal",
            2,
            42,
            tmp_path / "bad",
            "protocol",
        )


def test_censored_likelihood_has_correct_probabilities_and_tail_gradients() -> None:
    mean = torch.tensor([0.0] * 5, dtype=torch.float64, requires_grad=True)
    scale = torch.ones(5, dtype=torch.float64, requires_grad=True)
    lower = torch.tensor([0, -float("inf"), 0, -1, 100.0], dtype=torch.float64)
    upper = torch.tensor([0, 0, float("inf"), 1, float("inf")], dtype=torch.float64)
    loss = censored_normal_nll(mean, scale, lower, upper)
    np.testing.assert_allclose(
        loss[:4].detach(), [0.9189385332, np.log(2), np.log(2), 0.3817151463]
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()
    assert mean.grad[-1] < 0


def test_interval_likelihood_is_stable_in_both_tails() -> None:
    mean = torch.tensor([-100.0, 100.0], dtype=torch.float64, requires_grad=True)
    scale = torch.ones(2, dtype=torch.float64, requires_grad=True)
    loss = censored_normal_nll(mean, scale, torch.zeros(2), torch.ones(2))
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert mean.grad[0] < 0 < mean.grad[1]
    with pytest.raises(ValueError, match="bound"):
        censored_normal_nll(mean, scale, torch.ones(2), torch.zeros(2))


def test_bounds_do_not_impute_missing_or_consensus_measurements() -> None:
    rows = pd.DataFrame(
        dict(
            objective=["measured_mic"] * 3 + ["qmap_consensus"],
            lower_um=[8, 64, None, 4],
            upper_um=[8, None, None, 4],
        )
    )
    low, high, usable = measured_bounds(rows)
    np.testing.assert_equal(low[:2], [3, 6])
    assert np.isposinf(high[1])
    np.testing.assert_equal(usable, [True, True, False, False])
    rows.loc[0, "lower_um"] = 0
    with pytest.raises(ValueError, match="positive"):
        measured_bounds(rows)


def test_folds_keep_groups_and_exclude_outer_validation_from_inner() -> None:
    groups = {"a": "x", "b": "x", "c": "y", "d": "z", "e": "w"}
    outer = fold_assignments(groups, 3, 42)
    assert outer["a"] == outer["b"]
    train = {s: g for s, g in groups.items() if outer[s] != 0}
    inner = fold_assignments(train, 3, 42)
    assert set(inner) == set(train)
    assert not set(inner) & {s for s, f in outer.items() if f == 0}
    assert fold_assignments({"a": "x", "b": "x"}, 5, 42) == {"a": -1, "b": -1}


def test_global_identity_is_symmetric_and_includes_gaps() -> None:
    assert global_identity("ACDEFGHK", "ACDEFGHK") == 1
    a, b = "ACDEFGHK", "ACDEFGHKLL"
    assert global_identity(a, b) == global_identity(b, a) == 0.8


def test_metrics_exclude_censoring_and_report_coverage() -> None:
    rows = pd.DataFrame(dict(exact_regression=[True, False, True], mic_um=[2, 64, 4]))
    metric = mic_metrics(rows, np.array([2.0, 0.0, np.nan]))
    assert metric["mae"] == 1
    assert metric["exact_rows"] == 1
    assert metric["predicted_rows"] == 2
    assert metric["within1"] == 1
    assert metric["pcc"] is None


def test_prediction_handoff_preserves_explicit_strain_fallback() -> None:
    from robust_apex_qd.research.prediction_mic import (
        blend_strain_predictions,
        prediction_records,
    )

    mean = np.full((2, 18), np.nan)
    mean[:, :7] = 1.0
    mean[:, 7] = [2.0, 3.0]
    apex = np.full((2, 11), 5.0)
    frame = prediction_records(
        ["ACDEFGHK", "ACDEFGHL"], mean, np.full((2, 18), np.nan), apex, "new", "apex"
    )
    assert frame.supported.sum() == 2
    assert frame.loc[~frame.supported, "prediction"].eq(5).all()
    assert frame.loc[~frame.supported, "model_sha256"].eq("apex").all()
    sequences = ["ACDEFGHK", "ACDEFGHL"]
    np.testing.assert_equal(blend_strain_predictions(frame, apex, 1, sequences), apex)
    np.testing.assert_equal(
        blend_strain_predictions(frame, apex, 0.5, sequences),
        blend_strain_predictions(frame.sample(frac=1, random_state=3), apex, 0.5, sequences),
    )
    assert np.isnan(frame.scale).all()
    frame["target_level"] = "species"
    with pytest.raises(ValueError, match="strain"):
        blend_strain_predictions(frame, apex, 0.5, sequences)


def test_baseline_fit_does_not_mutate_outer_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    runner = importlib.import_module("finish_mic_research")
    rows = pd.DataFrame(
        dict(
            observation_id=["a", "b", "c"],
            sequence=["A", "B", "C"],
            homology_group=["a", "b", "c"],
            sequence_index=[0, 1, 2],
            species_index=[0] * 3,
            strain_index=[-1] * 3,
            objective=["measured_mic"] * 3,
            exact_regression=[True, False, True],
            mic_um=[2.0, 4.0, 8.0],
            consensus_um=[None] * 3,
        )
    )
    membership = np.array([True, True, False])
    runner.baseline_fold(
        rows, np.array([[0.0], [1.0], [2.0]]), membership, ~membership, 1.0, tmp_path / "fit"
    )
    np.testing.assert_equal(membership, [True, True, False])
