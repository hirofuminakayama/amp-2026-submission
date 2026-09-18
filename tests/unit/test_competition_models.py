import numpy as np
import pandas as pd
import pytest
import torch

from robust_apex_qd.research.competition_models import (
    bounded_loss,
    checked_masks,
    fit_ridge_heads,
    predict_ridge_heads,
    supported_blend,
)


def test_interval_loss_respects_both_censor_boundaries() -> None:
    p = torch.tensor([2.0, 5.0, 1.0, 6.0], requires_grad=True)
    loss = bounded_loss(
        p, torch.tensor([-float("inf"), 4.0, 2.0, 5.0]), torch.tensor([3.0, float("inf"), 2.0, 5.0])
    )
    assert loss.tolist() == [0.0, 0.0, 1.0, 1.0]
    loss.sum().backward()
    assert p.grad is not None
    assert p.grad.tolist() == [0.0, 0.0, -2.0, 2.0]


def test_fold_guard_rejects_group_crossing() -> None:
    rows = pd.DataFrame({"homology_group": ["a", "a"], "homology_fold": [0, 1]})
    with pytest.raises(ValueError, match="cross"):
        checked_masks(rows, 0)


def test_ridge_uses_training_only_and_round_trips(tmp_path) -> None:
    x = np.array([[0.0], [1.0], [10000.0]])
    state = fit_ridge_heads(x[:2], np.array([0, 0]), np.array([1.0, 2.0]), 2, 1.0)
    assert state["mean"][0] == 0.5
    values = predict_ridge_heads(state, x)
    assert np.isnan(values[:, 1]).all()
    np.savez(tmp_path / "state.npz", **state)
    loaded = dict(np.load(tmp_path / "state.npz"))
    np.testing.assert_equal(values, predict_ridge_heads(loaded, x))


def test_unsupported_heads_fallback_is_explicit() -> None:
    p, mask = supported_blend(np.array([[1.0, np.nan]]), np.array([[3.0, 4.0]]), 0.25)
    np.testing.assert_equal(p, [[1.5, 4.0]])
    np.testing.assert_equal(mask, [[True, False]])


def test_target_diagnostics_keep_classification_units_and_missing_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    diagnostic = importlib.import_module("finish_competition_models").target_diagnostics
    rows = pd.DataFrame(
        dict(
            species=["x"] * 3,
            sequence=["A", "B", "C"],
            observation_id=["a", "b", "c"],
            active16=[True, False, True],
            exact_regression=[True, True, True],
            mic_um=[2, 32, 4],
        )
    )
    result = diagnostic(rows, np.array([-2.0, 1.0, np.nan]), "negative_activity_logit", "species")
    assert result[0]["coverage"] == 2
    assert result[0]["top20_activity"] == 1.0
    assert result[0]["mae"] is None
