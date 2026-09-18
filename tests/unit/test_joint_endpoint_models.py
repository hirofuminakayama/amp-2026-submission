import numpy as np
import pytest

from robust_apex_qd.research.joint_endpoint_models import fit_joint_head, predict_joint_head


def test_joint_head_excludes_outer_features_and_labels() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2000.0, 3000.0]])
    sequence_rows = np.array([0, 1, 2, 0, 1, 2])
    heads = np.array([0, 0, 0, 7, 7, 7])
    lo = np.array([1.0, 2.0, 9000.0, 5.0, 6.0, 8000.0])
    hi = lo.copy()
    groups = np.array(["a", "b", "c"])
    tr, vi = np.array([0, 1]), np.array([2])
    first = fit_joint_head(
        x, sequence_rows, heads, lo, hi, groups, tr, vi, shared=True, seed=42, epochs=3
    )
    lo[[2, 5]], hi[[2, 5]] = -9000.0, -9000.0
    again = fit_joint_head(
        x, sequence_rows, heads, lo, hi, groups, tr, vi, shared=True, seed=42, epochs=3
    )
    np.testing.assert_equal(predict_joint_head(first, x), predict_joint_head(again, x))
    assert first["mean"].tolist() == [0.5, 0.5]
    assert np.isnan(predict_joint_head(first, x)[:, 1:7]).all()
    with pytest.raises(ValueError, match="group"):
        fit_joint_head(
            x,
            sequence_rows,
            heads,
            lo,
            hi,
            np.array(["a", "b", "a"]),
            tr,
            vi,
            shared=True,
            seed=42,
            epochs=3,
        )
