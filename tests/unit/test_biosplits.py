import numpy as np

from robust_apex_qd.research.biosplits import extend_identity, shared_folds


def test_new_endpoint_can_bridge_old_groups_and_requires_new_folds() -> None:
    sequences = ["a", "b", "hc"]
    matrix = np.array([[1, 0.2, 0.7], [0.2, 1, 0.7], [0.7, 0.7, 1]])
    result = shared_folds(sequences, matrix, threshold=0.6, outer_count=5, inner_count=3, seed=42)
    assert len(set(result["groups"].values())) == 1
    assert set(result["outer"].values()) == {-1}
    assert result["inner"] == {}


def test_extension_reuses_old_cells_and_only_aligns_new_pairs() -> None:
    calls = []

    def identity(a: str, b: str) -> float:
        calls.append((a, b))
        return 0.1

    old = np.array([[1, 0.2], [0.2, 1]], dtype=np.float32)
    matrix = extend_identity(["b", "c"], old, ["a", "b", "c"], identity)
    np.testing.assert_equal(matrix[1:, 1:], old)
    assert len(calls) == 2


def test_inner_split_never_contains_outer_validation() -> None:
    result = shared_folds(
        list("abcdef"), np.eye(6), threshold=0.6, outer_count=3, inner_count=2, seed=42
    )
    for fold, inner in result["inner"].items():
        assert all(result["outer"][s] != int(fold) for s in inner)
        assert len(inner) == 4
