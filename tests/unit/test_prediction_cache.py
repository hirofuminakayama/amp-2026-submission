from pathlib import Path

import numpy as np
import pytest

from robust_apex_qd.research.prediction_cache import PredictionCache, isolated_rows


def test_cache_reorders_by_sequence_and_rejects_stale_inputs(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "cache")
    values = np.array([[1.0, 2.0], [3.0, np.nan]])
    contract = {"weights": "a", "features": "b", "configuration": "c"}
    cache.write(["AA", "BB"], values, contract)
    np.testing.assert_equal(cache.read(["BB", "AA"], contract), values[[1, 0]])
    for changed in [{**contract, "weights": "different"}, {**contract, "features": "new"}]:
        with pytest.raises(ValueError):
            cache.read(["AA", "BB"], changed)
    with pytest.raises(ValueError):
        cache.read(["AA", "AA"], contract)
    with pytest.raises(ValueError):
        cache.read(["AA", "CC"], contract)
    (tmp_path / "cache" / "values.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        cache.read(["AA", "BB"], contract)


def test_training_partition_rejects_sequence_and_group_leakage() -> None:
    isolated_rows(["A", "B"], ["x", "y"], np.array([0]), np.array([1]))
    for sequences, groups in [(["A", "A"], ["x", "y"]), (["A", "B"], ["x", "x"])]:
        with pytest.raises(ValueError):
            isolated_rows(sequences, groups, np.array([0]), np.array([1]))
