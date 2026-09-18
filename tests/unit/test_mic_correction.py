from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_apex_qd.research.mic_correction import correction_masks, paired_comparison


def test_corrections_only_change_training_and_share_validation() -> None:
    rows = pd.DataFrame(
        dict(
            observation_id=["a", "b", "c", "d"],
            homology_fold=[0, 0, 1, 1],
            exact_regression=[True, True, True, False],
        )
    )
    original, corrected, valid = correction_masks(rows, {"a"}, 1)
    assert original.tolist() == [True, True, False, False]
    assert corrected.tolist() == [False, True, False, False]
    assert valid.tolist() == [False, False, True, True]
    _, _, valid = correction_masks(rows, {"a"}, 0)
    assert valid.tolist() == [False, True, False, False]
    with pytest.raises(ValueError):
        correction_masks(rows, {"missing"}, 0)


def frame(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        dict(
            observation_id=[str(i) for i in range(count)],
            component_id=[str(i) for i in range(count)],
            species=["species"] * count,
            exact_regression=[True] * count,
            mic_um=[1.0] * count,
            prediction_original=np.ones(count),
            prediction_corrected=np.zeros(count),
        )
    )


def test_component_bootstrap_preserves_pairing_and_insufficient_ci() -> None:
    result = paired_comparison(frame(5))
    assert result["delta_macro_mae"] == -1
    assert result["ci95"] == [-1, -1]
    assert result["components"] == 5
    assert paired_comparison(frame(4))["ci95"] is None


def test_censored_and_unsupported_rows_do_not_enter_exact_pair_metric() -> None:
    rows = frame(5)
    rows.loc[0, "exact_regression"] = False
    rows.loc[1, "prediction_corrected"] = np.nan
    result = paired_comparison(rows)
    assert result["common_exact_rows"] == 3
    assert result["excluded_rows"] == 2
    assert result["ci95"] is None
    assert paired_comparison(rows.iloc[:0])["delta_macro_mae"] is None


def test_completed_fit_reuse_requires_hashes_membership_and_settings(tmp_path: Path) -> None:
    import json

    from robust_apex_qd.features.embeddings import file_sha256
    from robust_apex_qd.research.mic_correction import verify_completed_fit

    metadata = dict(
        training_ids=["a"],
        validation_ids=["b"],
        arm={"epochs": 10},
        seed=42,
        serialization_equal=True,
    )
    (tmp_path / "fit.json").write_text(json.dumps(metadata))
    (tmp_path / "weights.pt").write_bytes(b"fixture")
    manifest = dict(
        inputs_sha256={"fixture": "abc"},
        artifacts_sha256={p.name: file_sha256(p) for p in tmp_path.iterdir()},
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    verify_completed_fit(tmp_path, ["a"], ["b"], {"epochs": 10}, 42, {"fixture": "abc"})
    with pytest.raises(ValueError):
        verify_completed_fit(tmp_path, ["a"], ["b"], {"epochs": 1}, 42, {"fixture": "abc"})
    with pytest.raises(ValueError):
        verify_completed_fit(tmp_path, ["b"], ["a"], {"epochs": 10}, 42, {"fixture": "abc"})
    (tmp_path / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_completed_fit(tmp_path, ["a"], ["b"], {"epochs": 10}, 42, {"fixture": "abc"})
