import numpy as np
import pandas as pd
import pytest

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS
from robust_apex_qd.research.selection import (
    align_apex_scores,
    bounded_quotas,
    complete_panel_activity,
    keyed_subset,
    library_rows,
    selection_scores,
)


def test_historical_panel_uses_canonical_strain_mapping_and_rejects_incomplete_panels() -> None:
    strains = list(APEX_PATHOGENS)
    strains[0] = "Acinetobacter baumannii ATCC 19606"
    rows = pd.DataFrame({"peptide_id": [1] * 11, "strain": strains, "active": [1] * 11})
    assert complete_panel_activity(rows).tolist() == [1.0]
    with pytest.raises(ValueError, match="complete"):
        complete_panel_activity(rows.iloc[:-1])


def test_library_membership_does_not_reintroduce_rejected_duplicate_pool_rows() -> None:
    pool = pd.DataFrame({"sequence": ["a", "a", "b"], "valid": [True, False, True]})
    assert library_rows(pool, ["a", "b"]).index.tolist() == [0, 2]
    with pytest.raises(ValueError):
        library_rows(pool, ["missing"])


def test_pool_scores_align_by_sequence_and_reject_missing_valid_predictions() -> None:
    tensor = np.asarray([np.full((8, 11), 8.0), np.full((8, 11), 32.0)])
    scores = align_apex_scores(["a", "b"], tensor, ["b", "invalid", "a"], [True, False, True])
    assert scores["B0"][0] == -32.0
    assert np.isnan(scores["B0"][1])
    assert scores["B0"][2] == -8.0
    with pytest.raises(ValueError):
        align_apex_scores(["a", "b"], tensor, ["missing"], [True])


def test_keyed_subsets_are_nested_order_invariant_and_shared_across_libraries() -> None:
    values = [f"sequence{i}" for i in range(30)]
    small = keyed_subset(values, 8, 42)
    assert small == keyed_subset(list(reversed(values)), 8, 42)
    assert set(small) <= set(keyed_subset(values, 16, 42))
    ranked = keyed_subset(values, 30, 42)
    assert keyed_subset(ranked[5:], 8, 42) == ranked[5:13]
    with pytest.raises(ValueError):
        keyed_subset(["a", "a"], 1, 42)


def test_reference_quotas_respect_capacity_and_redistribute_deficits() -> None:
    result = bounded_quotas(10, {0: 9.0, 1: 1.0}, {0: 2, 1: 20})
    assert result == {0: 2, 1: 8}
    assert bounded_quotas(10, {0: 3.0, 1: 1.0}, {0: 20, 1: 20}) == {0: 8, 1: 2}
    with pytest.raises(ValueError):
        bounded_quotas(30, {0: 1.0}, {0: 20})


def test_balanced_aggregation_gives_equal_weight_to_bacterial_groups() -> None:
    tensor = np.full((2, 8, 11), 32.0, dtype=np.float32)
    tensor[0, :, :7] = 8.0
    tensor[1, :, 7:] = 8.0
    scores = selection_scores(tensor)
    assert scores["B2"][0] > scores["B2"][1]
    assert scores["balanced"][0] == scores["balanced"][1] == 0.5
    assert scores["tail90"].tolist() == [-5.0, -5.0]
