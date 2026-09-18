import numpy as np
import pandas as pd
import pytest

from robust_apex_qd.research.scale import merge_pool_sources, mix_library


def test_cached_predictions_align_by_sequence_and_leave_new_rows_missing() -> None:
    from robust_apex_qd.research.scale import align_cached_rows

    values, available = align_cached_rows(
        ["K", "L", "A"], ["A", "K"], np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    np.testing.assert_array_equal(available, [True, False, True])
    np.testing.assert_array_equal(values[[0, 2]], [[3.0, 4.0], [1.0, 2.0]])
    assert np.isnan(values[1]).all()
    with pytest.raises(ValueError, match="unique"):
        align_cached_rows(["A"], ["A", "A"], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="row count"):
        align_cached_rows(["A"], ["A"], np.zeros((2, 2)))


def test_pool_union_tracks_duplicates_references_and_source_order() -> None:
    first, second, known = "A" * 12, "K" * 13, "L" * 14
    pool, provenance, inventory = merge_pool_sources(
        {"old": [first, known, "X" * 12], "new": [first, second, second]}, {known}
    )
    assert pool.sequence.tolist() == [first, second]
    assert pool.candidate_id.is_unique
    assert len(provenance) == 6
    assert inventory.raw_count.tolist() == [3, 3]
    assert inventory.additional_unique.tolist() == [1, 1]
    assert provenance.query("sequence == @first").sequence_sha256.nunique() == 1
    assert set(provenance.reason) == {"", "exact_reference_overlap", "invalid", "duplicate"}


def test_mixture_reports_actual_supply_without_duplicating_donors() -> None:
    base = pd.DataFrame({"sequence": ["A", "B", "C", "D", "E"], "raw_order": range(5)})
    donor = pd.DataFrame({"sequence": ["F", "F", "A"], "raw_order": [5, 6, 0]})
    library, counts = mix_library(base, donor, size=5, fraction=0.5)
    assert len(library) == library.sequence.nunique() == 5
    assert counts["desired_donor_count"] == 2
    assert counts["actual_donor_count"] == 1
    assert counts["actual_fraction"] == 0.2
    assert library.sequence.tolist() == ["F", "A", "B", "C", "D"]
    with pytest.raises(ValueError, match="Insufficient"):
        mix_library(base.iloc[:1], donor, size=5, fraction=0.5)


def test_reference_mix_keeps_length_totals_and_respects_cluster_capacity() -> None:
    from robust_apex_qd.research.scale import reference_mix

    pool = pd.DataFrame(
        dict(
            sequence=list("ABCDEF"),
            length=[10] * 4 + [20] * 2,
            embedding_cluster=[0, 0, 1, 1, 0, 1],
            raw_order=range(6),
            physchem_ood=[0.0] * 6,
            embedding_ood=[0.0] * 6,
        )
    )
    l2 = pool.iloc[[0, 1, 4]]
    reference = pd.DataFrame(dict(length=[10, 20], cluster=[1, 1]))
    result = reference_mix(pool, l2, reference, 1.0)
    assert result.sequence.tolist() == ["C", "D", "F"]
    assert result.length.value_counts().to_dict() == {10: 2, 20: 1}
    assert reference_mix(pool, l2, reference, 0).sequence.tolist() == l2.sequence.tolist()


def test_activity_group_averages_model_ranks_without_ranking_the_mean_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    module = importlib.import_module("evaluate_competition_pool")
    frame = pd.DataFrame(
        dict(
            apex_activity=[3, 1, 2],
            apex_weak_species=[1, 2, 3],
            safety=[1, 1, 1],
            hc50_complete_median=[100, 100, 100],
            fbd=[1, 1, 1],
            diversity=[2, 2, 2],
            library_diversity=[0.5] * 3,
        )
    )
    for name in module.FAMILIES:
        frame[f"{name}_mean_log2"] = [1, 2, 3]
    actual = module.score_scenarios(frame, {"scenarios": {"equal": [0.2] * 5}})
    expected = (
        frame.apex_activity.rank(pct=True)
        + 5 * frame.physchem_mean_log2.rank(pct=True, ascending=False)
    ) / 6
    pd.testing.assert_series_equal(actual.family_activity, expected, check_names=False)


def test_global_ranking_rejects_unequal_subset_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    rank = importlib.import_module("report_competition_scale").rank_comparisons
    frame = pd.DataFrame(dict(id=["a", "a", "b"], seed=[42, 43, 42], subset_size=[1000] * 3))
    with pytest.raises(ValueError, match="coverage"):
        rank(frame, {})
    with pytest.raises(ValueError, match="Duplicate"):
        rank(pd.concat([frame, frame.iloc[:1]]), {})


def test_registered_ranker_extension_preserves_pool_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    requests = importlib.import_module("compare_competition_pool").requests
    config = dict(library_variants=["L2"], rankers=["B1"], constraint_variants=["current"])
    rows = requests(config, ["B1", "linear650-w0.5"])
    assert [row["id"] for row in rows] == ["library-L2", "rank-linear650-w0.5"]
    assert len(requests(config)) == 1
    assert config["rankers"] == ["B1"]
    with pytest.raises(ValueError, match="unique"):
        requests(config, ["B1", "B1"])
