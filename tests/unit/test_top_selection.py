import numpy as np
import pytest
from pydantic import ValidationError

from robust_apex_qd.ranking.objectives import configured_ranker_score, resolve_ranker
from robust_apex_qd.selection.top import (
    TopCandidate,
    TopSelectionConfig,
    select_top_with_fallback,
    top_selection_config_from_mapping,
)


def _candidates(count: int) -> tuple[TopCandidate, ...]:
    return tuple(
        TopCandidate(
            candidate_id=f"cand_{index:03d}",
            sequence=f"ACDEFGHIK{chr(65 + index)}",
            raw_order=index,
            final_score=1.0 - index / 100,
            embedding_cluster=index % 4,
            physchem_hard_reject=False,
            median_log2_mic=float(index),
        )
        for index in range(count)
    )


def test_selector_enforces_constraints_subset_tiebreak_and_repeat() -> None:
    candidates = _candidates(8)

    def challenge(sequence: str, references: object) -> float:
        return 0.79 if sequence.endswith("A") else 0.2

    def known(sequence: str, references: object) -> float:
        return 0.61 if sequence.endswith("B") else 0.2

    def pairwise(first: str, second: str) -> float:
        return 0.51 if {first[-1], second[-1]} == {"C", "D"} else 0.1

    config = TopSelectionConfig(
        prefilter_sizes=(8,),
        cluster_caps=(5,),
        pairwise_thresholds=(0.50,),
    )
    first = select_top_with_fallback(
        candidates,
        challenge_references=("REFERENCE",),
        known_references=("KNOWN",),
        top_k=4,
        config=config,
        challenge_similarity=challenge,
        known_similarity=known,
        pairwise_similarity=pairwise,
    )
    second = select_top_with_fallback(
        candidates,
        challenge_references=("REFERENCE",),
        known_references=("KNOWN",),
        top_k=4,
        config=config,
        challenge_similarity=challenge,
        known_similarity=known,
        pairwise_similarity=pairwise,
    )

    assert first == second
    assert len(first.selected) == 4
    assert {row.candidate.candidate_id for row in first.selected} <= {
        candidate.candidate_id for candidate in candidates
    }
    assert [row.candidate.candidate_id for row in first.selected] == [
        "cand_002",
        "cand_004",
        "cand_005",
        "cand_006",
    ]
    assert all(row.challenge_similarity <= 0.78 for row in first.selected)
    assert all(row.known_similarity <= 0.60 for row in first.selected)


def test_fallback_order_changes_only_documented_constraints() -> None:
    candidates = tuple(
        TopCandidate(
            candidate_id=f"cand_{index}",
            sequence=f"SEQ{index}",
            raw_order=index,
            final_score=10 - index,
            embedding_cluster=0,
            physchem_hard_reject=False,
            median_log2_mic=float(index),
        )
        for index in range(12)
    )
    config = TopSelectionConfig()
    result = select_top_with_fallback(
        candidates,
        challenge_references=("ref",),
        known_references=("known",),
        top_k=6,
        config=config,
        challenge_similarity=lambda sequence, references: 0.1,
        known_similarity=lambda sequence, references: 0.1,
        pairwise_similarity=lambda first, second: 0.1,
    )

    assert result.relaxation_step == 2
    assert result.prefilter_size == 10_000
    assert result.cluster_cap == 8
    assert result.pairwise_threshold == 0.50
    assert config.challenge_similarity_max == 0.78
    assert config.known_similarity_max == 0.60


def test_top_config_rejects_manual_allowlist() -> None:
    with pytest.raises(ValidationError):
        TopSelectionConfig.model_validate({"manual_allowlist": ["cand_1"]})


def test_external_hard_reject_runs_before_similarity_callbacks() -> None:
    candidates = (
        _candidates(1)[0].model_copy(update={"external_hard_reject": True}),
        _candidates(2)[1],
    )

    def challenge(sequence: str, references: object) -> float:
        assert not sequence.endswith("A")
        return 0.1

    result = select_top_with_fallback(
        candidates,
        challenge_references=("reference",),
        known_references=("known",),
        top_k=1,
        config=TopSelectionConfig(
            prefilter_sizes=(2,), cluster_caps=(2,), pairwise_thresholds=(0.5,)
        ),
        challenge_similarity=challenge,
        known_similarity=lambda sequence, references: 0.1,
        pairwise_similarity=lambda first, second: 0.1,
    )

    assert result.selected[0].candidate == candidates[1]
    assert result.rejection_counts == {"external_hard_reject": 1}


def test_active_config_selects_ranker_fallback_and_constraints() -> None:
    apex_rows = [
        {"official_broad_mean_mic_uM": "8", "median_log2_mic": "5"},
        {"official_broad_mean_mic_uM": "32", "median_log2_mic": "1"},
    ]

    assert resolve_ranker({"enabled": True, "adopted_ranker": "B1"}) == "B1"
    assert (
        resolve_ranker(
            {
                "enabled": False,
                "adopted_ranker": "B1",
                "fallback": "official_linear_mean",
            }
        )
        == "B0"
    )
    np.testing.assert_array_equal(
        configured_ranker_score(apex_rows, "B0"),
        np.asarray([1.0, 0.0]),
    )
    np.testing.assert_array_equal(
        configured_ranker_score(apex_rows, "B1"),
        np.asarray([0.0, 1.0]),
    )

    config = top_selection_config_from_mapping(
        {
            "selection_challenge_similarity_max": 0.78,
            "known_local_similarity_max": 0.60,
            "pairwise_local_similarity_max": 0.45,
            "prefilter_sizes": [4000, 9000],
            "max_per_embedding_cluster": 4,
            "fallback_cluster_cap": 7,
            "fallback_pairwise_local_similarity_max": 0.52,
        }
    )
    assert config.prefilter_sizes == (4000, 9000, 9000, 9000)
    assert config.cluster_caps == (4, 4, 7, 7)
    assert config.pairwise_thresholds == (0.45, 0.45, 0.45, 0.52)
