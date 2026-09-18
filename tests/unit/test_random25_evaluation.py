import numpy as np
import pytest

from robust_apex_qd.evaluation.models import CandidateEvaluation
from robust_apex_qd.evaluation.random25 import simulate_random_draws


def _candidate(index: int) -> CandidateEvaluation:
    return CandidateEvaluation(
        rank=index + 1,
        candidate_id=f"cand_{index}",
        sequence="ACDEFGHIK" + "A" * index,
        final_score=1.0 - index / 10,
        apex_vote16=index / 10,
        apex_consensus16=index % 2,
        gram_negative_success16=index / 20,
        gram_positive_success16=index / 25,
        mdr_proxy=index / 30,
        apex_broad_mean_mic_u_m=float(index + 1),
        apex_model_disagreement=float(index) / 100,
        embedding_cluster=index % 3,
        pep_hard_filter_pass=index % 2 == 0,
        spps_difficulty_score=float(index % 4),
    )


def test_random_draws_are_seeded_without_replacement_and_summarized() -> None:
    candidates = tuple(_candidate(index) for index in range(10))

    first = simulate_random_draws(candidates, draws=20, sample_size=4, seed=42, variant="B1")
    second = simulate_random_draws(candidates, draws=20, sample_size=4, seed=42, variant="B1")

    assert first == second
    assert len(first.rows) == 20
    assert all(len(set(row.selected_ranks)) == 4 for row in first.rows)
    assert first.summary["apex_vote16"].count == 20
    values = np.asarray([row.apex_vote16 for row in first.rows])
    assert first.summary["apex_vote16"].p50 == pytest.approx(np.quantile(values, 0.50))


def test_random_draws_reject_invalid_sample_size() -> None:
    with pytest.raises(ValueError, match="sample_size"):
        simulate_random_draws(
            tuple(_candidate(index) for index in range(3)), draws=2, sample_size=4
        )
