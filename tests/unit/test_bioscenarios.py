import numpy as np
import pytest

from robust_apex_qd.research.bioscenarios import (
    biological_rank_scores,
    joint_trials,
    marginal_joint_probability,
    random25_scenarios,
    select_portfolio,
)


def test_shared_failure_changes_tail_not_expected_hits() -> None:
    trials = np.zeros((100, 100, 1), dtype=bool)
    trials[:50] = True
    result = random25_scenarios(trials, draws=10000, seed=42)
    assert result["analytic_expected_species_average_hits"] == 12.5
    assert result["sampling_only"]["std"] == 0
    assert result["predictive_scenario"]["p10"] == 0
    assert result["predictive_scenario"]["mean"] == pytest.approx(12.5, abs=0.4)
    assert result["predictive_scenario"]["std"] > 12


def test_cvar_portfolio_avoids_shared_failures() -> None:
    scores = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert select_portfolio(scores, ["a", "b", "c"], count=2, objective="mean") == [0, 1]
    assert select_portfolio(scores, ["a", "b", "c"], count=2, objective="cvar") == [0, 2]


def test_zero_residuals_obey_joint_event_and_missing_is_rejected() -> None:
    mic = np.array([[3.0, 5.0], [4.0, 4.0]])
    hc = np.array([7.0, 6.0])
    result = joint_trials(
        mic, hc, [np.zeros(4), np.zeros(4)], np.zeros(4), groups=["a", "b"], draws=10, ratio=8
    )
    np.testing.assert_equal(result[0], [[True, False], [False, False]])
    mic[0, 0] = np.nan
    with pytest.raises(ValueError, match="coverage"):
        joint_trials(mic, hc, [np.zeros(4), np.zeros(4)], np.zeros(4), groups=["a", "b"], draws=10)


def test_cluster_cap_is_not_silently_relaxed() -> None:
    with pytest.raises(ValueError, match="supply"):
        select_portfolio(np.ones((3, 3)), ["a", "b", "c"], count=3, groups=["g", "g", "g"], cap=2)


def test_marginal_probabilities_are_invariant_to_chunks_and_permutation() -> None:
    mic = np.array([[3.0, 4.0], [5.0, 1.0], [2.0, 2.0]])
    hc = np.array([7.0, 6.0, 3.0])
    errors = [np.array([-1.0, 0.0, 1.0])] * 2
    he = np.array([-2.0, 0.0, 2.0])
    expected = marginal_joint_probability(mic, hc, errors, he)
    small = marginal_joint_probability(mic, hc, errors, he, batch_size=1)
    reverse = marginal_joint_probability(mic[::-1], hc[::-1], errors, he)
    np.testing.assert_array_equal(expected, small)
    np.testing.assert_array_equal(expected, reverse[::-1])


def test_pool_scores_prefer_lower_mic_and_higher_joint_probability() -> None:
    result = biological_rank_scores(
        np.array([[1.0, 2.0], [6.0, 7.0]]),
        np.array([[0.8, 0.9], [0.1, 0.2]]),
        np.array([-2.0, -8.0]),
    )
    assert all(scores[0] > scores[1] for scores in result.values())
    assert result["bio-MIC-apex50"][0] == -1.75
