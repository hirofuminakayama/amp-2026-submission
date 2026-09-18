import numpy as np
import pytest

from robust_apex_qd.ranking.objectives import (
    RankerGroundTruth,
    broad_objectives,
    conservative_mdr_proxy,
    percentile_score,
    weighted_quality,
)


def test_broad_mean_tail_and_mdr_proxy_use_fixed_pathogen_mapping() -> None:
    probabilities = np.asarray(
        [[0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]],
        dtype=np.float64,
    )

    broad_mean, broad_tail = broad_objectives(probabilities)
    mdr = conservative_mdr_proxy(probabilities)

    assert broad_mean[0] == pytest.approx(probabilities.mean())
    assert broad_tail[0] == pytest.approx(np.quantile(probabilities, 0.10))
    assert mdr[0] == pytest.approx(np.mean([0.8, 0.6, 0.6, 0.5, 0.3, 0.1, 0.05, 0.01]))


def test_percentile_ties_and_weighted_penalties_are_deterministic() -> None:
    values = np.asarray([2.0, 1.0, 1.0, 4.0])

    higher = percentile_score(values)
    lower = percentile_score(values, higher_is_better=False)
    quality = weighted_quality(
        broad_mean=np.asarray([0.5, 0.5]),
        broad_tail=np.asarray([0.2, 0.4]),
        mdr_proxy=np.asarray([0.7, 0.3]),
        disagreement=np.asarray([0.1, 0.9]),
        physchem_ood=np.asarray([0.1, 0.9]),
        embedding_ood=np.asarray([0.1, 0.9]),
    )

    np.testing.assert_allclose(higher, [2 / 3, 1 / 6, 1 / 6, 1.0])
    np.testing.assert_allclose(lower, [1 / 3, 5 / 6, 5 / 6, 0.0])
    assert quality[0] > quality[1]


def test_wet_lab_target_and_apex_proxy_are_distinct_schema_fields() -> None:
    ground_truth = RankerGroundTruth(
        peptide_id="pep_1",
        sequence="ACDEFGHI",
        measured_success_rate_16=0.5,
        measured_mic50_u_m=16.0,
        measured_mic90_u_m=64.0,
        measured_mdr_success_rate=0.25,
        apex_proxy_broad_probability=0.75,
        apex_proxy_mdr_probability=0.6,
    )

    assert ground_truth.measured_success_rate_16 != ground_truth.apex_proxy_broad_probability
    assert "measured_success_rate_16" in ground_truth.model_dump()
    assert "apex_proxy_broad_probability" in ground_truth.model_dump()
