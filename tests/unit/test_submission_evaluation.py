import numpy as np
import pytest

from robust_apex_qd.evaluation.submission import summarize_numeric, summarize_pathogen_groups


def test_numeric_summary_uses_fixed_quantiles() -> None:
    summary = summarize_numeric(np.asarray([1.0, 2.0, 3.0, 4.0]))

    assert summary.count == 4
    assert summary.mean == 2.5
    assert summary.p50 == 2.5
    assert summary.minimum == 1
    assert summary.maximum == 4


def test_pathogen_groups_use_fixed_apex_order() -> None:
    probabilities = np.zeros((2, 11), dtype=np.float64)
    probabilities[:, :7] = 0.8
    probabilities[:, 7:] = 0.2

    groups = summarize_pathogen_groups(probabilities)

    assert groups["gram_negative"].mean == pytest.approx(0.8)
    assert groups["gram_positive"].mean == pytest.approx(0.2)
    assert groups["mdr_proxy"].count == 2
