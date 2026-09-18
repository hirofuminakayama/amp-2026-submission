from robust_apex_qd.evaluation.developability import evaluate_developability


def test_developability_reports_spps_and_cysteine_hard_filters() -> None:
    odd_cysteine = evaluate_developability("ACDEFGHIK")
    hydrophobic_run = evaluate_developability("AVILMFWYAVILMFWY")

    assert not odd_cysteine.hard_filter_pass
    assert "odd_cysteine_count" in odd_cysteine.hard_filter_reasons
    assert hydrophobic_run.spps_difficulty_score >= 3
    assert hydrophobic_run.aggregation_risk in {"Medium", "High"}
    assert 0 <= odd_cysteine.developability_score <= 100


def test_developability_is_repeatable_and_accepts_canonical_peptide() -> None:
    first = evaluate_developability("GIGKFLHSAKKFGKAFVGEIMKS")
    second = evaluate_developability("GIGKFLHSAKKFGKAFVGEIMKS")

    assert first == second
    assert first.length == 23
    assert first.developability_class in {"Excellent", "Good", "Moderate", "Low"}
