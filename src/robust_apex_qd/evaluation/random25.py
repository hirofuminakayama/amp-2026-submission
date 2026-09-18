from collections.abc import Sequence

import numpy as np

from robust_apex_qd.evaluation.models import (
    CandidateEvaluation,
    Random25Row,
    RandomDrawEvaluation,
)
from robust_apex_qd.evaluation.submission import summarize_numeric


def _optional_values(
    candidates: Sequence[CandidateEvaluation],
    indices: np.ndarray,
    field: str,
) -> list[float]:
    values = [getattr(candidates[int(index)], field) for index in indices]
    return [float(value) for value in values if value is not None]


def simulate_random_draws(
    candidates: Sequence[CandidateEvaluation],
    *,
    draws: int = 10_000,
    sample_size: int = 25,
    seed: int = 42,
    variant: str = "B1",
) -> RandomDrawEvaluation:
    if draws <= 0:
        raise ValueError("draws must be positive")
    if sample_size <= 0 or sample_size > len(candidates):
        raise ValueError("sample_size must be between one and the candidate count")
    rng = np.random.default_rng(seed)
    rows: list[Random25Row] = []
    for draw_index in range(draws):
        indices = rng.choice(len(candidates), size=sample_size, replace=False)
        selected = [candidates[int(index)] for index in indices]
        hc50_values = _optional_values(candidates, indices, "hemopi2_hc50_u_m")
        selectivity_values = _optional_values(candidates, indices, "selectivity_proxy")
        hemolytic_values = [candidate.hemopi2_hemolytic for candidate in selected]
        complete_hemolysis = all(value is not None for value in hemolytic_values)
        rows.append(
            Random25Row(
                variant=variant,
                draw=draw_index + 1,
                selected_ranks=tuple(candidate.rank for candidate in selected),
                apex_vote16=float(np.mean([candidate.apex_vote16 for candidate in selected])),
                apex_consensus16=float(
                    np.mean([candidate.apex_consensus16 for candidate in selected])
                ),
                gram_negative_success16=float(
                    np.mean([candidate.gram_negative_success16 for candidate in selected])
                ),
                gram_positive_success16=float(
                    np.mean([candidate.gram_positive_success16 for candidate in selected])
                ),
                mdr_proxy=float(np.mean([candidate.mdr_proxy for candidate in selected])),
                apex_broad_mean_mic_u_m=float(
                    np.median([candidate.apex_broad_mean_mic_u_m for candidate in selected])
                ),
                apex_model_disagreement=float(
                    np.mean([candidate.apex_model_disagreement for candidate in selected])
                ),
                embedding_cluster_coverage=len(
                    {candidate.embedding_cluster for candidate in selected}
                ),
                pep_hard_filter_pass_fraction=float(
                    np.mean([candidate.pep_hard_filter_pass for candidate in selected])
                ),
                spps_favorable_fraction=float(
                    np.mean([candidate.spps_difficulty_score <= 3 for candidate in selected])
                ),
                hemopi2_non_hemolytic_fraction=(
                    float(np.mean([not bool(value) for value in hemolytic_values]))
                    if complete_hemolysis
                    else None
                ),
                hemopi2_hc50_median_u_m=float(np.median(hc50_values))
                if len(hc50_values) == sample_size
                else None,
                selectivity_proxy_median=float(np.median(selectivity_values))
                if len(selectivity_values) == sample_size
                else None,
            )
        )
    numeric_fields = (
        "apex_vote16",
        "apex_consensus16",
        "gram_negative_success16",
        "gram_positive_success16",
        "mdr_proxy",
        "apex_broad_mean_mic_u_m",
        "apex_model_disagreement",
        "embedding_cluster_coverage",
        "pep_hard_filter_pass_fraction",
        "spps_favorable_fraction",
        "hemopi2_non_hemolytic_fraction",
        "hemopi2_hc50_median_u_m",
        "selectivity_proxy_median",
    )
    summary = {}
    for field in numeric_fields:
        values = [getattr(row, field) for row in rows]
        if all(value is not None for value in values):
            summary[field] = summarize_numeric(np.asarray(values, dtype=np.float64))
    return RandomDrawEvaluation(
        variant=variant,
        draws=draws,
        sample_size=sample_size,
        seed=seed,
        rows=tuple(rows),
        summary=summary,
    )
