from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.stats import rankdata

from robust_apex_qd.apex.ensemble import APEX_PATHOGEN_COUNT


class RankerGroundTruth(BaseModel):
    model_config = ConfigDict(frozen=True)

    peptide_id: str
    sequence: str
    measured_success_rate_16: float
    measured_mic50_u_m: float
    measured_mic90_u_m: float
    measured_mdr_success_rate: float
    apex_proxy_broad_probability: float
    apex_proxy_mdr_probability: float


def _probability_matrix(values: NDArray[np.floating]) -> NDArray[np.float64]:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != APEX_PATHOGEN_COUNT:
        raise ValueError("Pathogen probabilities must have shape [sequence, 11]")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Pathogen probabilities must be finite values between zero and one")
    return probabilities


def broad_objectives(
    pathogen_probabilities: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    probabilities = _probability_matrix(pathogen_probabilities)
    return (
        probabilities.mean(axis=1),
        np.quantile(probabilities, 0.10, axis=1),
    )


def conservative_mdr_proxy(
    pathogen_probabilities: NDArray[np.floating],
) -> NDArray[np.float64]:
    probabilities = _probability_matrix(pathogen_probabilities)
    proxies = np.column_stack(
        (
            probabilities[:, 0],
            probabilities[:, 3],
            probabilities[:, 1:4].min(axis=1),
            probabilities[:, 4],
            probabilities[:, 5:7].min(axis=1),
            probabilities[:, 8],
            probabilities[:, 9],
            probabilities[:, 10],
        )
    )
    return proxies.mean(axis=1)


def percentile_score(
    values: NDArray[np.floating],
    *,
    higher_is_better: bool = True,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Percentile input must be a finite one-dimensional array")
    if len(array) == 1:
        return np.ones(1, dtype=np.float64)
    ranks = rankdata(array, method="average")
    percentiles = (ranks - 1.0) / (len(array) - 1.0)
    return percentiles if higher_is_better else 1.0 - percentiles


def resolve_ranker(ranking_config: Mapping[str, object]) -> str:
    fallback = str(ranking_config.get("fallback", "official_linear_mean"))
    if fallback != "official_linear_mean":
        raise ValueError(f"Unsupported ranking fallback: {fallback}")
    if not bool(ranking_config.get("enabled", True)):
        return "B0"
    ranker = str(ranking_config.get("adopted_ranker", "B0"))
    if ranker not in {"B0", "B1"}:
        raise ValueError(f"Top selection does not support configured ranker {ranker}")
    return ranker


def configured_ranker_score(
    apex_rows: Sequence[Mapping[str, str]],
    ranker: str,
) -> NDArray[np.float64]:
    field_by_ranker = {
        "B0": "official_broad_mean_mic_uM",
        "B1": "median_log2_mic",
    }
    if ranker not in field_by_ranker:
        raise ValueError(f"Top selection does not support configured ranker {ranker}")
    field = field_by_ranker[ranker]
    try:
        values = np.asarray([float(row[field]) for row in apex_rows], dtype=np.float64)
    except (KeyError, ValueError) as error:
        raise ValueError(f"APEX rows do not provide finite numeric {field}") from error
    return percentile_score(values, higher_is_better=False)


def weighted_quality(
    *,
    broad_mean: NDArray[np.floating],
    broad_tail: NDArray[np.floating],
    mdr_proxy: NDArray[np.floating],
    disagreement: NDArray[np.floating],
    physchem_ood: NDArray[np.floating],
    embedding_ood: NDArray[np.floating],
    broad_weight: float = 0.60,
    tail_weight: float = 0.20,
    mdr_weight: float = 0.20,
    uncertainty_penalty: float = 0.15,
    physchem_penalty: float = 0.10,
    embedding_penalty: float = 0.05,
) -> NDArray[np.float64]:
    components = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            broad_mean,
            broad_tail,
            mdr_proxy,
            disagreement,
            physchem_ood,
            embedding_ood,
        )
    )
    lengths = {len(values) for values in components}
    if len(lengths) != 1 or any(values.ndim != 1 for values in components):
        raise ValueError("All quality components must be aligned one-dimensional arrays")
    mean, tail, mdr, uncertainty, physchem, embedding = components
    return (
        broad_weight * percentile_score(mean)
        + tail_weight * percentile_score(tail)
        + mdr_weight * percentile_score(mdr)
        - uncertainty_penalty * percentile_score(uncertainty)
        - physchem_penalty * percentile_score(physchem)
        - embedding_penalty * percentile_score(embedding)
    )
