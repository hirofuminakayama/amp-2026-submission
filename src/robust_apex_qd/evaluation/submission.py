from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from robust_apex_qd.apex.ensemble import APEX_PATHOGEN_COUNT
from robust_apex_qd.evaluation.models import NumericSummary

GRAM_NEGATIVE_INDICES = tuple(range(7))
GRAM_POSITIVE_INDICES = tuple(range(7, 11))


def summarize_numeric(values: NDArray[np.floating]) -> NumericSummary:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Numeric summary input must be a non-empty finite vector")
    quantiles = np.quantile(array, (0.05, 0.25, 0.50, 0.75, 0.95))
    return NumericSummary(
        count=len(array),
        mean=float(array.mean()),
        standard_deviation=float(array.std()),
        minimum=float(array.min()),
        p05=float(quantiles[0]),
        p25=float(quantiles[1]),
        p50=float(quantiles[2]),
        p75=float(quantiles[3]),
        p95=float(quantiles[4]),
        maximum=float(array.max()),
    )


def pathogen_group_values(
    pathogen_probabilities: NDArray[np.floating],
) -> Mapping[str, NDArray[np.float64]]:
    probabilities = np.asarray(pathogen_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != APEX_PATHOGEN_COUNT:
        raise ValueError("Pathogen probabilities must have shape [sequence, 11]")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Pathogen probabilities must be finite values between zero and one")
    mdr_values = np.column_stack(
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
    ).mean(axis=1)
    return {
        "gram_negative": probabilities[:, GRAM_NEGATIVE_INDICES].mean(axis=1),
        "gram_positive": probabilities[:, GRAM_POSITIVE_INDICES].mean(axis=1),
        "mdr_proxy": mdr_values,
    }


def summarize_pathogen_groups(
    pathogen_probabilities: NDArray[np.floating],
) -> dict[str, NumericSummary]:
    return {
        name: summarize_numeric(values)
        for name, values in pathogen_group_values(pathogen_probabilities).items()
    }
