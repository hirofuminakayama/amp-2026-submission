"""Explicit strain support and fallback for measured-MIC predictor handoff."""

import numpy as np
import pandas as pd

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS
from robust_apex_qd.research.competition_models import SPECIES


def prediction_records(
    sequences: list[str],
    means: np.ndarray,
    scales: np.ndarray,
    apex: np.ndarray,
    model_hash: str,
    apex_hash: str,
) -> pd.DataFrame:
    if len(set(sequences)) != len(sequences):
        raise ValueError("Unique sequence keys required")
    if means.shape != (len(sequences), len(SPECIES) + len(APEX_PATHOGENS)):
        raise ValueError("Prediction head shape mismatch")
    if scales.shape != means.shape or apex.shape != (len(sequences), len(APEX_PATHOGENS)):
        raise ValueError("Scale/APEX shape mismatch")
    if not np.isfinite(apex).all() or np.isinf(means).any() or np.isinf(scales).any():
        raise ValueError("Finite predictions or explicit missing heads required")
    if np.any(scales[np.isfinite(scales)] <= 0):
        raise ValueError("Prediction scales must be positive")
    rows = []
    for head, strain in enumerate(APEX_PATHOGENS):
        p = means[:, head + len(SPECIES)]
        support = np.isfinite(p)
        rows.append(
            pd.DataFrame(
                dict(
                    sequence=sequences,
                    target_id=strain,
                    target_level="strain",
                    unit="log2_uM",
                    prediction=np.where(support, p, apex[:, head]),
                    scale=np.where(support, scales[:, head + len(SPECIES)], np.nan),
                    supported=support,
                    fallback_reason=np.where(support, "", "no_supported_strain_head"),
                    model_sha256=np.where(support, model_hash, apex_hash),
                    requested_model_sha256=model_hash,
                )
            )
        )
    return pd.concat(rows, ignore_index=True)


def blend_strain_predictions(
    new: pd.DataFrame, apex: np.ndarray, weight: float, sequences: list[str]
) -> np.ndarray:
    if not 0 <= weight <= 1:
        raise ValueError("Convex APEX weight required")
    if not new.target_level.eq("strain").all() or not new.unit.eq("log2_uM").all():
        raise ValueError("Only aligned strain log2 MIC predictions may be blended")
    if len(set(sequences)) != len(sequences) or set(sequences) != set(new.sequence):
        raise ValueError("Explicit APEX sequence keys must match prediction keys")
    if new.duplicated(["sequence", "target_id"]).any():
        raise ValueError("Duplicate prediction keys")
    wide = new.pivot(index="sequence", columns="target_id", values="prediction")
    values = wide.reindex(index=sequences, columns=APEX_PATHOGENS).to_numpy()
    if values.shape != apex.shape or not np.isfinite(values).all():
        raise ValueError("Incomplete prediction coverage")
    return weight * apex + (1 - weight) * values
