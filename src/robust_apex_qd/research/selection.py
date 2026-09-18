"""Fixed sampling and uncalibrated selection comparisons for saved candidate pools."""

import hashlib

import numpy as np
import pandas as pd

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS, aggregate_predictions
from robust_apex_qd.calibration.model import STRAIN_TO_PATHOGEN


def complete_panel_activity(measurements: pd.DataFrame) -> pd.Series:
    rows = measurements.assign(pathogen=measurements.strain.map(STRAIN_TO_PATHOGEN))
    if rows.pathogen.isna().any() or rows.duplicated(["peptide_id", "pathogen"]).any():
        raise ValueError("Require unique known strain measurements")
    if (
        rows.empty
        or not rows.groupby("peptide_id").pathogen.nunique().eq(len(APEX_PATHOGENS)).all()
    ):
        raise ValueError("Historical comparison requires complete pathogen panels")
    return rows.groupby("peptide_id").active.mean().sort_index()


def library_rows(pool: pd.DataFrame, sequences: list[str]) -> pd.DataFrame:
    selected = pool[pool.valid & pool.sequence.isin(sequences)].copy()
    if len(set(sequences)) != len(sequences) or len(selected) != len(sequences):
        raise ValueError("Library membership must identify one valid pool row per sequence")
    if selected.sequence.duplicated().any() or set(selected.sequence) != set(sequences):
        raise ValueError("Library contains missing or duplicate valid pool sequences")
    return selected


def align_apex_scores(
    archive_sequences: list[str], tensor: np.ndarray, sequences: list[str], valid: list[bool]
) -> dict[str, np.ndarray]:
    if len(archive_sequences) != len(set(archive_sequences)) or len(archive_sequences) != len(
        tensor
    ):
        raise ValueError("APEX sequence identifiers must uniquely match the tensor")
    if len(sequences) != len(valid):
        raise ValueError("Validity flags must align with pool rows")
    index = {sequence: i for i, sequence in enumerate(archive_sequences)}
    if any(
        accepted and sequence not in index
        for sequence, accepted in zip(sequences, valid, strict=True)
    ):
        raise ValueError("A valid candidate is missing APEX predictions")
    scores = selection_scores(tensor)
    return {
        name: np.asarray([values[index[s]] if s in index else np.nan for s in sequences])
        for name, values in scores.items()
    }


def keyed_subset(sequences: list[str], size: int, seed: int) -> list[str]:
    if len(set(sequences)) != len(sequences) or not 0 < size <= len(sequences):
        raise ValueError("Require unique sequences and a feasible positive subset size")
    return sorted(
        sequences,
        key=lambda sequence: (hashlib.sha256(f"{seed}:{sequence}".encode()).digest(), sequence),
    )[:size]


def bounded_quotas(
    size: int, weights: dict[int, float], capacity: dict[int, int]
) -> dict[int, int]:
    if size < 0 or size > sum(capacity.values()) or any(v < 0 for v in capacity.values()):
        raise ValueError("Insufficient or invalid cluster capacity")
    if any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Cluster weights must be finite and nonnegative")
    quotas = {key: 0 for key in sorted(capacity)}
    while sum(quotas.values()) < size:
        remaining = size - sum(quotas.values())
        available = [key for key in quotas if quotas[key] < capacity[key]]
        mass = {key: weights.get(key, 0.0) for key in available}
        if not sum(mass.values()):
            mass = {key: float(capacity[key] - quotas[key]) for key in available}
        total = sum(mass.values())
        ideal = {key: remaining * mass[key] / total for key in available}
        additions = {key: min(int(ideal[key]), capacity[key] - quotas[key]) for key in available}
        for key in available:
            quotas[key] += additions[key]
        remaining = size - sum(quotas.values())
        for key in sorted(available, key=lambda key: (-(ideal[key] % 1), key)):
            if remaining and quotas[key] < capacity[key]:
                quotas[key] += 1
                remaining -= 1
    return quotas


def selection_scores(tensor: np.ndarray) -> dict[str, np.ndarray]:
    aggregates = aggregate_predictions(tensor)
    votes = (tensor <= 16).mean(axis=1)
    return {
        "B0": -aggregates.official_broad_mean_mic_u_m,
        "B1": -aggregates.median_log2_mic,
        "B2": aggregates.vote16,
        "balanced": 0.5 * (votes[:, :7].mean(axis=1) + votes[:, 7:].mean(axis=1)),
        "tail90": -aggregates.q90_log2_mic,
    }
