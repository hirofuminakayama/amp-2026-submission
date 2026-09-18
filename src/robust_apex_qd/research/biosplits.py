"""Extend a verified MIC identity matrix before sharing folds across endpoints."""

from collections.abc import Callable
from typing import Any

import numpy as np

from robust_apex_qd.research.mic_data import fold_assignments, similarity_groups


def extend_identity(
    old_sequences: list[str],
    old_matrix: np.ndarray,
    sequences: list[str],
    identity: Callable[[str, str], float],
) -> np.ndarray:
    if len(set(sequences)) != len(sequences) or not set(old_sequences) <= set(sequences):
        raise ValueError("Endpoint union must contain unique old sequences")
    if old_matrix.shape != (len(old_sequences), len(old_sequences)):
        raise ValueError("Old identity matrix does not match sequence order")
    if not np.isfinite(old_matrix).all() or not np.allclose(old_matrix, old_matrix.T):
        raise ValueError("Identity matrix must be finite and symmetric")
    positions = {s: i for i, s in enumerate(sequences)}
    old_positions = [positions[s] for s in old_sequences]
    old_set = set(old_positions)
    matrix = np.eye(len(sequences), dtype=np.float32)
    matrix[np.ix_(old_positions, old_positions)] = old_matrix
    for i, left in enumerate(sequences):
        for j, right in enumerate(sequences[:i]):
            if i in old_set and j in old_set:
                continue
            value = identity(left, right)
            if not 0 <= value <= 1:
                raise ValueError("Identity must be a fraction")
            matrix[i, j] = matrix[j, i] = value
    return matrix


def shared_folds(
    sequences: list[str],
    matrix: np.ndarray,
    *,
    threshold: float,
    outer_count: int,
    inner_count: int,
    seed: int,
) -> dict[str, Any]:
    groups = similarity_groups(sequences, matrix, threshold)
    outer = fold_assignments(groups, outer_count, seed)
    inner = {
        str(fold): fold_assignments(
            {s: g for s, g in groups.items() if outer[s] != fold}, inner_count, seed
        )
        for fold in sorted(set(outer.values()))
        if fold >= 0
    }
    folds = np.array([outer[s] for s in sequences])
    for i in range(len(sequences)):
        other = folds != folds[i]
        if other.any() and float(matrix[i, other].max()) > threshold + 1e-7:
            raise ValueError("Endpoint groups cross the outer-fold boundary")
    return dict(
        groups=groups,
        outer=outer,
        inner=inner,
        threshold=threshold,
        development_only=True,
        prior_holdout_reused=True,
        method="BLOSUM45 global; gap open -5 / extend -1; matches / aligned length",
        alignment_scope="MIC implementation reused; official QMAP tie parity unestablished",
    )
