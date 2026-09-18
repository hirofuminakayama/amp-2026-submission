"""Sequence-based consensus ranking with the registered selection constraints."""

import numpy as np
import pandas as pd

from robust_apex_qd.ranking.objectives import percentile_score
from robust_apex_qd.research.exploration import ExplorationConstraints, select_exploration_top
from robust_apex_qd.selection.top import maximum_levenshtein_ratio, maximum_local_similarity


def consensus_rank(pool: pd.DataFrame, predictions: dict[str, np.ndarray]) -> np.ndarray:
    if len(predictions) != 5 or not np.isfinite(pool.species).all():
        raise ValueError("Require complete APEX and five predictor families")
    ranks = [percentile_score(pool.species.to_numpy())]
    for prediction in predictions.values():
        if prediction.shape != (len(pool), 7) or not np.isfinite(prediction).all():
            raise ValueError("Require complete aligned seven-species predictions")
        ranks.append(percentile_score(-prediction.mean(1)))
    return np.mean(ranks, axis=0)


def select_adopted(
    pool: pd.DataFrame,
    library: list[str],
    top_k: int,
    challenge: set[str],
    known: set[str],
) -> pd.DataFrame:
    if pool.sequence.duplicated().any() or len(set(library)) != len(library):
        raise ValueError("Require unique pool and library sequences")
    frame = pool[pool.sequence.isin(library)].copy()
    if len(frame) != len(library):
        raise ValueError("Pool predictions do not cover the library")
    frame["score"] = percentile_score(frame.rankmean.to_numpy())
    return select_exploration_top(
        frame,
        "score",
        ExplorationConstraints(challenge_max=0.80),
        top_k,
        lambda sequence: (
            maximum_levenshtein_ratio(sequence, tuple(sorted(challenge))) if challenge else 0.0
        ),
        lambda sequence: maximum_local_similarity(sequence, tuple(sorted(known))) if known else 0.0,
    )
