"""Separate development and selection primitives for competition-scoped exploration."""

import hashlib
from collections import Counter
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import GroupKFold

from robust_apex_qd.research.data import CANONICAL, Observation, normalize_battle
from robust_apex_qd.validation.similarity import local_similarity


class ExplorationConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    challenge_max: float = Field(default=0.78, ge=0, le=0.80)
    known_max: float | None = Field(default=0.60, ge=0, le=1)
    pairwise_max: float | None = Field(default=0.50, ge=0, le=1)
    cluster_cap: int | None = Field(default=5, gt=0)
    physchem: Literal["hard", "soft", "none"] = "hard"
    prefilter: int | None = Field(default=10000, gt=0)


def development_row(observation: Observation) -> dict[str, Any]:
    row = observation.model_dump()
    canonical = 8 <= len(observation.sequence) <= 50 and not set(observation.sequence) - CANONICAL
    chemistry = observation.chemical_form in {"unknown", "reported_linear_free"}
    measured = observation.source == "battleamp" and observation.mic_um is not None
    consensus = observation.source == "qmap" and observation.consensus_um is not None
    row.update(
        usable=bool(canonical and chemistry and (measured or consensus)),
        strict_chemistry=observation.chemical_form == "reported_linear_free",
        exact_regression=bool(measured and observation.exact_mic),
        objective="measured_mic" if observation.source == "battleamp" else "qmap_consensus",
        lower_um=observation.mic_um if observation.relation in {"=", ">", ">="} else None,
        upper_um=observation.mic_um if observation.relation in {"=", "<", "<="} else None,
    )
    return row


def normalize_activity(
    raw: dict[str, str],
    peptides: dict[int, dict[str, Any]],
    chemistry: dict[int, dict[str, Any]],
    index: int,
) -> dict[str, Any]:
    identifier = int(raw["id"])
    if identifier not in peptides:
        raise ValueError("Unmapped measurement ID")
    form = chemistry.get(identifier)
    if form is not None and form["sequence"] != peptides[identifier]["sequence"]:
        form = None
    return development_row(normalize_battle(raw, peptides[identifier], form, index))


def actual_homology(first: str, second: str) -> bool:
    # Length and residue-count bounds prune impossible pairs, never establish an edge.
    if min(len(first), len(second)) < 0.8 * max(len(first), len(second)):
        return False
    if sum((Counter(first) & Counter(second)).values()) < 0.8 * max(len(first), len(second)):
        return False
    aligner = PairwiseAligner(
        mode="global", match_score=2, mismatch_score=-1, open_gap_score=-2, extend_gap_score=-0.5
    )
    alignment = aligner.align(first, second)[0]
    a, b = alignment[0], alignment[1]
    matches = sum(x == y for x, y in zip(a, b, strict=True))
    paired = sum(x != "-" and y != "-" for x, y in zip(a, b, strict=True))
    return matches / len(a) >= 0.8 and paired / min(len(first), len(second)) >= 0.8


def assign_folds(sequences: list[str], edges: list[tuple[str, str]], seed: int) -> pd.DataFrame:
    unique = sorted(set(sequences))
    parent = {s: s for s in unique}

    def root(s: str) -> str:
        while s != parent[s]:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for left, right in edges:
        a, b = sorted((root(left), root(right)))
        parent[b] = a
    frame = pd.DataFrame({"sequence": unique, "homology_group": [root(s) for s in unique]})
    for column, groups in [
        ("exact_fold", unique),
        ("homology_fold", frame.homology_group.tolist()),
    ]:
        hashed = [hashlib.sha256(f"{seed}:{s}".encode()).hexdigest() for s in groups]
        count = min(5, len(set(hashed)))
        frame[column] = -1
        if count >= 2:
            for fold, (_, test) in enumerate(GroupKFold(count).split(frame, groups=hashed)):
                frame.loc[test, column] = fold
    return frame.set_index("sequence")


def oracle_union(pool: pd.DataFrame, rankers: list[str], count: int) -> set[str]:
    if count <= 0 or not rankers or pool.sequence.duplicated().any():
        raise ValueError("Require unique pool sequences, rankers and positive prefilter size")
    return set().union(
        *[
            set(
                pool.dropna(subset=[r])
                .sort_values([r, "raw_order", "sequence"], ascending=[False, True, True])
                .head(count)
                .sequence
            )
            for r in rankers
        ]
    )


def select_exploration_top(
    pool: pd.DataFrame,
    score: str,
    config: ExplorationConstraints,
    count: int,
    challenge: Callable[[str], float],
    known: Callable[[str], float],
) -> pd.DataFrame:
    ordered = pool[np.isfinite(pool[score])].sort_values(
        [score, "raw_order", "sequence"], ascending=[False, True, True]
    )
    if config.prefilter is not None:
        ordered = ordered.head(config.prefilter)
    selected, sequences = [], []
    clusters: Counter[int] = Counter()
    for row in ordered.itertuples():
        if (
            row.sequence in sequences
            or not 8 <= len(row.sequence) <= 50
            or set(row.sequence) - CANONICAL
        ):
            continue
        if config.physchem == "hard" and row.hard_reject:
            continue
        if config.cluster_cap is not None and clusters[row.embedding_cluster] >= config.cluster_cap:
            continue
        similarity = challenge(row.sequence)
        if similarity > config.challenge_max or similarity == 1.0:
            continue
        if config.known_max is not None and known(row.sequence) > config.known_max:
            continue
        if config.pairwise_max is not None and any(
            local_similarity(row.sequence, s) > config.pairwise_max for s in sequences
        ):
            continue
        selected.append(row.Index)
        sequences.append(row.sequence)
        clusters[row.embedding_cluster] += 1
        if len(selected) == count:
            return pool.loc[selected].copy()
    raise ValueError(f"infeasible: selected {len(selected)}/{count}")


def select_portfolio(
    pool: pd.DataFrame,
    fraction: float,
    config: ExplorationConstraints,
    count: int,
    challenge: Callable[[str], float],
    known: Callable[[str], float],
) -> pd.DataFrame:
    specialist_count = round(count * fraction)
    if not 0 < specialist_count < count:
        raise ValueError("Portfolio requires positive specialist and consensus quotas")
    frame = pool.copy()
    rankings = [
        frame.sort_values(
            [f"species_{i}", "raw_order", "sequence"], ascending=[False, True, True]
        ).index.tolist()
        for i in range(7)
    ]
    pointers = [0] * 7
    order, seen = [], set()
    for position in range(len(frame)):
        arm = position % 7
        while rankings[arm][pointers[arm]] in seen:
            pointers[arm] += 1
        index = rankings[arm][pointers[arm]]
        pointers[arm] += 1
        seen.add(index)
        order.append(index)
    frame["portfolio_score"] = 0.0
    frame.loc[order, "portfolio_score"] = np.arange(len(order), 0, -1) / len(order)
    specialists = select_exploration_top(
        frame, "portfolio_score", config, specialist_count, challenge, known
    )
    frame["portfolio_score"] = frame.consensus.rank(pct=True)
    frame.loc[specialists.index, "portfolio_score"] = 2 + np.arange(specialist_count, 0, -1)
    selected = select_exploration_top(frame, "portfolio_score", config, count, challenge, known)
    selected["portfolio_role"] = np.where(
        selected.index.isin(specialists.index), "specialist", "consensus"
    )
    if (selected.portfolio_role == "specialist").sum() != specialist_count:
        raise ValueError("infeasible: final portfolio lost specialist quota")
    return selected
