"""Versioned measured MIC bounds and deterministic similarity-isolated folds."""

import hashlib
from collections import defaultdict
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from pydantic import BaseModel, ConfigDict, Field, model_validator

from robust_apex_qd.research.data import Observation


class MICObservation(Observation):
    schema_version: int = 1
    objective: str
    lower_um: float | None = Field(default=None, gt=0)
    upper_um: float | None = Field(default=None, gt=0)
    duplicate_of: str | None = None
    assay_publication_verified: bool = False
    conversion_evidence: str

    @model_validator(mode="after")
    def check_bounds(self) -> "MICObservation":
        if (
            self.lower_um is not None
            and self.upper_um is not None
            and self.lower_um > self.upper_um
        ):
            raise ValueError("Inverted MIC bounds")
        return self


class MICConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    prior_models: str
    prior_report: str
    prior_refits: str
    development: str
    apex_development: str
    frozen_pool: str
    source_root: str
    outer_folds: int = Field(default=5, ge=2)
    inner_folds: int = Field(default=3, ge=2)
    identity_threshold: float = Field(default=0.6, gt=0, lt=1)
    seeds: list[int] = [42, 43, 44]
    epochs: int = Field(default=80, ge=1)
    batch_size: int = Field(default=256, ge=1)
    widths: list[int] = [32, 64]
    learning_rate: float = Field(default=0.001, gt=0)
    scale_floor: float = Field(default=0.1, gt=0)
    cpu_threads: int = Field(default=2, ge=1)
    gpu_hours: float = Field(default=96, gt=0)
    device: str = "cpu"


@lru_cache(maxsize=1)
def identity_aligner() -> PairwiseAligner:
    return PairwiseAligner(
        mode="global",
        substitution_matrix=substitution_matrices.load("BLOSUM45"),
        open_gap_score=-5,
        extend_gap_score=-1,
    )


def global_identity(left: str, right: str) -> float:
    """Matches / full global alignment length, with deterministic orientation and ties."""
    if not left or not right:
        raise ValueError("Empty alignment sequence")
    left, right = sorted((left, right))
    alignment = identity_aligner().align(left, right)[0]
    a, b = alignment[0], alignment[1]
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def fold_assignments(groups: dict[str, str], count: int, seed: int) -> dict[str, int]:
    members: dict[str, list[str]] = defaultdict(list)
    for sequence, group in groups.items():
        members[group].append(sequence)
    count = min(count, len(members))
    if count < 2:
        return dict.fromkeys(groups, -1)
    loads = [0] * count
    assignments = {}
    for group in sorted(
        members,
        key=lambda g: (-len(members[g]), hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()),
    ):
        fold = min(range(count), key=lambda f: (loads[f], f))
        for sequence in members[group]:
            assignments[sequence] = fold
        loads[fold] += len(members[group])
    return assignments


def similarity_groups(sequences: list[str], matrix: np.ndarray, threshold: float) -> dict[str, str]:
    if matrix.shape != (len(sequences), len(sequences)) or len(set(sequences)) != len(sequences):
        raise ValueError("Identity matrix must align with unique sequences")
    parent = list(range(len(sequences)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(sequences)):
        for j in np.flatnonzero(matrix[i, :i] > threshold):
            a, b = sorted((root(i), root(int(j))))
            parent[b] = a
    return {s: sequences[root(i)] for i, s in enumerate(sequences)}


def measured_bounds(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = rows.lower_um.to_numpy(dtype=float)
    high = rows.upper_um.to_numpy(dtype=float)
    if np.any(low[~np.isnan(low)] <= 0) or np.any(high[~np.isnan(high)] <= 0):
        raise ValueError("MIC bounds must be positive")
    if np.isinf(low).any() or np.isinf(high).any():
        raise ValueError("Persist absent bounds as null, not infinity")
    usable = rows.objective.eq("measured_mic").to_numpy() & (~np.isnan(low) | ~np.isnan(high))
    lower = np.log2(np.where(np.isnan(low), 1.0, low))
    upper = np.log2(np.where(np.isnan(high), 1.0, high))
    lower[np.isnan(low)] = -np.inf
    upper[np.isnan(high)] = np.inf
    if np.any(lower > upper):
        raise ValueError("Inverted MIC bounds")
    return lower, upper, usable


def normalized_observations(rows: pd.DataFrame) -> list[dict[str, Any]]:
    fields = set(Observation.model_fields)
    result = []
    for raw in rows.to_dict("records"):
        record = {key: value for key, value in raw.items() if key in fields}
        for key, value in record.items():
            if isinstance(value, float) and np.isnan(value):
                record[key] = None
        observation = MICObservation(
            **record,
            objective=raw["objective"],
            lower_um=raw["lower_um"] if pd.notna(raw["lower_um"]) else None,
            upper_um=raw["upper_um"] if pd.notna(raw["upper_um"]) else None,
            conversion_evidence=raw["chemical_evidence"]
            if raw["raw_unit"] != "uM"
            else "source_uM",
        )
        result.append(observation.model_dump())
    return result
