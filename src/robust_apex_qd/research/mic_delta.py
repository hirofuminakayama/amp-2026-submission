"""Verified exact-MIC pair contracts and scaffold-balanced auxiliary loss."""

import math
from collections import defaultdict
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.nn import functional as F


class DeltaObservation(BaseModel):
    """Curated input; an evidence string is a provenance pointer, not automated verification."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        str_min_length=1,
        allow_inf_nan=False,
    )
    observation_id: str
    sequence: str
    scaffold_id: str
    target_id: str
    chemical_profile: str
    study_id: str
    comparable_assay_id: str
    verification_evidence: str
    mic_um: float = Field(gt=0)
    objective: Literal["measured_mic"]
    relation: Literal["="]
    assay_publication_verified: Literal[True]


class DeltaPair(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, str_min_length=1
    )
    schema_version: Literal[1] = 1
    left: DeltaObservation
    right: DeltaObservation
    partition: str

    @model_validator(mode="after")
    def comparable(self) -> "DeltaPair":
        if self.left.sequence == self.right.sequence:
            raise ValueError("Replicates of the same sequence are not mutation pairs")
        if self.left.observation_id == self.right.observation_id:
            raise ValueError("Duplicate observation ID")
        for field in (
            "target_id",
            "chemical_profile",
            "study_id",
            "comparable_assay_id",
            "scaffold_id",
        ):
            if getattr(self.left, field) != getattr(self.right, field):
                raise ValueError(f"Pair is not comparable: {field}")
        return self

    @property
    def delta_log2_um(self) -> float:
        """Right minus left; negative means the right endpoint has lower measured MIC."""
        return math.log2(self.right.mic_um) - math.log2(self.left.mic_um)


def build_delta_pairs(
    observations: list[DeltaObservation],
    endpoints: list[tuple[str, str]],
    partition_by_sequence: dict[str, str],
    partition: str,
) -> list[DeltaPair]:
    """Validate curated proposals after splitting; pass all split observations for leak checks.

    Assay IDs, chemistry profiles and scaffold IDs must already be reviewed and normalized.
    Similarity screening and publication verification belong to the input curation step.
    """
    by_id = {row.observation_id: row for row in observations}
    if len(by_id) != len(observations):
        raise ValueError("Duplicate observation ID")
    scaffold_partitions: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        assigned = partition_by_sequence.get(row.sequence)
        if not assigned or not assigned.strip():
            raise ValueError("Missing sequence partition")
        scaffold_partitions[row.scaffold_id].add(assigned)
    if any(len(parts) != 1 for parts in scaffold_partitions.values()):
        raise ValueError("Scaffold crosses a partition boundary")
    result = []
    seen: set[tuple[str, str]] = set()
    for left_id, right_id in endpoints:
        if left_id not in by_id or right_id not in by_id:
            raise ValueError("Unknown pair observation ID")
        left, right = by_id[left_id], by_id[right_id]
        if any(partition_by_sequence[row.sequence] != partition for row in (left, right)):
            raise ValueError("Pair endpoint is outside requested partition")
        key = (min(left_id, right_id), max(left_id, right_id))
        if key in seen:
            raise ValueError("Duplicate pair, including reversed orientation")
        seen.add(key)
        result.append(DeltaPair(left=left, right=right, partition=partition))
    return result


def delta_huber_loss(
    predictions_log2_um: torch.Tensor,
    observation_ids: list[str],
    pairs: list[DeltaPair],
    delta: float = 1.0,
) -> torch.Tensor:
    """Mean per scaffold of mean pair Huber loss, on a complete supplied pair batch.

    Predictions are for each observation's exact target/chemical profile. No batch-local row
    numbers are persisted. Empty eligible pairs contribute a differentiable zero auxiliary loss.
    """
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("Huber delta must be finite and positive")
    if (
        predictions_log2_um.ndim != 1
        or len(predictions_log2_um) != len(observation_ids)
        or len(set(observation_ids)) != len(observation_ids)
        or not predictions_log2_um.is_floating_point()
        or not bool(torch.isfinite(predictions_log2_um).all())
    ):
        raise ValueError("Expected finite floating predictions with unique aligned prediction IDs")
    if not pairs:
        return predictions_log2_um.sum() * 0
    by_id = {key: i for i, key in enumerate(observation_ids)}
    if any(row.observation_id not in by_id for p in pairs for row in (p.left, p.right)):
        raise ValueError("Missing prediction for pair endpoint")
    if len({p.partition for p in pairs}) != 1:
        raise ValueError("Pairs cross a partition boundary")
    keys = [tuple(sorted((p.left.observation_id, p.right.observation_id))) for p in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate pair")
    records: dict[str, DeltaObservation] = {}
    for pair in pairs:
        for row in (pair.left, pair.right):
            if row.observation_id in records and records[row.observation_id] != row:
                raise ValueError("Conflicting observation for prediction ID")
            records[row.observation_id] = row
    left = [by_id[p.left.observation_id] for p in pairs]
    right = [by_id[p.right.observation_id] for p in pairs]
    target = predictions_log2_um.new_tensor([p.delta_log2_um for p in pairs])
    losses = F.huber_loss(
        predictions_log2_um[right] - predictions_log2_um[left],
        target,
        reduction="none",
        delta=delta,
    )
    groups: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(pairs):
        groups[pair.left.scaffold_id].append(i)
    return torch.stack([losses[indices].mean() for indices in groups.values()]).mean()
