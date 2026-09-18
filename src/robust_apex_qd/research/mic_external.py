"""Label-free paper/homology components and explicit external-training boundaries."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from robust_apex_qd.features.embeddings import file_sha256


class ExternalObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    observation_id: str = Field(min_length=1)
    sequence: str = Field(pattern=r"^[ACDEFGHIKLMNPQRSTVWY]+$")
    paper_ids: list[str]
    species: str
    exposure: Literal["used", "unknown", "certified_unused"]
    exposure_evidence: str = Field(min_length=1)
    primary_eligible: bool

    @model_validator(mode="after")
    def check_unused(self) -> "ExternalObservation":
        if self.exposure == "certified_unused" and not self.paper_ids:
            raise ValueError("Unused certification requires paper identity")
        if any(not p for p in self.paper_ids):
            raise ValueError("Empty paper identity")
        return self


def build_external_split(
    rows: list[ExternalObservation],
    sequences: list[str],
    identity: np.ndarray,
    duplicate_links: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Join all paper/duplicate/strictly-above-60% edges before seed-fixed allocation."""
    ids = {r.observation_id: i for i, r in enumerate(rows)}
    if len(ids) != len(rows) or sequences != sorted({r.sequence for r in rows}):
        raise ValueError("Unique observation IDs and complete sorted sequence membership required")
    if (
        identity.shape != (len(sequences), len(sequences))
        or not np.isfinite(identity).all()
        or (identity < 0).any()
        or (identity > 1).any()
        or not np.array_equal(identity, identity.T)
        or not np.all(np.diag(identity) == 1)
    ):
        raise ValueError("Invalid complete native identity matrix")
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[root(i)] = root(j)

    sequence_owner: dict[str, int] = {}
    paper_owner: dict[str, int] = {}
    for i, row in enumerate(rows):
        union(i, sequence_owner.setdefault(row.sequence, i))
        for paper in row.paper_ids:
            union(i, paper_owner.setdefault(paper, i))
    for left, right in duplicate_links:
        if left not in ids or right not in ids:
            raise ValueError("Unresolved duplicate endpoint")
        union(ids[left], ids[right])
    threshold = np.asarray(0.6, dtype=identity.dtype)
    for i, sequence in enumerate(sequences):
        for j in np.flatnonzero(identity[i, :i] > threshold):
            union(sequence_owner[sequence], sequence_owner[sequences[j]])
    members: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        members[root(i)].append(i)
    components, partitions = {}, {}
    fresh = []
    for key, indices in members.items():
        names = sorted(rows[i].observation_id for i in indices)
        component = hashlib.sha256(json.dumps(names).encode()).hexdigest()
        components[key] = component
        exposures = {rows[i].exposure for i in indices}
        if "used" in exposures:
            partitions[component] = "development"
        elif "unknown" in exposures:
            partitions[component] = "diagnostic"
        else:
            fresh.append(component)
    fresh.sort()
    random.Random(42).shuffle(fresh)
    for i, component in enumerate(fresh):
        partitions[component] = "final_evaluation" if i < (len(fresh) + 1) // 2 else "new_training"
    return [
        dict(
            **row.model_dump(),
            component_id=components[root(i)],
            partition=partitions[components[root(i)]],
        )
        for i, row in enumerate(rows)
    ]


def validate_training_membership(rows: list[dict[str, Any]], contract: dict[str, Any]) -> None:
    allowed = set(contract["allowed_observation_ids"])
    forbidden_ids = set(contract["forbidden_observation_ids"])
    forbidden_sequences = set(contract["forbidden_sequences"])
    forbidden_papers = set(contract["forbidden_paper_ids"])
    forbidden_components = set(contract["forbidden_component_ids"])
    seen = set()
    for row in rows:
        identifier = row["observation_id"]
        papers = set(row.get("paper_ids", [])) | set(row.get("lineage", {}).get("study_ids", []))
        papers.update(p for p in [row.get("study"), row.get("paper_id")] if p)
        if (
            identifier not in allowed
            or identifier in forbidden_ids
            or identifier in seen
            or row.get("sequence") in forbidden_sequences
            or papers & forbidden_papers
            or row.get("component_id") in forbidden_components
            or row.get("external_partition") in {"final_evaluation", "diagnostic"}
        ):
            raise ValueError("Training input violates registered external evaluation membership")
        seen.add(identifier)


def load_training_rows(prepared: Path) -> pd.DataFrame:
    """Read trainer rows, validating external allowlists before fitting or feature selection."""
    path = prepared / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(r.get("external_partition") in {"final_evaluation", "diagnostic"} for r in rows):
        raise ValueError("External evaluation or diagnostic labels are not training inputs")
    contract_path = prepared / "training_contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        if file_sha256(path) != contract["rows_sha256"]:
            raise ValueError("External training rows changed")
        validate_training_membership(rows, contract)
        if not contract.get("folds_ready", True):
            raise ValueError("External training requires new development folds and features")
    elif any("external_partition" in r for r in rows):
        raise ValueError("External rows require a registered training contract")
    return pd.read_json(path, lines=True)
