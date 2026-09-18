"""Conservative export lineage and publication/benchmark quarantine contracts."""

import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.metadata import publication_keys
from robust_apex_qd.research.mic_data import MICObservation


class MeasurementLineage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    schema_version: Literal[2] = 2
    observation_id: str = Field(min_length=1)
    source_record: str = Field(min_length=1)
    metadata_path: str | None = None
    metadata_sha256: str | None = None
    assay_ids: list[int] = []
    study_ids: list[str] = []
    publication_candidates: list[str] = []
    linkage_evidence: str | None = None
    duplicate_status: str = "unresolved"
    duplicate_of: str | None = None
    raw_assays: list[dict[str, Any]] = []

    @model_validator(mode="after")
    def evidence_required(self) -> "MeasurementLineage":
        if self.study_ids and not (
            self.assay_ids and self.metadata_sha256 and self.linkage_evidence
        ):
            raise ValueError("Verified study linkage requires matched assay and source evidence")
        return self


class LineagedMICObservation(MICObservation):
    schema_version: Literal[2] = 2
    lineage: MeasurementLineage
    legacy_included: bool
    screen_included: bool

    @model_validator(mode="after")
    def check_lineage(self) -> "LineagedMICObservation":
        if self.observation_id != self.lineage.observation_id:
            raise ValueError("Lineage observation ID mismatch")
        if self.duplicate_of != self.lineage.duplicate_of:
            raise ValueError("Duplicate linkage mismatch")
        if self.objective == "qmap_consensus" and (
            self.lower_um is not None or self.upper_um is not None or self.exact_mic
        ):
            raise ValueError("Consensus cannot be represented as a measured interval")
        if self.objective == "measured_mic" and self.mic_um is not None:
            bounds = {
                "=": (self.mic_um, self.mic_um),
                ">": (self.mic_um, None),
                ">=": (self.mic_um, None),
                "<": (None, self.mic_um),
                "<=": (None, self.mic_um),
            }
            if (
                self.relation not in bounds
                or (self.lower_um, self.upper_um) != bounds[self.relation]
            ):
                raise ValueError("Censoring relation and normalized bounds disagree")
        return self


def reference_articles(token: Any, articles: list[dict[str, Any]]) -> list[str]:
    """DB display numbers are one-based article positions; reject ambiguous syntax."""
    if token is None or not re.fullmatch(r"\s*\d+(?:\s*,\s*\d+)*\s*", str(token)):
        return []
    indices = [int(part.strip()) - 1 for part in str(token).split(",")]
    if any(i < 0 or i >= len(articles) for i in indices):
        return []
    selected = [publication_keys([articles[i]]) for i in indices]
    if any(not keys for keys in selected):
        return []
    return sorted(set().union(*selected))


def duplicate_links(
    rows: pd.DataFrame, assay_ids: dict[str, list[int]]
) -> dict[str, tuple[str, str | None]]:
    if rows.observation_id.duplicated().any():
        raise ValueError("Observation IDs must be unique")
    result = {}
    for _, group in rows.groupby("export_key", sort=False):
        known: dict[int, str] = {}
        for identifier in group.observation_id:
            matches = assay_ids.get(identifier, [])
            if len(matches) == 1:
                key = matches[0]
                if key in known:
                    result[identifier] = ("same_source_assay", known[key])
                else:
                    result[identifier] = ("distinct_assay_record", None)
                    known[key] = identifier
            else:
                result[identifier] = (
                    "unresolved_identical_export" if len(group) > 1 else "unresolved_source_record",
                    None,
                )
    return result


def publication_groups(sequences: list[str], publications: dict[str, list[str]]) -> dict[str, str]:
    parent = {s: s for s in sequences}

    def root(s: str) -> str:
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    papers: dict[str, list[str]] = defaultdict(list)
    for s in sequences:
        for paper in publications.get(s, []):
            papers[paper].append(s)
    for members in papers.values():
        for member in members[1:]:
            a, b = sorted((root(members[0]), root(member)))
            parent[b] = a
    return {s: root(s) for s in sequences}


def retained_training(
    training: list[int], heldout: list[int], identity: np.ndarray, threshold: float
) -> list[int]:
    if not heldout or not 0 < threshold < 1:
        raise ValueError("Nonempty heldout and a fractional threshold are required")
    if identity.ndim != 2 or identity.shape[0] != identity.shape[1]:
        raise ValueError("Expected square identity matrix")
    cutoff = np.asarray(threshold, dtype=identity.dtype)
    maxima = identity[np.ix_(training, heldout)].max(axis=1)
    if not np.isfinite(maxima).all():
        raise ValueError("Missing identities cannot pass quarantine")
    test_set = set(heldout)
    return [i for i, m in zip(training, maxima, strict=True) if i not in test_set and m <= cutoff]


def official_training_mask(
    sequences: list[str], heldout: set[str], maxima: np.ndarray, threshold: float
) -> np.ndarray:
    """Official benchmark removal uses >=, unlike development connected-group edges."""
    if not heldout or not 0 < threshold < 1 or maxima.shape != (len(sequences),):
        raise ValueError("Nonempty test set and aligned maxima are required")
    if not np.isfinite(maxima).all() or np.any((maxima < 0) | (maxima > 1)):
        raise ValueError("Invalid test similarity")
    return (maxima < np.asarray(threshold, dtype=maxima.dtype)) & np.array(
        [s not in heldout for s in sequences]
    )


def capture_execution(output: Path) -> None:
    """Snapshot source bodies and locks before running a read-only data experiment."""
    sources = [
        *Path("src/robust_apex_qd").rglob("*.py"),
        *Path("scripts").glob("*mic*.py"),
        Path("scripts/run_competition_models.py"),
        Path("uv.lock"),
        Path("pyproject.toml"),
        Path("configs/research_sources.json"),
        Path("configs/mic_research.json"),
        Path("configs/mic_phase01_sources.json"),
    ]
    hashes = {}
    for path in sources:
        destination = output / "executed_sources" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[str(destination.relative_to(output))] = file_sha256(destination)
    manifest = dict(
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        artifacts_sha256=hashes,
    )
    (output / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
