"""Audit corrected endpoint eligibility before fitting a joint selection procedure."""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.bioaccuracy import EndpointObservation, chemistry_key, joint_hit_bounds
from robust_apex_qd.research.mic_data import fold_assignments, similarity_groups
from robust_apex_qd.research.mic_lineage import publication_groups

ChemistryStatus = Literal["known_linear_free_L", "known_modified", "unknown"]


def refit_mic_rows(
    original: pd.DataFrame, corrected: list[EndpointObservation], split: dict[str, Any]
) -> pd.DataFrame:
    """Retain feature indices while applying corrected ID membership and shared folds."""
    by_id = {row.observation_id: row for row in corrected if row.endpoint == "measured_mic"}
    if not set(by_id) <= set(original.observation_id):
        raise ValueError("Corrected MIC observations absent from original feature mapping")
    if original.observation_id.duplicated().any():
        raise ValueError("Duplicate original observation IDs")
    selected = original[original.observation_id.isin(by_id)].copy()
    if any(row.sequence != by_id[row.observation_id].sequence for row in selected.itertuples()):
        raise ValueError("Corrected sequence differs from original feature mapping")
    selected = selected[
        selected.observation_id.map(lambda key: chemistry_status(by_id[key]) != "known_modified")
    ].copy()
    selected["homology_group"] = selected.sequence.map(split["groups"])
    selected["homology_fold"] = selected.sequence.map(split["outer"])
    if selected[["homology_group", "homology_fold"]].isna().any().any():
        raise ValueError("Missing shared-fold mapping")
    selected["reviewed_chemistry"] = selected.observation_id.map(
        lambda key: chemistry_status(by_id[key])
    )
    return selected.reset_index(drop=True)


class ChemistryEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str
    observation_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    sources_sha256: dict[str, str] = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    nterminal: str | None = None
    cterminal: str | None = None
    bonds: str | None = None
    stereochemistry: Literal["L", "D", "mixed"] | None = None
    chemistry_support: dict[str, str]


def apply_chemistry_evidence(
    rows: list[EndpointObservation], records: list[ChemistryEvidence]
) -> list[EndpointObservation]:
    """Apply reviewed chemistry only to the exact observation and source version."""
    by_id = {row.observation_id: row for row in rows}
    if len(by_id) != len(rows) or len({r.observation_id for r in records}) != len(records):
        raise ValueError("Duplicate observation or evidence identifier")
    verified: dict[str, str] = {}
    for record in records:
        row = by_id.get(record.observation_id)
        if row is None:
            raise ValueError("Evidence observation is missing")
        if hashlib.sha256(row.model_dump_json().encode()).hexdigest() != record.observation_sha256:
            raise ValueError("Evidence observation hash mismatch")
        for path, expected in record.sources_sha256.items():
            if path not in verified:
                verified[path] = file_sha256(Path(path))
            actual = verified[path]
            if actual != expected:
                raise ValueError("Evidence source hash mismatch")
        updates = record.model_dump(
            include={"nterminal", "cterminal", "bonds", "stereochemistry", "chemistry_support"}
        )
        by_id[row.observation_id] = EndpointObservation.model_validate(row.model_dump() | updates)
    return [by_id[row.observation_id] for row in rows]


def primary_joint_coverage(
    rows: list[EndpointObservation], split: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit molecular pairs and usable validation folds without changing selection rules."""
    eligible = [r for r in rows if chemistry_status(r) == "known_linear_free_L"]
    human: dict[str, list[EndpointObservation]] = defaultdict(list)
    for row in eligible:
        if row.endpoint == "measured_hc50" and row.rbc_species == "human":
            human[chemistry_key(row)].append(row)
    paired: dict[tuple[str, str], dict[str, Any]] = {}
    for mic in eligible:
        if mic.endpoint != "measured_mic":
            continue
        molecule = chemistry_key(mic)
        for hc in human.get(molecule, []):
            bounds = joint_hit_bounds(mic, hc, ratio=8, threshold=16)
            if bounds is None:
                continue
            key = (molecule, mic.species)
            entry = paired.setdefault(
                key,
                dict(
                    molecule_id=molecule,
                    sequence=mic.sequence,
                    species=mic.species,
                    component=split["groups"][mic.sequence],
                    outer=split["outer"][mic.sequence],
                    hit_lower=bounds[0],
                    hit_upper=bounds[1],
                ),
            )
            entry["hit_lower"] = min(entry["hit_lower"], bounds[0])
            entry["hit_upper"] = max(entry["hit_upper"], bounds[1])
    pairs = list(paired.values())
    coverage = []
    for species in sorted({row["species"] for row in pairs}):
        species_pairs = [row for row in pairs if row["species"] == species]
        outer_folds = sorted({row["outer"] for row in species_pairs})
        inner_ready = {}
        for outer, assignment in split["inner"].items():
            inner_ready[outer] = sorted(
                {
                    assignment[row["sequence"]]
                    for row in species_pairs
                    if row["sequence"] in assignment
                }
            ) == list(range(3))
        coverage.append(
            dict(
                species=species,
                molecules=len(species_pairs),
                outer_folds=outer_folds,
                all_inner_covered=len(inner_ready) == 5 and all(inner_ready.values()),
            )
        )
    supported = bool(coverage) and all(
        row["outer_folds"] == list(range(5)) and row["all_inner_covered"] for row in coverage
    )
    return dict(
        primary_joint_pairs=len(pairs),
        paired_molecules=len({row["molecule_id"] for row in pairs}),
        paired_components=len({row["component"] for row in pairs}),
        species_coverage=coverage,
        nested_supported=supported,
        reason="primary pairs do not cover all registered outer/inner folds"
        if not supported
        else "coverage available; full procedure implementation and budget audit still required",
    ), pairs


def chemistry_status(row: EndpointObservation) -> ChemistryStatus:
    """Missing structural annotations are not evidence of an all-L molecule."""
    fields = ("nterminal", "cterminal", "bonds")
    if any(getattr(row, name) for name in fields) or any(
        value == "reported_modified" for value in row.chemistry_support.values()
    ):
        return "known_modified"
    stereo = (row.stereochemistry or "").strip().upper()
    if stereo in {"D", "ALL-D", "MIXED", "D/L"}:
        return "known_modified"
    if (
        all(row.chemistry_support.get(name) == "reported_free" for name in fields)
        and stereo in {"L", "ALL-L"}
        and row.chemistry_support.get("stereochemistry") == "reported_L"
    ):
        return "known_linear_free_L"
    return "unknown"


def corrected_observations(
    rows: list[EndpointObservation], corrections: list[str]
) -> tuple[list[EndpointObservation], list[dict[str, Any]]]:
    """Exclude identified observations without transferring chemistry across assays."""
    if len(set(corrections)) != len(corrections):
        raise ValueError("Duplicate correction identifiers")
    counts = Counter(row.observation_id for row in rows)
    if any(n != 1 for n in counts.values()):
        raise ValueError("Duplicate endpoint identifiers")
    excluded = set(corrections)
    return [row for row in rows if row.observation_id not in excluded], [
        dict(observation_id=identifier, matches=counts[identifier])
        for identifier in sorted(excluded)
    ]


def joint_development_split(
    sequences: list[str], identity: np.ndarray, publications: dict[str, list[str]]
) -> dict[str, Any]:
    """Keep the transitive union of paper and homology links within every fold."""
    if (
        len(set(sequences)) != len(sequences)
        or identity.shape != (len(sequences), len(sequences))
        or not np.isfinite(identity).all()
        or not np.allclose(identity, identity.T)
        or np.any(identity < 0)
        or np.any(identity > 1)
    ):
        raise ValueError("Invalid shared identity matrix or sequence order")
    homology = similarity_groups(sequences, identity, 0.6)
    links = {s: [f"homology:{homology[s]}", *publications.get(s, [])] for s in sequences}
    groups = publication_groups(sequences, links)
    outer = fold_assignments(groups, 5, 42)
    inner = {
        str(fold): fold_assignments({s: g for s, g in groups.items() if outer[s] != fold}, 3, 42)
        for fold in sorted(set(outer.values()))
        if fold >= 0
    }
    # Audit both paper links and transitive components for each level.
    for assignment in [outer, *inner.values()]:
        seen: dict[str, int] = {}
        for sequence, fold in assignment.items():
            for key in [f"component:{groups[sequence]}", *links[sequence]]:
                if seen.setdefault(key, fold) != fold:
                    raise ValueError("Cross-endpoint component crosses a fold boundary")
    return dict(
        groups=groups,
        outer=outer,
        inner=inner,
        folds_ready=set(outer.values()) == set(range(5))
        and all(set(v.values()) == set(range(3)) for v in inner.values()),
        threshold=0.6,
        seed=42,
        development_only=True,
        prior_holdout_reused=True,
        paper_unknown_sequences=sum(not publications.get(s) for s in sequences),
        boundary_checks="homology/paper transitive components, outer and inner",
    )
