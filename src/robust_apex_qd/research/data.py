"""Conservative measurement normalization and label-blind sequence grouping."""

import hashlib
import json
import math
import re
from typing import Any, Literal

import Levenshtein
from Bio.SeqUtils import molecular_weight
from pydantic import BaseModel, ConfigDict

from robust_apex_qd.calibration.model import STRAIN_TO_PATHOGEN

CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    observation_id: str
    source: Literal["battleamp", "qmap"]
    source_id: int
    sequence: str
    target: str
    species: str
    apex_pathogen: str | None = None
    target_level: Literal["strain", "species"]
    chemical_form: str
    nterminal: str | None = None
    cterminal: str | None = None
    bonds: str | None = None
    chemical_evidence: str
    raw_value: str
    raw_unit: str | None = None
    relation: str | None = None
    mic_um: float | None = None
    active16: int | None = None
    exact_mic: bool = False
    consensus_um: float | None = None
    medium: str | None = None
    cfu: str | None = None
    note: str | None = None
    study: str | None = None
    exclusion_reasons: list[str]
    missing_fields: list[str]
    primary_eligible: bool = False
    apex_training_overlap: str = "unknown"


def parse_mic(text: str) -> tuple[str | None, float | None]:
    normalized = text.strip().replace("≤", "<=").replace("≥", ">=")
    match = re.fullmatch(r"(<=|>=|<|>|=)?\s*(\d+(?:\.\d*)?|\.\d+)([eE][+-]?\d+)?", normalized)
    if match is None:
        return None, None
    value = float(match[2] + (match[3] or ""))
    if not math.isfinite(value) or value <= 0:
        return None, None
    return match[1] or "=", value


def activity_label(relation: str | None, value: float | None) -> int | None:
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    if relation == "=":
        return int(value <= 16)
    if relation in {"<", "<="} and value <= 16:
        return 1
    if (relation == ">" and value >= 16) or (relation == ">=" and value > 16):
        return 0
    return None


def chemical_form(row: dict[str, Any] | None, sequence: str) -> str:
    if not sequence or set(sequence) - CANONICAL:
        return "noncanonical"
    if row is None or row.get("sequence") != sequence:
        return "unknown"
    if not {"bonds", "nterminal", "cterminal"} <= row.keys():
        return "unknown"
    if row["bonds"] or row["nterminal"] or row["cterminal"]:
        return "modified"
    return "reported_linear_free"


def normalize_qmap(raw: dict[str, Any]) -> list[Observation]:
    sequence = str(raw["sequence"])
    form = chemical_form(raw, sequence)
    rows = []
    for target, values in sorted(raw["targets"].items()):
        consensus = float(values[2])
        if not math.isfinite(consensus) or consensus <= 0:
            consensus = None
        rows.append(
            Observation(
                observation_id=f"qmap:{raw['id']}:{target}",
                source="qmap",
                source_id=int(raw["id"]),
                sequence=sequence,
                target=target,
                species=target,
                target_level="species",
                chemical_form=form,
                nterminal=raw.get("nterminal"),
                cterminal=raw.get("cterminal"),
                bonds=json.dumps(raw.get("bonds")),
                chemical_evidence="qmap_reported_fields",
                raw_value=json.dumps(values),
                raw_unit="µM",
                consensus_um=consensus,
                exclusion_reasons=["consensus_not_raw_measurement"],
                missing_fields=[
                    "raw_censoring",
                    "strain",
                    "medium",
                    "cfu",
                    "study",
                    "stereochemistry",
                ],
            )
        )
    return rows


def normalize_battle(
    raw: dict[str, str], peptide: dict[str, str], chemistry: dict[str, Any] | None, index: int
) -> Observation:
    sequence = peptide["sequence"].strip()
    form = chemical_form(chemistry, sequence)
    # Cross-release disagreement cannot establish a chemical form.
    if chemistry is not None and (
        bool(peptide.get("nTerminus")) != bool(chemistry.get("nterminal"))
        or bool(peptide.get("cTerminus")) != bool(chemistry.get("cterminal"))
    ):
        form = "conflicting_metadata"
    relation, value = parse_mic(raw.get("concentration", ""))
    unit = raw.get("unit", "").strip()
    reasons = []
    mic_um = None
    if value is None:
        reasons.append("invalid_or_ambiguous_concentration")
    elif unit in {"µM", "μM", "uM"}:
        mic_um = value
    elif unit in {"µg/ml", "μg/ml", "ug/ml"}:
        if form == "reported_linear_free":
            mic_um = value * 1000 / molecular_weight(sequence, seq_type="protein")
        else:
            reasons.append("unverified_chemical_form_for_mass_conversion")
    else:
        reasons.append("unsupported_or_missing_unit")
    if form != "reported_linear_free":
        reasons.append(f"chemical_form:{form}")
    if not 8 <= len(sequence) <= 50:
        reasons.append("outside_submission_length")
    target = raw["targetSpecies"].strip()
    species = " ".join(target.split()[:2])
    pathogen = STRAIN_TO_PATHOGEN.get(target)
    if pathogen is None:
        reasons.append("no_exact_apex_strain_match")
    active = activity_label(relation, mic_um)
    if active is None:
        reasons.append("activity16_not_determined")
    missing = [name for name in ["medium", "cfu", "note"] if not raw.get(name)]
    return Observation(
        observation_id=f"battleamp:{index}",
        source="battleamp",
        source_id=int(raw["id"]),
        sequence=sequence,
        target=target,
        species=species,
        apex_pathogen=pathogen,
        target_level="species" if target == species else "strain",
        chemical_form=form,
        nterminal=peptide.get("nTerminus") or None,
        cterminal=peptide.get("cTerminus") or None,
        bonds=json.dumps(chemistry["bonds"]) if chemistry is not None else None,
        chemical_evidence="qmap_same_id_sequence" if chemistry is not None else "bonds_unknown",
        raw_value=raw.get("concentration", ""),
        raw_unit=unit or None,
        relation=relation,
        mic_um=mic_um,
        active16=active,
        exact_mic=relation == "=" and mic_um is not None,
        medium=raw.get("medium") or None,
        cfu=raw.get("cfu") or None,
        note=raw.get("note") or None,
        exclusion_reasons=reasons,
        missing_fields=[*missing, "study", "stereochemistry"],
        primary_eligible=not reasons and pathogen is not None,
    )


def conservative_identity(left: str, right: str) -> float:
    """LCS / longest length upper-bounds matches / alignment length for any alignment."""
    if not left or not right:
        return 0.0
    matches = round(Levenshtein.ratio(left, right) * (len(left) + len(right)) / 2)
    return matches / max(len(left), len(right))


def grouped_split(
    sequences: list[str],
    historical: set[str],
    threshold: float = 0.6,
    seed: int = 42,
    linked_sequences: list[list[str]] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    if not 0 < threshold <= 1:
        raise ValueError("Identity threshold must be in (0, 1]")
    unique = sorted(set(sequences) | historical)
    parent = list(range(len(unique)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    positions = {sequence: i for i, sequence in enumerate(unique)}
    for linked in linked_sequences or []:
        members = [positions[seq] for seq in linked if seq in positions]
        for member in members[1:]:
            a, b = root(members[0]), root(member)
            parent[max(a, b)] = min(a, b)

    for i, left in enumerate(unique):
        for j in range(i):
            right = unique[j]
            if min(len(left), len(right)) / max(len(left), len(right)) < threshold:
                continue
            if conservative_identity(left, right) >= threshold:
                a, b = root(i), root(j)
                parent[max(a, b)] = min(a, b)
    historical_roots = {root(i) for i, seq in enumerate(unique) if seq in historical}
    groups, splits = {}, {}
    for i, sequence in enumerate(unique):
        group = hashlib.sha256(unique[root(i)].encode()).hexdigest()
        bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) / 2**32
        groups[sequence] = group
        splits[sequence] = (
            "historical"
            if root(i) in historical_roots
            else "train"
            if bucket < 0.7
            else "development"
            if bucket < 0.85
            else "holdout"
        )
    return groups, splits
