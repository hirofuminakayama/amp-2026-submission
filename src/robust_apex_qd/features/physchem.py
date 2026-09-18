import csv
import gzip
import io
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pydantic import BaseModel, ConfigDict

CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
FEATURE_NAMES = (
    "length",
    "charge_ph_7_4",
    "charge_density",
    "gravy",
    "hydrophobic_moment",
    "aromaticity",
    "boman_index",
    "isoelectric_point",
    "molecular_weight",
    "shannon_entropy",
    "longest_homopolymer",
    "max_aa_fraction",
    "cysteine_count",
)
EPSILON = 1e-6
ROBUST_Z_CLIP = 10.0

EISENBERG_HYDROPHOBICITY = {
    "A": 0.62,
    "C": 0.29,
    "D": -0.90,
    "E": -0.74,
    "F": 1.19,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "K": -1.50,
    "L": 1.06,
    "M": 0.64,
    "N": -0.78,
    "P": 0.12,
    "Q": -0.85,
    "R": -2.53,
    "S": -0.18,
    "T": -0.05,
    "V": 1.08,
    "W": 0.81,
    "Y": 0.26,
}

BOMAN_SOLUBILITY = {
    "A": 0.17,
    "C": 0.24,
    "D": -1.23,
    "E": -2.02,
    "F": 1.13,
    "G": 0.01,
    "H": -0.96,
    "I": 0.31,
    "K": -0.99,
    "L": 0.56,
    "M": 0.23,
    "N": -0.42,
    "P": 0.45,
    "Q": -0.58,
    "R": -1.01,
    "S": -0.13,
    "T": -0.14,
    "V": 0.07,
    "W": 1.85,
    "Y": 0.94,
}


class FeatureStatistics(BaseModel):
    model_config = ConfigDict(frozen=True)

    median: float
    iqr: float
    q0_005: float
    q0_01: float
    q0_99: float
    q0_995: float


class PhyschemReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    reference_sha256: str
    reference_count: int
    reference_filter: str
    feature_order: tuple[str, ...]
    epsilon: float
    robust_z_clip: float
    statistics: dict[str, FeatureStatistics]


@dataclass(frozen=True)
class PhyschemScore:
    physchem_ood: float
    soft_penalty: float
    extreme_feature_count: int
    hard_reject: bool


def _hydrophobic_moment(sequence: str) -> float:
    angle = math.radians(100.0)
    x_component = sum(
        EISENBERG_HYDROPHOBICITY[amino_acid] * math.cos(index * angle)
        for index, amino_acid in enumerate(sequence)
    )
    y_component = sum(
        EISENBERG_HYDROPHOBICITY[amino_acid] * math.sin(index * angle)
        for index, amino_acid in enumerate(sequence)
    )
    return math.hypot(x_component, y_component) / len(sequence)


def _shannon_entropy(counts: Counter[str], length: int) -> float:
    return -sum((count / length) * math.log2(count / length) for count in counts.values() if count)


def _longest_homopolymer(sequence: str) -> int:
    longest = 1
    current = 1
    for previous, amino_acid in pairwise(sequence):
        if amino_acid == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def compute_features(sequence: str) -> dict[str, float]:
    normalized = sequence.strip().upper()
    if not normalized:
        raise ValueError("Sequence must not be empty")
    invalid = set(normalized) - CANONICAL_AMINO_ACIDS
    if invalid:
        raise ValueError(f"Sequence contains non-canonical amino acids: {sorted(invalid)}")

    analysis = ProteinAnalysis(normalized)
    counts = Counter(normalized)
    length = len(normalized)
    charge = float(analysis.charge_at_pH(7.4))
    features = {
        "length": float(length),
        "charge_ph_7_4": charge,
        "charge_density": charge / length,
        "gravy": float(analysis.gravy()),
        "hydrophobic_moment": _hydrophobic_moment(normalized),
        "aromaticity": float(analysis.aromaticity()),
        "boman_index": sum(BOMAN_SOLUBILITY[amino_acid] for amino_acid in normalized) / length,
        "isoelectric_point": float(analysis.isoelectric_point()),
        "molecular_weight": float(analysis.molecular_weight()),
        "shannon_entropy": _shannon_entropy(counts, length),
        "longest_homopolymer": float(_longest_homopolymer(normalized)),
        "max_aa_fraction": max(counts.values()) / length,
        "cysteine_count": float(counts.get("C", 0)),
    }
    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("Physicochemical feature order drifted")
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("Physicochemical feature computation produced a non-finite value")
    return features


def fit_reference(
    sequences: Iterable[str],
    *,
    reference_sha256: str,
) -> PhyschemReference:
    materialized_sequences = list(sequences)
    feature_rows = [compute_features(sequence) for sequence in materialized_sequences]
    if not feature_rows:
        raise ValueError("Reference must contain at least one sequence")
    matrix = np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in feature_rows],
        dtype=np.float64,
    )
    statistics: dict[str, FeatureStatistics] = {}
    for column_index, name in enumerate(FEATURE_NAMES):
        values = matrix[:, column_index]
        q0_005, q0_01, q0_25, median, q0_75, q0_99, q0_995 = np.quantile(
            values,
            (0.005, 0.01, 0.25, 0.5, 0.75, 0.99, 0.995),
        )
        statistics[name] = FeatureStatistics(
            median=float(median),
            iqr=float(q0_75 - q0_25),
            q0_005=float(q0_005),
            q0_01=float(q0_01),
            q0_99=float(q0_99),
            q0_995=float(q0_995),
        )
    return PhyschemReference(
        schema_version=2,
        reference_sha256=reference_sha256,
        reference_count=len(materialized_sequences),
        reference_filter="canonical_20_and_length_8_to_50",
        feature_order=FEATURE_NAMES,
        epsilon=EPSILON,
        robust_z_clip=ROBUST_Z_CLIP,
        statistics=statistics,
    )


def score_features(
    features: Mapping[str, float],
    reference: PhyschemReference,
) -> PhyschemScore:
    if tuple(features) != reference.feature_order:
        raise ValueError("Feature order does not match the physicochemical reference")
    robust_z: dict[str, float] = {}
    extreme_feature_count = 0
    for name in reference.feature_order:
        value = float(features[name])
        summary = reference.statistics[name]
        scale = max(summary.iqr, reference.epsilon)
        robust_z[name] = float(
            np.clip(
                (value - summary.median) / scale,
                -reference.robust_z_clip,
                reference.robust_z_clip,
            )
        )
        if value < summary.q0_005 or value > summary.q0_995:
            extreme_feature_count += 1

    physchem_ood = float(np.mean([abs(value) for value in robust_z.values()]))
    soft_names = (
        "charge_density",
        "gravy",
        "hydrophobic_moment",
        "cysteine_count",
    )
    soft_penalty = float(np.mean([abs(robust_z[name]) for name in soft_names]))
    entropy_is_extreme = (
        features["shannon_entropy"] < reference.statistics["shannon_entropy"].q0_005
    )
    hard_reject = entropy_is_extreme or extreme_feature_count >= 3
    return PhyschemScore(
        physchem_ood=physchem_ood,
        soft_penalty=soft_penalty,
        extreme_feature_count=extreme_feature_count,
        hard_reject=hard_reject,
    )


def write_reference(reference: PhyschemReference, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = reference.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_candidate_features(
    candidates_path: Path,
    output_path: Path,
    reference: PhyschemReference,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "candidate_id",
        "sequence",
        *FEATURE_NAMES,
        "physchem_ood",
        "soft_penalty",
        "extreme_feature_count",
        "hard_reject",
    )
    seen_ids: set[str] = set()
    row_count = 0
    with (
        gzip.open(candidates_path, "rt", newline="") as input_file,
        output_path.open("wb") as raw_output,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as gzip_output,
        io.TextIOWrapper(gzip_output, encoding="utf-8", newline="") as text_output,
    ):
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or not {"candidate_id", "sequence"} <= set(reader.fieldnames):
            raise ValueError("Candidate table must contain candidate_id and sequence")
        writer = csv.DictWriter(text_output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            candidate_id = row["candidate_id"].strip()
            sequence = row["sequence"].strip().upper()
            if not candidate_id or not sequence:
                raise ValueError("Candidate ID and sequence must not be empty")
            if candidate_id in seen_ids:
                raise ValueError(f"Duplicate candidate_id: {candidate_id}")
            seen_ids.add(candidate_id)
            features = compute_features(sequence)
            score = score_features(features, reference)
            output_row: dict[str, str | int] = {
                "candidate_id": candidate_id,
                "sequence": sequence,
                "physchem_ood": f"{score.physchem_ood:.10g}",
                "soft_penalty": f"{score.soft_penalty:.10g}",
                "extreme_feature_count": score.extreme_feature_count,
                "hard_reject": str(score.hard_reject),
            }
            output_row.update({name: f"{features[name]:.10g}" for name in FEATURE_NAMES})
            writer.writerow(output_row)
            row_count += 1
    return row_count
