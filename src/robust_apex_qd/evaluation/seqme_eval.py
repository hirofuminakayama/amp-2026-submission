import csv
import gzip
import hashlib
import importlib.metadata
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances

from robust_apex_qd.features.embeddings import (
    ESM2_EMBEDDING_DIM,
    ESM2_MODEL_NAME,
    ESM2_MODEL_REVISION,
    file_sha256,
    validate_reusable_embeddings,
)
from robust_apex_qd.features.physchem import compute_features
from robust_apex_qd.io.fasta import FastaRecord, read_fasta
from robust_apex_qd.validation.compliance import challenge_valid_records


def fixed_subset_indices(count: int, subset_size: int, *, seed: int) -> NDArray[np.int64]:
    if count <= 0 or subset_size <= 0 or subset_size > count:
        raise ValueError("Subset size must be between one and the row count")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=subset_size, replace=False)).astype(np.int64)


@dataclass(frozen=True)
class CachedEmbeddingLookup:
    sequences: tuple[str, ...]
    embeddings: NDArray[np.float32]

    def __post_init__(self) -> None:
        if len(self.sequences) != len(set(self.sequences)):
            raise ValueError("Cached embedding sequences must be unique")
        if self.embeddings.shape[0] != len(self.sequences) or self.embeddings.ndim != 2:
            raise ValueError("Cached embedding rows must align with sequences")
        if self.embeddings.dtype != np.float32 or not np.isfinite(self.embeddings).all():
            raise ValueError("Cached embeddings must be finite float32 values")

    def __call__(self, sequences: list[str]) -> NDArray[np.float32]:
        row_by_sequence = {sequence: index for index, sequence in enumerate(self.sequences)}
        missing = [sequence for sequence in sequences if sequence not in row_by_sequence]
        if missing:
            raise ValueError(f"Sequence is absent from the cached embeddings: {missing[0]}")
        indices = [row_by_sequence[sequence] for sequence in sequences]
        return np.asarray(self.embeddings[indices], dtype=np.float32)


class MetricResultLike(Protocol):
    value: float | int


class MetricLike(Protocol):
    def __call__(self, sequences: list[str]) -> MetricResultLike: ...


def decide_library_adoption(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    if "L0" not in metrics:
        raise ValueError("L0 baseline metrics are required")
    baseline = metrics["L0"]
    directions = {
        "fbd": "minimize",
        "ngram_jaccard": "minimize",
        "length_kl": "minimize",
        "charge_kl": "minimize",
        "gravy_kl": "minimize",
        "diversity": "maximize",
        "fkea": "maximize",
        "precision": "maximize",
        "recall": "maximize",
        "authenticity": "maximize",
        "conformity": "maximize",
    }
    decisions: dict[str, dict[str, Any]] = {}
    for variant, values in metrics.items():
        improved = [
            name
            for name, direction in directions.items()
            if name in values
            and (
                values[name] < baseline[name]
                if direction == "minimize"
                else values[name] > baseline[name]
            )
        ]
        eligible = (
            values["uniqueness"] == 1.0
            and values["novelty"] == 1.0
            and values["fbd"] <= baseline["fbd"] * 1.05
            and values["conformity"] >= baseline["conformity"] * 0.95
            and values["diversity"] >= baseline["diversity"] - 0.01
            and len(improved) >= 2
        )
        decisions[variant] = {
            "eligible": bool(eligible),
            "improved_metrics": improved,
            "improvement_count": len(improved),
        }
    eligible_candidates = [
        variant
        for variant in ("L1", "L2")
        if variant in decisions and decisions[variant]["eligible"]
    ]
    adopted = min(
        eligible_candidates,
        key=lambda variant: (-int(decisions[variant]["improvement_count"]), variant),
        default="L0",
    )
    return {
        "schema_version": 1,
        "adopted_variant": adopted,
        "variants": decisions,
        "rule": {
            "uniqueness": 1.0,
            "novelty": 1.0,
            "fbd_max_relative_to_L0": 1.05,
            "conformity_min_relative_to_L0": 0.95,
            "diversity_max_absolute_decrease": 0.01,
            "minimum_improved_metrics": 2,
        },
    }


def _subset_checksum(records: Sequence[FastaRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.header.encode())
        digest.update(b"\0")
        digest.update(record.sequence.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _markdown_table(report: pd.DataFrame) -> str:
    columns = list(report.columns)
    lines = (
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    )
    body = [
        "| "
        + " | ".join(f"{value:.6g}" if isinstance(value, float) else str(value) for value in row)
        + " |"
        for row in report.itertuples(index=False, name=None)
    ]
    return "\n".join((*lines, *body)) + "\n"


def _metric_value(metric: MetricLike, sequences: list[str]) -> float:
    return float(metric(sequences).value)


def _charge(sequences: list[str]) -> np.ndarray:
    return np.asarray([compute_features(sequence)["charge_ph_7_4"] for sequence in sequences])


def _hydrophobic_moment(sequences: list[str]) -> np.ndarray:
    return np.asarray([compute_features(sequence)["hydrophobic_moment"] for sequence in sequences])


def _length(sequences: list[str]) -> np.ndarray:
    return np.asarray([len(sequence) for sequence in sequences], dtype=np.float64)


def _gravy(sequences: list[str]) -> np.ndarray:
    return np.asarray([compute_features(sequence)["gravy"] for sequence in sequences])


def _combined_lookup(
    candidate_sequences: Sequence[str],
    candidate_embeddings: NDArray[np.float32],
    reference_sequences: Sequence[str],
    reference_embeddings: NDArray[np.float32],
) -> CachedEmbeddingLookup:
    rows: dict[str, NDArray[np.float32]] = {}
    for sequence, embedding in zip(candidate_sequences, candidate_embeddings, strict=True):
        rows[sequence] = embedding
    for sequence, embedding in zip(reference_sequences, reference_embeddings, strict=True):
        if sequence in rows and not np.array_equal(rows[sequence], embedding):
            raise ValueError("Duplicate sequence has inconsistent cached embeddings")
        rows[sequence] = embedding
    sequences = tuple(rows)
    embeddings = np.asarray([rows[sequence] for sequence in sequences], dtype=np.float32)
    return CachedEmbeddingLookup(sequences=sequences, embeddings=embeddings)


def validated_embedding_lookup(
    *,
    candidates_path: Path,
    candidate_embeddings_path: Path,
    reference_fasta_path: Path,
    reference_embeddings_path: Path,
    embedding_manifest_path: Path,
) -> tuple[list[str], list[str], CachedEmbeddingLookup]:
    with gzip.open(candidates_path, "rt", newline="") as file:
        candidate_rows = list(csv.DictReader(file))
    if not candidate_rows or not {"candidate_id", "sequence"} <= set(candidate_rows[0]):
        raise ValueError("Candidate metadata must contain candidate_id and sequence")
    candidate_ids = [row["candidate_id"] for row in candidate_rows]
    candidate_sequences = [row["sequence"] for row in candidate_rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate metadata IDs must be unique")
    reference_records = challenge_valid_records(read_fasta(reference_fasta_path))
    reference_ids = [record.header for record in reference_records]
    reference_sequences = [record.sequence for record in reference_records]
    validate_reusable_embeddings(
        manifest_path=embedding_manifest_path,
        candidate_embeddings_path=candidate_embeddings_path,
        reference_embeddings_path=reference_embeddings_path,
        candidate_ids=candidate_ids,
        candidate_sequences=candidate_sequences,
        reference_ids=reference_ids,
        reference_sequences=reference_sequences,
        candidate_input_sha256=file_sha256(candidates_path),
        reference_input_sha256=file_sha256(reference_fasta_path),
        model_name=ESM2_MODEL_NAME,
        model_revision=ESM2_MODEL_REVISION,
        embedding_dimension=ESM2_EMBEDDING_DIM,
    )
    candidate_embeddings = np.load(candidate_embeddings_path, mmap_mode="r")
    reference_embeddings = np.load(reference_embeddings_path, mmap_mode="r")
    return (
        candidate_sequences,
        reference_sequences,
        _combined_lookup(
            candidate_sequences,
            candidate_embeddings,
            reference_sequences,
            reference_embeddings,
        ),
    )


def _evaluate_variant(
    full_sequences: list[str],
    subset_sequences: list[str],
    reference_full: list[str],
    reference_subset: list[str],
    embedder: CachedEmbeddingLookup,
    *,
    seed: int,
) -> dict[str, float]:
    sm = import_module("seqme")

    reference_embedding_sample = embedder(reference_subset[: min(256, len(reference_subset))])
    distances = pairwise_distances(reference_embedding_sample)
    bandwidth = float(np.median(distances[np.triu_indices_from(distances, k=1)]))
    bandwidth = max(bandwidth, 1e-6)
    metrics = {
        "count": _metric_value(sm.metrics.Count(), full_sequences),
        "uniqueness": _metric_value(sm.metrics.Uniqueness(), full_sequences),
        "novelty": _metric_value(sm.metrics.Novelty(reference_full), full_sequences),
        "diversity": _metric_value(sm.metrics.Diversity(k=5, seed=seed), subset_sequences),
        "ngram_jaccard": _metric_value(
            sm.metrics.NGramJaccardSimilarity(reference_subset, n=3),
            subset_sequences,
        ),
        "fbd": _metric_value(sm.metrics.FBD(reference_subset, embedder), subset_sequences),
        "fkea": _metric_value(
            sm.metrics.FKEA(
                embedder,
                bandwidth,
                n_random_fourier_features=256,
                batch_size=256,
                device="cpu",
                seed=seed,
            ),
            subset_sequences,
        ),
        "precision": _metric_value(
            sm.metrics.Precision(
                5,
                reference_subset,
                embedder,
                strict=False,
                device="cpu",
            ),
            subset_sequences,
        ),
        "recall": _metric_value(
            sm.metrics.Recall(
                5,
                reference_subset,
                embedder,
                strict=False,
                device="cpu",
            ),
            subset_sequences,
        ),
        "authenticity": _metric_value(
            sm.metrics.AuthPct(reference_subset, embedder),
            subset_sequences,
        ),
        "conformity": _metric_value(
            sm.metrics.ConformityScore(
                reference_subset,
                [_charge, _hydrophobic_moment],
                seed=seed,
            ),
            subset_sequences,
        ),
        "length_kl": _metric_value(
            sm.metrics.KLDivergence(reference_subset, _length, seed=seed),
            subset_sequences,
        ),
        "charge_kl": _metric_value(
            sm.metrics.KLDivergence(reference_subset, _charge, seed=seed),
            subset_sequences,
        ),
        "gravy_kl": _metric_value(
            sm.metrics.KLDivergence(reference_subset, _gravy, seed=seed),
            subset_sequences,
        ),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("seqme evaluation produced a non-finite metric")
    return metrics


def evaluate_single_library(
    *,
    library_path: Path,
    candidates_path: Path,
    candidate_embeddings_path: Path,
    reference_fasta_path: Path,
    reference_embeddings_path: Path,
    embedding_manifest_path: Path,
    seed: int,
    subset_size: int,
) -> tuple[dict[str, float], dict[str, object]]:
    """Evaluate one run-scoped library while preserving exact/full versus sampled provenance."""
    _, reference_sequences, embedder = validated_embedding_lookup(
        candidates_path=candidates_path,
        candidate_embeddings_path=candidate_embeddings_path,
        reference_fasta_path=reference_fasta_path,
        reference_embeddings_path=reference_embeddings_path,
        embedding_manifest_path=embedding_manifest_path,
    )
    records = read_fasta(library_path)
    indices = fixed_subset_indices(len(records), min(subset_size, len(records)), seed=seed)
    subset = [records[index] for index in indices]
    reference_indices = fixed_subset_indices(
        len(reference_sequences),
        min(subset_size, len(reference_sequences)),
        seed=seed + 10_000,
    )
    reference_subset = [reference_sequences[index] for index in reference_indices]
    metrics = _evaluate_variant(
        [record.sequence for record in records],
        [record.sequence for record in subset],
        reference_sequences,
        reference_subset,
        embedder,
        seed=seed,
    )
    provenance: dict[str, object] = {
        "full_count": len(records),
        "subset_count": len(subset),
        "subset_seed": seed,
        "subset_sha256": _subset_checksum(subset),
        "seqme_version": importlib.metadata.version("seqme"),
        "embedding_manifest_sha256": file_sha256(embedding_manifest_path),
    }
    return metrics, provenance


def run_seqme_evaluation(
    *,
    variant_paths: dict[str, Path],
    candidates_path: Path,
    candidate_embeddings_path: Path,
    reference_fasta_path: Path,
    reference_embeddings_path: Path,
    embedding_manifest_path: Path,
    seed: int,
    subset_size: int,
    csv_path: Path,
    markdown_path: Path,
    subset_ids_path: Path,
    adoption_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "L0" not in variant_paths:
        raise ValueError("L0 baseline variant is required for library adoption")
    _, reference_sequences, embedder = validated_embedding_lookup(
        candidates_path=candidates_path,
        candidate_embeddings_path=candidate_embeddings_path,
        reference_fasta_path=reference_fasta_path,
        reference_embeddings_path=reference_embeddings_path,
        embedding_manifest_path=embedding_manifest_path,
    )
    reference_indices = fixed_subset_indices(
        len(reference_sequences),
        min(subset_size, len(reference_sequences)),
        seed=seed + 10_000,
    )
    reference_subset = [reference_sequences[index] for index in reference_indices]
    report_rows: list[dict[str, object]] = []
    metrics_by_variant: dict[str, dict[str, float]] = {}
    subset_lines: list[str] = []
    for variant_index, (variant, path) in enumerate(variant_paths.items()):
        records = read_fasta(path)
        indices = fixed_subset_indices(
            len(records),
            min(subset_size, len(records)),
            seed=seed + variant_index,
        )
        subset = [records[index] for index in indices]
        subset_lines.extend(f"{variant}\t{record.header}" for record in subset)
        metrics = _evaluate_variant(
            [record.sequence for record in records],
            [record.sequence for record in subset],
            reference_sequences,
            reference_subset,
            embedder,
            seed=seed,
        )
        metrics_by_variant[variant] = metrics
        report_rows.append(
            {
                "variant": variant,
                **metrics,
                "full_count": len(records),
                "subset_count": len(subset),
                "subset_sha256": _subset_checksum(subset),
                "seqme_version": importlib.metadata.version("seqme"),
                "embedding_manifest_sha256": file_sha256(embedding_manifest_path),
            }
        )
    report = pd.DataFrame(report_rows)
    adoption = decide_library_adoption(metrics_by_variant)
    adoption["seqme_version"] = importlib.metadata.version("seqme")
    adoption["seqme_license"] = "BSD-3-Clause"
    adoption["subset_seed"] = seed
    adoption["embedding_manifest_sha256"] = file_sha256(embedding_manifest_path)
    adoption["candidate_embeddings_sha256"] = file_sha256(candidate_embeddings_path)
    adoption["reference_embeddings_sha256"] = file_sha256(reference_embeddings_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(csv_path, index=False, float_format="%.10g", lineterminator="\n")
    markdown_path.write_text(_markdown_table(report))
    subset_ids_path.write_text("\n".join(subset_lines) + "\n")
    adoption_path.write_text(json.dumps(adoption, indent=2, sort_keys=True) + "\n")
    return report, adoption
