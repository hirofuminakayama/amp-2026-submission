import argparse
import csv
import gzip
import io
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from robust_apex_qd.features.embeddings import (
    ESM2_EMBEDDING_DIM,
    ESM2_MODEL_NAME,
    ESM2_MODEL_REVISION,
    Esm2BatchEncoder,
    compute_embedding_diagnostics,
    file_sha256,
    row_mapping_sha256,
    validate_reusable_embeddings,
    write_embedding_memmap,
)
from robust_apex_qd.io.fasta import read_fasta
from robust_apex_qd.validation.compliance import challenge_valid_records

ROOT = Path(__file__).resolve().parents[1]


def _read_candidate_rows(path: Path) -> tuple[list[str], list[str]]:
    identifiers: list[str] = []
    sequences: list[str] = []
    with gzip.open(path, "rt", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"candidate_id", "sequence"} <= set(reader.fieldnames):
            raise ValueError("Candidate table must contain candidate_id and sequence")
        for row in reader:
            identifiers.append(row["candidate_id"])
            sequences.append(row["sequence"])
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("Candidate IDs must be non-empty and unique")
    return identifiers, sequences


def _write_diagnostics(
    path: Path,
    identifiers: Sequence[str],
    embedding_ood: np.ndarray,
    cluster: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw_file,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as gzip_file,
        io.TextIOWrapper(gzip_file, encoding="utf-8", newline="") as text_file,
    ):
        writer = csv.DictWriter(
            text_file,
            fieldnames=("candidate_id", "embedding_ood", "embedding_cluster"),
            lineterminator="\n",
        )
        writer.writeheader()
        for identifier, ood, cluster_id in zip(
            identifiers,
            embedding_ood,
            cluster,
            strict=True,
        ):
            writer.writerow(
                {
                    "candidate_id": identifier,
                    "embedding_ood": f"{float(ood):.10g}",
                    "embedding_cluster": int(cluster_id),
                }
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute ESM2 embeddings and diagnostics")
    parser.add_argument("--candidates", type=Path, default=ROOT / "work/candidates.csv.gz")
    parser.add_argument(
        "--reference-fasta",
        type=Path,
        default=ROOT / "data/training/training.fasta",
    )
    parser.add_argument(
        "--candidate-embeddings",
        type=Path,
        default=ROOT / "work/candidate_embeddings.npy",
    )
    parser.add_argument(
        "--reference-embeddings",
        type=Path,
        default=ROOT / "work/reference_embeddings.npy",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=ROOT / "work/candidate_embedding_diagnostics.csv.gz",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT / "work/embedding_manifest.json",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--pca-reference-subset", type=int, default=10_000)
    parser.add_argument("--cluster-count", type=int, default=512)
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument("--reuse-embeddings", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    candidates_path = options.candidates.resolve()
    reference_path = options.reference_fasta.resolve()
    candidate_ids, candidate_sequences = _read_candidate_rows(candidates_path)
    reference_records = challenge_valid_records(read_fasta(reference_path))
    reference_ids = [record.header for record in reference_records]
    reference_sequences = [record.sequence for record in reference_records]
    candidate_embeddings_path = options.candidate_embeddings.resolve()
    reference_embeddings_path = options.reference_embeddings.resolve()
    candidate_input_sha256 = file_sha256(candidates_path)
    reference_input_sha256 = file_sha256(reference_path)

    if options.reuse_embeddings:
        validate_reusable_embeddings(
            manifest_path=options.manifest_output.resolve(),
            candidate_embeddings_path=candidate_embeddings_path,
            reference_embeddings_path=reference_embeddings_path,
            candidate_ids=candidate_ids,
            candidate_sequences=candidate_sequences,
            reference_ids=reference_ids,
            reference_sequences=reference_sequences,
            candidate_input_sha256=candidate_input_sha256,
            reference_input_sha256=reference_input_sha256,
            model_name=ESM2_MODEL_NAME,
            model_revision=ESM2_MODEL_REVISION,
            embedding_dimension=ESM2_EMBEDDING_DIM,
        )
    else:
        encoder = Esm2BatchEncoder(options.device)
        write_embedding_memmap(
            candidate_sequences,
            candidate_embeddings_path,
            options.batch_size,
            encoder,
        )
        write_embedding_memmap(
            reference_sequences,
            reference_embeddings_path,
            options.batch_size,
            encoder,
        )

    candidate_embeddings = np.load(candidate_embeddings_path, mmap_mode="r")
    reference_embeddings = np.load(reference_embeddings_path, mmap_mode="r")
    expected_candidate_shape = (len(candidate_sequences), ESM2_EMBEDDING_DIM)
    expected_reference_shape = (len(reference_sequences), ESM2_EMBEDDING_DIM)
    if candidate_embeddings.shape != expected_candidate_shape:
        raise ValueError(
            f"Candidate embedding shape {candidate_embeddings.shape} != {expected_candidate_shape}"
        )
    if reference_embeddings.shape != expected_reference_shape:
        raise ValueError(
            f"Reference embedding shape {reference_embeddings.shape} != {expected_reference_shape}"
        )

    diagnostics = compute_embedding_diagnostics(
        candidate_embeddings,
        reference_embeddings,
        seed=options.seed,
        pca_components=options.pca_components,
        cluster_count=options.cluster_count,
        pca_reference_subset=options.pca_reference_subset,
        thread_count=options.thread_count,
    )
    diagnostics_path = options.diagnostics_output.resolve()
    _write_diagnostics(
        diagnostics_path,
        candidate_ids,
        diagnostics.embedding_ood,
        diagnostics.cluster,
    )
    manifest = {
        "schema_version": 2,
        "model_name": ESM2_MODEL_NAME,
        "model_revision": ESM2_MODEL_REVISION,
        "embedding_dtype": "float32",
        "embedding_dimension": ESM2_EMBEDDING_DIM,
        "candidate_count": len(candidate_ids),
        "reference_count": len(reference_ids),
        "candidate_input_sha256": candidate_input_sha256,
        "reference_input_sha256": reference_input_sha256,
        "reference_filter": "canonical_20_and_length_8_to_50",
        "candidate_row_mapping_sha256": row_mapping_sha256(candidate_ids, candidate_sequences),
        "reference_row_mapping_sha256": row_mapping_sha256(reference_ids, reference_sequences),
        "candidate_embeddings_sha256": file_sha256(candidate_embeddings_path),
        "reference_embeddings_sha256": file_sha256(reference_embeddings_path),
        "diagnostics_sha256": file_sha256(diagnostics_path),
        "pca_components": options.pca_components,
        "pca_reference_subset": min(
            options.pca_reference_subset,
            len(reference_sequences),
        ),
        "pca_seed": options.seed,
        "pca_transform_sha256": diagnostics.pca_transform_sha256,
        "cluster_count": options.cluster_count,
        "clustering_seed": options.seed,
        "thread_count": options.thread_count,
        "batch_size": options.batch_size,
    }
    manifest_path = options.manifest_output.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote candidate embeddings {candidate_embeddings.shape}, "
        f"reference embeddings {reference_embeddings.shape}, and {options.cluster_count} clusters"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
