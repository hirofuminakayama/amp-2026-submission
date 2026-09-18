import argparse
import csv
import gzip
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from robust_apex_qd.generation.sampler import LengthPolicy, build_length_quotas
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.ranking.objectives import percentile_score
from robust_apex_qd.selection.library import LibraryCandidate, select_library
from robust_apex_qd.selection.top import (
    TopCandidate,
    TopSelectionConfig,
    select_top_with_fallback,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="") as file:
            return list(csv.DictReader(file))
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _unique_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Rows must have unique {key}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select L0/L1/L2 AMP candidate library")
    parser.add_argument("--variant", choices=("L0", "L1", "L2"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=ROOT / "work/candidates.csv.gz")
    parser.add_argument("--physchem", type=Path, default=ROOT / "work/candidate_physchem.csv.gz")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "work/candidate_embedding_diagnostics.csv.gz",
    )
    parser.add_argument("--apex", type=Path, default=ROOT / "work/apex_mean.csv")
    parser.add_argument(
        "--training-fasta",
        type=Path,
        default=ROOT / "data/training/training.fasta",
    )
    parser.add_argument(
        "--challenge-fasta",
        type=Path,
        default=ROOT / "data/antibacterial.fasta",
    )
    parser.add_argument("--size", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/library_selection.csv")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    raw_rows = _read_csv(options.candidates.resolve())
    physchem = _unique_by(_read_csv(options.physchem.resolve()), "candidate_id")
    embeddings = _unique_by(_read_csv(options.embeddings.resolve()), "candidate_id")
    apex = _unique_by(_read_csv(options.apex.resolve()), "sequence")
    candidates = tuple(
        LibraryCandidate(
            candidate_id=row["candidate_id"],
            sequence=row["sequence"],
            raw_order=int(row["raw_order"]),
            length=int(row["length"]),
            embedding_cluster=int(embeddings[row["candidate_id"]]["embedding_cluster"]),
            physchem_ood=float(physchem[row["candidate_id"]]["physchem_ood"]),
            embedding_ood=float(embeddings[row["candidate_id"]]["embedding_ood"]),
            valid=row["valid"] == "True",
        )
        for row in raw_rows
    )
    target_quotas = build_length_quotas(
        options.size,
        LengthPolicy.EMPIRICAL_TEMPERED,
        10,
        40,
        options.training_fasta.resolve(),
        0.75,
    )
    result = select_library(
        candidates,
        variant=options.variant,
        size=options.size,
        target_quotas=target_quotas,
    )
    median_log2 = np.asarray(
        [float(apex[candidate.sequence]["median_log2_mic"]) for candidate in result.selected]
    )
    quality = percentile_score(median_log2, higher_is_better=False)
    top_candidates = tuple(
        TopCandidate(
            candidate_id=candidate.candidate_id,
            sequence=candidate.sequence,
            raw_order=candidate.raw_order,
            final_score=float(score),
            embedding_cluster=candidate.embedding_cluster,
            physchem_hard_reject=physchem[candidate.candidate_id]["hard_reject"] == "True",
            median_log2_mic=float(apex[candidate.sequence]["median_log2_mic"]),
        )
        for candidate, score in zip(result.selected, quality, strict=True)
    )
    top = select_top_with_fallback(
        top_candidates,
        challenge_references=tuple(read_fasta_sequences(options.challenge_fasta.resolve())),
        known_references=tuple(read_fasta_sequences(options.training_fasta.resolve())),
        top_k=options.top_k,
        config=TopSelectionConfig(),
    )
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    library_path = output_dir / "library.fasta"
    write_fasta(
        [FastaRecord(candidate.candidate_id, candidate.sequence) for candidate in result.selected],
        library_path,
    )
    write_fasta(
        [
            FastaRecord(selected.candidate.candidate_id, selected.candidate.sequence)
            for selected in top.selected
        ],
        output_dir / "top.fasta",
    )
    with (output_dir / "ranking.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("rank", "candidate_id", "sequence", "final_score"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for rank, selected in enumerate(top.selected, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": selected.candidate.candidate_id,
                    "sequence": selected.candidate.sequence,
                    "final_score": f"{selected.candidate.final_score:.10g}",
                }
            )
    manifest = {
        "schema_version": 1,
        "library_count": len(result.selected),
        "top_count": len(top.selected),
        "library_variant": result.variant,
        "target_length_quotas": result.target_quotas,
        "actual_length_quotas": result.actual_quotas,
        "deficit_movements": result.movements,
        "cluster_coverage": result.cluster_coverage,
        "top_relaxation_step": top.relaxation_step,
        "manual_intervention": False,
        "output_sha256": {
            name: _sha256(output_dir / name)
            for name in ("library.fasta", "top.fasta", "ranking.tsv")
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report_path = options.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_csv(report_path) if report_path.exists() else []
    report_rows = [row for row in existing if row["variant"] != result.variant]
    report_rows.append(
        {
            "variant": result.variant,
            "target_length_histogram": json.dumps(result.target_quotas, sort_keys=True),
            "actual_length_histogram": json.dumps(result.actual_quotas, sort_keys=True),
            "deficit_redistribution": json.dumps(result.movements),
            "cluster_coverage": str(result.cluster_coverage),
            "library_sha256": _sha256(library_path),
        }
    )
    report_rows.sort(key=lambda row: row["variant"])
    with report_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "variant",
                "target_length_histogram",
                "actual_length_histogram",
                "deficit_redistribution",
                "cluster_coverage",
                "library_sha256",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(report_rows)
    print(
        f"Selected {len(result.selected)} {result.variant} rows with "
        f"{result.cluster_coverage} clusters into {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
