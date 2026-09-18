import argparse
import csv
import gzip
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from robust_apex_qd.io.fasta import FastaRecord, read_fasta, read_fasta_sequences, write_fasta
from robust_apex_qd.ranking.objectives import (
    broad_objectives,
    configured_ranker_score,
    conservative_mdr_proxy,
    resolve_ranker,
)
from robust_apex_qd.selection.top import (
    TopCandidate,
    select_top_with_fallback,
    top_selection_config_from_mapping,
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


def _rows_by_key(
    rows: list[dict[str, str]],
    key: str,
) -> dict[str, dict[str, str]]:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Rows must have unique {key}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select deterministic robust APEX Top-100")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/candidate.yaml")
    parser.add_argument("--candidates", type=Path, default=ROOT / "work/candidates.csv.gz")
    parser.add_argument("--apex", type=Path, default=ROOT / "work/apex_mean.csv")
    parser.add_argument(
        "--physchem",
        type=Path,
        default=ROOT / "work/candidate_physchem.csv.gz",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "work/candidate_embedding_diagnostics.csv.gz",
    )
    parser.add_argument(
        "--challenge-fasta",
        type=Path,
        default=ROOT / "data/antibacterial.fasta",
    )
    parser.add_argument(
        "--known-fasta",
        type=Path,
        default=ROOT / "data/training/training.fasta",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--library-size", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--library-fasta", type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    loaded_config = yaml.safe_load(options.config.resolve().read_text())
    if not isinstance(loaded_config, dict) or not isinstance(loaded_config.get("ranking"), dict):
        raise ValueError("Configuration must contain a ranking mapping")
    ranking_config = loaded_config["ranking"]
    ranker = resolve_ranker(ranking_config)
    selection_config = top_selection_config_from_mapping(ranking_config)
    candidate_rows = _read_csv(options.candidates.resolve())
    valid_rows = [row for row in candidate_rows if row["valid"] == "True"]
    if options.library_fasta is None:
        library_rows = valid_rows[: options.library_size]
    else:
        valid_by_sequence = {row["sequence"]: row for row in valid_rows}
        library_records = read_fasta(options.library_fasta.resolve())
        if any(record.sequence not in valid_by_sequence for record in library_records):
            raise ValueError("Library FASTA contains a sequence absent from valid candidates")
        library_rows = [valid_by_sequence[record.sequence] for record in library_records]
    if len(library_rows) != options.library_size:
        raise ValueError(f"Only {len(library_rows)} valid rows for the requested library")
    apex_by_sequence = _rows_by_key(_read_csv(options.apex.resolve()), "sequence")
    physchem_by_id = _rows_by_key(_read_csv(options.physchem.resolve()), "candidate_id")
    embedding_by_id = _rows_by_key(_read_csv(options.embeddings.resolve()), "candidate_id")
    aligned_apex_rows = [apex_by_sequence[row["sequence"]] for row in library_rows]
    final_scores = configured_ranker_score(
        aligned_apex_rows,
        ranker,
    )
    candidates: list[TopCandidate] = []
    for row, final_score in zip(library_rows, final_scores, strict=True):
        candidate_id = row["candidate_id"]
        physchem = physchem_by_id[candidate_id]
        embedding = embedding_by_id[candidate_id]
        if physchem["sequence"] != row["sequence"]:
            raise ValueError(f"Physchem sequence differs for {candidate_id}")
        candidates.append(
            TopCandidate(
                candidate_id=candidate_id,
                sequence=row["sequence"],
                raw_order=int(row["raw_order"]),
                final_score=float(final_score),
                embedding_cluster=int(embedding["embedding_cluster"]),
                physchem_hard_reject=physchem["hard_reject"] == "True",
                median_log2_mic=float(apex_by_sequence[row["sequence"]]["median_log2_mic"]),
            )
        )
    result = select_top_with_fallback(
        tuple(candidates),
        challenge_references=tuple(read_fasta_sequences(options.challenge_fasta.resolve())),
        known_references=tuple(read_fasta_sequences(options.known_fasta.resolve())),
        top_k=options.top_k,
        config=selection_config,
    )
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(
        [FastaRecord(row["candidate_id"], row["sequence"]) for row in library_rows],
        output_dir / "library.fasta",
    )
    write_fasta(
        [
            FastaRecord(row.candidate.candidate_id, row.candidate.sequence)
            for row in result.selected
        ],
        output_dir / "top.fasta",
    )
    ranking_path = output_dir / "ranking.tsv"
    fieldnames = (
        "rank",
        "candidate_id",
        "sequence",
        "broad_mean",
        "broad_tail",
        "mdr_proxy",
        "apex_median_log2_mic",
        "apex_model_disagreement",
        "physchem_ood",
        "embedding_ood",
        "challenge_similarity",
        "known_similarity",
        "pairwise_similarity",
        "embedding_cluster",
        "final_score",
    )
    with ranking_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for rank, selected in enumerate(result.selected, start=1):
            candidate = selected.candidate
            apex = apex_by_sequence[candidate.sequence]
            probability = np.asarray(
                [[float(value) for key, value in apex.items() if key.startswith("success16__")]]
            )
            broad_mean, broad_tail = broad_objectives(probability)
            mdr_proxy = conservative_mdr_proxy(probability)
            physchem = physchem_by_id[candidate.candidate_id]
            embedding = embedding_by_id[candidate.candidate_id]
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": candidate.candidate_id,
                    "sequence": candidate.sequence,
                    "broad_mean": f"{float(broad_mean[0]):.10g}",
                    "broad_tail": f"{float(broad_tail[0]):.10g}",
                    "mdr_proxy": f"{float(mdr_proxy[0]):.10g}",
                    "apex_median_log2_mic": f"{candidate.median_log2_mic:.10g}",
                    "apex_model_disagreement": apex["model_disagreement_mad_log2"],
                    "physchem_ood": physchem["physchem_ood"],
                    "embedding_ood": embedding["embedding_ood"],
                    "challenge_similarity": f"{selected.challenge_similarity:.10g}",
                    "known_similarity": f"{selected.known_similarity:.10g}",
                    "pairwise_similarity": f"{selected.pairwise_similarity:.10g}",
                    "embedding_cluster": candidate.embedding_cluster,
                    "final_score": f"{candidate.final_score:.10g}",
                }
            )
    manifest = {
        "schema_version": 1,
        "library_count": len(library_rows),
        "top_count": len(result.selected),
        "ranker": ranker,
        "manual_intervention": False,
        "relaxation_step": result.relaxation_step,
        "prefilter_size": result.prefilter_size,
        "cluster_cap": result.cluster_cap,
        "pairwise_similarity_max": result.pairwise_threshold,
        "challenge_similarity_max": selection_config.challenge_similarity_max,
        "known_similarity_max": selection_config.known_similarity_max,
        "rejection_counts": result.rejection_counts,
        "output_sha256": {
            name: _sha256(output_dir / name)
            for name in ("library.fasta", "top.fasta", "ranking.tsv")
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Selected {len(result.selected)} rows at fallback step {result.relaxation_step} "
        f"into {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
