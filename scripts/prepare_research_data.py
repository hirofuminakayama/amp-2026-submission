"""Normalize pinned MIC sources and freeze a conservative, label-blind research split."""

import argparse
import csv
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.research.data import (
    CANONICAL,
    conservative_identity,
    grouped_split,
    normalize_battle,
    normalize_qmap,
)
from robust_apex_qd.research.metadata import publication_keys


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    source_config = Path("configs/research_sources.json")
    source_manifest = json.loads(source_config.read_text())
    verify_hashes(
        {str(args.sources / row["path"]): row["sha256"] for row in source_manifest["sources"]}
    )
    inputs = [args.sources, Path("data"), Path("experimental")]
    if args.metadata is not None:
        inputs.append(args.metadata)
    fresh_output(args.output, inputs)
    # Freeze the protocol before reading measurement values or generating any predictions.
    protocol = {
        "seed": 42,
        "identity_threshold": 0.6,
        "similarity": "longest common subsequence / longer sequence length",
        "alignment": "global LCS; match=1, mismatch=0, gap=0; denominator=max(input lengths)",
        "grouping": "all-pairs threshold graph connected components including historical peptides",
        "allocation": (
            "sha256(seed:group) first 32 bits; <0.70 train, <0.85 development, else holdout"
        ),
        "historical": "entire connected components touching the existing 46 peptides",
        "qmap": "published five overlapping test memberships preserved separately; not CV folds",
        "qmap_identity": (
            "upstream BLOSUM45, gap open 5, extension 1; matches/max(alignment length,lengths)"
        ),
        "qmap_comparability": (
            "custom split is not the official benchmark; no benchmark result claimed"
        ),
        "homology_guarantee": (
            "LCS bound >= any alignment identity; conservative over-grouping possible"
        ),
        "preprocessing": "fit only on training portion of each fold",
        "holdout": "do not use labels for training, model selection, endpoint or weight changes",
        "adoption": (
            "blocked until APEX training overlap, study provenance and chemistry are resolved"
        ),
        "study_grouping": "merge all available publication-linked sequences before assignment",
    }
    studies: dict[str, set[str]] = defaultdict(set)
    if args.metadata is not None:
        path = args.metadata / "metadata_manifest.json"
        protocol["metadata_manifest_sha256"] = file_sha256(path)
        metadata = json.loads(path.read_text())
        for row in metadata["records"]:
            if row["status"] != "fetched":
                continue
            path = args.metadata / f"{row['id']}.json"
            if file_sha256(path) != row["sha256"]:
                raise ValueError("Metadata source hash mismatch")
            raw = json.loads(path.read_text())
            for key in publication_keys(raw.get("articles", [])):
                studies[key].add(raw["sequence"])
    protocol["publication_key_code_sha256"] = file_sha256(
        Path("src/robust_apex_qd/research/metadata.py")
    )
    write_json(args.output / "split_protocol.json", protocol)
    qmap = json.loads((args.sources / "qmap_hf/dbaasp.json").read_text())
    by_id = {row["id"]: row for row in qmap}
    if len(by_id) != len(qmap):
        raise ValueError("QMAP source IDs must be unique")
    with (args.sources / "battleamp/data/dbaasp/dbaasp_sequences.csv").open() as handle:
        peptide_rows = list(csv.DictReader(handle))
    peptides = {int(row["id"]): row for row in peptide_rows}
    if len(peptides) != len(peptide_rows):
        raise ValueError("BATTLE source IDs must be unique")
    observations = [observation for row in qmap for observation in normalize_qmap(row)]
    with (args.sources / "battleamp/data/dbaasp/dbaasp_activity.csv").open() as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            identifier = int(row["id"])
            if identifier not in peptides:
                raise ValueError("Activity row has no peptide record")
            chemistry = by_id.get(identifier)
            if chemistry is not None and chemistry["sequence"] != peptides[identifier]["sequence"]:
                chemistry = None
            observations.append(normalize_battle(row, peptides[identifier], chemistry, i))
    historical = set(pd.read_csv("experimental/mic.csv").sequence)
    sequences = sorted(
        {
            row.sequence
            for row in observations
            if 8 <= len(row.sequence) <= 50 and not set(row.sequence) - CANONICAL
        }
    )
    print(
        f"Normalized {len(observations)} observations; grouping {len(sequences)} sequences",
        flush=True,
    )
    groups, splits = grouped_split(
        sequences,
        historical,
        threshold=0.6,
        seed=42,
        linked_sequences=[sorted(values) for values in studies.values()],
    )
    study_crossings = sum(
        len({splits[seq] for seq in values if seq in splits}) > 1 for values in studies.values()
    )
    if study_crossings:
        raise ValueError("Known publication crosses partitions")
    training = set(read_fasta_sequences(Path("data/training/training.fasta")))
    all_sequences = sorted(set(sequences) | historical)
    with (args.output / "sequence_splits.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence",
                "group",
                "split",
                "generator_training_exact",
                "generator_training_max_bound",
                "apex_training_overlap",
                "study_overlap",
            ],
        )
        writer.writeheader()
        for i, sequence in enumerate(all_sequences):
            maximum = max(conservative_identity(sequence, known) for known in training)
            writer.writerow(
                {
                    "sequence": sequence,
                    "group": groups[sequence],
                    "split": splits[sequence],
                    "generator_training_exact": sequence in training,
                    "generator_training_max_bound": maximum,
                    "apex_training_overlap": "unknown",
                    "study_overlap": "unknown",
                }
            )
            if i % 3000 == 0:
                print(f"Training overlap audit {i}/{len(all_sequences)}", flush=True)
    # Independently enumerate cross-partition pairs; labels never enter this audit.
    cross_edges = 0
    max_cross = 0.0
    for i, left in enumerate(all_sequences):
        for right in all_sequences[:i]:
            if splits[left] == splits[right]:
                continue
            bound = conservative_identity(left, right)
            max_cross = max(max_cross, bound)
            cross_edges += int(bound >= 0.6)
    if cross_edges:
        raise ValueError("Cross-partition homology detected")
    public_memberships = []
    for fold in range(5):
        rows = json.loads((args.sources / f"qmap_hf/benchmark_split_{fold}.json").read_text())
        for row in rows:
            public_memberships.append(
                {"official_test_split": fold, "source_id": row["id"], "sequence": row["sequence"]}
            )
    pd.DataFrame(public_memberships).to_csv(
        args.output / "qmap_public_test_memberships.csv", index=False
    )
    frame = pd.DataFrame([row.model_dump() for row in observations])
    frame["group"] = frame.sequence.map(groups)
    frame["split"] = frame.sequence.map(splits).fillna("excluded_sequence")
    for split, subset in frame.groupby("split"):
        # Separate holdout artifact makes accidental training ingestion less likely.
        subset.to_json(args.output / f"observations_{split}.jsonl", orient="records", lines=True)
    frame.groupby(
        ["source", "species", "target", "target_level", "chemical_form", "split"], dropna=False
    ).agg(
        observations=("observation_id", "size"),
        sequences=("sequence", "nunique"),
        groups=("group", "nunique"),
        determined_activity=("active16", "count"),
        primary_eligible=("primary_eligible", "sum"),
    ).to_csv(args.output / "sample_counts.csv")
    exclusions = frame[["observation_id", "source", "exclusion_reasons"]].explode(
        "exclusion_reasons"
    )
    exclusions.dropna().to_csv(args.output / "exclusion_reasons.csv", index=False)
    frame[["observation_id", "source", "missing_fields"]].explode("missing_fields").to_csv(
        args.output / "missing_fields.csv", index=False
    )
    write_json(
        args.output / "leakage_audit.json",
        {
            "cross_partition_homology_edges": cross_edges,
            "max_cross_partition_identity_bound": max_cross,
            "historical_sequences": len(historical),
            "historical_component_sequences": sum(
                value == "historical" for value in splits.values()
            ),
            "apex_training_overlap": (
                "unknown; generator FASTA is not the complete APEX training set"
            ),
            "study_overlap": "unknown; source tables lack primary-paper identifiers",
            "qmap_public_memberships": len(public_memberships),
            "known_publication_crossings": study_crossings,
            "known_publications_grouped": len(studies),
        },
    )
    write_json(
        args.output / "split_manifest.json",
        {
            **protocol,
            "protocol_sha256": file_sha256(args.output / "split_protocol.json"),
            "sequence_assignments_sha256": file_sha256(args.output / "sequence_splits.csv"),
            "sequences_by_split": dict(Counter(splits.values())),
            "groups_by_split": {
                split: len({groups[s] for s in splits if splits[s] == split})
                for split in sorted(set(splits.values()))
            },
            "qmap_public_memberships_sha256": file_sha256(
                args.output / "qmap_public_test_memberships.csv"
            ),
        },
    )
    write_json(
        args.output / "dataset_manifest.json",
        {
            "schema_version": 1,
            "source_manifest": str(source_config),
            "source_manifest_sha256": file_sha256(source_config),
            "preparation_script_sha256": file_sha256(Path(__file__)),
            "normalization_sha256": file_sha256(Path("src/robust_apex_qd/research/data.py")),
            "uv_lock_sha256": file_sha256(Path("uv.lock")),
            "raw_observations_by_source": frame.source.value_counts().to_dict(),
            "primary_eligible_rows": int(frame.primary_eligible.sum()),
            "independent_adoption_ready": False,
            "limitations": [
                "APEX training provenance unknown",
                "primary study IDs unavailable",
                "QMAP consensus loses original censoring",
                "stereochemistry unverified",
                "cross-release chemical metadata is corroboration, not a primary-paper audit",
            ],
            "runtime_seconds": time.monotonic() - started,
            "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "artifacts_sha256": {
                str(p.relative_to(args.output)): file_sha256(p)
                for p in sorted(args.output.iterdir())
                if p.is_file()
            },
        },
    )
    print("Dataset and split manifests written", flush=True)


if __name__ == "__main__":
    main()
