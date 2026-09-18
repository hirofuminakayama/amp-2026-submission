"""Build development-only datasets and exact/global-homology folds from pinned public sources."""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.research.data import CANONICAL, normalize_qmap
from robust_apex_qd.research.exploration import (
    actual_homology,
    assign_folds,
    development_row,
    normalize_activity,
)
from robust_apex_qd.research.metadata import publication_keys


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/competition_exploration.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    source = Path(config["sources"])
    old = Path(config["old_dataset"])
    fresh_output(args.output, [source, old])
    write_json(args.output / "protocol.json", config)
    source_manifest = json.loads(Path("configs/research_sources.json").read_text())
    verify_hashes({str(source / r["path"]): r["sha256"] for r in source_manifest["sources"]})
    old_manifest = json.loads((old / "dataset_manifest.json").read_text())
    verify_hashes({str(old / k): v for k, v in old_manifest["artifacts_sha256"].items()})
    qmap = json.loads((source / "qmap_hf/dbaasp.json").read_text())
    chemistry = {int(r["id"]): r for r in qmap}
    peptides = pd.read_csv(
        source / "battleamp/data/dbaasp/dbaasp_sequences.csv", keep_default_na=False
    )
    if peptides.id.duplicated().any():
        raise ValueError("Ambiguous peptide IDs")
    peptide_index = {int(r["id"]): r for r in peptides.to_dict("records")}
    records = [development_row(r) for raw in qmap for r in normalize_qmap(raw)]
    with (source / "battleamp/data/dbaasp/dbaasp_activity.csv").open() as f:
        for i, raw in enumerate(csv.DictReader(f)):
            records.append(normalize_activity(raw, peptide_index, chemistry, i))
    frame = pd.DataFrame(records)
    old_splits = pd.read_csv(old / "sequence_splits.csv").set_index("sequence")
    frame["old_partition"] = frame.sequence.map(old_splits.split).fillna("excluded_sequence")
    # Identical exports are collapsed; differing assays, chemistry or conditions remain separate.
    duplicate_key = [
        "source",
        "source_id",
        "sequence",
        "target",
        "raw_value",
        "raw_unit",
        "chemical_form",
        "medium",
        "cfu",
        "note",
    ]
    frame["duplicate_export"] = frame.duplicated(duplicate_key, keep="first")
    frame["included"] = frame.usable & ~frame.duplicate_export
    frame.to_json(args.output / "observations.jsonl", orient="records", lines=True)
    frame.groupby(
        [
            "source",
            "objective",
            "chemical_form",
            "target_level",
            "relation",
            "old_partition",
            "usable",
            "duplicate_export",
            "exact_regression",
        ],
        dropna=False,
    ).agg(
        observations=("observation_id", "size"),
        sequences=("sequence", "nunique"),
        included=("included", "sum"),
        determined_classification=("active16", "count"),
    ).reset_index().to_csv(args.output / "data_inventory.csv", index=False)
    frame[["observation_id", "exclusion_reasons", "usable", "duplicate_export"]].explode(
        "exclusion_reasons"
    ).to_csv(args.output / "exclusions.csv", index=False)
    development = frame[frame.included].copy()
    sequences = sorted(development.sequence.unique())
    alphabet = sorted(CANONICAL)
    counts = np.asarray([[s.count(a) for a in alphabet] for s in sequences], dtype=np.int16)
    lengths = np.asarray([len(s) for s in sequences])
    edges = []
    compared = 0
    for i, sequence in enumerate(sequences):
        lower = 0.8 * np.maximum(lengths[:i], lengths[i])
        eligible = (np.minimum(lengths[:i], lengths[i]) >= lower) & (
            np.minimum(counts[:i], counts[i]).sum(axis=1) >= lower
        )
        for j in np.flatnonzero(eligible):
            compared += 1
            if actual_homology(sequences[j], sequence):
                edges.append((sequences[j], sequence))
        if i % 1000 == 0:
            print(
                f"Homology {i}/{len(sequences)}; alignments={compared}; edges={len(edges)}",
                flush=True,
            )
    pd.DataFrame(edges, columns=pd.Index(["left", "right"])).to_csv(
        args.output / "homology_edges.csv", index=False
    )
    folds = assign_folds(sequences, edges, config["seeds"][0])
    folds.to_csv(args.output / "fold_assignments.csv")
    development = development.merge(
        folds, left_on="sequence", right_index=True, validate="many_to_one"
    )
    development.to_json(args.output / "development.jsonl", orient="records", lines=True)
    development.groupby(
        ["objective", "chemical_form", "species", "apex_pathogen"], dropna=False
    ).agg(
        observations=("sequence", "size"),
        sequences=("sequence", "nunique"),
        homology_groups=("homology_group", "nunique"),
        exact=("exact_regression", "sum"),
        classification=("active16", "count"),
    ).to_csv(args.output / "head_support.csv")
    cross_edges = sum(
        folds.loc[a, "homology_fold"] != folds.loc[b, "homology_fold"] for a, b in edges
    )
    if cross_edges or development.groupby("sequence").exact_fold.nunique().max() != 1:
        raise ValueError("Fold isolation failed")
    publication_rows = []
    metadata = Path("work/measured_activity_research/dbaasp-metadata-20260911-a")
    metadata_manifest = json.loads((metadata / "metadata_manifest.json").read_text())
    metadata_hashes = {
        str(metadata / f"{row['id']}.json"): row["sha256"]
        for row in metadata_manifest["records"]
        if row["status"] == "fetched"
    }
    verify_hashes(metadata_hashes)
    for path in sorted(metadata.glob("[0-9]*.json")):
        raw = json.loads(path.read_text())
        sequence = raw.get("sequence")
        if sequence not in folds.index:
            continue
        for key in publication_keys(raw.get("articles", [])):
            publication_rows.append(
                {
                    "publication_key": key,
                    "sequence": sequence,
                    "exact_fold": int(folds.loc[sequence, "exact_fold"]),
                    "homology_fold": int(folds.loc[sequence, "homology_fold"]),
                }
            )
    publications = pd.DataFrame(
        publication_rows,
        columns=pd.Index(["publication_key", "sequence", "exact_fold", "homology_fold"]),
    )
    publications.to_csv(args.output / "publication_sensitivity.csv", index=False)
    training = set(read_fasta_sequences(Path("data/training/training.fasta")))
    write_json(
        args.output / "overlap_report.json",
        {
            "exact_sequence_fold_crossings": 0,
            "homology_edge_crossings": int(cross_edges),
            "actual_alignments": compared,
            "homology_edges": len(edges),
            "apex_training_overlap": "unknown; exploratory model evaluation",
            "old_partition_reuse": development.groupby("old_partition").size().to_dict(),
            "generator_training_exact_sequences": len(set(sequences) & training),
            "duplicate_export_rows_collapsed": int(frame.duplicate_export.sum()),
            "deduplication_key": duplicate_key,
            "qmap_consensus": (
                "separate target, not an independent replication of DBAASP measurements"
            ),
            "publication_keys": int(publications.publication_key.nunique()),
            "publication_crossings_exact": int(
                (publications.groupby("publication_key").exact_fold.nunique() > 1).sum()
            ),
            "publication_role": "auxiliary sensitivity; no paper transitive quarantine",
        },
    )
    write_json(
        args.output / "split_manifest.json",
        {
            "config_sha256": file_sha256(args.config),
            "seed": config["seeds"][0],
            "raw_observations": len(frame),
            "development_observations": len(development),
            "sequences": len(sequences),
            "homology_groups": int(folds.homology_group.nunique()),
            "method": "GroupKFold on SHA256(seed:sequence/group); no labels used",
            "alignment": config["homology"],
            "reuse": config["development_reuse"],
            "single_group": "fold=-1; unsupported OOF, permitted shared-head training only",
            "old_dataset_manifest_sha256": file_sha256(old / "dataset_manifest.json"),
            "metadata_inputs_sha256": metadata_hashes,
            "source_manifest_sha256": file_sha256(Path("configs/research_sources.json")),
            "artifacts_sha256": {
                p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file()
            },
            "code_sha256": {
                str(p): file_sha256(p)
                for p in [
                    Path(__file__),
                    Path("src/robust_apex_qd/research/exploration.py"),
                    Path("src/robust_apex_qd/research/data.py"),
                ]
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"Complete: {len(development)} observations / {len(sequences)} sequences / "
        f"{folds.homology_group.nunique()} homology groups",
        flush=True,
    )


if __name__ == "__main__":
    main()
