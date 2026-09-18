"""Report possible study overlap and chemical metadata without rewriting measurements."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.metadata import publication_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh_output(args.output, [args.metadata, args.dataset])
    manifest_path = args.metadata / "metadata_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    splits = pd.read_csv(args.dataset / "sequence_splits.csv").set_index("sequence").split.to_dict()
    by_article: dict[str, set[str]] = defaultdict(set)
    coverage = []
    for row in manifest["records"]:
        if row["status"] != "fetched":
            continue
        path = args.metadata / f"{row['id']}.json"
        if file_sha256(path) != row["sha256"]:
            raise ValueError("Metadata source hash mismatch")
        partition = splits.get(row["sequence"], "excluded_sequence")
        raw = json.loads(path.read_text())
        for article in publication_keys(raw.get("articles", [])):
            if partition != "excluded_sequence":
                by_article[article].add(partition)
        modified = bool(
            row["nterminal"]
            or row["cterminal"]
            or row["intrachain_bonds"]
            or row["interchain_bonds"]
            or row["unusual_amino_acids"]
        )
        coverage.append(
            {
                "source_id": row["id"],
                "sequence": row["sequence"],
                "split": partition,
                "article_count": len(row["article_ids"]),
                "reported_modified": modified,
                "pubmed_ids": json.dumps(row["pubmed_ids"]),
            }
        )
    pd.DataFrame(coverage).to_csv(args.output / "metadata_coverage.csv", index=False)
    current = {row["source_id"]: row for row in coverage}
    conflicts = []
    for path in sorted(args.dataset.glob("observations_*.jsonl")):
        for line in path.open():
            observation = json.loads(line)
            if observation["source"] != "battleamp" or not observation["primary_eligible"]:
                continue
            metadata = current.get(observation["source_id"])
            if metadata is not None and (
                metadata["reported_modified"] or metadata["sequence"] != observation["sequence"]
            ):
                conflicts.append(
                    {
                        "observation_id": observation["observation_id"],
                        "source_id": observation["source_id"],
                        "split": observation["split"],
                    }
                )
    (args.output / "chemical_conflicts.json").write_text(json.dumps(conflicts, indent=2) + "\n")
    overlaps = [
        {"publication_key": article, "partitions": sorted(partitions)}
        for article, partitions in sorted(by_article.items())
        if len(partitions) > 1
    ]
    (args.output / "possible_study_overlap.json").write_text(json.dumps(overlaps, indent=2) + "\n")
    summary = {
        "audit_script_sha256": file_sha256(Path(__file__)),
        "publication_key_code_sha256": file_sha256(Path("src/robust_apex_qd/research/metadata.py")),
        "metadata_manifest_sha256": file_sha256(manifest_path),
        "dataset_manifest_sha256": file_sha256(args.dataset / "dataset_manifest.json"),
        "requested_records": len(manifest["records"]),
        "fetched_records": len(coverage),
        "records_with_article_candidates": sum(bool(row["article_count"]) for row in coverage),
        "candidate_articles": len(by_article),
        "possible_cross_partition_studies": len(overlaps),
        "technically_eligible_rows_with_current_chemical_conflicts": len(conflicts),
        "independent_adoption_ready": False,
        "interpretation": "Peptide article candidates are not verified assay-to-paper links. "
        "Current chemistry is not substituted into older source measurements. "
        "Possible study overlap requires curation before an independence claim.",
    }
    (args.output / "metadata_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
