"""Verify paper artifacts, cell replay, correction references and the preserved pilot."""

import argparse
import json
from collections import Counter
from pathlib import Path

from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.paper_mic import PaperObservation, read_source_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--curation", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--old-pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verified = {}
    for directory in [args.registry, args.curation, args.pilot]:
        manifest = json.loads((directory / "manifest.json").read_text())
        expected = dict(manifest["inputs_sha256"])
        expected.update({str(directory / k): v for k, v in manifest["artifacts_sha256"].items()})
        verified.update(verify_hashes(expected))
    rows = [
        PaperObservation.model_validate_json(line)
        for line in (args.curation / "paper_observations.jsonl").read_text().splitlines()
    ]
    grids = {}
    normalized = {}
    for row in rows:
        key = row.source_file, row.table_id
        if key not in grids:
            grids[key] = read_source_table(Path(row.source_file), row.table_id)
        if grids[key][row.row][row.column] != row.cell_text:
            raise ValueError("Saved original cell mismatch")
        try:
            relation, low, high = row.bounds
            normalized[row.observation_id] = dict(
                observation_id=row.observation_id, relation=relation, lower_um=low, upper_um=high
            )
        except ValueError:
            pass
    saved = {
        r["observation_id"]: r
        for r in map(
            json.loads, (args.curation / "normalized_bounds.jsonl").read_text().splitlines()
        )
    }
    if normalized != saved:
        raise ValueError("Saved MIC normalization mismatch")
    corrections = [
        json.loads(line)
        for line in (args.curation / "correction_ledger.jsonl").read_text().splitlines()
    ]
    original_path = json.loads((args.curation / "manifest.json").read_text())["inputs_sha256"]
    existing_path = next(
        p for p in original_path if p.endswith("/evaluation-final/observations.jsonl")
    )
    original = {
        r["observation_id"]: r
        for r in map(json.loads, Path(existing_path).read_text().splitlines())
    }
    by_id = {r.observation_id: r for r in rows}
    for item in corrections:
        prior = original[item["old_observation_id"]]
        paper = by_id[item["paper_observation_id"]]
        if (
            prior[item["field"]] != item["old_value"]
            or prior["sequence"] != paper.sequence
            or paper.paper_id not in prior["lineage"]["study_ids"]
            or paper.chemical_form != item["new_value"]
        ):
            raise ValueError("Correction no longer matches original")
    old = json.loads((args.old_pilot / "curation.json").read_text())
    new = json.loads((args.pilot / "curation.json").read_text())
    for key in ["verified_observations", "strict_pairs", "table_matches", "pair_folds"]:
        if old[key] != new[key]:
            raise ValueError("Single-paper pilot regression")
    for filename in ["observations.jsonl", "pairs.jsonl"]:
        before = [json.loads(s) for s in (args.old_pilot / filename).read_text().splitlines()]
        after = [json.loads(s) for s in (args.pilot / filename).read_text().splitlines()]
        if before != after:
            raise ValueError("Pilot observation/pair semantics changed")
    result = dict(
        verified_paths=len(verified),
        paper_observations=len(rows),
        normalized_bounds=len(normalized),
        chemical_form_counts=dict(Counter(r.chemical_form for r in rows)),
        corrections=len(corrections),
        primary_eligible=sum(r.primary_eligible for r in rows),
        preserved_pilot={k: old[k] for k in ["verified_observations", "strict_pairs"]},
        model_training_executed=False,
        limitation="Cell replay checks transcription, not independent laboratory validation.",
    )
    fresh_output(args.output, [args.registry, args.curation, args.pilot, args.old_pilot])
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "verified_hashes.json").write_text(json.dumps(verified, indent=2) + "\n")
    (args.output / "reviewer_sha256.txt").write_text(file_sha256(Path(__file__)) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
