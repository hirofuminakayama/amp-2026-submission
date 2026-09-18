"""Replay reviewed original-paper cells and reconcile without modifying legacy observations."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.mic_lineage import capture_execution
from robust_apex_qd.research.paper_mic import PaperObservation, compare_existing, read_source_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rules = json.loads(args.rules.read_text())
    registry = json.loads(args.registry.read_text())
    inputs = {str(p): file_sha256(p) for p in [args.rules, args.registry, args.existing]}
    inputs.update(rules["sources_sha256"])
    verify_hashes(inputs)
    full_ready = {r["paper_id"] for r in registry["papers"] if r["rights_status"] == "full_ready"}
    observations = [PaperObservation.model_validate(r) for r in rules["observations"]]
    if len({r.observation_id for r in observations}) != len(observations):
        raise ValueError("Repeated paper cell identity")
    grids = {}
    for row in observations:
        if row.paper_id not in full_ready:
            raise ValueError("Source not cleared for the common collection")
        if inputs.get(row.source_file) != row.source_sha256:
            raise ValueError("Source hash mismatch")
        key = row.source_file, row.table_id
        if key not in grids:
            grids[key] = read_source_table(Path(row.source_file), row.table_id)
        if grids[key][row.row][row.column] != row.cell_text:
            raise ValueError(f"Original cell differs: {row.observation_id}")
    existing = defaultdict(list)
    existing_by_id = {}
    for line in args.existing.open():
        r = json.loads(line)
        existing_by_id[r["observation_id"]] = r
        if r["objective"] == "measured_mic":
            existing[r["sequence"], r["target"]].append(r)
    corrections = rules.get("corrections", [])
    by_id = {r.observation_id: r for r in observations}
    seen_corrections = set()
    for correction in corrections:
        key = correction["old_observation_id"], correction["field"]
        if key in seen_corrections:
            raise ValueError("Repeated correction")
        seen_corrections.add(key)
        old = existing_by_id[correction["old_observation_id"]]
        row = by_id[correction["paper_observation_id"]]
        if (
            correction["field"] != "chemical_form"
            or correction["new_value"] != "modified"
            or row.chemical_form != "modified"
            or old["sequence"] != row.sequence
            or row.paper_id not in old["lineage"]["study_ids"]
            or old["chemical_form"] != correction["old_value"]
            or correction["source_sha256"] != row.source_sha256
            or not correction["reason"]
        ):
            raise ValueError("Unsupported chemical-form correction")
    audit, links, trainer, normalized_rows = [], [], [], []
    for row in observations:
        comparisons = [compare_existing(row, old) for old in existing[row.sequence, row.target]]
        links.extend(comparisons)
        # An unlinked legacy measurement may still be a duplicate. Do not add it silently.
        disposition = "legacy_overlap_quarantined" if comparisons else "new_original_observation"
        normalized = True
        reason = ""
        try:
            relation, lower, upper = row.bounds
            normalized_rows.append(
                dict(
                    observation_id=row.observation_id,
                    relation=relation,
                    lower_um=lower,
                    upper_um=upper,
                )
            )
        except ValueError as exc:
            normalized, reason = False, str(exc)
        if row.primary_eligible and not comparisons:
            trainer.append(row.trainer_record())
        audit.append(
            dict(
                observation_id=row.observation_id,
                paper_id=row.paper_id,
                disposition=disposition,
                legacy_matches=len(comparisons),
                primary_eligible=row.primary_eligible,
                bounds_available=normalized,
                reason=reason,
                chemical_form=row.chemical_form,
                species=row.species,
                medium=row.medium,
                cfu=row.cfu,
                source_file=row.source_file,
                table_id=row.table_id,
                row=row.row,
                column=row.column,
            )
        )
    fresh_output(args.output, [args.rules, args.registry, args.existing])
    capture_execution(args.output)
    for name, records in [
        ("paper_observations.jsonl", [r.model_dump() for r in observations]),
        ("duplicate_links.jsonl", links),
        ("correction_ledger.jsonl", corrections),
        ("normalized_bounds.jsonl", normalized_rows),
        ("trainer_additions.jsonl", trainer),
    ]:
        (args.output / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
    pd.DataFrame(audit).to_csv(args.output / "curation_audit.csv", index=False)
    coverage = (
        pd.DataFrame(audit)
        .groupby(["species", "chemical_form", "medium", "cfu"], dropna=False, as_index=False)
        .agg(observations=("observation_id", "size"))
    )
    coverage.to_csv(args.output / "coverage.csv", index=False)
    summary = dict(
        observations=len(observations),
        papers=len({r.paper_id for r in observations}),
        primary_eligible=sum(r.primary_eligible for r in observations),
        trainer_additions=len(trainer),
        corrections=len(corrections),
        dispositions=dict(Counter(a["disposition"] for a in audit)),
        match_statuses=dict(Counter(a["status"] for a in links)),
        limitation="No automated corrections; value agreement alone is not assay identity",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = dict(
        inputs_sha256=inputs,
        artifacts_sha256={
            str(p.relative_to(args.output)): file_sha256(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
