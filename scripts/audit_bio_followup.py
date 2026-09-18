"""Audit existing corrected biological evidence and report whether refitting is supported."""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_competition_bioaccuracy import (
    archive_sources,
    checked_manifest,
    finish_stage,
    read_observations,
    write_json,
)

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.bio_followup import (
    ChemistryEvidence,
    apply_chemistry_evidence,
    chemistry_status,
    corrected_observations,
    joint_development_split,
    primary_joint_coverage,
)
from robust_apex_qd.research.bioaccuracy import chemistry_key
from robust_apex_qd.research.mic_lineage import reference_articles


def run(output: Path, evidence_path: Path | None = None) -> dict[str, str]:
    bio = Path("work/competition_bioaccuracy")
    base = bio / "20260913-b"
    extended = bio / "20260913-c"
    mic = Path("work/mic_prediction/20260914-phase78-a/evaluation-v2")
    release = Path("work/competition_exploration/20260914-adoption-a/release")
    inputs: dict[str, str] = {}

    def read(path: Path) -> Any:
        inputs[str(path)] = file_sha256(path)
        return json.loads(path.read_text())

    for root, stages in [(base, ["prepare", "split", "joint"]), (extended, ["prepare"])]:
        for stage in stages:
            inputs.update(checked_manifest(root / stage / "manifest.json"))
    correction_path = mic / "training/correction_exclusions.jsonl"
    inputs[str(correction_path)] = file_sha256(correction_path)
    corrections = [json.loads(line) for line in correction_path.read_text().splitlines()]
    for correction in corrections:
        if correction["field"] != "chemical_form" or correction["new_value"] != "modified":
            raise ValueError("Unexpected correction type; review before applying")
    identifiers = [r["old_observation_id"] for r in corrections]
    evidence = []
    if evidence_path is not None:
        evidence = [ChemistryEvidence.model_validate(item) for item in read(evidence_path)]
        for record in evidence:
            for path, expected in record.sources_sha256.items():
                if path in inputs and inputs[path] != expected:
                    raise ValueError("Conflicting chemistry evidence source hashes")
                inputs[path] = expected
    correction_by_id = {r["old_observation_id"]: r for r in corrections}
    audit, coverage = [], []
    rows = []
    for root in [base, extended]:
        original = read_observations(root / "prepare/endpoint_observations.jsonl")
        rows, matches = corrected_observations(original, identifiers)
        available_ids = {row.observation_id for row in rows}
        rows = apply_chemistry_evidence(
            rows, [record for record in evidence if record.observation_id in available_ids]
        )
        by_id = {r.observation_id: r for r in original}
        for match in matches:
            row = by_id.get(match["observation_id"])
            audit.append(
                dict(
                    version=root.name,
                    **match,
                    sequence=row.sequence if row else None,
                    endpoint=row.endpoint if row else None,
                    old_molecule_id=chemistry_key(row) if row else None,
                    action="exclude_from_corrected_view" if row else "unmatched",
                    paper_observation_id=correction_by_id[match["observation_id"]][
                        "paper_observation_id"
                    ],
                )
            )
        counts = Counter((r.endpoint, r.rbc_species, chemistry_status(r)) for r in rows)
        for (endpoint, rbc, status), count in counts.items():
            coverage.append(
                dict(
                    version=root.name, endpoint=endpoint, rbc_species=rbc, chemistry=status, n=count
                )
            )
        write_json(
            output / f"{root.name}-counts.json",
            dict(original=len(original), retained=len(rows), excluded=len(original) - len(rows)),
        )
    if not {record.observation_id for record in evidence} <= {row.observation_id for row in rows}:
        raise ValueError("Chemistry evidence references absent corrected observations")
    write_json(
        output / "applied_chemistry_evidence.json", [record.model_dump() for record in evidence]
    )
    pd.DataFrame(audit).to_csv(output / "correction_impact.csv", index=False)
    training_path = base / "split/rows.jsonl"
    training = pd.read_json(training_path, lines=True)
    corrected_training = training[training.observation_id.isin(identifiers)].copy()
    corrected_training.to_csv(output / "training_correction_impact.csv", index=False)
    corrected_sequences = {row["sequence"] for row in audit if row["matches"]}
    same_sequence_hc50 = [
        row
        for row in rows
        if row.sequence in corrected_sequences and row.endpoint == "measured_hc50"
    ]
    write_json(
        output / "endpoint_correction_summary.json",
        dict(
            corrected_training_rows=len(corrected_training),
            corrected_training_exact_rows=int(corrected_training.exact_regression.sum()),
            corrected_sequences=len(corrected_sequences),
            hc50_same_sequence_rows=len(same_sequence_hc50),
            hc50_rows_excluded_by_sequence=0,
            rule=(
                "Only listed observation IDs are excluded; "
                "sequence matches do not transfer chemistry"
            ),
        ),
    )
    pd.DataFrame(coverage).to_csv(output / "chemistry_coverage.csv", index=False)
    with (output / "corrected_endpoint_observations.jsonl").open("w") as stream:
        for row in rows:
            stream.write(row.model_dump_json() + "\n")

    # Molecular labels involving corrected observations are invalidated as a whole.
    # Their old aggregate cannot be repaired by deleting a sequence or copying HC50 chemistry.
    affected = {r["old_molecule_id"] for r in audit if r["matches"]}
    labels = pd.read_csv(base / "joint/molecular_joint_labels.csv")
    labels[labels.molecule_id.isin(affected)].to_csv(
        output / "affected_joint_labels.csv", index=False
    )

    old_sequences = read(base / "split/sequences.json")
    old_identity = np.load(base / "split/identity.npy")
    sequences = sorted({r.sequence for r in rows})
    if not set(sequences) <= set(old_sequences):
        raise ValueError("New sequence requires extending the verified identity matrix")
    old_index = {s: i for i, s in enumerate(old_sequences)}
    indices = [old_index[s] for s in sequences]
    identity = old_identity[np.ix_(indices, indices)]
    mic_split = read(mic / "split_manifest.json")
    papers: dict[str, set[str]] = defaultdict(set)
    conservative_links: dict[str, set[str]] = defaultdict(set)
    for assignment in mic_split["assignments"]:
        sequence = assignment["sequence"]
        papers[sequence].update(assignment["paper_ids"])
        # Preserve existing transitive MIC links, even if an excluded row once bridged them.
        conservative_links[sequence].add(f"prior_mic_component:{assignment['component_id']}")
    metadata_audit = []
    for row in rows:
        if row.source != "dbaasp":
            continue
        path = bio / "20260913-a/hc50-metadata" / f"{row.source_id}.json"
        record = read(path)
        if record["sequence"] != row.sequence:
            raise ValueError("DBAASP metadata sequence differs from the registered endpoint")
        references = reference_articles(row.raw.get("reference"), record.get("articles", []))
        papers[row.sequence].update(references)
        metadata_audit.append(
            dict(
                observation_id=row.observation_id,
                sequence=row.sequence,
                endpoint=row.endpoint,
                metadata=str(path),
                reference=row.raw.get("reference"),
                paper_ids=json.dumps(references),
                explicit_stereo_fields=json.dumps([k for k in record if "stereo" in k.lower()]),
                unusual_amino_acids=len(record.get("unusualAminoAcids", [])),
                chemistry=chemistry_status(row),
                note="Empty unusual-AA annotations and canonical sequence do not establish all-L",
            )
        )
    pd.DataFrame(metadata_audit).to_csv(output / "metadata_audit.csv", index=False)
    cache_records = [
        read(path) for path in sorted((bio / "20260913-a/hc50-metadata").glob("[0-9]*.json"))
    ]
    write_json(
        output / "metadata_cache_summary.json",
        dict(
            records=len(cache_records),
            explicit_stereo_keys=sum(
                any("stereo" in key.lower() for key in record) for record in cache_records
            ),
            unusual_amino_acid_annotations=sum(
                bool(record.get("unusualAminoAcids")) for record in cache_records
            ),
            smiles_entries=sum(len(record.get("smiles", [])) for record in cache_records),
            manually_edited_smiles=sum(
                entry.get("manuallyEdited") is True
                for record in cache_records
                for entry in record.get("smiles", [])
            ),
            interpretation="Automatically generated SMILES are not assay-level stereo provenance",
        ),
    )
    links = {s: sorted(papers[s] | conservative_links[s]) for s in sequences}
    split = joint_development_split(sequences, identity, links)
    split.update(
        paper_unknown_sequences=sum(not papers[s] for s in sequences),
        conservative_prior_mic_components=True,
        source_identity_sha256=file_sha256(base / "split/identity.npy"),
        sequence_order_sha256=file_sha256(base / "split/sequences.json"),
        confirmed_chemistry_observations=sum(
            chemistry_status(row) == "known_linear_free_L" for row in rows
        ),
    )
    write_json(output / "split_manifest.json", split)
    write_json(output / "sequence_papers.json", {s: sorted(papers[s]) for s in sequences})
    old_split = read(base / "split/split_manifest.json")
    changed = sum(old_split["outer"][s] != split["outer"][s] for s in sequences)
    write_json(output / "split_change.json", dict(sequences=len(sequences), changed_outer=changed))

    reuse = []
    for stage_root in [
        bio / "20260913-a/shared-mic-baselines",
        base / "shared-mic-models",
        base / "shared-mic-models-s43",
        base / "shared-mic-models-s44",
        base / "hc50-measured",
    ]:
        manifest_path = stage_root / "manifest.json"
        manifest = read(manifest_path)
        old_hashes = [
            v for k, v in manifest["inputs_sha256"].items() if k.endswith("split_manifest.json")
        ]
        current_hash = file_sha256(output / "split_manifest.json")
        reuse.append(
            dict(
                stage=stage_root.name,
                manifest=str(manifest_path),
                reusable=False,
                split_hash_match=current_hash in old_hashes,
                other_contract_checks="not sufficient after split mismatch; no fit reused",
                reason="corrected input and shared paper/homology split require endpoint refit",
            )
        )
    pd.DataFrame(reuse).to_csv(output / "reuse_audit.csv", index=False)

    timings = []
    for parent in [bio, Path("work/mic_prediction"), Path("work/competition_exploration")]:
        for path in sorted(parent.glob("*/*/manifest.json")):
            if path.is_relative_to(output):
                continue
            manifest = read(path)
            timings.append(
                dict(
                    manifest=str(path),
                    seconds=manifest.get("seconds"),
                    accounting="stage wall time; not GPU utilization; possible overlap",
                )
            )
    pd.DataFrame(timings).to_csv(output / "historical_timings.csv", index=False)
    review = read(release / "completion_review.json")
    write_json(
        output / "budget.json",
        dict(
            shared_gpu_hour_ceiling=240,
            measured_release_pipeline_seconds=review["pipeline_seconds"],
            measured_release_process_seconds=review["process_seconds"],
            historical_timing_records=len(timings),
            historical_gpu_utilization_hours=None,
            remaining_gpu_hours=None,
            accounting="No GPU utilization ledger; overlapping stage wall times are not summed",
            additional_training_gpu_hours=0,
            training_smoke="not_started_pending_primary_cohort_gate",
            audit_device="CPU",
            maximum_concurrent_gpu_jobs=1,
        ),
    )
    primary = [r for r in rows if chemistry_status(r) == "known_linear_free_L"]
    human = [r for r in rows if r.endpoint == "measured_hc50" and r.rbc_species == "human"]
    eligible_human = [
        r for r in primary if r.endpoint == "measured_hc50" and r.rbc_species == "human"
    ]
    coverage_report, primary_pairs = primary_joint_coverage(rows, split)
    write_json(output / "primary_coverage.json", coverage_report)
    pd.DataFrame(
        primary_pairs,
        columns=pd.Index(
            [
                "molecule_id",
                "sequence",
                "species",
                "component",
                "outer",
                "hit_lower",
                "hit_upper",
            ]
        ),
    ).to_csv(output / "primary_joint_pairs.csv", index=False)
    if coverage_report["nested_supported"]:
        raise ValueError("Supported cohort: register budget and run the full nested procedure")
    insufficiency = dict(
        status="unsupported",
        reason=coverage_report["reason"],
        corrected_identifiers=len(identifiers),
        matched_by_version=dict(Counter(r["version"] for r in audit if r["matches"])),
        human_hc50_observations=len(human),
        eligible_human_hc50_observations=len(eligible_human),
        eligible_endpoint_observations=len(primary),
        primary_joint_pairs=coverage_report["primary_joint_pairs"],
        paired_components=coverage_report["paired_components"],
        nested_training="not_run",
        pool_comparison="not_run_no_selected_procedure",
        recommendation="retain_ddim120k_Lref_rankmean",
        independent_external_evaluation=False,
        restart="observation-level verified chemistry with provenance, followed by a fresh audit",
    )
    write_json(output / "insufficiency_report.json", insufficiency)
    (output / "biological_handoff.md").write_text(
        "# Biological follow-up handoff\n\n"
        "Recommendation: retain DDIM120k/Lref/rankmean. No new procedure was selected.\n\n"
        f"The correction ledger has {len(identifiers)} IDs; "
        "mapping counts are in correction_impact.csv. "
        f"The corrected input has {len(human)} measured human HC50 observations, "
        f"including {len(eligible_human)} with reviewed "
        "confirmed free termini, linear structure and all-L stereochemistry. Canonical uppercase "
        "sequence and empty unusual-amino-acid annotations do not establish that chemistry. "
        f"There are {len(primary_pairs)} primary molecule/species pairs in "
        f"{coverage_report['paired_components']} components; registered nested fold coverage "
        "is unsupported. See primary_coverage.json.\n\n"
        "Paper/homology components were audited across endpoints; both MIC and HC50 old fits "
        "are ineligible for this new split. No old OOF was relabelled as new validation. "
        "Nested fitting, a new Top selection, Random25 scenarios, and new artifact validators "
        "were not run because their prerequisite failed. This is not a finding that a new "
        "model performed worse, and does not establish experimental safety.\n\n"
        "Additional training GPU cost: zero. Historical wall-time records are preserved without "
        "double-counting them as GPU usage; the remaining shared GPU budget is unestablished. "
        "Remote clean-clone verification of the current dirty source remains unperformed.\n\n"
        "Usage/disclosure: work/mic_prediction/20260914-phase910-a/handoff/; current adoption: "
        "work/competition_exploration/20260914-adoption-a/adoption_decision.json. "
        "Existing outputs and adopted configuration are unchanged. Restart with verified "
        "observation-level chemical provenance in a fresh input version and rerun this audit.\n"
    )
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chemistry-evidence", type=Path)
    args = parser.parse_args()
    fresh_output(args.output, [Path("data"), Path("checkpoint")])
    started = time.monotonic()
    sources = [
        Path(__file__),
        Path("scripts/run_competition_bioaccuracy.py"),
        Path("configs/competition_bioaccuracy.json"),
        Path("uv.lock"),
        *Path("src/robust_apex_qd/research").glob("bio*.py"),
        Path("src/robust_apex_qd/research/mic_data.py"),
        Path("src/robust_apex_qd/research/mic_lineage.py"),
    ]
    inputs = archive_sources(args.output, sources)
    inputs.update(run(args.output, args.chemistry_evidence))
    finish_stage(args.output, inputs, started)
    print(json.dumps(dict(output=str(args.output), seconds=time.monotonic() - started)))


if __name__ == "__main__":
    main()
