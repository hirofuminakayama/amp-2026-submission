"""Build an observation-scoped chemistry evidence ledger from reviewed source rules."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from run_competition_bioaccuracy import archive_sources, finish_stage, read_observations, write_json

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.bio_followup import ChemistryEvidence, apply_chemistry_evidence
from robust_apex_qd.research.mic_lineage import reference_articles


def run(config: dict[str, Any], output: Path) -> dict[str, str]:
    paths = [Path(config[key]) for key in ["observations", "lineage", "research_review"]]
    paths += [Path(rule["paper"]) for rule in config["rules"]]
    inputs = {str(path): file_sha256(path) for path in paths}
    rows = read_observations(Path(config["observations"]))
    lineage = {
        row["observation_id"]: row["lineage"]
        for row in map(json.loads, Path(config["lineage"]).read_text().splitlines())
    }
    evidence, audit = [], []
    for row in rows:
        if row.endpoint not in {"measured_mic", "measured_hc50", "hemolysis_percent"}:
            continue
        rules = [rule for rule in config["rules"] if row.source_id in rule["source_ids"]]
        if not rules:
            continue
        metadata_path = Path(config["metadata"]) / f"{row.source_id}.json"
        if not metadata_path.exists():
            audit.append(dict(observation_id=row.observation_id, status="metadata_missing"))
            continue
        digest = file_sha256(metadata_path)
        inputs[str(metadata_path)] = digest
        metadata = json.loads(metadata_path.read_text())
        if metadata["sequence"] != row.sequence:
            raise ValueError("Metadata sequence mismatch")
        if row.endpoint == "measured_mic":
            linked = lineage.get(row.observation_id, {})
            if linked.get("metadata_sha256") != digest or not linked.get("assay_ids"):
                audit.append(dict(observation_id=row.observation_id, status="lineage_unverified"))
                continue
            papers = linked["study_ids"]
        else:
            papers = reference_articles(row.raw.get("reference"), metadata["articles"])
        for rule in rules:
            if papers != [rule["paper_id"]]:
                audit.append(
                    dict(observation_id=row.observation_id, status="other_or_ambiguous_paper")
                )
                continue
            support = {**row.chemistry_support, "stereochemistry": "reported_L"}
            support["cterminal"] = "reported_modified" if rule["cterminal"] else "reported_free"
            sources = {path: inputs[path] for path in [rule["paper"], config["research_review"]]}
            sources[str(metadata_path)] = digest
            if row.endpoint == "measured_mic":
                sources[config["lineage"]] = inputs[config["lineage"]]
            evidence.append(
                ChemistryEvidence(
                    observation_id=row.observation_id,
                    observation_sha256=hashlib.sha256(row.model_dump_json().encode()).hexdigest(),
                    sources_sha256=sources,
                    paper_id=rule["paper_id"],
                    locator=rule["locator"],
                    rationale=rule["rationale"],
                    nterminal=row.nterminal,
                    cterminal=rule["cterminal"],
                    bonds=row.bonds,
                    stereochemistry=rule["stereochemistry"],
                    chemistry_support=support,
                )
            )
            audit.append(
                dict(
                    observation_id=row.observation_id,
                    status="annotated",
                    paper_id=rule["paper_id"],
                    endpoint=row.endpoint,
                )
            )
    apply_chemistry_evidence(rows, evidence)
    write_json(output / "chemistry_evidence.json", [r.model_dump(mode="json") for r in evidence])
    old_rows = {row.observation_id: row for row in rows}
    exclusions = [
        dict(
            old_observation_id=record.observation_id,
            field="chemical_form",
            old_value="reported_linear_free",
            new_value="modified",
            paper_id=record.paper_id,
            reason=record.rationale,
            evidence_observation_sha256=record.observation_sha256,
            sources_sha256=record.sources_sha256,
        )
        for record in evidence
        if record.cterminal is not None
        and old_rows[record.observation_id].endpoint == "measured_mic"
    ]
    (output / "chemistry_training_exclusions.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in exclusions)
    )
    pd.DataFrame(audit).to_csv(output / "linkage_audit.csv", index=False)
    write_json(
        output / "summary.json",
        dict(annotated=len(evidence), audited=len(audit), new_training_exclusions=len(exclusions)),
    )
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bio_chemistry_review.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh_output(args.output, [Path("data"), Path("checkpoint")])
    started = time.monotonic()
    inputs = archive_sources(
        args.output,
        [
            args.config,
            Path(__file__),
            Path("src/robust_apex_qd/research/bio_followup.py"),
            Path("src/robust_apex_qd/research/mic_lineage.py"),
        ],
    )
    inputs.update(run(json.loads(args.config.read_text()), args.output))
    finish_stage(args.output, inputs, started)
    print(json.dumps(dict(output=str(args.output), seconds=time.monotonic() - started)))


if __name__ == "__main__":
    main()
