"""Freeze label-free external components, separate scoring labels, and report insufficiency."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.mic_external import ExternalObservation, build_external_split
from robust_apex_qd.research.mic_lineage import capture_execution
from robust_apex_qd.research.paper_mic import PaperObservation


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def checked(path: Path) -> dict[str, str]:
    manifest = json.loads((path / "manifest.json").read_text())
    hashes = dict(manifest["inputs_sha256"])
    hashes.update({str(path / n): h for n, h in manifest["artifacts_sha256"].items()})
    verify_hashes(hashes)
    hashes[str(path / "manifest.json")] = file_sha256(path / "manifest.json")
    return hashes


def finish(output: Path, inputs: dict[str, str], started: float) -> None:
    write_json(
        output / "manifest.json",
        dict(
            inputs_sha256=inputs,
            seconds=time.monotonic() - started,
            artifacts_sha256={
                str(p.relative_to(output)): file_sha256(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        ),
    )


def inventory(config: dict[str, Any], output: Path) -> dict[str, str]:
    registry, curation = Path(config["registry"]), Path(config["curation"])
    inputs = checked(registry)
    inputs.update(checked(curation))
    used = read_jsonl(registry / "used_observations.jsonl")
    originals = [
        PaperObservation.model_validate(r)
        for r in read_jsonl(curation / "paper_observations.jsonl")
    ]
    exposure = json.loads((registry / "exposure_manifest.json").read_text())
    known_papers = set(exposure["verified_used_studies"])
    candidate_papers = set(exposure["candidate_used_studies"])
    certified = config["certified_unused_papers"]
    if (known_papers | candidate_papers) & set(certified):
        raise ValueError("Unused certification contradicts registered exposure")
    for evidence in certified.values():
        if not evidence["reason"] or not evidence["evidence_sha256"]:
            raise ValueError("Unused certification needs explicit reviewed evidence")
        verify_hashes(evidence["evidence_sha256"])
        inputs.update(evidence["evidence_sha256"])
    rows = {}
    for r in used:
        rows[r["observation_id"]] = ExternalObservation(
            observation_id=r["observation_id"],
            sequence=r["sequence"],
            paper_ids=sorted(set(r["verified_study_ids"] + r["publication_candidates"])),
            species="legacy_supervised",
            exposure="used",
            exposure_evidence="registered fit/OOF membership",
            primary_eligible=False,
        )
    links = read_jsonl(curation / "duplicate_links.jsonl")
    legacy = {r["observation_id"]: r for r in read_jsonl(Path(config["legacy_observations"]))}
    inputs[config["legacy_observations"]] = file_sha256(Path(config["legacy_observations"]))
    for link in links:
        identifier = link["old_observation_id"]
        if identifier in rows:
            continue
        r = legacy[identifier]
        rows[identifier] = ExternalObservation(
            observation_id=identifier,
            sequence=r["sequence"],
            paper_ids=r.get("lineage", {}).get("study_ids", []),
            species=r["species"],
            exposure="unknown",
            exposure_evidence="legacy duplicate endpoint; no registered use proof",
            primary_eligible=False,
        )
    for r in originals:
        if r.observation_id in rows:
            raise ValueError("Original identity collides with legacy membership")
        status = (
            "used"
            if r.paper_id in known_papers
            else ("certified_unused" if r.paper_id in certified else "unknown")
        )
        rows[r.observation_id] = ExternalObservation(
            observation_id=r.observation_id,
            sequence=r.sequence,
            paper_ids=[r.paper_id],
            species=r.species,
            exposure=status,
            exposure_evidence=(
                "paper has prior supervised use"
                if status == "used"
                else certified[r.paper_id]["reason"]
                if status == "certified_unused"
                else "absence from registered fits is not proof of non-use"
            ),
            primary_eligible=r.primary_eligible,
        )
    ordered = [rows[k].model_dump() for k in sorted(rows)]
    write_jsonl(output / "observations.jsonl", ordered)
    write_json(output / "sequences.json", sorted({r["sequence"] for r in ordered}))
    write_json(
        output / "duplicate_edges.json",
        sorted({(r["paper_observation_id"], r["old_observation_id"]) for r in links}),
    )
    write_json(
        output / "exposure_summary.json",
        dict(
            rows=len(rows),
            known_supervised=len(used),
            originals=len(originals),
            exposure=dict(Counter(r["exposure"] for r in ordered)),
            unknowns=exposure["unknowns"],
            publication_candidates="candidate paper edges; assay provenance remains unverified",
        ),
    )
    return inputs


def freeze(
    config: dict[str, Any], inventory_path: Path, alignment: Path, output: Path
) -> dict[str, str]:
    inputs = checked(inventory_path)
    inputs.update(checked(alignment))
    curation = Path(config["curation"])
    inputs.update(checked(curation))
    sequences = json.loads((inventory_path / "sequences.json").read_text())
    align_manifest = json.loads((alignment / "manifest.json").read_text())
    if (
        align_manifest["inputs_sha256"].get(str(inventory_path / "sequences.json"))
        != file_sha256(inventory_path / "sequences.json")
        or json.loads((alignment / "sequences.json").read_text()) != sequences
        or any(
            align_manifest.get(k) != v
            for k, v in dict(
                engine="parasail.nw_stats_striped_sat",
                matrix="BLOSUM45",
                gap_open=5,
                gap_extend=1,
                orientation="lexicographic",
                denominator="max(stats.length, len(left), len(right))",
            ).items()
        )
    ):
        raise ValueError("Alignment membership or protocol differs")
    rows = [
        ExternalObservation.model_validate(r)
        for r in read_jsonl(inventory_path / "observations.jsonl")
    ]
    identity = np.load(alignment / "identity.npy", allow_pickle=False)
    assignments = build_external_split(
        rows, sequences, identity, json.loads((inventory_path / "duplicate_edges.json").read_text())
    )
    original = {
        r["observation_id"]: PaperObservation.model_validate(r)
        for r in read_jsonl(curation / "paper_observations.jsonl")
    }
    by_id = {r["observation_id"]: r for r in assignments}
    overlaps = {r["paper_observation_id"] for r in read_jsonl(curation / "duplicate_links.jsonl")}
    used_sequences = {r.sequence for r in rows if r.exposure == "used"}
    used_indices = [i for i, s in enumerate(sequences) if s in used_sequences]
    positions = {s: i for i, s in enumerate(sequences)}
    audit = []
    for identifier, r in original.items():
        a = by_id[identifier]
        maximum = (
            float(identity[positions[r.sequence], used_indices].max()) if used_indices else None
        )
        eligible = r.primary_eligible and identifier not in overlaps
        audit.append(
            dict(
                observation_id=identifier,
                paper_id=r.paper_id,
                species=r.species,
                component_id=a["component_id"],
                partition=a["partition"],
                exposure=a["exposure"],
                primary_eligible=eligible,
                max_known_supervised_identity=maximum,
                reason=(
                    "chemical_form_unit_species_or_length"
                    if not r.primary_eligible
                    else "legacy_overlap"
                    if identifier in overlaps
                    else a["partition"]
                    if a["partition"] in {"development", "diagnostic"}
                    else "eligible"
                ),
            )
        )
    pd.DataFrame(audit).to_csv(output / "overlap_audit.csv", index=False)
    primary_eval = [
        r["observation_id"]
        for r in audit
        if r["primary_eligible"] and r["partition"] == "final_evaluation"
    ]
    new_train = [
        r["observation_id"]
        for r in audit
        if r["primary_eligible"] and r["partition"] == "new_training"
    ]
    for sub in ["training", "inference", "scoring", "diagnostics"]:
        (output / sub).mkdir()
    # Inference receives an allowlist of metadata, never raw cells, bounds, or activity labels.
    write_jsonl(
        output / "inference/targets.jsonl",
        [
            dict(
                observation_id=k,
                sequence=original[k].sequence,
                target=original[k].target,
                species=original[k].species,
                component_id=by_id[k]["component_id"],
            )
            for k in primary_eval
        ],
    )
    write_jsonl(
        output / "scoring/labels.jsonl",
        [
            dict(
                **original[k].trainer_record(),
                external_partition="final_evaluation",
                component_id=by_id[k]["component_id"],
            )
            for k in primary_eval
        ],
    )
    write_jsonl(
        output / "diagnostics/observations.jsonl",
        [
            dict(
                **r.model_dump(),
                external_partition="diagnostic",
                component_id=by_id[k]["component_id"],
            )
            for k, r in original.items()
            if k not in set(primary_eval + new_train)
        ],
    )
    corrections = read_jsonl(curation / "correction_ledger.jsonl")
    excluded = {r["old_observation_id"] for r in corrections}
    training_path = Path(config["legacy_training_rows"])
    inputs[str(training_path)] = file_sha256(training_path)
    legacy_rows = read_jsonl(training_path)
    if not excluded <= {r["observation_id"] for r in legacy_rows}:
        raise ValueError("Correction endpoint missing from legacy training rows")
    training = []
    for r in legacy_rows:
        if r["observation_id"] in excluded:
            continue
        a = by_id.get(r["observation_id"])
        # All retained historical trainer rows must be in the audited graph.
        if a is None or a["partition"] != "development":
            raise ValueError("Historical training membership missing from exposure graph")
        clean = {
            k: v
            for k, v in r.items()
            if k not in {"homology_fold", "homology_group", "inner_folds"}
        }
        training.append(
            dict(**clean, external_partition="development", component_id=a["component_id"])
        )
    training.extend(
        dict(
            **original[k].trainer_record(),
            external_partition="new_training",
            component_id=by_id[k]["component_id"],
        )
        for k in new_train
    )
    write_jsonl(output / "training/rows.jsonl", training)
    write_jsonl(output / "training/correction_exclusions.jsonl", corrections)
    held = [r for r in assignments if r["partition"] in {"final_evaluation", "diagnostic"}]
    write_json(
        output / "training/training_contract.json",
        dict(
            allowed_observation_ids=[r["observation_id"] for r in training],
            forbidden_observation_ids=[r["observation_id"] for r in held],
            forbidden_sequences=sorted({r["sequence"] for r in held}),
            forbidden_paper_ids=sorted({p for r in held for p in r["paper_ids"]}),
            forbidden_component_ids=sorted({r["component_id"] for r in held}),
            rows_sha256=file_sha256(output / "training/rows.jsonl"),
            folds_ready=False,
            limitation="New development folds and features required before refitting",
        ),
    )
    write_json(
        output / "split_manifest.json",
        dict(
            schema_version=1,
            seed=42,
            assignments=assignments,
            primary_evaluation_ids=primary_eval,
            new_training_ids=new_train,
            component_counts=dict(
                Counter({r["component_id"]: r["partition"] for r in assignments}.values())
            ),
            row_counts=dict(Counter(r["partition"] for r in assignments)),
            old_oof_reused=False,
            alignment_sha256=file_sha256(alignment / "manifest.json"),
        ),
    )
    coverage = (
        pd.DataFrame(audit)
        .groupby(["partition", "species", "paper_id", "component_id"], as_index=False)
        .agg(observations=("observation_id", "size"), primary_eligible=("primary_eligible", "sum"))
    )
    coverage.to_csv(output / "coverage.csv", index=False)
    models, model_audit = [], []
    for model in config["comparators"]:
        paths = [Path(p) for p in model["files"]]
        hashes = {str(p): file_sha256(p) for p in paths}
        inputs.update(hashes)
        manifest_path = next(p for p in paths if p.name == "manifest.json")
        saved = json.loads(manifest_path.read_text())
        train_ids = saved.get("train_ids", saved.get("training_ids", []))
        if not train_ids or not set(train_ids) <= set(by_id):
            raise ValueError("Comparator training membership is unresolved")
        for p in paths:
            if (
                p.name in saved.get("artifacts_sha256", {})
                and hashes[str(p)] != saved["artifacts_sha256"][p.name]
            ):
                raise ValueError("Comparator checkpoint differs from saved fit")
        model_sequences = {by_id[k]["sequence"] for k in train_ids}
        model_papers = {p for k in train_ids for p in by_id[k]["paper_ids"]}
        model_indices = [positions[s] for s in sorted(model_sequences)]
        for identifier, r in original.items():
            model_audit.append(
                dict(
                    model=model["name"],
                    observation_id=identifier,
                    training_observations=len(train_ids),
                    training_sequences=len(model_sequences),
                    exact_sequence_overlap=r.sequence in model_sequences,
                    paper_overlap=r.paper_id in model_papers,
                    max_training_identity=float(
                        identity[positions[r.sequence], model_indices].max()
                    ),
                    primary_evaluation=identifier in primary_eval,
                )
            )
        models.append(
            dict(
                **model,
                files_sha256=hashes,
                overlap_audit="overlap_audit.csv",
                membership_scope="registered supervised exposure; unknown coverage retained",
            )
        )
    pd.DataFrame(model_audit).to_csv(output / "checkpoint_overlap_audit.csv", index=False)
    protocol = dict(
        schema_version=1,
        scoring_status="not_started",
        seed=42,
        primary_status="ready" if primary_eval else "insufficient",
        primary_evaluation_ids=primary_eval,
        comparisons=models,
        settings=config["comparison_settings"],
        metrics=config["metrics"],
        split_sha256=file_sha256(output / "split_manifest.json"),
        inference_sha256=file_sha256(output / "inference/targets.jsonl"),
        labels_sha256=file_sha256(output / "scoring/labels.jsonl"),
        limitations=json.loads((inventory_path / "exposure_summary.json").read_text())["unknowns"],
        allocation="component count; seed42; odd to evaluation; labels never rebalance",
        candidate_selection="development only; freeze checkpoints and predictions before scoring",
    )
    write_json(output / "evaluation_protocol.json", protocol)
    write_json(
        output / "insufficiency_report.json",
        dict(
            primary_evaluation_observations=len(primary_eval),
            new_training_observations=len(new_train),
            original_observations=len(original),
            paper_count=len({r.paper_id for r in original.values()}),
            exclusions=dict(Counter(r["reason"] for r in audit)),
            corrections_applied=len(excluded),
            diagnostic_observations=len(original) - len(primary_eval) - len(new_train),
            conclusion="evaluation insufficient; no efficacy conclusion"
            if not primary_eval
            else "protocol frozen; unscored",
            next_action="resolve chemistry and use evidence; additions require a fresh split",
        ),
    )
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["inventory", "freeze"], required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    started = time.monotonic()
    protected = [
        args.config,
        Path(config["registry"]),
        Path(config["curation"]),
        Path(config["legacy_observations"]),
        Path(config["legacy_training_rows"]),
    ]
    protected.extend(p for p in [args.inventory, args.alignment] if p is not None)
    fresh_output(args.output, protected)
    capture_execution(args.output)
    if args.stage == "inventory":
        inputs = inventory(config, args.output)
    else:
        if args.inventory is None or args.alignment is None:
            raise ValueError("Freeze requires inventory and native alignment")
        inputs = freeze(config, args.inventory, args.alignment, args.output)
    inputs[str(args.config)] = file_sha256(args.config)
    write_json(args.output / "config.json", config)
    finish(args.output, inputs, started)
    print(
        json.dumps(
            dict(stage=args.stage, output=str(args.output), seconds=time.monotonic() - started)
        )
    )


if __name__ == "__main__":
    main()
