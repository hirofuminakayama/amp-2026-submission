"""Verify frozen external boundaries, checkpoint membership and label-isolated exports."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_mic_external import checked, finish, read_jsonl, write_json

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.research.mic_external import (
    ExternalObservation,
    build_external_split,
    load_training_rows,
    validate_training_membership,
)
from robust_apex_qd.research.mic_lineage import capture_execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    inputs = {}
    for path in [args.inventory, args.alignment, args.evaluation]:
        inputs.update(checked(path))
    manifest = json.loads((args.evaluation / "split_manifest.json").read_text())
    protocol = json.loads((args.evaluation / "evaluation_protocol.json").read_text())
    rows = [
        ExternalObservation.model_validate(r)
        for r in read_jsonl(args.inventory / "observations.jsonl")
    ]
    sequences = json.loads((args.inventory / "sequences.json").read_text())
    matrix = np.load(args.alignment / "identity.npy", allow_pickle=False)
    links = json.loads((args.inventory / "duplicate_edges.json").read_text())
    rebuilt = build_external_split(rows, sequences, matrix, links)
    if rebuilt != manifest["assignments"]:
        raise ValueError("Split replay differs")
    by_id = {r["observation_id"]: r for r in rebuilt}
    component_sequences = defaultdict(set)
    component_papers = defaultdict(set)
    component_partitions = defaultdict(set)
    for r in rebuilt:
        component_sequences[r["sequence"]].add(r["component_id"])
        for p in r["paper_ids"]:
            component_papers[p].add(r["component_id"])
        component_partitions[r["component_id"]].add(r["partition"])
        if r["exposure"] == "used" and r["partition"] != "development":
            raise ValueError("Prior use left development")
        if (
            r["partition"] in {"new_training", "final_evaluation"}
            and r["exposure"] != "certified_unused"
        ):
            raise ValueError("Unknown use entered a new independent component")
    if any(
        len(v) != 1
        for d in [component_sequences, component_papers, component_partitions]
        for v in d.values()
    ):
        raise ValueError("Sequence, paper or component crossed partitions")
    for a, b in links:
        if by_id[a]["component_id"] != by_id[b]["component_id"]:
            raise ValueError("Duplicate crossed components")
    components = np.array([next(iter(component_sequences[s])) for s in sequences])
    prohibited = 0
    for i in range(len(sequences)):
        prohibited += int(
            ((matrix[i, :i] > np.float32(0.6)) & (components[:i] != components[i])).sum()
        )
    if prohibited:
        raise ValueError("Prohibited homology crossed components")
    training = read_jsonl(args.evaluation / "training/rows.jsonl")
    contract = json.loads((args.evaluation / "training/training_contract.json").read_text())
    validate_training_membership(training, contract)
    if set(contract["allowed_observation_ids"]) != {r["observation_id"] for r in training}:
        raise ValueError("Training allowlist differs")
    corrections = read_jsonl(args.evaluation / "training/correction_exclusions.jsonl")
    if {r["old_observation_id"] for r in corrections} & set(contract["allowed_observation_ids"]):
        raise ValueError("Corrected chemical form remains in training")
    try:
        load_training_rows(args.evaluation / "training")
    except ValueError as exc:
        if "new development folds" not in str(exc):
            raise
    else:
        raise ValueError("Unprepared new folds entered training")
    targets = read_jsonl(args.evaluation / "inference/targets.jsonl")
    labels = read_jsonl(args.evaluation / "scoring/labels.jsonl")
    expected = set(manifest["primary_evaluation_ids"])
    if expected != {r["observation_id"] for r in targets} or expected != {
        r["observation_id"] for r in labels
    }:
        raise ValueError("Inference/scoring identity differs")
    for r in targets:
        if set(r) != {"observation_id", "sequence", "target", "species", "component_id"}:
            raise ValueError("Inference includes unregistered fields or labels")
    if protocol["scoring_status"] != "not_started":
        raise ValueError("Final scoring already started")
    audit = pd.read_csv(args.evaluation / "checkpoint_overlap_audit.csv")
    position = {s: i for i, s in enumerate(sequences)}
    config = json.loads((args.evaluation / "config.json").read_text())
    originals = {
        r["observation_id"]: r
        for r in read_jsonl(Path(config["curation"]) / "paper_observations.jsonl")
    }
    for model in config["comparators"]:
        path = next(Path(p) for p in model["files"] if Path(p).name == "manifest.json")
        saved = json.loads(path.read_text())
        used = saved.get("train_ids", saved.get("training_ids", []))
        indices = [position[by_id[k]["sequence"]] for k in used]
        for r in audit[audit.model == model["name"]].itertuples(index=False):
            maximum = float(
                matrix[position[originals[r.observation_id]["sequence"]], indices].max()
            )
            if not np.isclose(maximum, r.max_training_identity, rtol=0, atol=1e-8):
                raise ValueError("Checkpoint overlap replay differs")
    fresh_output(args.output, [args.inventory, args.alignment, args.evaluation])
    capture_execution(args.output)
    write_json(
        args.output / "verification.json",
        dict(
            verified_paths=len(inputs),
            observations=len(rows),
            sequences=len(sequences),
            component_counts=manifest["component_counts"],
            row_counts=manifest["row_counts"],
            prohibited_cross_component_edges=prohibited,
            checkpoint_audit_rows=len(audit),
            primary_evaluation_observations=len(expected),
            training_rows=len(training),
            corrections_excluded=len(corrections),
            scoring_status=protocol["scoring_status"],
            split_replay=True,
            label_isolation=True,
            training_guard=True,
            primary_status=protocol["primary_status"],
        ),
    )
    write_json(args.output / "verified_hashes.json", inputs)
    finish(args.output, inputs, started)
    print((args.output / "verification.json").read_text())


if __name__ == "__main__":
    main()
