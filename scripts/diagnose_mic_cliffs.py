"""Separate reviewed and DB-only near-neighbor MIC diagnostics without adding training labels."""

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from run_mic_research import checked_manifest, finish_stage

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.research.activity_pairs import cliff_similarity
from robust_apex_qd.research.mic_lineage import capture_execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    inputs = checked_manifest(args.prepared / "manifest.json")
    inputs.update(checked_manifest(args.pairs / "manifest.json"))
    fresh_output(args.output, [args.prepared, args.pairs, args.run / "models"])
    capture_execution(args.output)
    rows = pd.read_json(args.prepared / "rows.jsonl", lines=True)
    rows = rows[rows.objective.eq("measured_mic") & rows.exact_regression]
    by_id = {r.observation_id: r for r in rows.itertuples()}
    verified = {
        tuple(sorted((p["left"]["observation_id"], p["right"]["observation_id"])))
        for p in map(json.loads, (args.pairs / "pairs.jsonl").read_text().splitlines())
    }
    groups = defaultdict(list)
    for line in (args.prepared / "observations.jsonl").open():
        r = json.loads(line)
        if r["observation_id"] in by_id and r["lineage"]["study_ids"]:
            groups[(tuple(r["lineage"]["study_ids"]), r["target"], r["chemical_form"])].append(
                r["observation_id"]
            )
    records = []
    for group in groups.values():
        for a, b in itertools.combinations(sorted(group), 2):
            left, right = by_id[a], by_id[b]
            if left.sequence == right.sequence:
                continue
            similarity = cliff_similarity(left.sequence, right.sequence)
            if similarity < 0.9:
                continue
            records.append(
                dict(
                    left_id=a,
                    right_id=b,
                    similarity=similarity,
                    delta=float(np.log2(right.mic_um / left.mic_um)),
                    evidence="reviewed"
                    if (a, b) in verified
                    else "DB-only diagnostic; assay comparability unknown",
                    same_fold=left.homology_fold == right.homology_fold,
                )
            )
    pairs = pd.DataFrame(records)
    pairs.to_csv(args.output / "pair_manifest.csv", index=False)
    metrics = []
    for stage in ["models", "baselines", "controls-v2", "delta"]:
        inputs.update(checked_manifest(args.run / stage / "manifest.json"))
        for path in sorted((args.run / stage).glob("*-oof.csv.gz")):
            frame = pd.read_csv(path).set_index("observation_id")
            for evidence, group in pairs.groupby("evidence"):
                for steps in [0, 1, 2]:
                    chosen = group[group.same_fold & (group.delta.abs() >= steps)]
                    estimate = (
                        frame.reindex(chosen.right_id).prediction.to_numpy()
                        - frame.reindex(chosen.left_id).prediction.to_numpy()
                    )
                    truth = chosen.delta.to_numpy()
                    usable = np.isfinite(estimate)
                    nonzero = usable & (truth != 0)
                    metrics.append(
                        dict(
                            model=path.name,
                            evidence=evidence,
                            minimum_dilutions=steps,
                            candidate_pairs=len(chosen),
                            evaluated_pairs=int(usable.sum()),
                            delta_mae=float(np.abs(estimate[usable] - truth[usable]).mean())
                            if usable.any()
                            else None,
                            direction_accuracy=float(
                                (np.sign(estimate[nonzero]) == np.sign(truth[nonzero])).mean()
                            )
                            if nonzero.any()
                            else None,
                            direction_rows=int(nonzero.sum()),
                        )
                    )
    pd.DataFrame(metrics).to_csv(args.output / "cliff_metrics.csv", index=False)
    finish_stage(
        args.output,
        inputs,
        started,
        scope=(
            "DB-study/target/chemistry grouped diagnostic; unknown-study pairs not enumerated; "
            "no added training pairs"
        ),
    )


if __name__ == "__main__":
    main()
