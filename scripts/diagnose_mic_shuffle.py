"""Label-free composition-preserving perturbation diagnostics for a fitted predictor."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_competition_models import predict
from run_mic_research import checked_manifest, finish_stage, write_json

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.research.mic_lineage import capture_execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    inputs = checked_manifest(args.pairs / "manifest.json")
    inputs.update(checked_manifest(args.handoff / "finetune8/manifest.json"))
    fresh_output(args.output, [args.pairs, args.handoff])
    capture_execution(args.output)
    sequences = sorted(
        {
            json.loads(line)["sequence"]
            for line in (args.pairs / "observations.jsonl").read_text().splitlines()
        }
    )
    rng = np.random.default_rng(42)
    shuffled = ["".join(rng.permutation(list(s))) for s in sequences]
    assert all(Counter(a) == Counter(b) for a, b in zip(sequences, shuffled, strict=True))
    config = json.loads(Path("configs/competition_models.json").read_text())
    state = torch.load(args.handoff / "finetune8/weights.pt", weights_only=False)
    arm = json.loads((args.handoff / "finetune8/refit.json").read_text())["arm"]
    ordered = sequences + shuffled
    # This encoder consumes sequences; placeholder features only provide the row count.
    species, _ = predict(config, state, np.zeros((len(ordered), 320)), ordered, arm)
    records = []
    for i, (original, shuffle) in enumerate(zip(sequences, shuffled, strict=True)):
        records.append(
            dict(
                sequence=original,
                shuffled_sequence=shuffle,
                composition_equal=True,
                changed=original != shuffle,
                mean_absolute_prediction_change=float(
                    np.nanmean(np.abs(species[i] - species[i + len(sequences)]))
                ),
                measured_label=None,
            )
        )
    pd.DataFrame(records).to_csv(args.output / "shuffle_diagnostics.csv", index=False)
    write_json(
        args.output / "interpretation.json",
        dict(
            seed=42,
            pairs=len(records),
            experimental_activity="unknown; no synthetic inactive labels",
            role="predictor sequence-order sensitivity only; no wet-lab conclusion",
        ),
    )
    finish_stage(args.output, inputs, started)


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
