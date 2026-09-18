"""Fixed-protocol masked joint-trunk versus independent-trunk endpoint ablation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_bio_selection import score_labels
from run_competition_bioaccuracy import (
    archive_sources,
    checked_manifest,
    finish_stage,
    read_observations,
    write_json,
)
from threadpoolctl import threadpool_limits
from train_bio_measured_hc50 import bound_metrics

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.research.bioaccuracy import chemistry_key, concentration_bounds
from robust_apex_qd.research.competition_models import SPECIES
from robust_apex_qd.research.joint_endpoint_models import fit_joint_head, predict_joint_head


def run(root: Path, output: Path) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for stage in ["prepare", "split", "features", "joint", "hc50-embeddings"]:
        inputs.update(checked_manifest(root / stage / "manifest.json"))
    observations = read_observations(root / "prepare/endpoint_observations.jsonl")
    selected = [
        r
        for r in observations
        if r.endpoint == "measured_mic"
        or (r.endpoint == "measured_hc50" and r.rbc_species == "human")
    ]
    selected = [r for r in selected if concentration_bounds(r) is not None]
    sequences = json.loads((root / "features/sequences.json").read_text())
    index = {s: i for i, s in enumerate(sequences)}
    x = np.load(root / "hc50-embeddings/features.npy")
    splits = json.loads((root / "split/split_manifest.json").read_text())
    groups = np.array([splits["groups"][s] for s in sequences])
    outer = np.array([splits["outer"][s] for s in sequences])
    seq = np.array([index[r.sequence] for r in selected])
    heads = np.array(
        [7 if r.endpoint == "measured_hc50" else SPECIES.index(r.species) for r in selected]
    )
    bounds = [concentration_bounds(r) for r in selected]
    with np.errstate(divide="ignore"):
        low = np.array([np.log2(b[0]) for b in bounds if b is not None])
        high = np.array([np.log2(b[1]) for b in bounds if b is not None])
    labels = pd.read_csv(root / "joint/molecular_joint_labels.csv")
    labels = labels[
        (labels.evidence == "measured_mic+measured_hc50")
        & (labels.rbc_species == "human")
        & (labels.ratio == 8)
    ].copy()
    mapping = {chemistry_key(r): r.sequence for r in observations}
    labels["sequence"] = labels.molecule_id.map(mapping)
    write_json(
        output / "protocol.json",
        dict(
            width=32,
            epochs=80,
            learning_rate=0.003,
            seeds=[42, 43, 44],
            outer_folds=5,
            configuration="fixed before outer evaluation; no outer-based retuning",
            loss="masked interval squared; train-only scaling; equal MIC/HC50 endpoint mass",
            supervision="measured species MIC and human RBC HC50, no fabricated absent targets",
            control="independent width32 trunks per target; more parameters than one shared trunk",
            selection_score="mean signed log2 margin min(4-MIC, HC50-MIC-3); not probability",
        ),
    )
    comparisons, selection_metrics = [], []
    for shared in [False, True]:
        arm = "shared" if shared else "independent"
        for seed in [42, 43, 44]:
            oof = np.full((len(x), 8), np.nan)
            for fold in range(5):
                train, valid = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
                model = fit_joint_head(
                    x, seq, heads, low, high, groups, train, valid, shared=shared, seed=seed
                )
                destination = output / f"{arm}-s{seed}-f{fold}.pt"
                torch.save(model, destination)
                loaded = torch.load(destination, weights_only=False)
                oof[valid] = predict_joint_head(model, x[valid])
                np.testing.assert_equal(oof[valid], predict_joint_head(loaded, x[valid]))
            predicted = oof[seq, heads]
            for endpoint, mask in [("measured_mic", heads < 7), ("human_hc50", heads == 7)]:
                comparisons.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        endpoint=endpoint,
                        **bound_metrics(low[mask], high[mask], predicted[mask]),
                    )
                )
            for fold in range(5):
                cohort = labels[labels.sequence.map(splits["outer"]) == fold]
                ss = sorted(cohort.sequence.unique())
                values = oof[[index[s] for s in ss]]
                margins = np.minimum(4 - values[:, :7], values[:, [7]] - values[:, :7] - 3).mean(
                    axis=1
                )
                scores = dict(zip(ss, margins, strict=True))
                selection_metrics.extend(
                    dict(arm=arm, seed=seed, fold=fold, **r) for r in score_labels(cohort, scores)
                )
            np.save(output / f"{arm}-s{seed}-oof.npy", oof)
            print(json.dumps(dict(arm=arm, seed=seed, completed_folds=5)), flush=True)
    write_json(output / "endpoint_comparison.json", comparisons)
    pd.DataFrame(selection_metrics).to_csv(output / "joint_head_selection.csv", index=False)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "joint-heads"
    fresh_output(output, [args.root / "prepare"])
    start = time.monotonic()
    inputs = archive_sources(
        output,
        [
            Path(__file__),
            Path("scripts/evaluate_bio_selection.py"),
            Path("scripts/train_bio_measured_hc50.py"),
            Path("scripts/train_competition_hc50.py"),
            Path("scripts/run_competition_bioaccuracy.py"),
            Path("uv.lock"),
            Path("src/robust_apex_qd/research/joint_endpoint_models.py"),
            *Path("src/robust_apex_qd/research").glob("bio*.py"),
        ],
    )
    torch.set_num_threads(2)
    with threadpool_limits(2):
        inputs.update(run(args.root, output))
    finish_stage(output, inputs, start)


if __name__ == "__main__":
    main()
