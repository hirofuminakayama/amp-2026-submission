"""Refit previously fixed GPU controls on an explicitly supplied new development split."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_competition_models import fit, predict
from run_mic_research import checked_manifest, finish_stage, write_json
from threadpoolctl import threadpool_limits
from train_mic_models import evaluate

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import FastaRecord, write_fasta
from robust_apex_qd.research.mic_external import load_training_rows
from robust_apex_qd.research.mic_lineage import capture_execution
from robust_apex_qd.research.prediction_cache import isolated_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--families", nargs="+", default=["finetune8", "ablation-interval"])
    parser.add_argument(
        "--prior", type=Path, default=Path("work/competition_exploration/20260912-b/phase3-refits")
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("work/competition_exploration/20260912-b/phase3-r3/esm8"),
    )
    args = parser.parse_args()
    started = time.monotonic()
    inputs = checked_manifest(args.prepared / "manifest.json")
    inputs.update(checked_manifest(args.features / "manifest.json"))
    config_path = Path("configs/competition_models.json")
    config = json.loads(config_path.read_text())
    selected_path = args.prior / "selected_models.json"
    inputs.update(
        {
            str(p): file_sha256(p)
            for p in [config_path, selected_path, Path(config["esm8_checkpoint"])]
        }
    )
    arms = [a for a in json.loads(selected_path.read_text()) if a["artifact_key"] in args.families]
    fresh_output(args.output, [args.prepared, args.prior, args.features])
    capture_execution(args.output)
    config["run_root"] = str(args.output)
    sequences = json.loads((args.prepared / "sequences.json").read_text())
    (args.output / "prepare").mkdir()
    write_fasta(
        [FastaRecord(str(i), s) for i, s in enumerate(sequences)],
        args.output / "prepare/sequences.fasta",
    )
    rows = load_training_rows(args.prepared)
    rows = rows[rows.objective.eq("measured_mic")].reset_index(drop=True)
    features = np.load(args.features / "features.npy")
    write_json(
        args.output / "protocol.json",
        dict(
            config=config,
            arms=arms,
            seed=args.seed,
            selection="previously fixed control settings; no new outer-label tuning",
        ),
    )
    results = []
    for arm in arms:
        prediction = np.full(len(rows), np.nan)
        for fold in sorted(rows.homology_fold.unique()):
            valid = rows.homology_fold.to_numpy() == fold
            train = ~valid
            if arm.get("loss") != "interval":
                train &= rows.exact_regression.to_numpy(bool)
            isolated_rows(
                rows.sequence.tolist(),
                rows.homology_group.tolist(),
                np.flatnonzero(train),
                np.flatnonzero(valid),
            )
            dest = args.output / arm["artifact_key"] / f"fold{fold}"
            dest.mkdir(parents=True)
            state = fit(config, rows, features, train, arm, args.seed, dest)
            subset = rows.loc[valid]
            species, strains = predict(
                config, state, features[subset.sequence_index], subset.sequence.tolist(), arm
            )
            values = species[np.arange(len(subset)), subset.species_index.to_numpy(int)]
            known = subset.strain_index.to_numpy(int) >= 0
            values[known] = strains[np.flatnonzero(known), subset.strain_index.to_numpy(int)[known]]
            prediction[valid] = values
            loaded = torch.load(dest / "weights.pt", weights_only=False)
            again = predict(
                config,
                loaded,
                features[subset.sequence_index.iloc[:16]],
                subset.sequence.iloc[:16].tolist(),
                arm,
            )
            np.testing.assert_allclose(species[:16], again[0], atol=1e-5, equal_nan=True)
            np.testing.assert_allclose(strains[:16], again[1], atol=1e-5, equal_nan=True)
            write_json(
                dest / "fit.json",
                dict(
                    training_ids=rows.loc[train, "observation_id"].tolist(),
                    validation_ids=subset.observation_id.tolist(),
                    serialization_equal=True,
                    arm=arm,
                ),
            )
            finish_stage(dest, inputs, started)
            print(
                arm["artifact_key"],
                int(fold),
                evaluate(subset, values, np.full(len(subset), np.nan)),
                flush=True,
            )
        rows.assign(prediction=prediction, model=arm["artifact_key"], unit="log2_uM").to_csv(
            args.output / f"{arm['artifact_key']}-oof.csv.gz", index=False
        )
        results.append(
            dict(
                model=arm["artifact_key"], **evaluate(rows, prediction, np.full(len(rows), np.nan))
            )
        )
        pd.DataFrame(results).to_csv(args.output / "model_comparison.csv", index=False)
    finish_stage(args.output, inputs, started, development_only=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        main()
