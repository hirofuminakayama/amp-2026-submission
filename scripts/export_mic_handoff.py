"""Refit new-split control winners and export immutable saved-pool predictions and rankings."""

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

from robust_apex_qd.apex.ensemble import load_prediction_archive
from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import FastaRecord, write_fasta
from robust_apex_qd.research.mic_data import MICConfig
from robust_apex_qd.research.mic_lineage import capture_execution
from robust_apex_qd.research.prediction_cache import PredictionCache
from robust_apex_qd.research.prediction_mic import blend_strain_predictions, prediction_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    config = MICConfig.model_validate_json(Path("configs/mic_research.json").read_text())
    inputs = {}
    for stage in [args.prepared, args.controls, args.baselines, Path(config.prior_models) / "esm8"]:
        inputs.update(checked_manifest(stage / "manifest.json"))
    refits = Path(config.prior_refits)
    pool_file = refits / "pool_sequences.csv"
    feature_path = refits / "esm8.npy"
    config_path = Path("configs/competition_models.json")
    original = json.loads(config_path.read_text())
    apex_path = Path(config.frozen_pool) / "work/apex_predictions.npz"
    for path in [
        pool_file,
        feature_path,
        config_path,
        apex_path,
        Path(original["esm8_checkpoint"]),
    ]:
        inputs[str(path)] = file_sha256(path)
    fresh_output(args.output, [args.prepared, args.controls, args.baselines, refits])
    capture_execution(args.output)
    sequences = pd.read_csv(pool_file).sequence.tolist()
    px = np.load(feature_path)
    x = np.load(Path(config.prior_models) / "esm8/features.npy")
    if len(px) != len(sequences) or len(set(sequences)) != len(sequences):
        raise ValueError("Pool features require unique aligned sequences")
    archive = load_prediction_archive(apex_path)
    lookup = {s: i for i, s in enumerate(archive.sequences)}
    apex = np.log2(archive.mic_u_m.mean(1))[[lookup[s] for s in sequences]]
    rows = pd.read_json(args.prepared / "rows.jsonl", lines=True)
    rows = rows[rows.objective.eq("measured_mic")].reset_index(drop=True)
    train = rows.exact_regression.to_numpy(bool)
    training_sequences = json.loads((args.prepared / "sequences.json").read_text())
    original["run_root"] = str(args.output)
    (args.output / "prepare").mkdir()
    write_fasta(
        [FastaRecord(str(i), s) for i, s in enumerate(training_sequences)],
        args.output / "prepare/sequences.fasta",
    )
    selected = json.loads((args.controls / "protocol.json").read_text())["arms"]
    fine = next(a for a in selected if a["family"] == "finetune8")
    alphas = pd.Series(
        [r["alpha"] for r in json.loads((args.baselines / "selection.json").read_text())]
    ).value_counts()
    alpha = min(alphas.index, key=lambda a: (-alphas[a], a))
    arms = [dict(id="linear8", artifact_key="linear8", family="linear8", alpha=float(alpha)), fine]
    sweep = []
    for arm in arms:
        name = arm["artifact_key"]
        dest = args.output / name
        dest.mkdir()
        state = fit(original, rows, x, train, arm, 42, dest)
        checkpoint = dest / ("weights.npz" if name == "linear8" else "weights.pt")
        dependencies = dict(
            weights=file_sha256(checkpoint),
            features=file_sha256(feature_path),
            feature_config=file_sha256(config_path),
            sequences=file_sha256(pool_file),
            apex=file_sha256(apex_path),
            predictor=file_sha256(Path("scripts/run_competition_models.py")),
        )
        species, strains = predict(original, state, px, sequences, arm)
        mean = np.column_stack([species, strains])
        scale = np.full_like(mean, np.nan)
        cache = PredictionCache(dest / "cache")
        cache.write(sequences, np.column_stack([mean, scale]), dependencies)
        np.testing.assert_equal(
            cache.read(sequences[::-1], dependencies), np.column_stack([mean, scale])[::-1]
        )
        if name == "linear8":
            loaded = dict(np.load(checkpoint))
        else:
            loaded = torch.load(checkpoint, weights_only=False)
        again = predict(original, loaded, px[:32], sequences[:32], arm)
        np.testing.assert_allclose(species[:32], again[0], atol=1e-5, equal_nan=True)
        np.testing.assert_allclose(strains[:32], again[1], atol=1e-5, equal_nan=True)
        frame = prediction_records(
            sequences, mean, scale, apex, dependencies["weights"], dependencies["apex"]
        )
        frame.to_csv(dest / "candidate_predictions.csv.gz", index=False)
        np.savez(dest / "candidate_predictions.npz", species=species, strain=strains, scale=scale)
        for weight in [0, 0.25, 0.5, 0.75, 1]:
            score = np.median(blend_strain_predictions(frame, apex, weight, sequences), axis=1)
            pd.DataFrame(dict(sequence=sequences, score=score)).sort_values(
                ["score", "sequence"]
            ).to_csv(dest / f"ranking-w{weight:g}.csv.gz", index=False)
            if weight == 1:
                np.testing.assert_equal(score, np.median(apex, axis=1))
            sweep.append(
                dict(
                    model=name,
                    method=f"w{weight:g}",
                    apex_weight=weight,
                    sequences=len(sequences),
                    supported=int(frame.supported.sum()),
                )
            )
        pure = np.median(blend_strain_predictions(frame, apex, 0, sequences), axis=1)
        rankmean = (
            pd.Series(pure).rank(pct=True) + pd.Series(np.median(apex, axis=1)).rank(pct=True)
        ) / 2
        pd.DataFrame(dict(sequence=sequences, score=rankmean)).sort_values(
            ["score", "sequence"]
        ).to_csv(dest / "ranking-rankmean.csv.gz", index=False)
        sweep.append(
            dict(
                model=name,
                method="rankmean",
                apex_weight=None,
                sequences=len(sequences),
                supported=int(frame.supported.sum()),
            )
        )
        write_json(
            dest / "refit.json",
            dict(
                arm=arm,
                training_ids=rows.loc[train, "observation_id"].tolist(),
                seed=42,
                dependencies=dependencies,
                serialization_equal=True,
                cache_reorder_equal=True,
            ),
        )
        finish_stage(dest, inputs, start)
        print(f"Exported {name}: {len(sequences)} sequences", flush=True)
    pd.DataFrame(sweep).to_csv(args.output / "ensemble_sweep.csv", index=False)
    write_json(
        args.output / "handoff.json",
        dict(
            selected=[a["artifact_key"] for a in arms],
            adopted=False,
            purpose="new-split best control plus linear comparator; research handoff",
            limitation=(
                "single baseline pool; no assay-missing candidate assumptions; "
                "full generation not performed"
            ),
        ),
    )
    finish_stage(args.output, inputs, start)


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        main()
