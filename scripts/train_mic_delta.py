"""Nested auxiliary-loss ablation using only curated training-partition pairs."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from run_mic_research import checked_manifest, finish_stage, write_json
from threadpoolctl import threadpool_limits
from train_mic_models import evaluate, feature_matrix

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.research.mic_data import MICConfig, measured_bounds
from robust_apex_qd.research.mic_delta import DeltaPair
from robust_apex_qd.research.mic_external import load_training_rows
from robust_apex_qd.research.mic_lineage import capture_execution
from robust_apex_qd.research.mic_models import fit_regressor, predict_regressor
from robust_apex_qd.research.prediction_cache import isolated_rows


def fit_partition(
    rows: pd.DataFrame,
    features: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    pairs: list[DeltaPair],
    weight: float,
    settings: dict[str, Any],
    output: Path,
) -> np.ndarray:
    isolated_rows(rows.sequence.tolist(), rows.homology_group.tolist(), train, valid)
    training_ids = rows.iloc[train].observation_id.tolist()
    eligible = [
        p
        for p in pairs
        if p.left.observation_id in set(training_ids)
        and p.right.observation_id in set(training_ids)
    ]
    # Original outer-fold labels are provenance; current training membership was checked above.
    eligible = [p.model_copy(update={"partition": "training"}) for p in eligible]
    low, high, _ = measured_bounds(rows)
    heads = np.where(rows.strain_index >= 0, rows.strain_index + 7, rows.species_index).astype(int)
    bundle = fit_regressor(
        features[train],
        heads[train],
        low[train],
        high[train],
        settings,
        42,
        pairs=eligible,
        observation_ids=training_ids,
        delta_weight=weight,
    )
    output.mkdir(parents=True, exist_ok=False)
    torch.save(bundle, output / "weights.pt")
    mean, _ = predict_regressor(bundle, features[valid])
    again, _ = predict_regressor(
        torch.load(output / "weights.pt", weights_only=True), features[valid]
    )
    np.testing.assert_equal(mean, again)
    write_json(
        output / "fit.json",
        dict(
            training_ids=training_ids,
            validation_ids=rows.iloc[valid].observation_id.tolist(),
            pairs=[p.model_dump() for p in eligible],
            delta_weight=weight,
            settings=settings,
            serialization_equal=True,
        ),
    )
    finish_stage(output, {}, time.monotonic())
    return mean[np.arange(len(valid)), heads[valid]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    inputs = checked_manifest(args.prepared / "manifest.json")
    inputs.update(checked_manifest(args.pairs / "manifest.json"))
    inputs.update(checked_manifest(args.models / "manifest.json"))
    config = MICConfig.model_validate_json(Path("configs/mic_research.json").read_text())
    fresh_output(args.output, [args.prepared, args.pairs, args.models])
    capture_execution(args.output)
    rows = load_training_rows(args.prepared)
    rows = rows[rows.objective.eq("measured_mic")].reset_index(drop=True)
    features = feature_matrix(config, "esm8-exact")[rows.sequence_index.to_numpy(int)]
    pairs = [
        DeltaPair.model_validate_json(line)
        for line in (args.pairs / "pairs.jsonl").read_text().splitlines()
    ]
    splits = json.loads((args.prepared / "split_manifest.json").read_text())
    widths = {
        c["fold"]: c["selected_width"]
        for c in json.loads((args.models / "esm8-exact-selection.json").read_text())
    }
    predictions = {w: np.full(len(rows), np.nan) for w in [0.0, 0.1, 0.3]}
    selected = np.full(len(rows), np.nan)
    choices = []
    for fold in sorted(rows.homology_fold.unique()):
        outer = rows.homology_fold.to_numpy() != fold
        inner = rows.sequence.map(splits["inner"][str(fold)]).fillna(-1).to_numpy(int)
        settings = dict(
            width=widths[fold],
            heads=18,
            scale_floor=config.scale_floor,
            device="cpu",
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            loss="interval",
            family="esm8-exact-delta",
        )
        scores = []
        for weight in predictions:
            inner_prediction = np.full(len(rows), np.nan)
            for inside in sorted(set(inner[outer])):
                tr = np.flatnonzero(
                    outer & (inner != inside) & rows.exact_regression.to_numpy(bool)
                )
                vi = np.flatnonzero(outer & (inner == inside))
                inner_prediction[vi] = fit_partition(
                    rows,
                    features,
                    tr,
                    vi,
                    pairs,
                    weight,
                    settings,
                    args.output / "fits" / f"o{fold}-i{inside}-w{weight:g}",
                )
            metric = evaluate(
                rows.loc[outer], inner_prediction[outer], np.full(outer.sum(), np.nan)
            )
            scores.append(dict(weight=weight, **metric))
            tr = np.flatnonzero(outer & rows.exact_regression.to_numpy(bool))
            vi = np.flatnonzero(~outer)
            predictions[weight][vi] = fit_partition(
                rows,
                features,
                tr,
                vi,
                pairs,
                weight,
                settings,
                args.output / "fits" / f"o{fold}-w{weight:g}",
            )
        winner = min(scores, key=lambda s: (s["macro_mae"], s["weight"]))["weight"]
        selected[~outer] = predictions[winner][~outer]
        choices.append(dict(fold=int(fold), selected_weight=winner, inner_scores=scores))
        print(f"delta outer {fold} selected {winner}", flush=True)
    comparisons, cliff = [], []
    by_id = {r.observation_id: i for i, r in enumerate(rows.itertuples())}
    all_predictions = {f"delta-w{w:g}": p for w, p in predictions.items()}
    all_predictions["delta-selected"] = selected
    for name, values in all_predictions.items():
        rows.assign(prediction=values, model=name).to_csv(
            args.output / f"{name}-oof.csv.gz", index=False
        )
        comparisons.append(dict(model=name, **evaluate(rows, values, np.full(len(rows), np.nan))))
        for threshold in [0, 1, 2]:
            chosen = [p for p in pairs if abs(p.delta_log2_um) >= threshold]
            truth = np.array([p.delta_log2_um for p in chosen])
            estimate = np.array(
                [
                    values[by_id[p.right.observation_id]] - values[by_id[p.left.observation_id]]
                    for p in chosen
                ]
            )
            nonzero = truth != 0
            cliff.append(
                dict(
                    model=name,
                    minimum_dilutions=threshold,
                    pairs=len(chosen),
                    folds=sorted({p.partition for p in chosen}),
                    delta_mae=float(np.abs(estimate - truth).mean()) if len(chosen) else None,
                    direction_accuracy=float(
                        (np.sign(estimate[nonzero]) == np.sign(truth[nonzero])).mean()
                    )
                    if nonzero.any()
                    else None,
                    direction_rows=int(nonzero.sum()),
                    limitation=(
                        "single-paper pilot; pair-containing heldout fold "
                        "has no curated training pairs"
                    ),
                )
            )
    pd.DataFrame(comparisons).to_csv(args.output / "model_comparison.csv", index=False)
    pd.DataFrame(cliff).to_csv(args.output / "cliff_metrics.csv", index=False)
    write_json(args.output / "selection.json", choices)
    reference = pd.read_csv(args.models / "esm8-exact-oof.csv.gz").set_index("observation_id")
    np.testing.assert_allclose(
        predictions[0.0], reference.reindex(rows.observation_id).prediction.to_numpy(), atol=1e-5
    )
    write_json(
        args.output / "absolute_parity.json",
        dict(rows=len(rows), zero_weight_equal=True, tolerance=1e-5),
    )
    finish_stage(args.output, inputs, start)


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        main()
