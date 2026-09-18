"""Nested MIC regression with fold-local assay encoding and reusable completed fits."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from run_mic_research import checked_manifest, finish_stage, write_json
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.mic_data import MICConfig, measured_bounds
from robust_apex_qd.research.mic_external import load_training_rows
from robust_apex_qd.research.mic_models import (
    censored_normal_nll,
    fit_regressor,
    mic_metrics,
    predict_regressor,
)

FAMILIES = [
    "esm8-exact",
    "esm8-interval",
    "esm8-normal",
    "esm8-physchem-normal",
    "esm650-physchem-normal",
    "esm8-assay-normal",
]


def assay_features(rows: pd.DataFrame, training: np.ndarray) -> tuple[np.ndarray, list[list[str]]]:
    categorical = rows[["medium", "cfu"]].fillna("__missing__").astype(str)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    encoder.fit(categorical.iloc[training])
    missing = rows[["medium", "cfu"]].isna().to_numpy(dtype=np.float32)
    return np.column_stack([encoder.transform(categorical), missing]), [
        c.tolist() for c in encoder.categories_
    ]


def feature_matrix(config: MICConfig, family: str) -> np.ndarray:
    root = Path(config.prior_models)
    name = "esm650" if "650" in family else "esm8"
    x = np.load(root / name / "features.npy")
    if "physchem" in family or "assay" in family:
        x = np.column_stack([x, pd.read_csv(root / "prepare/physchem.csv").to_numpy(np.float32)])
    return x.astype(np.float32)


def evaluate(rows: pd.DataFrame, mean: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    result = mic_metrics(rows, mean)
    species_mae, top20 = [], []
    for _species, group in rows.assign(prediction=mean).groupby("species"):
        m = mic_metrics(group, group.prediction.to_numpy())
        if m["mae"] is not None:
            species_mae.append(m["mae"])
        active = group[group.active16.notna() & np.isfinite(group.prediction)].sort_values(
            ["prediction", "sequence", "observation_id"]
        )
        if len(active):
            top20.append(
                float(active.head(max(1, int(np.ceil(0.2 * len(active))))).active16.mean())
            )
    result["macro_mae"] = float(np.mean(species_mae)) if species_mae else None
    result["macro_top20"] = float(np.mean(top20)) if top20 else None
    low, high, usable = measured_bounds(rows)
    supported = usable & np.isfinite(mean)
    censored = supported & (low != high)
    result["censored_rows"] = int(censored.sum())
    result["bound_violation"] = (
        float(((mean[censored] < low[censored]) | (mean[censored] > high[censored])).mean())
        if censored.any()
        else None
    )
    distribution = supported & np.isfinite(scale)
    result["nll"] = (
        float(
            censored_normal_nll(
                torch.tensor(mean[distribution], dtype=torch.float64),
                torch.tensor(scale[distribution], dtype=torch.float64),
                torch.tensor(low[distribution]),
                torch.tensor(high[distribution]),
            ).mean()
        )
        if distribution.any()
        else None
    )
    exact = distribution & (low == high)
    result["coverage90_exact"] = (
        float((np.abs(mean[exact] - low[exact]) <= 1.644853627 * scale[exact]).mean())
        if exact.any()
        else None
    )
    result["width90_exact"] = (
        float((2 * 1.644853627 * scale[exact]).mean()) if exact.any() else None
    )
    return result


def run_fit(
    config: MICConfig,
    rows: pd.DataFrame,
    x: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    family: str,
    width: int,
    seed: int,
    output: Path,
    protocol_hash: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    start = time.monotonic()
    low, high, usable = measured_bounds(rows)
    if "exact" in family:
        usable &= low == high
    training = training[usable[training]]
    if not len(training):
        raise ValueError("No supported training measurements")
    if set(rows.iloc[training].homology_group) & set(rows.iloc[validation].homology_group):
        raise ValueError("Training and validation groups cross")
    features = x[rows.sequence_index.to_numpy(int)]
    categories = []
    if "assay" in family:
        extra, categories = assay_features(rows, training)
        features = np.column_stack([features, extra])
    settings = dict(
        width=width,
        heads=18,
        scale_floor=config.scale_floor,
        device=config.device,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        loss="normal" if "normal" in family else "interval",
        family=family,
    )
    train_ids = rows.iloc[training].observation_id.tolist()
    valid_ids = rows.iloc[validation].observation_id.tolist()
    manifest = output / "manifest.json"
    if manifest.exists():
        checked_manifest(manifest)
        saved = json.loads(manifest.read_text())
        if (
            saved["protocol_hash"] != protocol_hash
            or saved["training_ids"] != train_ids
            or saved["validation_ids"] != valid_ids
        ):
            raise ValueError("Completed MIC fit protocol/IDs changed")
        bundle = torch.load(output / "weights.pt", weights_only=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        heads = np.where(rows.strain_index >= 0, rows.strain_index + 7, rows.species_index).astype(
            int
        )
        bundle = fit_regressor(
            features[training], heads[training], low[training], high[training], settings, seed
        )
        bundle["assay_categories"] = categories
        torch.save(bundle, output / "weights.pt")
        first = predict_regressor(bundle, features[validation[:32]])
        second = predict_regressor(
            torch.load(output / "weights.pt", weights_only=True), features[validation[:32]]
        )
        for a, b in zip(first, second, strict=True):
            np.testing.assert_equal(a, b)
        write_json(
            output / "fit.json",
            dict(
                settings=settings,
                seed=seed,
                training_ids=train_ids,
                validation_ids=valid_ids,
                assay_categories=categories,
                loss_curve=bundle["loss_curve"],
                serialization_equal=True,
            ),
        )
        finish_stage(
            output,
            {},
            start,
            protocol_hash=protocol_hash,
            training_ids=train_ids,
            validation_ids=valid_ids,
        )
    mean, scale = predict_regressor(bundle, features[validation])
    head = np.where(
        rows.iloc[validation].strain_index >= 0,
        rows.iloc[validation].strain_index + 7,
        rows.iloc[validation].species_index,
    ).astype(int)
    p, sd = mean[np.arange(len(validation)), head], scale[np.arange(len(validation)), head]
    return p, sd, evaluate(rows.iloc[validation], p, sd)


def train(config: MICConfig, prepared: Path, output: Path, families: list[str]) -> None:
    started = time.monotonic()
    inputs = checked_manifest(prepared / "manifest.json")
    for name in ["prepare", "esm8", "esm650"]:
        inputs.update(checked_manifest(Path(config.prior_models) / name / "manifest.json"))
    execution = {
        "config": config.model_dump(),
        "families": families,
        "inputs_sha256": inputs,
        "code_sha256": {
            str(p): file_sha256(p)
            for p in [
                Path(__file__),
                Path("scripts/run_mic_research.py"),
                *Path("src/robust_apex_qd/research").glob("mic_*.py"),
            ]
        },
    }
    protocol = output / "protocol.json"
    if protocol.exists():
        if json.loads(protocol.read_text()) != execution:
            raise ValueError("MIC training source/config/input changed; use fresh output")
    else:
        fresh_output(output, [prepared, Path(config.prior_models)])
        write_json(protocol, execution)
    protocol_hash = file_sha256(protocol)
    rows = load_training_rows(prepared)
    rows = rows[rows.objective.eq("measured_mic")].reset_index(drop=True)
    splits = json.loads((prepared / "split_manifest.json").read_text())
    if rows.homology_fold.max() < 1:
        raise ValueError("Insufficient outer groups for OOF")
    all_predictions, comparisons = [], []
    for family in families:
        x = feature_matrix(config, family)
        outer_means, outer_scales = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
        choices = []
        for fold in sorted(rows.homology_fold.unique()):
            valid = np.flatnonzero(rows.homology_fold == fold)
            training = np.flatnonzero(rows.homology_fold != fold)
            inner = rows.sequence.map(splits["inner"][str(fold)]).fillna(-1).to_numpy(int)
            if len(set(inner[training])) < 2:
                raise ValueError("Insufficient inner groups for model selection")
            scores = []
            for width in config.widths:
                inner_mean = np.full(len(rows), np.nan)
                for inner_fold in sorted(set(inner[training])):
                    vi = training[inner[training] == inner_fold]
                    tr = training[inner[training] != inner_fold]
                    dest = (
                        output
                        / "fits"
                        / family
                        / f"outer{fold}-inner{inner_fold}-w{width}-s{config.seeds[0]}"
                    )
                    p, _sd, _metric = run_fit(
                        config, rows, x, tr, vi, family, width, config.seeds[0], dest, protocol_hash
                    )
                    inner_mean[vi] = p
                metric = evaluate(
                    rows.iloc[training], inner_mean[training], np.full(len(training), np.nan)
                )
                scores.append(dict(width=width, **metric))
            eligible = [s for s in scores if s["macro_mae"] is not None]
            if not eligible:
                raise ValueError("No inner exact labels for selecting MIC model")
            width = sorted(
                eligible, key=lambda s: (s["macro_mae"], -(s["macro_top20"] or 0), s["width"])
            )[0]["width"]
            dest = output / "fits" / family / f"outer{fold}-selected-w{width}-s{config.seeds[0]}"
            p, sd, metric = run_fit(
                config,
                rows,
                x,
                training,
                valid,
                family,
                width,
                config.seeds[0],
                dest,
                protocol_hash,
            )
            outer_means[valid], outer_scales[valid] = p, sd
            choices.append(
                dict(
                    fold=int(fold), selected_width=width, inner_scores=scores, outer_metrics=metric
                )
            )
            print(f"{family} outer {fold} MAE={metric['macro_mae']}", flush=True)
        prediction = rows[
            [
                "observation_id",
                "sequence",
                "species",
                "target",
                "apex_pathogen",
                "homology_fold",
                "homology_group",
            ]
        ].copy()
        prediction["prediction"], prediction["scale"] = outer_means, outer_scales
        prediction["model"], prediction["unit"] = family, "log2_uM"
        prediction["target_level"] = np.where(rows.strain_index >= 0, "strain", "species")
        prediction.to_csv(output / f"{family}-oof.csv.gz", index=False)
        write_json(output / f"{family}-selection.json", choices)
        comparisons.append(
            dict(family=family, seed=config.seeds[0], **evaluate(rows, outer_means, outer_scales))
        )
        all_predictions.append(prediction)
        pd.DataFrame(comparisons).to_csv(output / "model_comparison.csv", index=False)
        pd.concat(all_predictions).to_csv(output / "oof_predictions.csv.gz", index=False)
    finish_stage(
        output,
        inputs,
        started,
        claim="nested development OOF; prior selection history remains",
        repeats="not yet run",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mic_research.json"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    args = parser.parse_args()
    config = MICConfig.model_validate_json(args.config.read_text())
    torch.set_num_threads(config.cpu_threads)
    with threadpool_limits(config.cpu_threads):
        train(config, args.prepared, args.output, args.families)


if __name__ == "__main__":
    main()
