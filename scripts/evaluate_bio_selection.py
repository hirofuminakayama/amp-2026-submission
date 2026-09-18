"""Outer-fold joint selection using only inner-fold endpoint errors and model choices."""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from run_competition_bioaccuracy import (
    archive_sources,
    checked_manifest,
    finish_stage,
    read_observations,
    write_json,
)
from run_competition_models import predict
from threadpoolctl import threadpool_limits

from robust_apex_qd.apex.ensemble import aggregate_predictions, load_prediction_archive
from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.bioaccuracy import chemistry_key
from robust_apex_qd.research.biomodels import HC50Bundle, predict_hc50
from robust_apex_qd.research.bioscenarios import joint_trials, random25_scenarios, select_portfolio
from robust_apex_qd.research.competition_models import SPECIES
from robust_apex_qd.research.mic_models import predict_regressor


def score_labels(labels: pd.DataFrame, scores: dict[str, float]) -> list[dict[str, Any]]:
    records = []
    for species, cohort in labels.groupby("species"):
        cohort = cohort.assign(score=cohort.sequence.map(scores))
        if not np.isfinite(cohort.score).all():
            raise ValueError("Complete prediction coverage required before selection")
        cohort = cohort.sort_values(["score", "molecule_id"], ascending=[False, True])
        for name, k in [
            ("top20pct", max(1, math.ceil(len(cohort) * 0.2))),
            ("p10", 10),
            ("p25", 25),
            ("p100", 100),
        ]:
            enough = len(cohort) >= k
            chosen = cohort.head(k)
            records.append(
                dict(
                    species=species,
                    metric=name,
                    available=len(cohort),
                    requested=k,
                    supported=enough,
                    lower=float(chosen.hit_lower.mean()) if enough else None,
                    upper=float(chosen.hit_upper.mean()) if enough else None,
                )
            )
    return records


def run(
    root: Path,
    mic_root: Path,
    apex_path: Path,
    output: Path,
    mic_models: Path | None = None,
    mic_family: str = "linear8",
    mic_training_seed: int = 42,
) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for stage in ["prepare", "features", "split", "joint", "hc50-embeddings", "hc50-measured"]:
        inputs.update(checked_manifest(root / stage / "manifest.json"))
    inputs.update(checked_manifest(mic_root / "manifest.json"))
    mic_manifest = json.loads((mic_root / "manifest.json").read_text())
    split_hash = file_sha256(root / "split/split_manifest.json")
    old_splits = [
        v for k, v in mic_manifest["inputs_sha256"].items() if k.endswith("split_manifest.json")
    ]
    if split_hash not in old_splits:
        raise ValueError("MIC fits must use the identical cross-endpoint split")
    widths = {}
    if mic_family != "linear8":
        if mic_models is None:
            raise ValueError("Shared-fold MIC model directory required")
        inputs.update(checked_manifest(mic_models / "manifest.json"))
        saved = json.loads((mic_models / "manifest.json").read_text())
        if split_hash not in [
            v for k, v in saved["inputs_sha256"].items() if k.endswith("split_manifest.json")
        ]:
            raise ValueError("Neural MIC and HC50 shared split differ")
        widths = {
            int(r["fold"]): r["selected_width"]
            for r in json.loads((mic_models / f"{mic_family}-selection.json").read_text())
        }
    inputs[str(apex_path)] = file_sha256(apex_path)
    sequences = json.loads((root / "features/sequences.json").read_text())
    index = {s: i for i, s in enumerate(sequences)}
    splits = json.loads((root / "split/split_manifest.json").read_text())
    outer = np.array([splits["outer"][s] for s in sequences])
    x = np.load(root / "hc50-embeddings/features.npy")
    physical = np.load(root / "features/standard-local0-interactions0.npy")
    observations = read_observations(root / "prepare/endpoint_observations.jsonl")
    human = [r for r in observations if r.endpoint == "measured_hc50" and r.rbc_species == "human"]
    mapping = {chemistry_key(r): r.sequence for r in observations}
    labels = pd.read_csv(root / "joint/molecular_joint_labels.csv")
    labels["sequence"] = labels.molecule_id.map(mapping)
    primary = labels[
        (labels.evidence == "measured_mic+measured_hc50")
        & (labels.rbc_species == "human")
        & (labels.ratio == 8)
    ]
    mic_rows = pd.read_json(root / "split/rows.jsonl", lines=True)
    mic_rows = mic_rows[mic_rows.objective.eq("measured_mic") & mic_rows.exact_regression].copy()
    archive = load_prediction_archive(apex_path)
    b1 = dict(
        zip(archive.sequences, -aggregate_predictions(archive.mic_u_m).median_log2_mic, strict=True)
    )
    alphas = {
        int(r["fold"]): r["alpha"] for r in json.loads((mic_root / "selection.json").read_text())
    }
    arms = ["median", "standard-exact", "standard-interval", "esm8-exact", "esm8-interval"]
    metrics, portfolios, audits, prediction_rows, sensitivity = [], [], [], [], []
    write_json(
        output / "protocol.json",
        dict(
            primary="human measured MIC+HC50 molecular lower-bound macro top20%",
            family_selection="inner folds only; maximize primary score, ties arm id",
            mic_family=mic_family,
            mic_training_seed=mic_training_seed,
            selectors=["S0_B1", "MIC_only", "S1_mean", "S2_cap5", "S3_cvar"],
            assessment="outer5 shared OOD60; development reuse, not untouched holdout",
            residuals="inner OOF exact labels averaged per sequence/species; censoring retained",
            caveat="residuals exclude censored outcomes and may understate high-HC50 uncertainty",
            ratios=[4, 8, 16],
            correlations=[-0.5, 0.0, 0.5],
            dependence=["independent", "cluster", "species"],
            seeds=[42, 43, 44],
            draws=1000,
            random25_draws=10000,
            s0="actual APEX B1 score; external model training overlap unknown",
        ),
    )
    for fold in sorted(set(outer)):
        inside = np.array([splits["inner"][str(fold)].get(s, -1) for s in sequences])
        train = outer != fold
        mic_inner = np.full((len(x), 7), np.nan)
        for inner in sorted(set(inside[train])):
            state = dict(
                np.load(
                    mic_root
                    / "fits"
                    / f"outer{fold}-inner{inner}-a{alphas[fold]:g}"
                    / "weights.npz"
                )
            )
            mask = train & (inside == inner)
            if mic_models is not None and mic_family != "linear8":
                path = (
                    mic_models
                    / "fits"
                    / mic_family
                    / f"outer{fold}-inner{inner}-w{widths[fold]}-s{mic_training_seed}/weights.pt"
                )
                inputs.update(checked_manifest(path.parent / "manifest.json"))
                mic_inner[mask] = predict_regressor(torch.load(path, weights_only=True), x[mask])[
                    0
                ][:, :7]
            else:
                mic_inner[mask] = predict({}, state, x[mask], [], dict(family="linear8"))[0]
        state = dict(np.load(mic_root / "fits" / f"outer{fold}-selected" / "weights.npz"))
        if mic_models is not None and mic_family != "linear8":
            path = (
                mic_models
                / "fits"
                / mic_family
                / f"outer{fold}-selected-w{widths[fold]}-s{mic_training_seed}/weights.pt"
            )
            inputs.update(checked_manifest(path.parent / "manifest.json"))
            mic_outer = predict_regressor(torch.load(path, weights_only=True), x)[0][:, :7]
        else:
            mic_outer = predict({}, state, x, [], dict(family="linear8"))[0]
        residuals = []
        for species in SPECIES:
            cohort = mic_rows[
                (mic_rows.species == species) & (mic_rows.homology_fold != fold)
            ].copy()
            cohort["residual"] = [
                np.log2(r.mic_um) - mic_inner[index[r.sequence], SPECIES.index(species)]
                for r in cohort.itertuples()
            ]
            residuals.append(cohort.groupby("sequence").residual.mean().to_numpy())
        candidates: list[dict[str, Any]] = []
        inner_labels = primary[primary.sequence.map(splits["outer"]) != fold]
        inner_sequences = sorted(inner_labels.sequence.unique())
        ii = [index[s] for s in inner_sequences]
        for arm in arms:
            features = x if arm.startswith("esm8") else physical
            audit = json.loads(
                (root / "hc50-measured" / arm / f"outer{fold}-audit.json").read_text()
            )
            alpha = audit["selected_alpha"]
            hc_inner = np.full(len(x), np.nan)
            for inner in sorted(set(inside[train])):
                model = HC50Bundle.model_validate_json(
                    (
                        root / "hc50-measured" / arm / f"outer{fold}-inner{inner}-a{alpha:g}.json"
                    ).read_text()
                )
                mask = train & (inside == inner)
                hc_inner[mask] = predict_hc50(model, features[mask], model.feature_sha256)
            exact = [r for r in human if r.relation == "=" and splits["outer"][r.sequence] != fold]
            errors = pd.DataFrame(
                [
                    dict(
                        sequence=r.sequence,
                        residual=math.log2(float(r.value_um)) - hc_inner[index[r.sequence]],
                    )
                    for r in exact
                    if r.value_um is not None
                ]
            )
            hc_errors = errors.groupby("sequence").residual.mean().to_numpy()
            trial = joint_trials(
                mic_inner[ii],
                hc_inner[ii],
                residuals,
                hc_errors,
                groups=[splits["groups"][s] for s in inner_sequences],
            )
            scores = dict(zip(inner_sequences, trial.mean(axis=(0, 2)), strict=True))
            score = np.mean(
                [
                    r["lower"]
                    for r in score_labels(inner_labels, scores)
                    if r["metric"] == "top20pct"
                ]
            )
            candidates.append(dict(arm=arm, inner_score=float(score), hc_errors=hc_errors))
        best = sorted(candidates, key=lambda r: (-r["inner_score"], r["arm"]))[0]
        arm = best["arm"]
        model = HC50Bundle.model_validate_json(
            (root / "hc50-measured" / arm / f"outer{fold}-selected.json").read_text()
        )
        hc_outer = predict_hc50(
            model, x if arm.startswith("esm8") else physical, model.feature_sha256
        )
        audits.append(
            dict(
                fold=int(fold),
                selected_arm=arm,
                candidates=[{k: v for k, v in r.items() if k != "hc_errors"} for r in candidates],
                train_sequences=[s for s in sequences if splits["outer"][s] != fold],
                mic_residual_counts=[len(r) for r in residuals],
                hc_residual_count=len(best["hc_errors"]),
            )
        )
        np.savez_compressed(
            output / f"fold{fold}-residuals.npz",
            hc50=best["hc_errors"],
            **{f"mic{i}": r for i, r in enumerate(residuals)},
        )
        outer_labels = labels[labels.sequence.map(splits["outer"]) == fold]
        ss = sorted(outer_labels.sequence.unique())
        jj = [index[s] for s in ss]
        groups = [splits["groups"][s] for s in ss]
        for i, sequence in enumerate(ss):
            prediction_rows.append(
                dict(
                    sequence=sequence,
                    fold=int(fold),
                    hc50_arm=arm,
                    hc50_log2_um=float(hc_outer[jj[i]]),
                    **{f"mic{k}": float(mic_outer[jj[i], k]) for k in range(7)},
                )
            )
        for ratio in [4, 8, 16]:
            for seed in [42, 43, 44]:
                trials = joint_trials(
                    mic_outer[jj],
                    hc_outer[jj],
                    residuals,
                    best["hc_errors"],
                    groups=groups,
                    ratio=ratio,
                    seed=seed,
                )
                scores_by_method = {
                    "S0_B1": {s: float(b1[s]) for s in ss},
                    "MIC_only": dict(zip(ss, -np.median(mic_outer[jj], axis=1), strict=True)),
                    "S1_mean": dict(zip(ss, trials.mean(axis=(0, 2)), strict=True)),
                }
                count = min(100, len(ss))
                orders = {}
                for method, objective, cap in [
                    ("S1_mean", "mean", None),
                    ("S2_cap5", "mean", 5),
                    ("S3_cvar", "cvar", None),
                ]:
                    if (
                        cap is not None
                        and sum(min(n, cap) for n in Counter(groups).values()) < count
                    ):
                        portfolios.append(
                            dict(
                                fold=int(fold),
                                selector=method,
                                ratio=ratio,
                                seed=seed,
                                status="insufficient group-cap supply",
                            )
                        )
                        continue
                    order = select_portfolio(
                        trials.mean(axis=2),
                        ss,
                        count=count,
                        objective="cvar" if objective == "cvar" else "mean",
                        groups=groups,
                        cap=cap,
                    )
                    orders[method] = order
                    if method != "S1_mean":
                        # Preserve the greedy order for unselected candidates as well.
                        remaining = sorted(
                            set(range(len(ss))) - set(order),
                            key=lambda i: (-scores_by_method["S1_mean"][ss[i]], ss[i]),
                        )
                        scores_by_method[method] = {
                            ss[i]: float(len(ss) - rank) for rank, i in enumerate(order + remaining)
                        }
                for (evidence, rbc), cohort in outer_labels[outer_labels.ratio == ratio].groupby(
                    ["evidence", "rbc_species"]
                ):
                    for method, scores in scores_by_method.items():
                        metrics.extend(
                            dict(
                                fold=int(fold),
                                seed=seed,
                                ratio=ratio,
                                evidence=evidence,
                                rbc_species=rbc,
                                selector=method,
                                hc50_arm=arm,
                                **r,
                            )
                            for r in score_labels(cohort, scores)
                        )
                if ratio == 8 and seed == 42:
                    for method, order in orders.items():
                        portfolios.append(
                            dict(
                                fold=int(fold),
                                selector=method,
                                count=count,
                                sequences=[ss[i] for i in order],
                                **random25_scenarios(trials[:, order], sample_size=25),
                            )
                        )
                    for correlation in [-0.5, 0.0, 0.5]:
                        for dependence in ["independent", "cluster", "species"]:
                            varied = joint_trials(
                                mic_outer[jj],
                                hc_outer[jj],
                                residuals,
                                best["hc_errors"],
                                groups=groups,
                                correlation=correlation,
                                dependence=dependence,
                            )
                            for method, order in orders.items():
                                sensitivity.append(
                                    dict(
                                        fold=int(fold),
                                        selector=method,
                                        correlation=correlation,
                                        dependence=dependence,
                                        **random25_scenarios(varied[:, order], draws=10000),
                                    )
                                )
        print(json.dumps(dict(fold=int(fold), selected_hc50=arm, cohort=len(ss))), flush=True)
    pd.DataFrame(metrics).to_csv(output / "nested_selection_results.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "outer_endpoint_predictions.csv", index=False)
    write_json(output / "selection_audit.json", audits)
    write_json(output / "random25.json", portfolios)
    write_json(output / "predictive_scenarios.json", sensitivity)
    frame = pd.DataFrame(metrics)
    primary_metrics = frame[
        (frame.evidence == "measured_mic+measured_hc50")
        & (frame.rbc_species == "human")
        & (frame.ratio == 8)
        & frame.supported
    ]
    primary_metrics.groupby(["selector", "metric", "species"]).agg(
        lower=("lower", "mean"),
        upper=("upper", "mean"),
        assessed_rows=("lower", "size"),
    ).reset_index().groupby(["selector", "metric"]).agg(
        lower=("lower", "mean"),
        upper=("upper", "mean"),
        assessed_rows=("assessed_rows", "sum"),
        species=("species", "nunique"),
    ).reset_index().to_csv(output / "selector_comparison.csv", index=False)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mic-baselines", type=Path, required=True)
    parser.add_argument("--mic-models", type=Path)
    parser.add_argument("--mic-training-seed", type=int, default=42)
    parser.add_argument(
        "--mic-family",
        choices=["linear8", "esm8-exact", "esm8-interval", "esm8-normal"],
        default="linear8",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--apex",
        type=Path,
        default=Path("work/competition_exploration/20260912-b/phase3-apex/predictions.npz"),
    )
    args = parser.parse_args()
    output = args.output or args.root / "nested-selection"
    fresh_output(output, [args.mic_baselines, args.apex.parent])
    started = time.monotonic()
    inputs = archive_sources(
        output,
        [
            Path(__file__),
            Path("scripts/run_competition_bioaccuracy.py"),
            Path("scripts/run_competition_models.py"),
            Path("src/robust_apex_qd/research/mic_models.py"),
            Path("uv.lock"),
            *Path("src/robust_apex_qd/research").glob("bio*.py"),
        ],
    )
    with threadpool_limits(2):
        torch.set_num_threads(2)
        inputs.update(
            run(
                args.root,
                args.mic_baselines,
                args.apex,
                output,
                args.mic_models,
                args.mic_family,
                args.mic_training_seed,
            )
        )
    finish_stage(output, inputs, started)


if __name__ == "__main__":
    main()
