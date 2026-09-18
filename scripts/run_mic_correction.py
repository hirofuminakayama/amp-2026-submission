"""Compare fixed finetune controls on shared paper/homology development folds."""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from prepare_mic_external import checked, finish, read_jsonl, write_json, write_jsonl
from run_competition_models import fit, predict
from threadpoolctl import threadpool_limits
from train_mic_models import evaluate

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.research.mic_correction import (
    correction_masks,
    paired_comparison,
    verify_completed_fit,
)
from robust_apex_qd.research.mic_data import fold_assignments
from robust_apex_qd.research.mic_external import validate_training_membership
from robust_apex_qd.research.mic_lineage import capture_execution
from robust_apex_qd.research.prediction_cache import isolated_rows


def prepare(cfg: dict[str, Any], output: Path) -> dict[str, str]:
    external = Path(cfg["external"])
    inputs = checked(external)
    protocol = json.loads((external / "evaluation_protocol.json").read_text())
    if protocol["scoring_status"] != "not_started":
        raise ValueError("External scoring already used")
    split = json.loads((external / "split_manifest.json").read_text())
    if split["new_training_ids"] or split["primary_evaluation_ids"]:
        raise ValueError(
            "This bounded correction workflow requires the registered empty external collection"
        )
    by_id = {r["observation_id"]: r for r in split["assignments"]}
    groups = {
        r["sequence"]: r["component_id"]
        for r in split["assignments"]
        if r["partition"] == "development"
    }
    folds = fold_assignments(groups, 5, 42)
    if set(folds.values()) != {0, 1, 2, 3, 4}:
        raise ValueError("Insufficient development components")
    old = Path(cfg["prepared"])
    inputs.update(checked(old))
    rows = pd.read_json(old / "rows.jsonl", lines=True)
    rows = rows[rows.objective.eq("measured_mic")].copy().reset_index(drop=True)
    for r in rows.itertuples():
        if by_id[r.observation_id]["partition"] != "development":
            raise ValueError("External row in development")
    rows["component_id"] = [by_id[k]["component_id"] for k in rows.observation_id]
    rows["homology_group"] = rows.component_id
    rows["homology_fold"] = [folds[s] for s in rows.sequence]
    rows["prediction_level"] = np.where(
        rows.strain_index >= 0, "strain_head", "species_head_unknown_strain"
    )
    sequences = json.loads((old / "sequences.json").read_text())
    feature_order = Path(cfg["feature_sequences"])
    if read_fasta_sequences(feature_order) != sequences:
        raise ValueError("Frozen feature order differs")
    if any(sequences[int(r.sequence_index)] != r.sequence for r in rows.itertuples()):
        raise ValueError("Sequence indices differ")
    for token in ["features", "feature_sequences", "selected_models", "training_config"]:
        path = Path(cfg[token])
        inputs[str(path)] = file_sha256(path)
    feature_manifest = Path(cfg["features"]).parent / "manifest.json"
    if (
        json.loads(feature_manifest.read_text())["artifacts_sha256"]["features.npy"]
        != inputs[cfg["features"]]
    ):
        raise ValueError("Frozen features differ from their saved manifest")
    inputs[str(feature_manifest)] = file_sha256(feature_manifest)
    features = np.load(cfg["features"], allow_pickle=False)
    if len(features) != len(sequences) or not np.isfinite(features).all():
        raise ValueError("Features invalid")
    corrections = {
        r["old_observation_id"]
        for r in read_jsonl(external / "training/correction_exclusions.jsonl")
    }
    if not corrections <= set(rows.observation_id):
        raise ValueError("Missing correction endpoint")
    contract = json.loads((external / "training/training_contract.json").read_text())
    # The original-data control intentionally retains the documented legacy chemistry.
    contract["allowed_observation_ids"] = rows.observation_id.tolist()
    validate_training_membership(rows.to_dict("records"), contract)
    settings = next(
        a
        for a in json.loads(Path(cfg["selected_models"]).read_text())
        if a["artifact_key"] == "finetune8"
    )
    training_config = json.loads(Path(cfg["training_config"]).read_text())
    checkpoint = Path(training_config["esm8_checkpoint"])
    inputs[str(checkpoint)] = file_sha256(checkpoint)
    write_jsonl(output / "rows.jsonl", rows.to_dict("records"))
    write_json(output / "sequences.json", sequences)
    write_json(output / "corrections.json", sorted(corrections))
    contract.update(folds_ready=True, rows_sha256=file_sha256(output / "rows.jsonl"))
    write_json(output / "training_contract.json", contract)
    write_json(
        output / "development_split.json",
        dict(
            groups=groups,
            folds=folds,
            seed=42,
            assignment="greedy sequence-count balance of complete frozen components",
            scope="development only",
            old_oof_reused=False,
        ),
    )
    write_json(
        output / "protocol.json",
        dict(
            arm=settings,
            training_config=training_config,
            seeds=[42, 43, 44],
            data_arms=["original", "corrected", "corrected_plus_new_training"],
            equivalent_arm=dict(
                corrected_plus_new_training="corrected; no new training observations"
            ),
            primary_external_rows=0,
            external_scoring="not_executed_insufficient",
            comparison_rows=sorted(set(rows.observation_id) - corrections),
            original_rows=len(rows),
            corrected_rows=len(rows) - len(corrections),
            training_objective="exact measured MIC; fixed prior control; no tuning",
            candidate_policy="retain finetune8-w0.25; no new export or adoption",
            bootstrap=dict(
                seed=42, repetitions=1000, unit="paper_homology_component", minimum_components=5
            ),
        ),
    )
    rows.groupby(["homology_fold", "species"], as_index=False).agg(
        observations=("observation_id", "size")
    ).to_csv(output / "fold_coverage.csv", index=False)
    return inputs


def train(
    cfg: dict[str, Any],
    prepared: Path,
    output: Path,
    smoke: bool,
    reuse: Path | None = None,
) -> dict[str, str]:
    inputs = checked(prepared)
    if reuse is not None:
        for token in ["scripts/run_competition_models.py", "uv.lock"]:
            if file_sha256(Path(token)) != file_sha256(reuse / "executed_sources" / token):
                raise ValueError("Completed fit training engine or environment changed")
    protocol = json.loads((prepared / "protocol.json").read_text())
    rows = pd.read_json(prepared / "rows.jsonl", lines=True)
    features = np.load(cfg["features"], allow_pickle=False)
    inputs[cfg["features"]] = file_sha256(Path(cfg["features"]))
    sequences = json.loads((prepared / "sequences.json").read_text())
    corrections = set(json.loads((prepared / "corrections.json").read_text()))
    config = protocol["training_config"]
    config["run_root"] = str(output)
    (output / "prepare").mkdir()
    write_fasta(
        [FastaRecord(str(i), s) for i, s in enumerate(sequences)],
        output / "prepare/sequences.fasta",
    )
    arm = dict(protocol["arm"])
    if smoke:
        arm["epochs"] = 1
    reuse_events = []
    for seed in [42] if smoke else protocol["seeds"]:
        for name in ["corrected"] if smoke else ["original", "corrected"]:
            predictions = []
            for fold in [0] if smoke else sorted(rows.homology_fold.unique()):
                started = time.monotonic()
                original, corrected, valid = correction_masks(rows, corrections, int(fold))
                mask = original if name == "original" else corrected
                if smoke:
                    train_indices = np.flatnonzero(mask)[:32]
                    valid_indices = np.flatnonzero(valid)[:16]
                    mask = np.zeros(len(rows), bool)
                    mask[train_indices] = True
                    valid = np.zeros(len(rows), bool)
                    valid[valid_indices] = True
                isolated_rows(
                    rows.sequence.tolist(),
                    rows.component_id.tolist(),
                    np.flatnonzero(mask),
                    np.flatnonzero(valid),
                )
                dest = output / f"{name}-s{seed}" / f"fold{fold}"
                prior = reuse / f"{name}-s{seed}" / f"fold{fold}" if reuse else None
                if prior is not None and (prior / "manifest.json").exists():
                    verify_completed_fit(
                        prior,
                        rows.loc[mask, "observation_id"].tolist(),
                        rows.loc[valid, "observation_id"].tolist(),
                        arm,
                        seed,
                        inputs,
                    )
                    shutil.copytree(prior, dest)
                    reuse_events.append(
                        dict(
                            source=str(prior),
                            destination=str(dest),
                            source_manifest_sha256=file_sha256(prior / "manifest.json"),
                        )
                    )
                    write_json(output / "reused_fits.json", reuse_events)
                    predictions.append(pd.read_csv(dest / "predictions.csv.gz"))
                    print(json.dumps(dict(reused_fit=str(prior))), flush=True)
                    continue
                dest.mkdir(parents=True)
                state = fit(config, rows, features, mask, arm, seed, dest)
                subset = rows.loc[valid].copy()
                species, strains = predict(
                    config, state, features[subset.sequence_index], subset.sequence.tolist(), arm
                )
                values = species[np.arange(len(subset)), subset.species_index.to_numpy(int)].copy()
                known = subset.strain_index.to_numpy(int) >= 0
                values[known] = strains[
                    np.flatnonzero(known), subset.strain_index.to_numpy(int)[known]
                ]
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
                subset["prediction"] = values
                subset["arm"] = name
                subset["seed"] = seed
                subset.to_csv(dest / "predictions.csv.gz", index=False)
                write_json(
                    dest / "fit.json",
                    dict(
                        training_ids=rows.loc[mask, "observation_id"].tolist(),
                        validation_ids=subset.observation_id.tolist(),
                        arm=arm,
                        seed=seed,
                        species_support=state["species_support"].tolist(),
                        strain_support=state["strain_support"].tolist(),
                        serialization_equal=True,
                        scope="smoke" if smoke else "development OOF",
                    ),
                )
                finish(dest, inputs, started)
                predictions.append(subset)
                print(
                    json.dumps(
                        dict(
                            arm=name, seed=seed, fold=int(fold), seconds=time.monotonic() - started
                        )
                    ),
                    flush=True,
                )
                del state, loaded
                torch.cuda.empty_cache()
            pd.concat(predictions).to_csv(output / f"{name}-s{seed}-oof.csv.gz", index=False)
    return inputs


def report(prepared: Path, trained: Path, output: Path) -> dict[str, str]:
    inputs = checked(prepared)
    inputs.update(checked(trained))
    comparisons = []
    paired = []
    component_metrics = []
    for seed in [42, 43, 44]:
        frames = {
            name: pd.read_csv(trained / f"{name}-s{seed}-oof.csv.gz")
            for name in ["original", "corrected"]
        }
        expected = set(json.loads((prepared / "protocol.json").read_text())["comparison_rows"])
        for name, r in frames.items():
            if r.observation_id.duplicated().any() or set(r.observation_id) != expected:
                raise ValueError("OOF membership differs")
            for level, group in [("all", r), *list(r.groupby("prediction_level"))]:
                metrics = evaluate(group, group.prediction.to_numpy(), np.full(len(group), np.nan))
                comparisons.append(
                    dict(
                        arm=name,
                        seed=seed,
                        scope="development_oof",
                        prediction_level=level,
                        effective_species=int(
                            group.loc[
                                group.exact_regression & group.prediction.notna(), "species"
                            ].nunique()
                        ),
                        **metrics,
                    )
                )
        joined = (
            frames["original"]
            .rename(columns={"prediction": "prediction_original"})
            .merge(
                frames["corrected"][["observation_id", "prediction"]].rename(
                    columns={"prediction": "prediction_corrected"}
                ),
                on="observation_id",
                validate="one_to_one",
            )
        )
        joined.to_csv(output / f"paired-s{seed}.csv.gz", index=False)
        paired.append(dict(seed=seed, **paired_comparison(joined)))
        for component, g in joined.groupby("component_id"):
            component_metrics.append(
                dict(seed=seed, component_id=component, **paired_comparison(g))
            )
    # Equivalent data are explicitly aliased; no fabricated extra fits or independent evidence.
    comparisons.extend(
        {**r, "arm": "corrected_plus_new_training", "equivalent_to": "corrected"}
        for r in list(comparisons)
        if r["arm"] == "corrected"
    )
    for model in [
        "original",
        "corrected",
        "corrected_plus_new_training",
        "saved_linear8",
        "saved_finetune8",
        "APEX",
    ]:
        comparisons.append(
            dict(
                arm=model,
                scope="primary_external",
                rows=0,
                predicted_rows=0,
                status="insufficient",
                reason="frozen external primary collection empty",
                macro_mae=None,
            )
        )
    pd.DataFrame(comparisons).to_csv(output / "comparison.csv", index=False)
    write_json(output / "paired_comparison.json", paired)
    write_jsonl(output / "component_metrics.jsonl", component_metrics)
    write_json(
        output / "evaluation_status.json",
        dict(
            primary_external="not_executed_insufficient",
            expansion_effect="not_estimable_no_new_training_rows",
            delta_ablation="not_run_no_external_pairs",
            external_bootstrap="not_run_zero_components",
            candidate="retain finetune8-w0.25",
            adopted=False,
            development_comparison="original versus corrected; common new paper/homology OOF",
        ),
    )
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["prepare", "smoke", "train", "report"], required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--trained", type=Path)
    parser.add_argument("--reuse-completed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    cfg = json.loads(args.config.read_text())
    fresh_output(
        args.output,
        [
            args.config,
            Path(cfg["external"]),
            Path(cfg["prepared"]),
            *[p for p in [args.prepared, args.trained, args.reuse_completed] if p is not None],
        ],
    )
    capture_execution(args.output)
    if args.stage == "prepare":
        inputs = prepare(cfg, args.output)
    elif args.prepared is None:
        raise ValueError("Prepared dataset required")
    elif args.stage == "report":
        if args.trained is None:
            raise ValueError("Completed training required")
        inputs = report(args.prepared, args.trained, args.output)
    else:
        inputs = train(cfg, args.prepared, args.output, args.stage == "smoke", args.reuse_completed)
    inputs[str(args.config)] = file_sha256(args.config)
    write_json(args.output / "config.json", cfg)
    finish(args.output, inputs, started)
    print(json.dumps(dict(stage=args.stage, seconds=time.monotonic() - started)), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        main()
