"""Add matched baselines, export selected MIC heads, and inventory unresolved data extensions."""

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
from train_mic_models import evaluate, feature_matrix

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS, load_prediction_archive
from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.mic_data import MICConfig, measured_bounds
from robust_apex_qd.research.mic_models import fit_regressor, predict_regressor
from robust_apex_qd.research.prediction_mic import blend_strain_predictions, prediction_records


def baseline_fold(
    rows: pd.DataFrame,
    features: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    alpha: float,
    output: Path,
) -> np.ndarray:
    train = train & rows.exact_regression.to_numpy(bool)
    if set(rows.loc[train].homology_group) & set(rows.loc[valid].homology_group):
        raise ValueError("Baseline group crossing")
    output.mkdir(parents=True, exist_ok=False)
    arm = dict(id="linear8", family="linear8", alpha=alpha)
    state = fit({}, rows, features, train, arm, 42, output)
    valid_rows = rows.loc[valid]
    species, strains = predict(
        {}, state, features[valid_rows.sequence_index], valid_rows.sequence.tolist(), arm
    )
    loaded = dict(np.load(output / "weights.npz"))
    again = predict(
        {}, loaded, features[valid_rows.sequence_index], valid_rows.sequence.tolist(), arm
    )
    np.testing.assert_equal(species, again[0])
    np.testing.assert_equal(strains, again[1])
    write_json(
        output / "manifest.json",
        dict(
            alpha=alpha,
            training_ids=rows.loc[train].observation_id.tolist(),
            validation_ids=valid_rows.observation_id.tolist(),
            serialization_equal=True,
            artifacts_sha256={"weights.npz": file_sha256(output / "weights.npz")},
        ),
    )
    p = species[np.arange(len(valid_rows)), valid_rows.species_index.to_numpy(int)]
    matched = valid_rows.strain_index.to_numpy(int) >= 0
    p[matched] = strains[np.flatnonzero(matched), valid_rows.strain_index.to_numpy(int)[matched]]
    return p


def baselines(config: MICConfig, prepared: Path, output: Path) -> None:
    started = time.monotonic()
    inputs = checked_manifest(prepared / "manifest.json")
    for name in ["prepare", "esm8"]:
        inputs.update(checked_manifest(Path(config.prior_models) / name / "manifest.json"))
    rows = pd.read_json(prepared / "rows.jsonl", lines=True)
    rows = rows[rows.objective.eq("measured_mic")].reset_index(drop=True)
    x = feature_matrix(config, "esm8-exact")
    splits = json.loads((prepared / "split_manifest.json").read_text())
    prediction = np.full(len(rows), np.nan)
    choices = []
    for fold in sorted(rows.homology_fold.unique()):
        if fold < 0:
            raise ValueError("Insufficient baseline folds")
        outer_train = rows.homology_fold.to_numpy() != fold
        inner = rows.sequence.map(splits["inner"][str(fold)]).fillna(-1).to_numpy(int)
        scores = []
        for alpha in [1.0, 10.0, 100.0]:
            p = np.full(len(rows), np.nan)
            for inside in sorted(set(inner[outer_train])):
                tr, vi = outer_train & (inner != inside), outer_train & (inner == inside)
                p[vi] = baseline_fold(
                    rows,
                    x,
                    tr,
                    vi,
                    alpha,
                    output / "fits" / f"outer{fold}-inner{inside}-a{alpha:g}",
                )
            scores.append(
                (
                    evaluate(
                        rows.loc[outer_train], p[outer_train], np.full(outer_train.sum(), np.nan)
                    )["macro_mae"],
                    alpha,
                )
            )
        alpha = sorted(scores)[0][1]
        prediction[~outer_train] = baseline_fold(
            rows, x, outer_train, ~outer_train, alpha, output / "fits" / f"outer{fold}-selected"
        )
        choices.append(dict(fold=int(fold), alpha=alpha, inner_scores=scores))
    rows["prediction"] = prediction
    rows["model"] = "linear8"
    rows.to_csv(output / "linear8-oof.csv.gz", index=False)
    write_json(output / "selection.json", choices)
    archive = load_prediction_archive(Path(config.apex_development))
    inputs[config.apex_development] = file_sha256(Path(config.apex_development))
    index = {s: i for i, s in enumerate(archive.sequences)}
    apex = np.log2(archive.mic_u_m.mean(1))
    matched = rows.strain_index.to_numpy() >= 0
    apex_p = np.full(len(rows), np.nan)
    apex_p[matched] = [
        apex[index[r.sequence], r.strain_index] for r in rows.loc[matched].itertuples()
    ]
    metrics = []
    for name, values in [("linear8", prediction), ("APEX", apex_p)]:
        for cohort, mask in [("all", np.ones(len(rows), bool)), ("matched_strain", matched)]:
            metrics.append(
                dict(
                    model=name,
                    cohort=cohort,
                    **evaluate(rows.loc[mask], values[mask], np.full(mask.sum(), np.nan)),
                )
            )
    pd.DataFrame(metrics).to_csv(output / "baseline_metrics.csv", index=False)
    finish_stage(output, inputs, started)


def handoff(config: MICConfig, prepared: Path, models: Path, output: Path) -> None:
    started = time.monotonic()
    inputs = checked_manifest(models / "manifest.json")
    inputs.update(checked_manifest(prepared / "manifest.json"))
    comparisons = pd.read_csv(models / "model_comparison.csv")
    # Available-assay OOF does not estimate missing-assay candidate inference performance.
    eligible = comparisons[~comparisons.family.str.contains("assay")]
    selected = sorted(
        set(
            [
                eligible.sort_values(["macro_mae", "family"]).iloc[0].family,
                eligible.sort_values(["macro_top20", "family"], ascending=[False, True])
                .iloc[0]
                .family,
            ]
        )
    )
    rows = pd.read_json(prepared / "rows.jsonl", lines=True)
    rows = rows[rows.objective.eq("measured_mic")].reset_index(drop=True)
    low, high, usable = measured_bounds(rows)
    refits = Path(config.prior_refits)
    for directory in refits.iterdir():
        if directory.is_dir() and (directory / "manifest.json").exists():
            inputs.update(checked_manifest(directory / "manifest.json"))
    sequences = pd.read_csv(refits / "pool_sequences.csv").sequence.tolist()
    archive_path = Path(config.frozen_pool) / "work/apex_predictions.npz"
    archive = load_prediction_archive(archive_path)
    inputs[str(archive_path)] = file_sha256(archive_path)
    index = {s: i for i, s in enumerate(archive.sequences)}
    apex = np.log2(archive.mic_u_m.mean(1))[[index[s] for s in sequences]]
    rankings = []
    for family in selected:
        choices = json.loads((models / f"{family}-selection.json").read_text())
        widths = pd.Series([c["selected_width"] for c in choices]).value_counts()
        width = int(sorted(widths.index, key=lambda w: (-widths[w], w))[0])
        mask = usable & ((low == high) if "exact" in family else True)
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
        x = feature_matrix(config, family)
        heads = np.where(rows.strain_index >= 0, rows.strain_index + 7, rows.species_index).astype(
            int
        )
        bundle = fit_regressor(
            x[rows.loc[mask, "sequence_index"]],
            heads[mask],
            low[mask],
            high[mask],
            settings,
            config.seeds[0],
        )
        dest = output / family
        dest.mkdir()
        torch.save(bundle, dest / "weights.pt")
        checkpoint_hash = file_sha256(dest / "weights.pt")
        feature_name = "esm650" if "650" in family else "esm8"
        px = np.load(refits / f"{feature_name}.npy")
        inputs[str(refits / f"{feature_name}.npy")] = file_sha256(refits / f"{feature_name}.npy")
        if "physchem" in family:
            px = np.column_stack([px, np.load(refits / "physchem.npy")])
        if len(px) != len(sequences):
            raise ValueError("Candidate feature alignment mismatch")
        mean, scale = predict_regressor(bundle, px)
        again = predict_regressor(torch.load(dest / "weights.pt", weights_only=True), px[:32])
        for a, b in zip((mean[:32], scale[:32]), again, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-5, equal_nan=True)
        frame = prediction_records(
            sequences, mean, scale, apex, checkpoint_hash, inputs[str(archive_path)]
        )
        frame.to_csv(dest / "candidate_predictions.csv.gz", index=False)
        np.savez(
            dest / "candidate_predictions.npz", species=mean[:, :7], strain=mean[:, 7:], scale=scale
        )
        write_json(
            dest / "manifest.json",
            dict(
                selected_width=width,
                selection="modal inner-selected width; smaller width on tie",
                training_ids=rows.loc[mask, "observation_id"].tolist(),
                heads=list(APEX_PATHOGENS),
                pool_sequence_sha256=file_sha256(refits / "pool_sequences.csv"),
                feature_sha256={
                    str(refits / f"{feature_name}.npy"): inputs[str(refits / f"{feature_name}.npy")]
                },
                artifacts_sha256={p.name: file_sha256(p) for p in dest.iterdir() if p.is_file()},
            ),
        )
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            blended = blend_strain_predictions(frame, apex, weight, sequences)
            score = np.median(blended, axis=1)
            order = sorted(range(len(sequences)), key=lambda i: (float(score[i]), sequences[i]))
            pd.DataFrame(dict(sequence=np.array(sequences)[order], score=score[order])).to_csv(
                dest / f"ranking-w{weight:g}.csv.gz", index=False
            )
            if weight == 1:
                np.testing.assert_equal(score, np.median(apex, axis=1))
            rankings.append(
                dict(
                    model=family,
                    apex_weight=weight,
                    sequences=len(sequences),
                    supported_strain_rows=int(frame.supported.sum()),
                    rank_contract="median log2 MIC; unconstrained pre-selection ranking",
                )
            )
    pd.DataFrame(rankings).to_csv(output / "ensemble_sweep.csv", index=False)
    write_json(
        output / "handoff.json",
        dict(
            selected=selected,
            adopted=False,
            purpose="predictor bundles and saved-pool rankings for the existing selection pipeline",
            limitation="not a validated submission; no assay-missing OOF for assay arm",
        ),
    )
    finish_stage(output, inputs, started)


def extensions(config: MICConfig, prepared: Path, output: Path) -> None:
    started = time.monotonic()
    inputs = checked_manifest(prepared / "manifest.json")
    rows = pd.read_json(prepared / "observations.jsonl", lines=True)
    measured = rows[rows.objective.eq("measured_mic")]
    verified = measured[measured.assay_publication_verified & measured.study.notna()]
    if len(verified):
        raise ValueError(
            "Verified assay rows require pair curation; empty-pair inventory no longer applies"
        )
    pd.DataFrame(
        columns=pd.Index(
            ["left_observation_id", "right_observation_id", "assay_key", "delta_log2_mic"]
        )
    ).to_csv(output / "pair_manifest.csv", index=False)
    measured.groupby(["target", "species", "target_level"]).agg(
        rows=("sequence", "size"), sequences=("sequence", "nunique")
    ).reset_index().assign(
        accession=None, accession_status="unverified; no genome source registered"
    ).to_csv(output / "target_inventory.csv", index=False)
    write_json(
        output / "extension_feasibility.json",
        dict(
            measured_rows=len(measured),
            verified_assay_rows=0,
            strict_pairs=0,
            publication_split="blocked by absent verified assay-to-study linkage",
            delta_training="not run: strict comparable-assay pairs unavailable",
            genome_pilot="not run: accession mapping and genome acquisition remain unimplemented",
            assays_available=int(measured.medium.notna().sum()),
            distinction="peptide-level candidate articles cannot establish exact assay provenance",
        ),
    )
    finish_stage(output, inputs, started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mic_research.json"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["baselines", "handoff", "extensions"], required=True)
    args = parser.parse_args()
    config = MICConfig.model_validate_json(args.config.read_text())
    if args.stage == "handoff" and args.models is None:
        parser.error("handoff requires --models")
    fresh_output(args.output, [args.prepared, Path(config.prior_refits), Path(config.prior_models)])
    write_json(args.output / "protocol.json", config.model_dump())
    torch.set_num_threads(config.cpu_threads)
    with threadpool_limits(config.cpu_threads):
        if args.stage == "handoff":
            handoff(config, args.prepared, args.models, args.output)
        elif args.stage == "baselines":
            baselines(config, args.prepared, args.output)
        else:
            extensions(config, args.prepared, args.output)


if __name__ == "__main__":
    main()
