"""Refit screened models, score a saved pool and compare aligned ensembles."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from run_competition_models import fit, load_features, metric, predict, training_mask, write_json
from run_research_models import load_esm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS, load_prediction_archive
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.features.physchem import compute_features
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.research.competition_models import supported_blend
from robust_apex_qd.research.exploration import ExplorationConstraints, select_exploration_top
from robust_apex_qd.selection.top import maximum_levenshtein_ratio, maximum_local_similarity


def target_diagnostics(
    rows: pd.DataFrame, values: np.ndarray, unit: str, grouping: str
) -> list[dict[str, Any]]:
    records = []
    for target, group in rows.assign(value=values).groupby(grouping):
        supported = group[np.isfinite(group.value)]
        active = supported[supported.active16.notna()].sort_values(
            ["value", "sequence", "observation_id"]
        )
        exact = supported[supported.exact_regression]
        records.append(
            dict(
                target=target,
                grouping=grouping,
                unit=unit,
                rows=len(group),
                coverage=len(supported),
                sequences=group.sequence.nunique(),
                active_rows=len(active),
                active_fraction=float(active.active16.mean()) if len(active) else None,
                top20_activity=float(
                    active.head(max(1, int(np.ceil(0.2 * len(active))))).active16.mean()
                )
                if len(active)
                else None,
                average_precision=float(
                    average_precision_score(active.active16.astype(int), -active.value)
                )
                if active.active16.nunique() == 2
                else None,
                mae=float(np.abs(exact.value - np.log2(exact.mic_um)).mean())
                if len(exact) and unit == "log2_uM"
                else None,
            )
        )
    return records


def best_models(root: Path) -> list[dict[str, Any]]:
    comparisons = pd.read_csv(root / "model_comparison.csv")
    candidates = comparisons[~comparisons.id.str.startswith("ablation")]
    grouped = candidates.groupby(["family", "id"])[["macro_top20", "mae"]].mean().reset_index()
    names = (
        grouped.sort_values(["macro_top20", "mae", "id"], ascending=[False, True, True])
        .groupby("family")
        .head(1)
        .id
    )
    extra = (
        comparisons[
            comparisons.id.str.startswith("ablation")
            & ~comparisons.id.isin(["ablation-classification", "ablation-exact-fold"])
        ]
        .sort_values(["macro_top20", "mae", "id"], ascending=[False, True, True])
        .head(2)
        .id
    )
    arms = []
    for name in [*names, *extra, "ablation-classification"]:
        arm = json.loads((root / "fits" / f"{name}-s42/fit_manifest.json").read_text())["arm"]
        arm["artifact_key"] = name if name.startswith("ablation") else arm["family"]
        arms.append(arm)
    return arms


def features_for_pool(
    config: dict[str, Any], pool: pd.DataFrame, output: Path, family: str
) -> np.ndarray:
    name = "physchem" if family == "physchem" else "esm650" if family == "linear650" else "esm8"
    path = output / f"{name}.npy"
    if path.exists():
        return np.load(path)
    if name == "physchem":
        features = []
        for s in pool.sequence:
            features.append(
                {
                    **compute_features(s),
                    **{f"aac_{a}": s.count(a) / len(s) for a in "ACDEFGHIKLMNPQRSTVWY"},
                }
            )
        values = pd.DataFrame(features).to_numpy(dtype=np.float32)
    elif name == "esm8":
        frozen = Path("work/phase11-clean-0f2f600/full_run_1/work")
        candidates = pd.read_csv(frozen / "candidates.csv.gz")
        index = {s: i for i, s in enumerate(candidates.sequence)}
        embeddings = np.load(frozen / "candidate_embeddings.npy")
        if len(embeddings) != len(candidates):
            raise ValueError("Saved embedding row alignment differs")
        values = embeddings[[index[s] for s in pool.sequence]]
    else:
        model, alphabet = load_esm(Path(config["esm650_checkpoint"]))
        model.eval().cuda()
        features = []
        sequences = pool.sequence.tolist()
        for start in range(0, len(sequences), 64):
            batch = sequences[start : start + 64]
            _, _, tokens = alphabet.get_batch_converter()(
                [(str(i), s) for i, s in enumerate(batch)]
            )
            with torch.no_grad():
                rep = model(tokens.cuda(), repr_layers=[33])["representations"][33]
            features.extend(
                rep[i, 1 : len(s) + 1].mean(0).cpu().numpy() for i, s in enumerate(batch)
            )
            if start % 1024 == 0:
                print(f"Pool ESM650 {start}/{len(sequences)}", flush=True)
        values = np.asarray(features, dtype=np.float32)
    np.save(path, values)
    return values


def refit(config: dict[str, Any], root: Path, output: Path) -> None:
    pool = pd.read_csv(Path(config["selection"]) / "prepare/pool.csv.gz")
    rows = pd.read_json(root / "prepare/rows.jsonl", lines=True)
    arms = best_models(root)
    write_json(output / "selected_models.json", arms)
    pool[["sequence"]].to_csv(output / "pool_sequences.csv", index=False)
    for arm in arms:
        dest = output / arm["artifact_key"]
        if (dest / "manifest.json").exists():
            audit = json.loads((dest / "manifest.json").read_text())
            for name, digest in audit["artifacts_sha256"].items():
                if file_sha256(dest / name) != digest:
                    raise ValueError("Refit artifact changed")
            continue
        dest.mkdir(exist_ok=False)
        start = time.monotonic()
        state = fit(
            config,
            rows,
            load_features(root, arm["family"]),
            training_mask(rows, arm),
            arm,
            42,
            dest,
        )
        x = features_for_pool(config, pool, output, arm["family"])
        p, s = predict(config, state, x, pool.sequence.tolist(), arm)
        loaded = (
            dict(np.load(dest / "weights.npz"))
            if (dest / "weights.npz").exists()
            else torch.load(dest / "weights.pt", weights_only=False)
        )
        p2, s2 = predict(config, loaded, x[:32], pool.sequence.iloc[:32].tolist(), arm)
        np.testing.assert_allclose(p[:32], p2, atol=1e-5, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(s[:32], s2, atol=1e-5, rtol=1e-5, equal_nan=True)
        np.savez(dest / "candidate_predictions.npz", species=p, strain=s)
        write_json(
            dest / "manifest.json",
            dict(
                arm=arm,
                unit="negative_activity_logit"
                if arm.get("loss") == "classification"
                else "log2_uM",
                strain_support=state["strain_support"].tolist(),
                reload_equal=True,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                dataset_sha256=file_sha256(root / "prepare/rows.jsonl"),
                train_ids=rows.loc[training_mask(rows, arm), "observation_id"].tolist(),
                source_sha256=file_sha256(Path(__file__)),
                seconds=time.monotonic() - start,
                train_pool_exact_overlap=len(
                    set(rows.loc[training_mask(rows, arm), "sequence"]) & set(pool.sequence)
                ),
                artifacts_sha256={p.name: file_sha256(p) for p in dest.iterdir() if p.is_file()},
            ),
        )
        print(f"Refit and pool: {arm['family']}", flush=True)


def report(config: dict[str, Any], root: Path, refits: Path, output: Path) -> None:
    rows = pd.read_json(root / "prepare/rows.jsonl", lines=True)
    archive = load_prediction_archive(Path(config["apex"]))
    ix = {s: i for i, s in enumerate(archive.sequences)}
    apex = np.log2(archive.mic_u_m.mean(1))
    exact = rows.apex_pathogen.notna() & rows.objective.eq("measured_mic")
    cohort = rows.loc[exact].copy().reset_index(drop=True)
    row_ix = [ix[s] for s in cohort.sequence]
    strain_ix = cohort.strain_index.to_numpy(int)
    cohort["apex"] = apex[row_ix, strain_ix]
    sweep = []
    all_oof = []
    diagnostics = []
    for grouping in ["species", "apex_pathogen"]:
        diagnostics.extend(
            dict(model="APEX", cohort="matched_strains", **item)
            for item in target_diagnostics(cohort, cohort.apex.to_numpy(), "log2_uM", grouping)
        )
    for path in sorted((root / "fits").glob("*/fit_manifest.json")):
        manifest = json.loads(path.read_text())
        frame = pd.read_csv(path.parent / "oof.csv").set_index("observation_id")
        name = path.parent.name
        unit = manifest["summary"]["unit"]
        measured = rows[rows.objective.eq("measured_mic")]
        for grouping in ["species", "apex_pathogen"]:
            diagnostics.extend(
                dict(
                    model=name,
                    cohort="matched_strains"
                    if grouping == "apex_pathogen"
                    else "all_seven_species",
                    **item,
                )
                for item in target_diagnostics(
                    measured,
                    frame.loc[measured.observation_id, "prediction"].to_numpy(),
                    unit,
                    grouping,
                )
            )
        values = frame.loc[cohort.observation_id, "prediction"].to_numpy()
        annotated = frame.reset_index().merge(
            rows[["observation_id", "objective"]], on="observation_id", validate="one_to_one"
        )
        all_oof.append(
            annotated.assign(
                model=name,
                unit=unit,
                prediction_target=(
                    "measured MIC head or activity logit; not auxiliary consensus head"
                ),
            )
        )
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0] if unit == "log2_uM" else []:
            blended, support = supported_blend(values, cohort.apex.to_numpy(), weight)
            sweep.append(
                dict(
                    model=name,
                    apex_weight=weight,
                    supported_rows=int(support.sum()),
                    fallback_rows=int((~support).sum()),
                    evaluation="matched_exact_strains",
                    **metric(cohort, blended),
                )
            )
        if unit != "log2_uM":
            classification_metrics = metric(cohort, values)
            classification_metrics["mae"] = None
            sweep.append(
                dict(
                    model=name,
                    apex_weight=None,
                    supported_rows=int(np.isfinite(values).sum()),
                    evaluation="classification_rank_only",
                    **classification_metrics,
                )
            )
    pd.concat(all_oof, ignore_index=True).to_csv(output / "oof_predictions.csv.gz", index=False)
    pd.DataFrame(sweep).to_csv(output / "ensemble_sweep.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "target_diagnostics.csv", index=False)
    # Calibration uses exogenous APEX features, fitted only on other homology groups.
    model_log = np.log2(archive.mic_u_m[row_ix])
    med = np.median(model_log, axis=1)[np.arange(len(cohort)), strain_ix]
    mad = np.median(np.abs(model_log - np.median(model_log, axis=1)[:, None, :]), axis=1)[
        np.arange(len(cohort)), strain_ix
    ]
    raw_vote = (archive.mic_u_m[row_ix] <= 16).mean(1)[np.arange(len(cohort)), strain_ix]
    calibration_rows = []
    calibrations = {}
    mask = cohort.active16.notna().to_numpy()
    y = cohort.loc[mask, "active16"].to_numpy(int)
    for name, x in [
        ("C0", med[:, None]),
        ("C1", med[:, None]),
        ("C2", np.column_stack([med, mad])),
    ]:
        probabilities = np.full(len(cohort), np.nan)
        audit = []
        for fold in sorted(cohort.homology_fold.unique()):
            train = mask & (cohort.homology_fold.to_numpy() != fold)
            valid = mask & (cohort.homology_fold.to_numpy() == fold)
            if name == "C0":
                probabilities[valid] = raw_vote[valid]
            else:
                model = LogisticRegression(max_iter=1000, random_state=42).fit(
                    x[train], cohort.loc[train, "active16"].to_numpy(int)
                )
                probabilities[valid] = model.predict_proba(x[valid])[:, 1]
            audit.append(
                dict(
                    fold=int(fold),
                    train_ids=cohort.loc[train, "observation_id"].tolist(),
                    validation_ids=cohort.loc[valid, "observation_id"].tolist(),
                )
            )
        calibration_rows.append(
            dict(
                calibrator=name,
                brier=float(np.mean((probabilities[mask] - y) ** 2)),
                rows=int(mask.sum()),
            )
        )
        cohort[name] = probabilities
        if name != "C0":
            model = LogisticRegression(max_iter=1000, random_state=42).fit(x[mask], y)
            calibrations[name] = dict(
                coef=model.coef_.tolist(), intercept=model.intercept_.tolist(), folds=audit
            )
    cohort.to_csv(output / "matched_strain_calibration_oof.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "calibration_comparison.csv", index=False)
    write_json(output / "calibration_fits.json", calibrations)
    pool = pd.read_csv(Path(config["selection"]) / "prepare/pool.csv.gz")
    archive = load_prediction_archive(Path(config["frozen_run"]) / "work/apex_predictions.npz")
    mapping = {s: i for i, s in enumerate(archive.sequences)}
    apex = np.log2(archive.mic_u_m[[mapping[s] for s in pool.sequence]].mean(1))
    control_sequences = read_fasta_sequences(
        Path(config["selection"]) / "tops/library-L2/top.fasta"
    )
    control = set(control_sequences)
    library = set(read_fasta_sequences(Path(config["selection"]) / "prepare/libraries/L2.fasta"))
    constraints = ExplorationConstraints.model_validate(
        json.loads(Path("configs/competition_exploration.json").read_text())["constraints"][
            "current"
        ]
    )
    eligible = pool.sequence.isin(library).to_numpy()
    refs = tuple(read_fasta_sequences(Path("data/antibacterial.fasta")))
    known_refs = tuple(read_fasta_sequences(Path("data/training/training.fasta")))
    cache = json.loads((Path(config["selection"]) / "tops/similarity_cache.json").read_text())

    def challenge(sequence: str) -> float:
        if sequence not in cache["challenge"]:
            cache["challenge"][sequence] = maximum_levenshtein_ratio(sequence, refs)
        return cache["challenge"][sequence]

    def known(sequence: str) -> float:
        if sequence not in cache["known"]:
            cache["known"][sequence] = maximum_local_similarity(sequence, known_refs)
        return cache["known"][sequence]

    rank_scores = {}
    changes = []

    def select_score(score: np.ndarray, name: str) -> pd.DataFrame | None:
        frame = pool.assign(score=score).loc[eligible]
        try:
            return select_exploration_top(frame, "score", constraints, 100, challenge, known)
        except ValueError as error:
            if not str(error).startswith("infeasible:"):
                raise
            changes.append(dict(model=name, status="infeasible", reason=str(error)))
            return None

    for arm in json.loads((refits / "selected_models.json").read_text()):
        name = arm["artifact_key"]
        values = np.load(refits / name / "candidate_predictions.npz")["strain"]
        classification = arm.get("loss") == "classification"
        base_probability = (archive.mic_u_m[[mapping[s] for s in pool.sequence]] <= 16).mean(1)
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            if classification:
                probability = 1 / (1 + np.exp(np.clip(values, -50, 50)))
                new, support = supported_blend(probability, base_probability, weight)
                score = new.mean(axis=1)
            else:
                new, support = supported_blend(values, apex, weight)
                score = -np.median(new, axis=1)
            if weight == 0.5:
                rank_scores[name] = pd.Series(score).rank(pct=True).to_numpy()
            top = select_score(score, f"{name}-w{weight}")
            if top is None:
                continue
            top[["sequence"]].assign(rank=np.arange(1, 101)).to_csv(
                output / f"{name}-w{weight:g}-top.csv", index=False
            )
            if weight == 1.0 and not classification and top.sequence.tolist() != control_sequences:
                raise ValueError("APEX-only candidate control no longer reproduces frozen Top")
            changes.append(
                dict(
                    model=name,
                    apex_weight=weight,
                    unit="activity probability" if classification else "log2_uM",
                    changed=100 - len(set(top.sequence) & control),
                    supported_heads=int(support[0].sum()),
                    fallback_heads=int((~support[0]).sum()),
                )
            )
    ranks = rank_scores
    for excluded in [None, *ranks]:
        score = np.mean([v for k, v in ranks.items() if k != excluded], axis=0)
        top = select_score(score, f"rankmean-without-{excluded}")
        if top is None:
            continue
        top[["sequence"]].to_csv(output / f"rankmean-without-{excluded}-top.csv", index=False)
        changes.append(
            dict(
                model=f"rankmean-without-{excluded}", changed=100 - len(set(top.sequence) & control)
            )
        )
    tensor = archive.mic_u_m[[mapping[s] for s in pool.sequence]]
    logs = np.log2(tensor)
    medians = np.median(logs, axis=1)
    deviations = np.median(np.abs(logs - medians[:, None, :]), axis=1)
    for calibrator in ["C0", "C1", "C2"]:
        if calibrator == "C0":
            probabilities = (tensor <= 16).mean(axis=1)
        else:
            calibration = calibrations[calibrator]
            coef = np.asarray(calibration["coef"])[0]
            linear = medians * coef[0] + calibration["intercept"][0]
            if calibrator == "C2":
                linear += deviations * coef[1]
            probabilities = 1 / (1 + np.exp(-np.clip(linear, -50, 50)))
        top = select_score(probabilities.mean(axis=1), f"APEX-{calibrator}")
        if top is None:
            continue
        top[["sequence"]].to_csv(output / f"APEX-{calibrator}-top.csv", index=False)
        changes.append(
            dict(
                model=f"APEX-{calibrator}-mean-strain-probability",
                changed=100 - len(set(top.sequence) & control),
                calibration_scope="pooled exact-strain development; transfer to other strains",
            )
        )
    write_json(output / "similarity_cache.json", cache)
    pd.DataFrame(changes).to_csv(output / "candidate_ensemble_sweep.csv", index=False)
    write_json(
        output / "fit_manifest.json",
        dict(
            development_only=True,
            apex_overlap="unknown",
            unit="log2_uM except explicitly classified rank-only outputs",
            heads=list(APEX_PATHOGENS),
            input_sha256={
                str(p): file_sha256(p)
                for p in [
                    root / "protocol.json",
                    root / "model_comparison.csv",
                    Path(config["apex"]),
                ]
            },
            artifacts_sha256={p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refits", type=Path)
    parser.add_argument("--stage", choices=["refit", "report"], required=True)
    args = parser.parse_args()
    config = json.loads((args.root / "protocol.json").read_text())["config"]
    config.update(
        selection="work/competition_exploration/20260912-a/phase2",
        apex="work/competition_exploration/20260912-b/phase3-apex/predictions.npz",
        frozen_run="work/phase11-clean-0f2f600/full_run_1",
    )
    args.output.mkdir(parents=True, exist_ok=args.stage == "refit")
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        if args.stage == "refit":
            refit(config, args.root, args.output)
        else:
            if args.refits is None:
                raise ValueError("--refits required for report")
            report(config, args.root, args.refits, args.output)


if __name__ == "__main__":
    main()
