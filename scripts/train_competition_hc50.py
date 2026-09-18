"""Shared-fold HC50 screening, feature ablations, and measured endpoint diagnostics."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from run_competition_bioaccuracy import (
    BioaccuracyConfig,
    archive_sources,
    checked_manifest,
    finish_stage,
    read_observations,
    write_json,
)
from run_research_models import load_esm
from threadpoolctl import threadpool_limits

from robust_apex_qd.evaluation.developability import evaluate_developability
from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.research.bioaccuracy import chemistry_key, concentration_bounds
from robust_apex_qd.research.biomodels import HC50Bundle, fit_hc50, hc50_signal, predict_hc50


def embeddings(config: BioaccuracyConfig, root: Path, output: Path) -> dict[str, str]:
    inputs = checked_manifest(root / "prepare/manifest.json")
    inputs.update(checked_manifest(config.prior_models / "esm8/manifest.json"))
    inputs.update(checked_manifest(config.prior_models / "prepare/manifest.json"))
    inputs[str(config.esm8_checkpoint)] = file_sha256(config.esm8_checkpoint)
    sequences = json.loads((root / "prepare/endpoint_sequences.json").read_text())
    previous = read_fasta_sequences(config.prior_models / "prepare/sequences.fasta")
    previous_features = np.load(config.prior_models / "esm8/features.npy", allow_pickle=False)
    if previous_features.shape != (len(previous), 320):
        raise ValueError("ESM8 feature shape mismatch")
    index = {s: i for i, s in enumerate(previous)}
    controls = previous[:8]
    missing = [s for s in sequences if s not in index]
    model, alphabet = load_esm(config.esm8_checkpoint)
    model.eval()
    convert = alphabet.get_batch_converter()
    inferred = {}
    for start in range(0, len(missing) + len(controls), 32):
        batch = (controls + missing)[start : start + 32]
        _, _, tokens = convert([(str(i), sequence) for i, sequence in enumerate(batch)])
        with torch.no_grad():
            representations = model(tokens, repr_layers=[6])["representations"][6]
        for i, sequence in enumerate(batch):
            inferred[sequence] = representations[i, 1 : len(sequence) + 1].mean(0).numpy()
    parity = np.array([inferred[s] for s in controls]) - previous_features[: len(controls)]
    if np.max(np.abs(parity)) > 2e-5:
        raise ValueError("ESM8 reuse failed CPU/source feature parity")
    values = np.array(
        [previous_features[index[s]] if s in index else inferred[s] for s in sequences]
    )
    np.save(output / "features.npy", values)
    write_json(output / "sequences.json", sequences)
    write_json(
        output / "embedding_contract.json",
        dict(
            checkpoint_sha256=file_sha256(config.esm8_checkpoint),
            dimension=320,
            pooling="mean residues excluding BOS/EOS/pad",
            device="cpu",
            reused=len(index),
            inferred=len(missing),
            controls=len(controls),
            maximum_parity_error=float(np.max(np.abs(parity))),
            tolerance=2e-5,
        ),
    )
    return inputs


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(y) & np.isfinite(prediction)
    if not finite.any():
        return dict(rows=0, mae=None, spearman=None)
    truth, values = y[finite], prediction[finite]
    correlation = (
        pd.Series(truth).corr(pd.Series(values), method="spearman")
        if len(set(truth)) > 1 and len(set(values)) > 1
        else np.nan
    )
    return dict(
        rows=int(finite.sum()),
        mae=float(np.mean(abs(truth - values))),
        spearman=float(correlation) if np.isfinite(correlation) else None,
    )


def train(config: BioaccuracyConfig, root: Path, output: Path) -> dict[str, str]:
    inputs = checked_manifest(root / "prepare/manifest.json")
    for stage in ["features", "split", "hc50-embeddings"]:
        inputs.update(checked_manifest(root / stage / "manifest.json"))
    observations = read_observations(root / "prepare/hc50_observations.jsonl")
    training_rows = [r for r in observations if r.endpoint == "consensus_hc50"]
    if len({chemistry_key(r) for r in training_rows}) != len(training_rows):
        raise ValueError("HC50 consensus must have one label per molecular profile")
    sequences = json.loads((root / "features/sequences.json").read_text())
    index = {s: i for i, s in enumerate(sequences)}
    selected = np.array([index[r.sequence] for r in training_rows])
    y = np.log2(np.array([r.value_um for r in training_rows], dtype=float))
    splits = json.loads((root / "split/split_manifest.json").read_text())
    groups = np.array([splits["groups"][r.sequence] for r in training_rows])
    outer = np.array([splits["outer"][r.sequence] for r in training_rows])
    if min(outer) < 0:
        raise ValueError("Insufficient common endpoint groups")
    arms = json.loads((root / "features/feature_contracts.json").read_text())
    arms.append(
        dict(id="esm8", sha256=file_sha256(root / "hc50-embeddings/embedding_contract.json"))
    )
    arms.append(dict(id="median", sha256="median-constant-v1"))
    summaries, all_oof, measured = [], [], []
    for arm in arms:
        name = arm["id"]
        features = np.load(
            root / "hc50-embeddings/features.npy"
            if name == "esm8"
            else root / "features/standard-local0-interactions0.npy"
            if name == "median"
            else root / "features" / f"{name}.npy",
            allow_pickle=False,
        )
        x = features[selected]
        prediction = np.full(len(y), np.nan)
        fold_scores = []
        for fold in sorted(set(outer)):
            training, validation = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
            inner = np.array(
                [splits["inner"][str(fold)].get(r.sequence, -1) for r in training_rows]
            )
            choices = []
            for alpha in [1.0] if name == "median" else config.ridge_alphas:
                inner_p = np.full(len(y), np.nan)
                for inside in sorted(set(inner[training])):
                    tr = training[inner[training] != inside]
                    vi = training[inner[training] == inside]
                    model = fit_hc50(
                        x,
                        y,
                        groups,
                        tr,
                        vi,
                        alpha=alpha,
                        feature_sha256=arm["sha256"],
                        median=name == "median",
                    )
                    inner_p[vi] = predict_hc50(model, x[vi], arm["sha256"])
                choices.append(
                    dict(alpha=alpha, **regression_metrics(y[training], inner_p[training]))
                )
            best = min(choices, key=lambda r: (r["mae"], r["alpha"]))
            model = fit_hc50(
                x,
                y,
                groups,
                training,
                validation,
                alpha=best["alpha"],
                feature_sha256=arm["sha256"],
                median=name == "median",
            )
            destination = output / name / f"fold{fold}"
            destination.mkdir(parents=True)
            (destination / "weights.json").write_text(model.model_dump_json(indent=2) + "\n")
            loaded = HC50Bundle.model_validate_json((destination / "weights.json").read_text())
            p = predict_hc50(model, x[validation], arm["sha256"])
            np.testing.assert_equal(p, predict_hc50(loaded, x[validation], arm["sha256"]))
            prediction[validation] = p
            write_json(
                destination / "audit.json",
                dict(
                    train_ids=[training_rows[i].observation_id for i in training],
                    validation_ids=[training_rows[i].observation_id for i in validation],
                    shared_split_sha256=file_sha256(root / "split/split_manifest.json"),
                    inner_scores=choices,
                    selected_alpha=best["alpha"],
                    reload_equal=True,
                ),
            )
            fold_scores.append(dict(fold=int(fold), **regression_metrics(y[validation], p)))
            for row in observations:
                if row.endpoint != "measured_hc50" or splits["outer"][row.sequence] != fold:
                    continue
                value = float(
                    predict_hc50(model, features[[index[row.sequence]]], arm["sha256"])[0]
                )
                bounds = concentration_bounds(row)
                if bounds is None:
                    continue
                low, high, _lc, _hc = bounds
                measured.append(
                    dict(
                        model=name,
                        observation_id=row.observation_id,
                        molecule_id=chemistry_key(row),
                        fold=int(fold),
                        rbc_species=row.rbc_species,
                        relation=row.relation,
                        value_um=row.value_um,
                        prediction_log2_um=value,
                        bound_violation=max(np.log2(low) - value, 0) if low else 0,
                        upper_violation=max(value - np.log2(high), 0) if np.isfinite(high) else 0,
                    )
                )
        summary = dict(model=name, **regression_metrics(y, prediction))
        summaries.append(summary)
        write_json(output / name / "fold_scores.json", fold_scores)
        for i, row in enumerate(training_rows):
            all_oof.append(
                dict(
                    model=name,
                    observation_id=row.observation_id,
                    molecule_id=chemistry_key(row),
                    sequence=row.sequence,
                    homology_group=groups[i],
                    fold=int(outer[i]),
                    endpoint=row.endpoint,
                    observed_log2_um=float(y[i]),
                    prediction_log2_um=float(prediction[i]),
                )
            )
        print(json.dumps(summary), flush=True)
    pd.DataFrame(all_oof).to_csv(output / "hc50_oof.csv", index=False)
    pd.DataFrame(measured).to_csv(output / "measured_hc50_predictions.csv", index=False)
    comparisons = pd.DataFrame(summaries)
    comparisons.to_csv(output / "hc50_comparison.csv", index=False)
    baseline = float(comparisons.loc[comparisons.model == "median", "mae"].iloc[0])
    signals = {
        str(r["model"]): hc50_signal(mae=r["mae"], spearman=r["spearman"], median_mae=baseline)
        for r in comparisons.to_dict("records")
        if r["model"] != "median"
    }
    eligible = comparisons[
        comparisons.model.isin(
            [name for name, signal in signals.items() if signal["scenario_candidate"]]
        )
    ].sort_values(["mae", "model"])
    write_json(
        output / "screen_decision.json",
        dict(
            shortlisted_models=eligible.model.tolist(),
            signals=signals,
            selection_usable=bool(len(eligible)),
            median_mae=baseline,
            independence="shared-fold development OOF, consensus labels",
            next="joint selection must choose the HC50 model within each inner training cohort",
            seeds="ridge and median fits deterministic; no stochastic repetitions claimed",
            raw_hc50="separate transferred-endpoint assessment; not pooled into consensus MAE",
        ),
    )
    return inputs


def filters(config: BioaccuracyConfig, root: Path, output: Path) -> dict[str, str]:
    inputs = checked_manifest(root / "prepare/manifest.json")
    inputs.update(checked_manifest(root / "joint/manifest.json"))
    inputs[str(Path("src/robust_apex_qd/evaluation/developability.py"))] = file_sha256(
        Path("src/robust_apex_qd/evaluation/developability.py")
    )
    labels = pd.read_json(root / "prepare/peptide_target_labels.jsonl", lines=True)
    sequences = sorted(labels.sequence.unique())
    reasons = {s: set(evaluate_developability(s).hard_filter_reasons) for s in sequences}
    rules = sorted(set.union(set(), *reasons.values()))
    rows = []
    for removed in ["none", "all", *rules]:
        keep = {s: not (r - {removed}) if removed != "all" else True for s, r in reasons.items()}
        frame = labels.assign(keep=labels.sequence.map(keep))
        for species, cohort in frame.groupby("species"):
            positive = cohort[cohort.hit_lower == 1]
            rows.append(
                dict(
                    removed_rule=removed,
                    species=species,
                    molecules=len(cohort),
                    retained=int(cohort.keep.sum()),
                    confirmed_active=len(positive),
                    active_retained=int(positive.keep.sum()),
                    active_retention=float(positive.keep.mean()) if len(positive) else None,
                )
            )
    pd.DataFrame(rows).to_csv(output / "filter_retention.csv", index=False)
    pd.DataFrame(
        [dict(sequence=s, reasons=json.dumps(sorted(r))) for s, r in reasons.items()]
    ).to_csv(output / "filter_reasons.csv", index=False)
    write_json(
        output / "ablation_registry.json",
        dict(
            rules=rules,
            primary_ratio=config.primary_ratio,
            scope="one-rule removal on measured MIC molecules; not selector adoption",
            joint_labels=str(root / "joint/molecular_joint_labels.csv"),
        ),
    )
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/competition_bioaccuracy.json"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["embeddings", "train", "filters"], required=True)
    args = parser.parse_args()
    config = BioaccuracyConfig.model_validate_json(args.config.read_text())
    output = args.root / f"hc50-{args.stage}"
    fresh_output(output, [config.prior_models, config.mic_prepare, config.frozen_pool])
    inputs = archive_sources(
        output,
        [
            args.config,
            Path(__file__),
            Path("scripts/run_competition_bioaccuracy.py"),
            Path("scripts/run_research_models.py"),
            *Path("src/robust_apex_qd/research").glob("bio*.py"),
        ],
    )
    write_json(output / "execution.json", dict(sources_sha256=inputs))
    write_json(output / "protocol.json", config.model_dump(mode="json"))
    start = time.monotonic()
    torch.set_num_threads(config.cpu_threads)
    with threadpool_limits(config.cpu_threads):
        inputs.update(
            dict(embeddings=embeddings, train=train, filters=filters)[args.stage](
                config, args.root, output
            )
        )
    finish_stage(output, inputs, start)


if __name__ == "__main__":
    main()
