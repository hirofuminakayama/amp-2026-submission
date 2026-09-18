"""Nested human-RBC HC50 comparison retaining right-censored measurements."""

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
from threadpoolctl import threadpool_limits
from train_competition_hc50 import regression_metrics

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.bioaccuracy import chemistry_key, concentration_bounds
from robust_apex_qd.research.biomodels import (
    HC50Bundle,
    fit_hc50,
    fit_interval_hc50,
    predict_hc50,
)


def bound_metrics(lower: np.ndarray, upper: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if not np.isfinite(prediction).all():
        raise ValueError("Complete measured HC50 prediction coverage required")
    exact = lower == upper
    violation = np.maximum(lower - prediction, 0) + np.maximum(prediction - upper, 0)
    return dict(
        rows=len(prediction),
        exact_rows=int(exact.sum()),
        interval_mae=float(violation.mean()),
        exact=regression_metrics(lower[exact], prediction[exact]),
    )


def fit_arm(
    arm: str,
    features: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    alpha: float,
    feature_hash: str,
) -> HC50Bundle:
    if arm.endswith("interval"):
        return fit_interval_hc50(
            features,
            low,
            high,
            groups,
            train,
            valid,
            alpha=alpha,
            feature_sha256=feature_hash,
            epochs=200,
        )
    exact_train = train[low[train] == high[train]]
    result = fit_hc50(
        features,
        low,
        groups,
        exact_train,
        valid,
        alpha=alpha,
        feature_sha256=feature_hash,
        median=arm == "median",
    )
    return result.model_copy(update={"endpoint": "measured_hc50"})


def train(config: BioaccuracyConfig, root: Path, output: Path) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for stage in ["prepare", "features", "split", "hc50-embeddings"]:
        inputs.update(checked_manifest(root / stage / "manifest.json"))
    rows = [
        r
        for r in read_observations(root / "prepare/hc50_observations.jsonl")
        if r.endpoint == "measured_hc50" and r.rbc_species == "human"
    ]
    sequences = json.loads((root / "features/sequences.json").read_text())
    index = {s: i for i, s in enumerate(sequences)}
    splits = json.loads((root / "split/split_manifest.json").read_text())
    bounds = [concentration_bounds(r) for r in rows]
    if any(b is None for b in bounds):
        raise ValueError("HC50 bounds required")
    with np.errstate(divide="ignore"):
        low = np.array([np.log2(b[0]) for b in bounds if b is not None])
        high = np.array([np.log2(b[1]) for b in bounds if b is not None])
    outer = np.array([splits["outer"][r.sequence] for r in rows])
    groups = np.array([splits["groups"][r.sequence] for r in rows])
    matrices = {
        "standard": np.load(root / "features/standard-local0-interactions0.npy"),
        "esm8": np.load(root / "hc50-embeddings/features.npy"),
    }
    contracts = json.loads((root / "features/feature_contracts.json").read_text())
    hashes = {
        "standard": next(
            c["sha256"] for c in contracts if c["id"] == "standard-local0-interactions0"
        ),
        "esm8": file_sha256(root / "hc50-embeddings/embedding_contract.json"),
    }
    arms = ["median", "standard-exact", "standard-interval", "esm8-exact", "esm8-interval"]
    write_json(
        output / "protocol.json",
        dict(
            endpoint="measured_hc50",
            primary_rbc="human",
            arms=arms,
            alphas=config.ridge_alphas,
            inner_criterion="interval MAE on all human measurements, then alpha",
            weighting="interval: equal homology groups; exact/median: equal exact observations",
            epochs=200,
            seed="deterministic zero initialization/full batch",
            caveat="loss and supervision/weighting differ; not a pure loss-only ablation",
        ),
    )
    summaries, records = [], []
    for arm in arms:
        feature = "esm8" if arm.startswith("esm8") else "standard"
        x = matrices[feature][[index[r.sequence] for r in rows]]
        feature_hash = hashes[feature]
        oof = np.full(len(rows), np.nan)
        for fold in sorted(set(outer)):
            tr, vi = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
            inner = np.array([splits["inner"][str(fold)].get(r.sequence, -1) for r in rows])
            scores = []
            for alpha in [1.0] if arm == "median" else config.ridge_alphas:
                ip = np.full(len(rows), np.nan)
                for inside in sorted(set(inner[tr])):
                    it, iv = tr[inner[tr] != inside], tr[inner[tr] == inside]
                    model = fit_arm(arm, x, low, high, groups, it, iv, alpha, feature_hash)
                    ip[iv] = predict_hc50(model, x[iv], feature_hash)
                    dest = output / arm / f"outer{fold}-inner{inside}-a{alpha:g}.json"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(model.model_dump_json(indent=2) + "\n")
                scores.append(dict(alpha=alpha, **bound_metrics(low[tr], high[tr], ip[tr])))
            best = min(scores, key=lambda r: (r["interval_mae"], r["alpha"]))
            model = fit_arm(arm, x, low, high, groups, tr, vi, best["alpha"], feature_hash)
            oof[vi] = predict_hc50(model, x[vi], feature_hash)
            dest = output / arm / f"outer{fold}-selected.json"
            dest.write_text(model.model_dump_json(indent=2) + "\n")
            loaded = HC50Bundle.model_validate_json(dest.read_text())
            np.testing.assert_array_equal(oof[vi], predict_hc50(loaded, x[vi], feature_hash))
            write_json(
                output / arm / f"outer{fold}-audit.json",
                dict(
                    training_ids=[rows[i].observation_id for i in tr],
                    validation_ids=[rows[i].observation_id for i in vi],
                    inner_scores=scores,
                    selected_alpha=best["alpha"],
                    reload_equal=True,
                    shared_split_sha256=file_sha256(root / "split/split_manifest.json"),
                ),
            )
        summary = dict(arm=arm, **bound_metrics(low, high, oof))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        for i, row in enumerate(rows):
            records.append(
                dict(
                    arm=arm,
                    observation_id=row.observation_id,
                    molecule_id=chemistry_key(row),
                    sequence=row.sequence,
                    fold=int(outer[i]),
                    homology_group=groups[i],
                    value_um=row.value_um,
                    relation=row.relation,
                    rbc_species=row.rbc_species,
                    prediction_log2_um=float(oof[i]),
                )
            )
        # Export all-data deployment bundle separately from the nested assessment.
        alpha = float(
            np.median(
                [
                    json.loads((output / arm / f"outer{fold}-audit.json").read_text())[
                        "selected_alpha"
                    ]
                    for fold in sorted(set(outer))
                ]
            )
        )
        model = fit_arm(
            arm,
            x,
            low,
            high,
            groups,
            np.arange(len(rows)),
            np.array([], dtype=int),
            alpha,
            feature_hash,
        )
        (output / arm / "refit.json").write_text(model.model_dump_json(indent=2) + "\n")
    write_json(output / "comparison.json", summaries)
    pd.DataFrame(records).to_csv(output / "measured_hc50_oof.csv", index=False)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/competition_bioaccuracy.json"))
    args = parser.parse_args()
    config = BioaccuracyConfig.model_validate_json(args.config.read_text())
    output = args.root / "hc50-measured"
    fresh_output(output, [config.prior_models, config.mic_prepare, config.metadata])
    start = time.monotonic()
    sources = [
        args.config,
        Path(__file__),
        Path("scripts/train_competition_hc50.py"),
        Path("scripts/run_competition_bioaccuracy.py"),
        Path("uv.lock"),
        *Path("src/robust_apex_qd/research").glob("bio*.py"),
    ]
    inputs = archive_sources(output, sources)
    torch.set_num_threads(config.cpu_threads)
    with threadpool_limits(config.cpu_threads):
        inputs.update(train(config, args.root, output))
    finish_stage(output, inputs, start)


if __name__ == "__main__":
    main()
