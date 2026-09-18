"""Run resumable, training-only MIC predictor comparisons in isolated stage directories."""

import argparse
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import esm
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS, load_prediction_archive
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.features.physchem import compute_features, fit_reference, score_features
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.research.data import conservative_identity
from robust_apex_qd.research.models import (
    evaluate_predictions,
    fold_masks,
    paired_interval,
    ridge_fold,
    select_candidate,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(config: dict[str, Any], output: Path, config_path: Path) -> None:
    data = Path(config["dataset"])
    fold_directory = Path(config["fold_directory"])
    manifest = json.loads((data / "dataset_manifest.json").read_text())
    for name in ["observations_train.jsonl", "sequence_splits.csv"]:
        if file_sha256(data / name) != manifest["artifacts_sha256"][name]:
            raise ValueError(f"Frozen input changed: {name}")
    if file_sha256(data / "sequence_splits.csv") != config["assignment_sha256"]:
        raise ValueError("Outer assignment changed")
    if file_sha256(fold_directory / "development_folds.csv") != config["folds_sha256"]:
        raise ValueError("Inner folds changed")
    rows = pd.read_json(data / "observations_train.jsonl", lines=True)
    if not rows.split.eq("train").all():
        raise ValueError("Only the outer training partition may be opened")
    rows = rows[rows.primary_eligible].copy()
    folds = pd.read_csv(fold_directory / "development_folds.csv")
    rows = rows.merge(folds, on=["sequence", "group"], validate="many_to_one", how="left")
    if (
        rows.validation_fold.isna().any()
        or rows.groupby("group").validation_fold.nunique().max() != 1
    ):
        raise ValueError("Missing or inconsistent inner fold")
    if rows.duplicated(["sequence", "apex_pathogen"]).any():
        raise ValueError(
            "Repeated strain/sequence observations require an explicit aggregation rule"
        )
    counts = rows.groupby("apex_pathogen").group.nunique()
    if set(counts[counts >= 5].index) != set(config["primary_pathogens"]):
        raise ValueError("Primary strain support differs from the locked protocol")
    rows = rows.sort_values("observation_id").reset_index(drop=True)
    rows.to_csv(output / "rows.csv", index=False)
    sequences = sorted(rows.sequence.unique())
    write_fasta(
        [FastaRecord(f"r{i}", s) for i, s in enumerate(sequences)], output / "sequences.fasta"
    )
    references = sorted(
        set(
            s
            for s in read_fasta_sequences(Path(config["reference"]))
            if 8 <= len(s) <= 50 and set(s) <= set("ACDEFGHIKLMNPQRSTVWY")
        )
    )
    reference = fit_reference(references, reference_sha256=file_sha256(Path(config["reference"])))
    write_json(output / "physchem_reference.json", reference.model_dump())
    features = []
    for sequence in sequences:
        props = compute_features(sequence)
        features.append(
            {
                "sequence": sequence,
                **props,
                "physchem_ood": score_features(props, reference).physchem_ood,
                "reference_similarity_bound": max(
                    conservative_identity(sequence, s) for s in references
                ),
                **{f"aac_{a}": sequence.count(a) / len(sequence) for a in "ACDEFGHIKLMNPQRSTVWY"},
            }
        )
    pd.DataFrame(features).to_csv(output / "features.csv", index=False)
    rows.groupby(["apex_pathogen", "validation_fold"]).agg(
        rows=("sequence", "size"), groups=("group", "nunique"), exact=("exact_mic", "sum")
    ).to_csv(output / "support.csv")
    write_json(
        output / "input_manifest.json",
        {
            "config_sha256": file_sha256(config_path),
            "dataset_manifest_sha256": file_sha256(data / "dataset_manifest.json"),
            "train_sha256": file_sha256(data / "observations_train.jsonl"),
            "rows_sha256": file_sha256(output / "rows.csv"),
            "features_sha256": file_sha256(output / "features.csv"),
            "reference_sha256": file_sha256(Path(config["reference"])),
            "folds_sha256": config["folds_sha256"],
            "holdout_opened": False,
        },
    )
    write_json(output / "protocol.json", config)


def load_esm(checkpoint: Path) -> tuple[Any, Any]:
    # Trusted, explicitly chosen local FAIR checkpoint; loading does not fetch remote code.
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return esm.pretrained.load_model_and_alphabet_core(checkpoint.stem, weights, None)


def encode(config: dict[str, Any], root: Path, output: Path, large: bool) -> None:
    checkpoint = Path(config["esm650_checkpoint" if large else "esm8_checkpoint"]).expanduser()
    model, alphabet = load_esm(checkpoint)
    model.eval().to(config["device"])
    convert = alphabet.get_batch_converter()
    sequences = read_fasta_sequences(root / "prepare/sequences.fasta")
    features = []
    first_seconds = None
    for start in range(0, len(sequences), config["batch_size"]):
        batch = sequences[start : start + config["batch_size"]]
        _, _, tokens = convert([(str(i), s) for i, s in enumerate(batch)])
        began = time.monotonic()
        with torch.no_grad():
            result = model(tokens.to(config["device"]), repr_layers=[model.num_layers])
        tensor = result["representations"][model.num_layers]
        features.extend(
            tensor[i, 1 : len(s) + 1].mean(0).cpu().numpy() for i, s in enumerate(batch)
        )
        if first_seconds is None:
            first_seconds = time.monotonic() - began
            print(f"Embedding smoke: {len(batch)} sequences, {first_seconds:.2f}s", flush=True)
    np.save(output / "features.npy", np.asarray(features, dtype=np.float32), allow_pickle=False)
    write_json(
        output / "embedding_manifest.json",
        {
            "checkpoint": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "sequences": sequences,
            "dimension": len(features[0]),
            "first_batch_seconds": first_seconds,
            "pooling": "mean residues excluding BOS/EOS/pad",
        },
    )


def neural_fold(
    config: dict[str, Any],
    rows: pd.DataFrame,
    features: np.ndarray,
    strains: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    seed: int,
    path: Path,
    finetune: bool,
) -> np.ndarray:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = config["device"]
    center = float(np.log2(rows.loc[train, "mic_um"]).mean())
    encoder = None
    if finetune:
        encoder, alphabet = load_esm(Path(config["esm8_checkpoint"]).expanduser())
        encoder.to(device).train()
        _, _, tokens = alphabet.get_batch_converter()(
            [(str(i), s) for i, s in enumerate(rows.sequence)]
        )
        inputs = tokens.to(device)
        head = torch.nn.Linear(320, 11).to(device)
        parameters = list(encoder.parameters()) + list(head.parameters())
        epochs, lr = config["finetune_epochs"], config["finetune_lr"]
        scaler = None
    else:
        scaler = StandardScaler().fit(features[train])
        inputs = torch.tensor(scaler.transform(features), dtype=torch.float32, device=device)
        head = torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], config["mlp_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(config["mlp_hidden"], 11),
        ).to(device)
        parameters = list(head.parameters())
        epochs, lr = config["mlp_epochs"], config["mlp_lr"]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=config["weight_decay"])
    targets = torch.tensor(
        np.log2(rows.mic_um.to_numpy()) - center, dtype=torch.float32, device=device
    )
    target_indices = torch.tensor(strains, device=device)
    lengths = torch.tensor(rows.sequence.str.len().to_numpy(), device=device)
    rng = np.random.default_rng(seed)

    def predict(indices: np.ndarray) -> torch.Tensor:
        if encoder is None:
            embeddings = inputs[indices]
        else:
            representation = encoder(inputs[indices], repr_layers=[6])["representations"][6]
            mask = (torch.arange(representation.shape[1], device=device)[None, :] > 0) & (
                torch.arange(representation.shape[1], device=device)[None, :]
                <= lengths[indices, None]
            )
            embeddings = (representation * mask[:, :, None]).sum(1) / lengths[indices, None]
        return head(embeddings).gather(1, target_indices[indices, None]).squeeze(1)

    for _ in range(epochs):
        shuffled = rng.permutation(np.flatnonzero(train))
        for start in range(0, len(shuffled), config["batch_size"]):
            indices = shuffled[start : start + config["batch_size"]]
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(predict(indices), targets[indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
    head.eval()
    if encoder is not None:
        encoder.eval()
    with torch.no_grad():
        predictions = np.concatenate(
            [
                predict(batch).cpu().numpy() + center
                for batch in np.array_split(
                    np.flatnonzero(valid),
                    max(1, math_ceil_count(valid.sum(), config["batch_size"])),
                )
            ]
        )
    state = {"head": head.state_dict(), "center": center}
    if encoder is not None:
        state["encoder"] = encoder.state_dict()
    torch.save(state, path.with_suffix(".pt"))
    if scaler is not None:
        np.savez(
            path.with_suffix(".npz"), mean=np.asarray(scaler.mean_), scale=np.asarray(scaler.scale_)
        )
    return predictions


def math_ceil_count(count: int, batch_size: int) -> int:
    return (count + batch_size - 1) // batch_size


def train_models(config: dict[str, Any], root: Path, output: Path, name: str) -> None:
    rows = pd.read_csv(root / "prepare/rows.csv")
    strains = np.asarray([APEX_PATHOGENS.index(p) for p in rows.apex_pathogen])
    if name == "physchem":
        feature_table = pd.read_csv(root / "prepare/features.csv").set_index("sequence")
        features = (
            feature_table.loc[rows.sequence]
            .drop(columns=["physchem_ood", "reference_similarity_bound"])
            .to_numpy()
        )
    else:
        stage = "esm650" if name == "linear650" else "esm8"
        sequences = read_fasta_sequences(root / "prepare/sequences.fasta")
        index = {sequence: i for i, sequence in enumerate(sequences)}
        features = np.load(root / stage / "features.npy")[[index[s] for s in rows.sequence]]
    targets = np.log2(rows.mic_um.to_numpy())
    audits = []
    for seed in config["seeds"]:
        predictions = np.full(len(rows), np.nan)
        for fold in sorted(rows.validation_fold.unique()):
            start = time.monotonic()
            train, valid = fold_masks(rows, int(fold))
            path = output / f"seed{seed}-fold{fold}"
            if train.sum() < 2 or not valid.any():
                raise ValueError("Insufficient training or validation observations")
            if name in ("physchem", "linear8", "linear650"):
                values, scaler, model = ridge_fold(
                    features, strains, targets, train, valid, config["ridge_alpha"]
                )
                np.savez(
                    path.with_suffix(".npz"),
                    coef=model.coef_,
                    intercept=model.intercept_,
                    mean=np.asarray(scaler.mean_),
                    scale=np.asarray(scaler.scale_),
                )
            else:
                values = neural_fold(
                    config, rows, features, strains, train, valid, seed, path, name == "finetune8"
                )
            supported = np.isin(strains[valid], np.unique(strains[train]))
            values[~supported] = np.nan
            predictions[valid] = values
            audit = {
                "seed": seed,
                "fold": int(fold),
                "train_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
                "unseen_strain_rows": int((~supported).sum()),
                "train_ids": rows.loc[train, "observation_id"].tolist(),
                "validation_ids": rows.loc[valid, "observation_id"].tolist(),
                "runtime_seconds": time.monotonic() - start,
                "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else 0,
            }
            audits.append(audit)
            print(f"{name} seed={seed} fold={fold} {audit['runtime_seconds']:.2f}s", flush=True)
        result = rows[
            ["observation_id", "sequence", "apex_pathogen", "group", "validation_fold"]
        ].copy()
        result["prediction"] = predictions
        result.to_csv(output / f"oof-seed{seed}.csv", index=False)
    write_json(output / "fold_audit.json", audits)


def report(config: dict[str, Any], root: Path, output: Path) -> None:
    rows = pd.read_csv(root / "prepare/rows.csv")
    archive = load_prediction_archive(root / "apex/predictions.npz")
    sequence_index = {s: i for i, s in enumerate(archive.sequences)}
    tensor = np.asarray(
        [
            archive.mic_u_m[sequence_index[s], :, APEX_PATHOGENS.index(p)]
            for s, p in zip(rows.sequence, rows.apex_pathogen, strict=True)
        ]
    )
    rows["apex_log2"] = np.log2(tensor.mean(axis=1))
    log_tensor = np.log2(tensor)
    rows["disagreement"] = np.median(
        np.abs(log_tensor - np.median(log_tensor, axis=1)[:, None]), axis=1
    )
    rows = rows.merge(
        pd.read_csv(root / "prepare/features.csv"), on="sequence", validate="many_to_one"
    )
    predictions = {"APEX": "apex_log2"}
    seed_metrics = []
    for name in config["models"]:
        values = []
        for seed in config["seeds"]:
            oof = pd.read_csv(root / name / f"oof-seed{seed}.csv")
            if oof.observation_id.tolist() != rows.observation_id.tolist():
                raise ValueError("OOF row identity mismatch")
            values.append(oof.prediction.to_numpy())
            scored = rows.assign(prediction=oof.prediction.to_numpy())
            seed_metrics.append(
                {
                    "model": name,
                    "seed": seed,
                    **evaluate_predictions(scored, "prediction", config["primary_pathogens"]),
                }
            )
        rows[name] = np.mean(values, axis=0)
        rows[f"{name}_apex_mean"] = (rows[name] + rows.apex_log2) / 2
        predictions[name] = name
        predictions[f"{name}_apex_mean"] = f"{name}_apex_mean"
    comparisons, strain_metrics = [], []
    primary = rows[rows.apex_pathogen.isin(config["primary_pathogens"])]
    for name, prediction in predictions.items():
        print(f"Reporting {name}", flush=True)
        metrics = evaluate_predictions(primary, prediction, config["primary_pathogens"])
        intervals = paired_interval(
            primary,
            prediction,
            config["primary_pathogens"],
            config["bootstrap_seed"],
            config["bootstrap_iterations"],
        )
        exact = primary[primary.exact_mic & primary[prediction].notna()]
        target = np.log2(exact.mic_um)
        residual = exact[prediction] - target
        baseline_residual = exact.apex_log2 - target
        overlap = []
        for _pathogen, subset in primary.groupby("apex_pathogen"):
            k = max(1, math_ceil_count(len(subset), 5))
            a = set(subset.sort_values([prediction, "sequence"]).head(k).sequence)
            b = set(subset.sort_values(["apex_log2", "sequence"]).head(k).sequence)
            overlap.append(len(a & b) / len(a | b))
        comparisons.append(
            {
                "model": name,
                **metrics,
                **intervals,
                "error_correlation_apex": residual.corr(baseline_residual)
                if residual.nunique() > 1 and baseline_residual.nunique() > 1
                else None,
                "top20_jaccard_apex": float(np.mean(overlap)),
                "adoption_eligible": False,
                "status": "exploratory_only",
            }
        )
        for pathogen, subset in rows.groupby("apex_pathogen"):
            strain_metrics.append(
                {
                    "model": name,
                    "pathogen": pathogen,
                    "primary": pathogen in config["primary_pathogens"],
                    **evaluate_predictions(subset, prediction, [str(pathogen)]),
                    **paired_interval(
                        subset, prediction, [str(pathogen)], config["bootstrap_seed"], 200
                    ),
                }
            )
    pd.DataFrame(comparisons).to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame(strain_metrics).to_csv(output / "strain_metrics.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(output / "seed_metrics.csv", index=False)
    rows.to_csv(output / "oof_predictions.csv", index=False)
    diagnostics = []
    strata = {
        "strain": rows.apex_pathogen,
        "species": rows.species,
        "length": pd.cut(rows.length, [0, 20, 30, 50], include_lowest=True).astype(str),
        "charge": pd.cut(rows.charge_ph_7_4, [-np.inf, 0, 5, np.inf]).astype(str),
        "hydrophobicity": pd.cut(rows.gravy, [-np.inf, 0, 1, np.inf]).astype(str),
        "physchem_ood": pd.cut(rows.physchem_ood, [-np.inf, 1, 2, np.inf]).astype(str),
        "reference_similarity_bound": pd.cut(
            rows.reference_similarity_bound, [0, 0.4, 0.6, 1], include_lowest=True
        ).astype(str),
        "disagreement": pd.cut(rows.disagreement, [-np.inf, 0.5, 1, np.inf]).astype(str),
    }
    for dimension, bins in strata.items():
        for label in sorted(bins.unique()):
            subset = rows[bins == label]
            exact = subset[subset.exact_mic]
            group_errors = exact.assign(error=(exact.apex_log2 - np.log2(exact.mic_um)).abs())
            groups = sorted(group_errors.group.unique())
            means = []
            rng = np.random.default_rng(config["bootstrap_seed"])
            errors = {
                g: group_errors.loc[group_errors.group == g, "error"].to_numpy() for g in groups
            }
            if len(groups) > 1:
                means = [
                    float(
                        np.concatenate([errors[g] for g in rng.choice(groups, len(groups))]).mean()
                    )
                    for _ in range(config["bootstrap_iterations"])
                ]
            ci = np.quantile(means, [0.025, 0.975]) if means else [None, None]
            diagnostics.append(
                {
                    "dimension": dimension,
                    "stratum": label,
                    **evaluate_predictions(
                        subset, "apex_log2", sorted(subset.apex_pathogen.unique())
                    ),
                    "mae_ci_low": ci[0],
                    "mae_ci_high": ci[1],
                    "interval_status": "group_bootstrap" if means else "insufficient_groups",
                }
            )
    pd.DataFrame(diagnostics).to_csv(output / "apex_diagnostics.csv", index=False)


def decide(config: dict[str, Any], root: Path, output: Path) -> None:
    comparison = root / "report/model_comparison.csv"
    manifest = json.loads((root / "report/run_manifest.json").read_text())
    if file_sha256(comparison) != manifest["artifacts_sha256"]["model_comparison.csv"]:
        raise ValueError("Comparison table changed since reporting")
    table = pd.read_csv(comparison)
    expected = {"APEX", *config["models"], *[f"{name}_apex_mean" for name in config["models"]]}
    if set(table.model) != expected or table.model.duplicated().any():
        raise ValueError("Comparison must contain every registered candidate exactly once")
    candidates = table.to_dict("records")
    best = select_candidate(candidates)
    standalone = select_candidate([r for r in candidates if r["model"] in config["models"]])
    table["exploratory_selected"] = table.model == best["model"]
    table["review_reason"] = [
        "Frozen comparator; training overlap unknown"
        if name == "APEX"
        else "Exploratory finalist; paired interval and independence do not justify adoption"
        if name == best["model"]
        else "Lower registered primary score or worse exact-MIC tie-break; not selected"
        for name in table.model
    ]
    table.to_csv(output / "candidate_review.csv", index=False)
    write_json(
        output / "decision.json",
        {
            "exploratory_candidate": best["model"],
            "best_standalone_candidate": standalone["model"],
            "selection": "All registered standalone/mean candidates; rounded top20 then exact MAE",
            "primary_top20_active": best["macro_top20_active"],
            "paired_delta_interval95": [best["delta_ci_low"], best["delta_ci_high"]],
            "independent_adoption_ready": False,
            "adopted_policy": "B1/L2/C0",
            "reason": "Exploratory OOF selection; checkpoint training overlap remains unknown",
            "comparison_sha256": file_sha256(comparison),
            "holdout_opened": False,
            "supersedes_legacy_report_decision": (root / "report/decision.json").exists(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research_models.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=[
            "prepare",
            "apex",
            "esm8",
            "esm650",
            "physchem",
            "linear8",
            "mlp8",
            "linear650",
            "finetune8",
            "report",
            "decision",
        ],
        required=True,
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.output
    output = root / args.stage
    if args.stage != "prepare":
        original = json.loads((root / "prepare/input_manifest.json").read_text())
        for path, expected in [
            (args.config, original["config_sha256"]),
            (root / "prepare/rows.csv", original["rows_sha256"]),
            (root / "prepare/features.csv", original["features_sha256"]),
        ]:
            if file_sha256(path) != expected:
                raise ValueError(f"Prepared inputs changed: {path}")
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with threadpool_limits(limits=config["cpu_threads"]):
        if args.stage == "prepare":
            prepare(config, output, args.config)
        elif args.stage == "apex":
            subprocess.run(
                [
                    sys.executable,
                    "apex/APEX_predict_ensemble.py",
                    "--input",
                    str(root / "prepare/sequences.fasta"),
                    "--output",
                    str(output / "predictions.npz"),
                    "--device",
                    "cpu",
                    "--batch-size",
                    str(config["batch_size"]),
                ],
                check=True,
            )
        elif args.stage in ("esm8", "esm650"):
            encode(config, root, output, args.stage == "esm650")
        elif args.stage == "report":
            report(config, root, output)
        elif args.stage == "decision":
            decide(config, root, output)
        else:
            train_models(config, root, output, args.stage)
    write_json(
        output / "run_manifest.json",
        {
            "stage": args.stage,
            "runtime_seconds": time.monotonic() - start,
            "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "child_peak_ram_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0,
            "config_sha256": file_sha256(args.config),
            "script_sha256": file_sha256(Path(__file__)),
            "module_sha256": file_sha256(Path("src/robust_apex_qd/research/models.py")),
            "uv_lock_sha256": file_sha256(Path("uv.lock")),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "artifacts_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
        },
    )
    print(f"Completed {args.stage}: {time.monotonic() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
