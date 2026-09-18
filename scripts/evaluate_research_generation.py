"""Audit completed small generation runs with a frozen, unused selection proxy."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_research_generation import jobs

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.features.physchem import (
    PhyschemReference,
    compute_features,
    score_features,
)
from robust_apex_qd.generation.sampler import LengthPolicy, build_length_quotas
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.research.generation_evaluation import frozen_ridge_predict


def validate_run_protocol(
    name: str, manifest: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    registered = jobs(config)
    if name in registered:
        expected = registered[name]
        if manifest.get("config") != config or manifest.get("job") != expected:
            raise ValueError("Run protocol differs from the frozen experiment")
    else:
        paired_names = {f"hydramp-s{seed}": f"paired-s{seed}" for seed in config["seeds"]}
        if name not in paired_names:
            raise ValueError("Run protocol identity is not registered")
        paired = registered[paired_names[name]]
        expected = {key: paired[key] for key in ["seed", "count", "min_length", "max_length"]}
        expected.update(filter_out=True, n_attempts=1, softmax=True)
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("HydrAMP protocol differs from the frozen paired experiment")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research_generation.json"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    model_root = Path(config["measured_run"])
    model_config = json.loads((model_root / "prepare/protocol.json").read_text())
    strains = model_config["primary_pathogens"]
    columns = (
        pd.read_csv(model_root / "prepare/features.csv", nrows=0)
        .columns.drop(["sequence", "physchem_ood", "reference_similarity_bound"])
        .tolist()
    )
    model_paths = [model_root / f"physchem/seed42-fold{i}.npz" for i in range(5)]
    frozen_manifest = json.loads((model_root / "physchem/run_manifest.json").read_text())
    for path in model_paths:
        if file_sha256(path) != frozen_manifest["artifacts_sha256"][path.name]:
            raise ValueError(f"Frozen predictor changed: {path}")
    states = [{k: v for k, v in np.load(path).items()} for path in model_paths]
    train_features = pd.read_csv(model_root / "prepare/features.csv")[columns].to_numpy()
    reference_path = model_root / "prepare/physchem_reference.json"
    reference = PhyschemReference.model_validate_json(reference_path.read_text())
    known = set(read_fasta_sequences(Path(config["reference"])))
    challenge = set(read_fasta_sequences(Path(config["challenge"])))
    run_paths = sorted(args.root.glob("*/run_manifest.json"))
    run_paths = [p for p in run_paths if "smoke" not in p.parent.name]
    expected_jobs = jobs(config)
    expected_names = (set(expected_jobs) - {"smoke"}) | {
        f"hydramp-s{seed}" for seed in config["seeds"]
    }
    if not args.allow_partial and {p.parent.name for p in run_paths} != expected_names:
        raise ValueError("Completed run identities differ from the registered experiment")
    results = []
    all_rows = []
    checks = []
    input_hashes = {str(p): file_sha256(p) for p in [args.config, reference_path, *model_paths]}
    for extra in [
        model_root / "prepare/features.csv",
        model_root / "prepare/protocol.json",
        model_root / "physchem/run_manifest.json",
        Path("src/robust_apex_qd/research/models.py"),
        Path("src/robust_apex_qd/features/physchem.py"),
        Path(model_config["esm8_checkpoint"]).expanduser(),
    ]:
        input_hashes[str(extra)] = file_sha256(extra)
    baseline = json.loads(Path(config["baseline_manifest"]).read_text())
    for name, expected in baseline["sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Baseline input changed: {name}")
        input_hashes[name] = expected
    for manifest_path in run_paths:
        path = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest["artifacts_sha256"].items():
            if file_sha256(path / name) != expected:
                raise ValueError(f"Output changed: {path / name}")
        for name, expected in manifest["input_sha256"].items():
            if file_sha256(Path(name)) != expected:
                raise ValueError(f"Generation input changed: {name}")
            input_hashes[name] = expected
        sequences = read_fasta_sequences(path / "raw.fasta")
        job = validate_run_protocol(path.name, manifest, config)
        if len(sequences) != job["count"]:
            raise ValueError("Generated count differs from registered count")
        valid = [
            s
            for s in sequences
            if job["min_length"] <= len(s) <= job["max_length"]
            and set(s) <= set("ACDEFGHIKLMNPQRSTVWY")
        ]
        if "quotas" in manifest and Counter(map(len, sequences)) != {
            int(k): v for k, v in manifest["quotas"].items()
        }:
            raise ValueError("Generated lengths differ from quotas")
        props = [compute_features(s) for s in valid]
        feature_rows = [
            {**p, **{f"aac_{a}": s.count(a) / len(s) for a in "ACDEFGHIKLMNPQRSTVWY"}}
            for s, p in zip(valid, props, strict=True)
        ]
        features = pd.DataFrame(feature_rows)[columns].to_numpy()
        predictions = np.stack(
            [
                np.mean(
                    [
                        frozen_ridge_predict(features, APEX_PATHOGENS.index(strain), state)
                        for state in states
                    ],
                    axis=0,
                )
                for strain in strains
            ],
            axis=1,
        )
        table = pd.DataFrame(
            {
                "run": path.name,
                "sequence": valid,
                "length": list(map(len, valid)),
                "physchem_ood": [score_features(p, reference).physchem_ood for p in props],
                "proxy_active16_fraction": (predictions <= 4).mean(axis=1),
                "proxy_mean_log2_mic": predictions.mean(axis=1),
                "features_outside_measured_train_range": (
                    (features < train_features.min(axis=0))
                    | (features > train_features.max(axis=0))
                ).mean(axis=1),
            }
        )
        for i, strain in enumerate(strains):
            table[f"log2_mic_{strain}"] = predictions[:, i]
        table.to_csv(args.output / f"{path.name}-predictions.csv.gz", index=False)
        all_rows.append(table)
        results.append(
            {
                "run": path.name,
                "method": "HydrAMP" if "hydramp" in path.name else "AMP-Diffusion",
                "seed": job["seed"],
                "steps": job.get("steps"),
                "temperature": job.get("temperature"),
                "count": len(sequences),
                "requested_count": job["count"],
                "min_length": job["min_length"],
                "max_length": job["max_length"],
                "valid_rate": len(valid) / len(sequences),
                "duplicate_rate": 1 - len(set(valid)) / len(valid),
                "known_exact_rate": sum(s in known for s in valid) / len(valid),
                "challenge_exact_rate": sum(s in challenge for s in valid) / len(valid),
                "mean_length": table.length.mean(),
                "mean_charge": np.mean([p["charge_ph_7_4"] for p in props]),
                "mean_gravy": np.mean([p["gravy"] for p in props]),
                "mean_entropy": np.mean([p["shannon_entropy"] for p in props]),
                "mean_physchem_ood": table.physchem_ood.mean(),
                "proxy_active16_fraction": table.proxy_active16_fraction.mean(),
                "proxy_mean_log2_mic": table.proxy_mean_log2_mic.mean(),
                "mean_features_outside_train_range": (
                    table.features_outside_measured_train_range.mean()
                ),
                "runtime_seconds": manifest["runtime_seconds"],
                "peak_ram_bytes": manifest["peak_ram_bytes"],
                "peak_cuda_bytes": manifest["peak_cuda_bytes"],
                "emitted_count": manifest.get("raw_emitted_count", len(sequences)),
                "emitted_length_acceptance": len(valid)
                / manifest.get("raw_emitted_count", len(sequences)),
                "sampling_denominator": "classifier-filtered; decoder denominator unknown"
                if "hydramp" in path.name
                else "raw decoder samples",
                "status": "small_run_complete",
                "scale": False,
                "reason": "Independent measured activity evidence remains insufficient",
            }
        )
        checks.append(
            {
                "run": path.name,
                "manifest_sha256": file_sha256(manifest_path),
                "verified_artifacts": len(manifest["artifacts_sha256"]),
                "count": len(sequences),
                "requested_count": job["count"],
                "quota_check": "quotas" in manifest,
            }
        )
    combined = pd.concat(all_rows, ignore_index=True)
    combined.groupby(["run", "length"]).agg(
        count=("sequence", "size"),
        proxy_active16_fraction=("proxy_active16_fraction", "mean"),
        proxy_mean_log2_mic=("proxy_mean_log2_mic", "mean"),
    ).to_csv(args.output / "length_stratified.csv")
    # Common-length weighting prevents length mixtures from masquerading as activity effects.
    matched = []
    groups = (
        [
            sorted(t.run.unique())
            for t in [
                combined[combined.run.str.startswith("length-")],
                combined[combined.run.str.startswith("steps-")],
                combined[combined.run.str.startswith(("paired-", "hydramp-"))],
            ]
        ]
        if len(combined)
        else []
    )
    for group_id, names in enumerate(groups):
        if len(names) < 2:
            continue
        common = set.intersection(*[set(combined[combined.run == n].length) for n in names])
        weights = Counter(len(s) for s in known if len(s) in common)
        total = sum(weights.values())
        for name in names:
            table = combined[combined.run == name].groupby("length").mean(numeric_only=True)
            matched.append(
                {
                    "comparison_group": group_id,
                    "run": name,
                    "common_lengths": ",".join(map(str, sorted(common))),
                    "reference_count": total,
                    "sample_common_length_fraction": float(
                        combined.loc[combined.run == name, "length"].isin(common).mean()
                    ),
                    "proxy_active16_fraction": sum(
                        weights[k] * table.loc[k, "proxy_active16_fraction"] for k in common
                    )
                    / total,
                    "proxy_mean_log2_mic": sum(
                        weights[k] * table.loc[k, "proxy_mean_log2_mic"] for k in common
                    )
                    / total,
                }
            )
    pd.DataFrame(matched).to_csv(args.output / "length_standardized.csv", index=False)
    overlaps = []
    names = sorted(combined.run.unique())
    for i, left in enumerate(names):
        a = set(combined.loc[combined.run == left, "sequence"])
        for right in names[i + 1 :]:
            b = set(combined.loc[combined.run == right, "sequence"])
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "intersection": len(a & b),
                    "jaccard": len(a & b) / len(a | b),
                }
            )
    pd.DataFrame(overlaps).to_csv(args.output / "sequence_overlap.csv", index=False)
    assets_path = args.root / "asset_review.json"
    if assets_path.exists():
        for row in json.loads(assets_path.read_text()):
            results.append(row)
        input_hashes[str(assets_path)] = file_sha256(assets_path)
    results.append(
        {
            "run": "pool-120000",
            "method": "AMP-Diffusion",
            "count": 0,
            "requested_count": config["pool_candidate_count"],
            "status": "not_scaled",
            "scale": False,
            "reason": "No independent activity evidence to justify full-pool expansion",
        }
    )
    pd.DataFrame(results).to_csv(args.output / "generation_comparison.csv", index=False)
    # Use only 1000-step control batch measurements to project generation, excluding evaluation.
    batches = []
    for p in run_paths:
        m = json.loads(p.read_text())
        if m.get("job", {}).get("steps") == 1000:
            batches.extend(m["batches"])
    resources: dict[str, Any] = {
        "full_run_executed": False,
        "evaluation_runtime_included": False,
        "prefix_subset_assumed": False,
    }
    if batches:
        x = np.array([[1, b["count"]] for b in batches])
        y = np.array([b["seconds"] for b in batches])
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        for size in [60000, 120000]:
            quotas = build_length_quotas(
                size, LengthPolicy.EMPIRICAL_TEMPERED, 10, 40, Path(config["reference"]), 0.75
            )
            batch_sizes = [
                min(128, q - start) for q in quotas.values() for start in range(0, q, 128)
            ]
            resources[str(size)] = {
                "batches": len(batch_sizes),
                "projected_generation_seconds": float(
                    sum(max(0, coef[0] + coef[1] * n) for n in batch_sizes)
                ),
                "quotas": quotas,
            }
        resources["fit_batch_count"] = len(batches)
        resources["seconds_intercept_and_per_sequence"] = coef.tolist()
        resources["warning"] = "Small-run extrapolation, not end-to-end runtime guarantee"
    (args.output / "resource_projection.json").write_text(json.dumps(resources, indent=2) + "\n")
    decision = {
        "adopted_policy": "B1/L2/C0",
        "scaled_candidates": [],
        "independent_activity_evidence": False,
        "holdout_opened": False,
        "partial": args.allow_partial,
        "exploration_closed": not args.allow_partial,
        "reason": "Unused objective proxy is exploratory, not an independent measured evaluation",
    }
    (args.output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    input_hashes[str(Path(__file__))] = file_sha256(Path(__file__))
    input_hashes["src/robust_apex_qd/research/generation_evaluation.py"] = file_sha256(
        Path("src/robust_apex_qd/research/generation_evaluation.py")
    )
    manifest = {
        "runs": checks,
        "input_sha256": input_hashes,
        "artifacts_sha256": {p.name: file_sha256(p) for p in args.output.iterdir()},
    }
    (args.output / "verification.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(results)[["run", "count", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
