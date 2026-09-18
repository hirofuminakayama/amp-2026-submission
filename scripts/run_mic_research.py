"""Audit prior MIC models and run isolated, versioned quantitative MIC experiments."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from robust_apex_qd.apex.ensemble import load_prediction_archive
from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.research.mic_data import (
    MICConfig,
    fold_assignments,
    global_identity,
    measured_bounds,
    normalized_observations,
    similarity_groups,
)
from robust_apex_qd.research.mic_models import mic_metrics


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def checked_manifest(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text())
    expected = {str(path.parent / k): v for k, v in manifest["artifacts_sha256"].items()}
    verify_hashes(expected)
    return {str(path): file_sha256(path), **expected}


def finish_stage(output: Path, inputs: dict[str, str], started: float, **extra: Any) -> None:
    write_json(
        output / "manifest.json",
        dict(
            inputs_sha256=inputs,
            seconds=time.monotonic() - started,
            **extra,
            artifacts_sha256={
                p.name: file_sha256(p)
                for p in sorted(output.iterdir())
                if p.is_file() and p.name != "manifest.json"
            },
        ),
    )


def audit(config: MICConfig, output: Path) -> None:
    started = time.monotonic()
    prior, report, refits = (
        Path(config.prior_models),
        Path(config.prior_report),
        Path(config.prior_refits),
    )
    inputs = {}
    for path in [prior / "prepare/manifest.json", report / "fit_manifest.json"]:
        inputs.update(checked_manifest(path))
    arms = json.loads((refits / "selected_models.json").read_text())
    chosen = [a for a in arms if a["artifact_key"] in {"linear8", "finetune8", "ablation-interval"}]
    for arm in chosen:
        inputs.update(checked_manifest(refits / arm["artifact_key"] / "manifest.json"))
        inputs.update(checked_manifest(prior / "fits" / f"{arm['id']}-s42/fit_manifest.json"))
    rows = pd.read_json(prior / "prepare/rows.jsonl", lines=True)
    rows = rows[rows.objective.eq("measured_mic")].copy()
    oof = pd.read_csv(report / "oof_predictions.csv.gz")
    values = {}
    for arm in chosen:
        name = f"{arm['id']}-s42"
        subset = oof[oof.model.eq(name) & oof.unit.eq("log2_uM")].set_index("observation_id")
        values[name] = subset.reindex(rows.observation_id).prediction.to_numpy()
    archive = load_prediction_archive(Path(config.apex_development))
    inputs[config.apex_development] = file_sha256(Path(config.apex_development))
    index = {s: i for i, s in enumerate(archive.sequences)}
    apex = np.log2(archive.mic_u_m.mean(axis=1))
    values["APEX"] = np.array(
        [
            apex[index[r.sequence], int(r.strain_index)] if r.strain_index >= 0 else np.nan
            for r in rows.itertuples()
        ]
    )
    common = np.all(np.isfinite(np.array(list(values.values()))), axis=0)
    slices = []
    rows["length_band"] = pd.cut(rows.sequence.str.len(), [0, 15, 25, 50]).astype(str)
    rows["mic_band"] = pd.cut(rows.mic_um, [0, 10, 32, np.inf], right=False).astype(str)
    rows["study_status"] = np.where(rows.study.notna(), "reported", "missing")
    for name, prediction in values.items():
        for cohort, mask in [
            ("all_supported", np.ones(len(rows), bool)),
            ("common_strain", common),
        ]:
            frame = rows.loc[mask].assign(prediction=prediction[mask])
            slices.append(
                dict(
                    model=name,
                    cohort=cohort,
                    grouping="all",
                    target="all",
                    **mic_metrics(frame, frame.prediction.to_numpy()),
                )
            )
            for column in [
                "species",
                "apex_pathogen",
                "length_band",
                "mic_band",
                "chemical_form",
                "source",
                "study_status",
                "relation",
            ]:
                for key, group in frame.groupby(column, dropna=False, observed=True):
                    slices.append(
                        dict(
                            model=name,
                            cohort=cohort,
                            grouping=column,
                            target=str(key),
                            **mic_metrics(group, group.prediction.to_numpy()),
                        )
                    )
    pd.DataFrame(slices).to_csv(output / "error_slices.csv", index=False)
    pd.DataFrame(values, index=rows.observation_id).to_csv(output / "aligned_predictions.csv.gz")
    pool_seq = pd.read_csv(refits / "pool_sequences.csv").sequence.tolist()
    frozen = load_prediction_archive(Path(config.frozen_pool) / "work/apex_predictions.npz")
    index = {s: i for i, s in enumerate(frozen.sequences)}
    pool_apex = np.log2(frozen.mic_u_m.mean(axis=1))[[index[s] for s in pool_seq]]
    scores = {"B1": np.median(pool_apex, axis=1)}
    for arm in chosen:
        new = np.load(refits / arm["artifact_key"] / "candidate_predictions.npz")["strain"]
        supported = np.isfinite(new)
        p = np.where(supported, new, pool_apex)
        scores[arm["artifact_key"]] = np.median(p, axis=1)
    order = {
        name: sorted(range(len(pool_seq)), key=lambda i: (float(score[i]), pool_seq[i]))[:100]
        for name, score in scores.items()
    }
    pd.DataFrame(
        [
            dict(
                model=k,
                top100_overlap_with_B1=len(set(v) & set(order["B1"])),
                pool_rows=len(pool_seq),
                constraints="unconstrained diagnostic; not a submission",
                safety="not recomputed; existing validated Tops remain separate",
            )
            for k, v in order.items()
        ]
    ).to_csv(output / "fixed_pool_rank_changes.csv", index=False)
    write_json(
        output / "baseline_audit.json",
        dict(
            models=list(values),
            measured_rows=len(rows),
            common_rows=int(common.sum()),
            study_known=int(rows.study.notna().sum()),
            medium_known=int(rows.medium.notna().sum()),
            apex_training_overlap="unknown",
            split="prior 80% development OOF",
            prediction_units="log2_uM",
            verified_files=len(inputs),
        ),
    )
    summary = pd.DataFrame(slices)
    table = summary[(summary.grouping == "all") & (summary.cohort == "common_strain")][
        ["model", "exact_rows", "mae", "within1"]
    ]
    (output / "bottleneck_report.md").write_text(
        "# MIC bottleneck diagnostic\n\n"
        "Measured MIC only; prior development OOF, not independent validation.\n\n"
        + table.to_csv(index=False)
        + "\nNo verified assay-to-study links are present. "
        "Assay effects and strain mismatch remain confounded.\n"
        + "APEX is compared on matched strains only; unsupported targets are not invented.\n"
        + "Fixed-pool rank changes measure sensitivity, not wet-lab improvement. "
        "Next: OOD60 and censored likelihood.\n"
    )
    finish_stage(output, inputs, started)


def prepare(config: MICConfig, output: Path) -> None:
    started = time.monotonic()
    prior = Path(config.prior_models)
    inputs = checked_manifest(prior / "prepare/manifest.json")
    for feature in ["esm8", "esm650"]:
        inputs.update(checked_manifest(prior / feature / "manifest.json"))
    inputs.update(checked_manifest(Path(config.development) / "split_manifest.json"))
    rows = pd.read_json(prior / "prepare/rows.jsonl", lines=True)
    records = normalized_observations(rows)
    pd.DataFrame(records).to_json(output / "observations.jsonl", orient="records", lines=True)
    measured_bounds(rows)
    sequences = read_fasta_sequences(prior / "prepare/sequences.fasta")
    if sequences != sorted(rows.sequence.unique()):
        raise ValueError("Feature sequence order differs")
    (output / "sequences.json").write_text(json.dumps(sequences) + "\n")
    identities = np.eye(len(sequences), dtype=np.float32)
    for i, left in enumerate(sequences):
        for j in range(i):
            identities[i, j] = identities[j, i] = global_identity(left, sequences[j])
        if i % 250 == 0:
            print(f"Identity {i}/{len(sequences)}", flush=True)
    np.save(output / "identity.npy", identities)
    groups = similarity_groups(sequences, identities, config.identity_threshold)
    outer = fold_assignments(groups, config.outer_folds, config.seeds[0])
    rows["prior_homology_fold"] = rows.homology_fold
    rows["homology_group"] = rows.sequence.map(groups)
    rows["homology_fold"] = rows.sequence.map(outer)
    rows.to_json(output / "rows.jsonl", orient="records", lines=True)
    folds = np.array([outer[s] for s in sequences])
    maxima = []
    for i, sequence in enumerate(sequences):
        other = folds != folds[i]
        maximum = float(identities[i, other].max()) if other.any() else None
        if maximum is not None and maximum > config.identity_threshold + 1e-7:
            raise ValueError("Cross-fold identity exceeds registered threshold")
        maxima.append(dict(sequence=sequence, fold=int(folds[i]), max_train_identity=maximum))
    pd.DataFrame(maxima).to_csv(output / "cross_split_identity.csv", index=False)
    inner = {}
    for fold in sorted(set(outer.values())):
        if fold < 0:
            continue
        inner[str(fold)] = fold_assignments(
            {s: g for s, g in groups.items() if outer[s] != fold},
            config.inner_folds,
            config.seeds[0],
        )
    write_json(
        output / "split_manifest.json",
        dict(
            outer=outer,
            inner=inner,
            groups=groups,
            threshold=config.identity_threshold,
            method=(
                "BLOSUM45 global; open -5, extension -1; matches/aligned length; "
                "deterministic first optimal alignment"
            ),
            implementation=(
                "Biopython; Parasail tied-alignment parity not established; "
                "QMAP-inspired, not official benchmark reproduction"
            ),
            development_only=True,
            prior_holdout_reused=True,
            apex_overlap="unknown",
        ),
    )
    source_rows = pd.read_json(Path(config.development) / "observations.jsonl", lines=True)
    source_rows[source_rows.duplicate_export][
        ["observation_id", "source", "source_id", "sequence", "target", "raw_value", "raw_unit"]
    ].to_csv(output / "duplicate_audit.csv", index=False)
    write_json(
        output / "observation_manifest.json",
        dict(
            schema_version=1,
            rows=len(rows),
            measured=int(rows.objective.eq("measured_mic").sum()),
            consensus=int(rows.objective.eq("qmap_consensus").sum()),
            sequences=len(sequences),
            study_verified=0,
            publication_split="unavailable: no verified assay-to-publication mapping",
            duplicate_audit=(
                "prior source export deduplication preserved; no new measurements acquired"
            ),
            qmap_official="not executed; development OOD60 only",
        ),
    )
    finish_stage(output, inputs, started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mic_research.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["audit", "prepare"], required=True)
    args = parser.parse_args()
    config = MICConfig.model_validate_json(args.config.read_text())
    fresh_output(
        args.output,
        [
            Path(config.prior_models),
            Path(config.prior_refits),
            Path(config.development),
            Path(config.frozen_pool),
        ],
    )
    write_json(args.output / "protocol.json", config.model_dump())
    sources = [Path(__file__), *Path("src/robust_apex_qd/research").glob("mic_*.py")]
    write_json(
        args.output / "execution.json",
        dict(
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            diff_sha256=hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD"])
            ).hexdigest(),
            config_sha256=file_sha256(args.config),
            code_sha256={str(p): file_sha256(p) for p in sources},
            stage=args.stage,
        ),
    )
    torch.set_num_threads(config.cpu_threads)
    with threadpool_limits(config.cpu_threads):
        {"audit": audit, "prepare": prepare}[args.stage](config, args.output)


if __name__ == "__main__":
    main()
