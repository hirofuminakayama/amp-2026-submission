"""Complete saved-model and pool audits using sequence-keyed diagnostics only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_mic_research import audit, checked_manifest, finish_stage, write_json
from scipy.stats import spearmanr

from robust_apex_qd.apex.ensemble import APEX_PATHOGENS
from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.research.mic_data import MICConfig
from robust_apex_qd.research.mic_lineage import capture_execution
from robust_apex_qd.research.mic_models import mic_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mic_research.json"))
    parser.add_argument(
        "--prepared", type=Path, default=Path("work/mic_prediction/20260913-a/prepare")
    )
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument(
        "--pools", type=Path, default=Path("work/competition_exploration/20260913-scale/processed")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = MICConfig.model_validate_json(args.config.read_text())
    fresh_output(
        args.output,
        [
            args.prepared,
            args.lineage,
            args.alignment,
            args.pools,
            Path(config.prior_models),
            Path(config.prior_report),
            Path(config.prior_refits),
        ],
    )
    capture_execution(args.output)
    started = time.monotonic()
    audit(config, args.output)
    inputs = json.loads((args.output / "manifest.json").read_text())["inputs_sha256"]
    for path in [
        args.lineage / "manifest.json",
        args.alignment / "manifest.json",
        args.prepared / "manifest.json",
    ]:
        inputs.update(checked_manifest(path))
    manifests = []
    for root in [
        Path(config.prior_models),
        Path(config.prior_report),
        Path(config.prior_refits),
        args.pools,
    ]:
        for path in sorted(root.rglob("*manifest.json")):
            payload = json.loads(path.read_text())
            if "artifacts_sha256" in payload:
                inputs.update(checked_manifest(path))
                manifests.append(str(path))
    # Verify registered dependency hashes, not just the containers that record them.
    for path in sorted(args.pools.rglob("input_sha256.json")):
        expected = json.loads(path.read_text())
        verify_hashes(expected)
        inputs.update(expected)
    rows = pd.read_json(Path(config.prior_models) / "prepare/rows.jsonl", lines=True)
    seq = json.loads((args.prepared / "sequences.json").read_text())
    matrix = np.load(args.alignment / "identity.npy", mmap_mode="r")
    folds = rows.groupby("sequence").homology_fold.first().reindex(seq).to_numpy()
    if rows.groupby("sequence").homology_fold.nunique().max() != 1:
        raise ValueError("Legacy folds cross identical sequence")
    identity = {s: float(matrix[i, folds != folds[i]].max()) for i, s in enumerate(seq)}
    pd.DataFrame(
        dict(sequence=seq, prior_fold=folds, max_train_identity=[identity[s] for s in seq])
    ).to_csv(args.output / "prior_cross_split_identity.csv", index=False)
    provenance = pd.read_csv(args.lineage / "duplicate_audit.csv", keep_default_na=False).set_index(
        "observation_id"
    )
    measured = rows[rows.objective.eq("measured_mic")].copy()
    measured["identity_band"] = pd.cut(
        measured.sequence.map(identity), [-0.01, 0.4, 0.6, 0.8, 1], include_lowest=True
    ).astype(str)
    measured["db_study"] = measured.observation_id.map(provenance.study_ids).replace("", "unknown")
    measured["db_study_status"] = np.where(
        measured.db_study.eq("unknown"), "unknown", "db_attributed_not_paper_verified"
    )
    predictions = (
        pd.read_csv(args.output / "aligned_predictions.csv.gz")
        .set_index("observation_id")
        .reindex(measured.observation_id)
    )
    common = np.isfinite(predictions.to_numpy()).all(axis=1)
    extra = []
    for model in predictions:
        for cohort, mask in [
            ("all_supported", np.ones(len(measured), bool)),
            ("common_strain", common),
        ]:
            frame = measured.loc[mask].assign(prediction=predictions[model].to_numpy()[mask])
            for grouping in ["identity_band", "db_study", "db_study_status", "target"]:
                for target, group in frame.groupby(grouping):
                    extra.append(
                        dict(
                            model=model,
                            cohort=cohort,
                            grouping=grouping,
                            target=str(target),
                            **mic_metrics(group, group.prediction.to_numpy()),
                        )
                    )
    pd.concat(
        [pd.read_csv(args.output / "error_slices.csv"), pd.DataFrame(extra)], ignore_index=True
    ).to_csv(args.output / "error_slices.csv", index=False)
    pool_audit, rank_rows, disagreements = [], [], []
    for root in sorted(args.pools.iterdir()):
        if not (root / "models/manifest.json").is_file():
            continue
        sequences = read_fasta_sequences(root / "prepare/sequences.fasta")
        model_pool = pd.read_csv(root / "models/pool.csv.gz")
        apex_pool = pd.read_csv(root / "apex/pool.csv.gz")
        if (
            len(set(sequences)) != len(sequences)
            or sequences != model_pool.sequence.tolist()
            or sequences != apex_pool.sequence.tolist()
        ):
            raise ValueError(f"Pool sequence order differs: {root}")
        tensor = np.load(root / "apex/tensor.npy", mmap_mode="r")
        if tensor.shape != (len(sequences), 8, 11):
            raise ValueError("Unexpected APEX tensor shape")
        apex = np.log2(tensor.mean(axis=1))
        predictions_pool = {"B1": apex}
        for family in ["linear8", "finetune8"]:
            p = np.load(root / "models" / f"{family}.npz")["strain"]
            if p.shape != apex.shape:
                raise ValueError("Strain prediction shape mismatch")
            predictions_pool[family] = np.where(np.isfinite(p), p, apex)
        # The interval refit exists only for the original baseline pool.
        if root.name == "baseline":
            old_sequences = pd.read_csv(
                Path(config.prior_refits) / "pool_sequences.csv"
            ).sequence.tolist()
            index = {s: i for i, s in enumerate(old_sequences)}
            p = np.load(Path(config.prior_refits) / "ablation-interval/candidate_predictions.npz")[
                "strain"
            ][[index[s] for s in sequences]]
            predictions_pool["interval"] = np.where(np.isfinite(p), p, apex)
        safety_path = root / "safety-v2-combined/predictions.csv"
        safety = (
            pd.read_csv(safety_path).set_index("sequence")
            if safety_path.is_file()
            else pd.DataFrame(columns=pd.Index(["hc50"]))
        )
        if not safety.index.is_unique:
            raise ValueError("Ambiguous safety sequence keys")
        hc50 = safety.reindex(sequences).hc50.to_numpy(float)
        scores = {name: np.median(p, axis=1) for name, p in predictions_pool.items()}
        order = {name: np.lexsort((np.array(sequences), p))[:100] for name, p in scores.items()}
        for name, p in predictions_pool.items():
            selected = order[name]
            safe = hc50[selected]
            rank_rows.append(
                dict(
                    pool=root.name,
                    model=name,
                    target="all_median",
                    rows=len(sequences),
                    top100_overlap=len(set(selected) & set(order["B1"])),
                    spearman_with_apex=float(spearmanr(scores[name], scores["B1"]).statistic),
                    hc50_proxy_known=int(np.isfinite(safe).sum()),
                    hc50_proxy_mean=float(np.nanmean(safe)) if np.isfinite(safe).any() else None,
                    safety_missing_reason=""
                    if np.isfinite(safe).all()
                    else "no_saved_proxy_for_sequence",
                    constraints="unconstrained diagnostic; not submission",
                )
            )
            for j, pathogen in enumerate(APEX_PATHOGENS):
                selected = np.lexsort((np.array(sequences), p[:, j]))[:100]
                base = np.lexsort((np.array(sequences), apex[:, j]))[:100]
                rank_rows.append(
                    dict(
                        pool=root.name,
                        model=name,
                        target=pathogen,
                        rows=len(sequences),
                        top100_overlap=len(set(selected) & set(base)),
                        spearman_with_apex=float(spearmanr(p[:, j], apex[:, j]).statistic),
                    )
                )
            delta = scores[name] - scores["B1"]
            for i in np.argsort(-np.abs(delta), kind="stable")[:100]:
                disagreements.append(
                    dict(
                        pool=root.name,
                        model=name,
                        sequence=sequences[i],
                        delta_log2_um=float(delta[i]),
                        apex_log2_um=float(scores["B1"][i]),
                        model_log2_um=float(scores[name][i]),
                        hc50_proxy=float(hc50[i]) if np.isfinite(hc50[i]) else None,
                    )
                )
        pool_audit.append(
            dict(
                pool=root.name,
                rows=len(sequences),
                sequence_sha256=hashlib.sha256(("\n".join(sequences) + "\n").encode()).hexdigest(),
                tensor_shape=list(tensor.shape),
                safety_known=int(np.isfinite(hc50).sum()),
                interval_coverage="complete"
                if root.name == "baseline"
                else "not_previously_inferred",
            )
        )
    pd.DataFrame(rank_rows).to_csv(args.output / "fixed_pool_rank_changes.csv", index=False)
    pd.DataFrame(disagreements).to_csv(args.output / "pool_disagreements.csv", index=False)
    write_json(args.output / "pool_audit.json", pool_audit)
    summary = json.loads((args.output / "baseline_audit.json").read_text())
    summary.update(
        verified_files=len(inputs),
        verified_manifests=len(manifests),
        db_attributed_measured=int(measured.db_study.ne("unknown").sum()),
        all_prior_fold_maximum_identity=float(max(identity.values())),
        original_paper_assay_verified=0,
    )
    write_json(args.output / "baseline_audit.json", summary)
    with (args.output / "bottleneck_report.md").open("a") as f:
        f.write(
            "\n## Extended audit\n\n"
            + json.dumps(summary, indent=2)
            + (
                "\n\nAll completed processed pools are sequence-aligned and "
                "hash-verified. Strain rank changes and unconstrained Top100 "
                "safety proxy coverage are in fixed_pool_rank_changes.csv. HC50 is"
                " a saved prediction, not measured safety. The interval model has "
                "no saved expanded-pool inference. DB-attributed study slices do "
                "not establish comparable assays or independent paper validation. "
                "Prior folds and predictions remain unchanged; similarity is "
                "diagnosed using the new reference engine.\n"
            )
        )
    inputs[str(Path(__file__))] = file_sha256(Path(__file__))
    finish_stage(args.output, inputs, started)


if __name__ == "__main__":
    main()
