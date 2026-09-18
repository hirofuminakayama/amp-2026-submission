"""Version reference-engine OOD folds and evaluate publication and official-test sensitivity."""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_mic_research import checked_manifest, finish_stage, write_json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.mic_data import fold_assignments, similarity_groups
from robust_apex_qd.research.mic_lineage import (
    capture_execution,
    official_training_mask,
    publication_groups,
)


def fit_diagnostic(
    rows: pd.DataFrame,
    features: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    target_column: str,
    output: Path,
) -> pd.DataFrame:
    """Fixed species-ridge diagnostic: fit every learned quantity on training rows only."""
    if set(rows.loc[training].sequence) & set(rows.loc[validation].sequence):
        raise ValueError("Training and validation sequences overlap")
    output.mkdir(parents=True, exist_ok=False)
    frames, fits = [], []
    for species, valid in rows.loc[validation].groupby("species"):
        train = rows.loc[training & rows.species.eq(species).to_numpy()]
        prediction = np.full(len(valid), np.nan)
        baseline = np.full(len(valid), np.nan)
        if len(train) >= 2:
            x = features[train.sequence_index.to_numpy(int)]
            y = np.log2(train[target_column].to_numpy(float))
            scaler = StandardScaler().fit(x)
            model = Ridge(alpha=100.0).fit(scaler.transform(x), y)
            test = features[valid.sequence_index.to_numpy(int)]
            prediction = model.predict(scaler.transform(test))
            baseline[:] = np.median(y)
            path = output / f"species-{len(fits)}.npz"
            np.savez(
                path,
                mean=scaler.mean_,
                scale=scaler.scale_,
                coef=model.coef_,
                intercept=model.intercept_,
            )
            loaded = np.load(path)
            again = ((test - loaded["mean"]) / loaded["scale"]) @ loaded["coef"] + loaded[
                "intercept"
            ]
            np.testing.assert_allclose(again, prediction, rtol=1e-5, atol=1e-5)
        fits.append(
            dict(
                species=species,
                training_ids=train.observation_id.tolist(),
                validation_ids=valid.observation_id.tolist(),
                supported=len(train) >= 2,
            )
        )
        frames.append(
            valid[["observation_id", "sequence", "species", target_column]].assign(
                prediction=prediction,
                median_prediction=baseline,
                target_log2=np.log2(valid[target_column].to_numpy(float)),
            )
        )
    write_json(
        output / "manifest.json",
        dict(
            fits=fits,
            alpha=100,
            units="log2_uM",
            purpose="fixed diagnostic; no hyperparameter selection",
            artifacts_sha256={p.name: file_sha256(p) for p in output.glob("*.npz")},
        ),
    )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(frame: pd.DataFrame, **labels: Any) -> list[dict[str, Any]]:
    result = []
    for species, rows in frame.groupby("species"):
        for model, column in [("ridge100", "prediction"), ("median", "median_prediction")]:
            usable = rows[column].notna()
            error = (rows.loc[usable, column] - rows.loc[usable, "target_log2"]).to_numpy()
            result.append(
                dict(
                    **labels,
                    species=species,
                    model=model,
                    rows=len(rows),
                    supported_rows=int(usable.sum()),
                    mae=float(np.mean(abs(error))) if len(error) else None,
                    within1=float(np.mean(abs(error) <= 1)) if len(error) else None,
                    unsupported_reason=""
                    if usable.all()
                    else "fewer_than_two_training_observations",
                )
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared", type=Path, default=Path("work/mic_prediction/20260913-a/prepare")
    )
    parser.add_argument(
        "--prior", type=Path, default=Path("work/competition_exploration/20260912-b/phase3-r3")
    )
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument(
        "--qmap", type=Path, default=Path("work/measured_activity_research/sources/qmap_hf")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh_output(args.output, [args.prepared, args.prior, args.lineage, args.alignment, args.qmap])
    capture_execution(args.output)
    started = time.monotonic()
    inputs = {}
    for path in [
        args.prepared / "manifest.json",
        args.lineage / "manifest.json",
        args.alignment / "manifest.json",
        args.prior / "esm8/manifest.json",
    ]:
        inputs.update(checked_manifest(path))
    sequences = json.loads((args.prepared / "sequences.json").read_text())
    matrix = np.load(args.alignment / "identity.npy", mmap_mode="r")
    old_matrix = np.load(args.prepared / "identity.npy", mmap_mode="r")
    old_split = json.loads((args.prepared / "split_manifest.json").read_text())
    rows = pd.read_json(args.prepared / "rows.jsonl", lines=True)
    features = np.load(args.prior / "esm8/features.npy").astype(np.float64)
    groups = similarity_groups(sequences, matrix, 0.6)
    outer = fold_assignments(groups, 5, 42)
    inner = {
        str(f): fold_assignments({s: g for s, g in groups.items() if outer[s] != f}, 3, 42)
        for f in sorted(set(outer.values()))
        if f >= 0
    }
    maxima = []
    partitions = {"outer": outer, **{f"inner-{f}": value for f, value in inner.items()}}
    seq_index = {s: i for i, s in enumerate(sequences)}
    for scope, assignment in partitions.items():
        for sequence, fold in assignment.items():
            training = [seq_index[s] for s, f in assignment.items() if f != fold and f >= 0]
            maximum = (
                float(matrix[seq_index[sequence], training].max())
                if training and fold >= 0
                else None
            )
            if maximum is not None and maximum > float(np.float32(0.6)):
                raise ValueError("Reference identity crosses new split")
            maxima.append(
                dict(
                    scope=scope,
                    sequence=sequence,
                    fold=fold,
                    max_train_identity=maximum,
                    unsupported_reason="" if maximum is not None else "insufficient_groups",
                )
            )
    pd.DataFrame(maxima).to_csv(args.output / "cross_split_identity.csv", index=False)
    rows["biopython_ood60_fold"] = rows.homology_fold
    rows["homology_fold"] = rows.sequence.map(outer)
    rows["homology_group"] = rows.sequence.map(groups)
    rows.to_json(args.output / "rows.jsonl", orient="records", lines=True)
    shutil.copyfile(args.alignment / "identity.npy", args.output / "identity.npy")
    write_json(args.output / "sequences.json", sequences)
    write_json(
        args.output / "split_manifest.json",
        dict(
            schema_version=2,
            outer=outer,
            inner=inner,
            groups=groups,
            threshold=0.6,
            engine_manifest=str(args.alignment / "manifest.json"),
            prior_split_sha256=file_sha256(args.prepared / "split_manifest.json"),
            development_only=True,
            prior_holdout_reused=True,
            previous_oof_reusable=False,
        ),
    )
    cutoff = np.float32(0.6)
    changed = np.triu((matrix > cutoff) != (old_matrix > cutoff), 1)
    indices = np.argwhere(changed)
    pd.DataFrame(
        [
            dict(
                left=sequences[i],
                right=sequences[j],
                biopython=float(old_matrix[i, j]),
                parasail=float(matrix[i, j]),
                old_fold_left=old_split["outer"][sequences[i]],
                old_fold_right=old_split["outer"][sequences[j]],
            )
            for i, j in indices
        ]
    ).to_csv(args.output / "alignment_threshold_changes.csv", index=False)
    write_json(
        args.output / "alignment_comparison.json",
        dict(
            pairs=len(sequences) * (len(sequences) - 1) // 2,
            differing_pairs=int(np.triu(matrix != old_matrix, 1).sum()),
            threshold_changes=len(indices),
            new_groups=len(set(groups.values())),
            new_fold_sizes=pd.Series(outer).value_counts().sort_index().to_dict(),
            old_split_reference_engine_violations=int(
                sum(
                    matrix[i, j] > cutoff
                    and old_split["outer"][sequences[i]] != old_split["outer"][sequences[j]]
                    for i, j in indices
                )
            ),
        ),
    )
    lineage_rows = {}
    with (args.output / "observations.jsonl").open("w") as output_handle:
        for line in (args.lineage / "observations.jsonl").open():
            item = json.loads(line)
            if item["screen_included"]:
                lineage_rows[item["observation_id"]] = item["lineage"]
                output_handle.write(line)
    shutil.copyfile(args.lineage / "duplicate_audit.csv", args.output / "duplicate_audit.csv")
    observation_manifest = json.loads((args.lineage / "observation_manifest.json").read_text())
    observation_manifest.update(
        observations=len(lineage_rows), full_source_manifest=str(args.lineage / "manifest.json")
    )
    write_json(args.output / "observation_manifest.json", observation_manifest)
    publications: dict[str, list[str]] = {}
    for row in rows.to_dict("records"):
        publications.setdefault(row["sequence"], []).extend(
            lineage_rows[row["observation_id"]]["study_ids"]
        )
    pub_groups = publication_groups(sequences, publications)
    pub_folds = fold_assignments(pub_groups, 5, 42)
    paper_rows = [
        dict(sequence=s, study=p, fold=pub_folds[s])
        for s, papers in publications.items()
        for p in sorted(set(papers))
    ]
    paper_frame = pd.DataFrame(paper_rows)
    if not paper_frame.empty and paper_frame.groupby("study").fold.nunique().max() != 1:
        raise ValueError("Database publication crosses folds")
    paper_frame.to_csv(args.output / "publication_membership.csv", index=False)
    write_json(
        args.output / "publication_split.json",
        dict(
            groups=pub_groups,
            outer=pub_folds,
            known_sequences=sum(bool(p) for p in publications.values()),
            unknown_sequences=sum(not p for p in publications.values()),
            linkage=(
                "DB-attributed measurement, not paper-verified assay; unknowns "
                "grouped by sequence only"
            ),
            homology_isolated=False,
        ),
    )
    metrics, publication_audit = [], []
    exact = rows.objective.eq("measured_mic").to_numpy() & rows.exact_regression.to_numpy(bool)
    for split_name, assignment in [("parasail_ood60", outer), ("db_publication", pub_folds)]:
        row_folds = rows.sequence.map(assignment).to_numpy()
        outputs = []
        for fold in sorted(set(assignment.values())):
            if fold < 0:
                continue
            train, valid = exact & (row_folds != fold), exact & (row_folds == fold)
            pred = fit_diagnostic(
                rows,
                features,
                train,
                valid,
                "mic_um",
                args.output / "diagnostics" / f"{split_name}-{fold}",
            )
            outputs.append(pred.assign(fold=fold))
        frame = pd.concat(outputs, ignore_index=True)
        frame.to_csv(args.output / f"{split_name}_oof.csv.gz", index=False)
        metrics.extend(summarize(frame, split=split_name, cohort="all_exact"))
        known = frame.observation_id.map(lambda i: bool(lineage_rows[i]["study_ids"]))
        metrics.extend(summarize(frame.loc[known], split=split_name, cohort="db_attributed_exact"))
        for sequence in sequences:
            other = [seq_index[s] for s in sequences if assignment[s] != assignment[sequence]]
            publication_audit.append(
                dict(
                    split=split_name,
                    sequence=sequence,
                    fold=assignment[sequence],
                    study_known=bool(publications[sequence]),
                    max_train_identity=float(matrix[seq_index[sequence], other].max())
                    if other
                    else None,
                )
            )
    pd.DataFrame(metrics).to_csv(args.output / "publication_sensitivity.csv", index=False)
    pd.DataFrame(publication_audit).to_csv(
        args.output / "publication_cross_split_identity.csv", index=False
    )
    alignment_protocol = json.loads((args.alignment / "manifest.json").read_text())
    if alignment_protocol.get("test_orientation") != "training_to_test":
        raise ValueError("Official benchmark needs directed training-to-test identities")
    test_sequences = json.loads((args.alignment / "test_sequences.json").read_text())
    test_index = {s: i for i, s in enumerate(test_sequences)}
    test_matrix = np.load(args.alignment / "test_identity.npy", mmap_mode="r")
    benchmark_summary, benchmark_metrics = [], []
    official_sets = []
    for split in range(5):
        path = args.qmap / f"benchmark_split_{split}.json"
        raw = json.loads(path.read_text())
        test_ids = {f"qmap:{item['id']}:{target}" for item in raw for target in item["targets"]}
        heldout = {item["sequence"] for item in raw}
        official_sets.append(heldout)
        maximum = test_matrix[:, [test_index[s] for s in sorted(heldout)]].max(axis=1)
        retained = set(
            np.array(sequences)[official_training_mask(sequences, heldout, maximum, 0.6)]
        )
        quarantine = pd.DataFrame(
            dict(
                sequence=sequences,
                max_official_test_identity=maximum,
                retained=[s in retained for s in sequences],
            )
        )
        stage = args.output / "official_tests" / str(split)
        stage.mkdir(parents=True)
        quarantine.to_csv(stage / "training_quarantine.csv", index=False)
        training = (
            rows.sequence.isin(retained).to_numpy() & rows.objective.eq("qmap_consensus").to_numpy()
        )
        validation = (
            rows.observation_id.isin(test_ids).to_numpy()
            & rows.objective.eq("qmap_consensus").to_numpy()
        )
        frame = fit_diagnostic(rows, features, training, validation, "consensus_um", stage / "fit")
        frame.to_csv(stage / "predictions.csv", index=False)
        benchmark_metrics.extend(summarize(frame, split=split, objective="qmap_consensus"))
        additional = rows.loc[
            rows.objective.eq("measured_mic"), ["observation_id", "sequence"]
        ].copy()
        additional["retained"] = additional.sequence.isin(retained)
        additional.to_csv(stage / "additional_training_quarantine.csv", index=False)
        exclusions = sorted(test_ids - set(rows.loc[validation].observation_id))
        write_json(
            stage / "excluded_test_observations.json",
            dict(
                observation_ids=exclusions,
                reason=(
                    "outside_existing_competition_feature/target/eligibility_scope; "
                    "not dropped based on test label"
                ),
            ),
        )
        record = dict(
            split=split,
            official_rows=len(raw),
            official_target_observations=len(test_ids),
            evaluated_observations=int(validation.sum()),
            excluded_observations=len(exclusions),
            train_consensus_observations=int(training.sum()),
            additional_measured_retained=int(additional.retained.sum()),
            additional_measured_used=False,
            max_retained_identity=float(maximum[[s in retained for s in sequences]].max())
            if retained
            else None,
            dataset_scope=(
                "existing competition feature/target coverage subset; not full official benchmark"
            ),
            objective="consensus, not measured MIC",
            historical_development_reuse=True,
        )
        write_json(
            stage / "manifest.json",
            dict(
                **record,
                source_sha256=file_sha256(path),
                artifacts_sha256={
                    str(p.relative_to(stage)): file_sha256(p)
                    for p in stage.rglob("*")
                    if p.is_file()
                },
            ),
        )
        benchmark_summary.append(record)
    pd.DataFrame(benchmark_summary).to_csv(args.output / "official_test_inventory.csv", index=False)
    pd.DataFrame(benchmark_metrics).to_csv(args.output / "official_test_metrics.csv", index=False)
    pd.DataFrame(
        [
            dict(left=i, right=j, sequence_overlap=len(a & b))
            for i, a in enumerate(official_sets)
            for j, b in enumerate(official_sets)
            if i < j
        ]
    ).to_csv(args.output / "official_test_overlap.csv", index=False)
    inputs[str(Path(__file__))] = file_sha256(Path(__file__))
    finish_stage(args.output, inputs, started, no_concatenated_official_oof=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
