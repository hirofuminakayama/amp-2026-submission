"""Audit correction fits and hand off development evidence plus external insufficiency."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from prepare_mic_external import checked, finish, read_jsonl, write_json, write_jsonl

from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.mic_correction import correction_masks, paired_comparison
from robust_apex_qd.research.mic_external import load_training_rows
from robust_apex_qd.research.mic_lineage import capture_execution
from robust_apex_qd.research.prediction_cache import isolated_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    inputs = {}
    for p in [args.prepared, args.trained, args.report]:
        inputs.update(checked(p))
    cfg = json.loads((args.prepared / "config.json").read_text())
    external = Path(cfg["external"])
    inputs.update(checked(external))
    rows = load_training_rows(args.prepared)
    features = np.load(cfg["features"], allow_pickle=False)
    corrections = set(json.loads((args.prepared / "corrections.json").read_text()))
    protocol = json.loads((args.prepared / "protocol.json").read_text())
    fit_records: list[dict[str, Any]] = []
    coverage_frames = []
    for seed in protocol["seeds"]:
        for name in ["original", "corrected"]:
            oof = pd.read_csv(args.trained / f"{name}-s{seed}-oof.csv.gz").set_index(
                "observation_id"
            )
            if set(oof.index) != set(protocol["comparison_rows"]) or oof.index.duplicated().any():
                raise ValueError("OOF coverage differs")
            coverage = (
                oof.assign(
                    supported=oof.prediction.notna(),
                    exact_supported=oof.exact_regression & oof.prediction.notna(),
                )
                .groupby(
                    ["species", "study", "component_id", "prediction_level"],
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    observations=("prediction", "size"),
                    supported=("supported", "sum"),
                    exact_supported=("exact_supported", "sum"),
                )
            )
            coverage["arm"], coverage["seed"] = name, seed
            coverage["study_scope"] = "DB-attributed; not original-assay verification"
            coverage_frames.append(coverage)
            for fold in sorted(rows.homology_fold.unique()):
                dest = args.trained / f"{name}-s{seed}" / f"fold{fold}"
                saved = json.loads((dest / "fit.json").read_text())
                original, corrected, valid = correction_masks(rows, corrections, int(fold))
                train = original if name == "original" else corrected
                if (
                    saved["training_ids"] != rows.loc[train, "observation_id"].tolist()
                    or saved["validation_ids"] != rows.loc[valid, "observation_id"].tolist()
                    or saved["arm"] != protocol["arm"]
                    or saved["seed"] != seed
                    or not saved["serialization_equal"]
                ):
                    raise ValueError("Fit membership/settings differ")
                isolated_rows(
                    rows.sequence.tolist(),
                    rows.component_id.tolist(),
                    np.flatnonzero(train),
                    np.flatnonzero(valid),
                )
                fold_rows = pd.read_csv(dest / "predictions.csv.gz").set_index("observation_id")
                if set(fold_rows.index) != set(saved["validation_ids"]):
                    raise ValueError("Fold predictions differ")
                np.testing.assert_allclose(
                    fold_rows.prediction,
                    oof.loc[fold_rows.index, "prediction"],
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
                state = torch.load(dest / "weights.pt", map_location="cpu", weights_only=False)
                chosen = rows.loc[train]
                np.testing.assert_allclose(
                    state["center"],
                    np.log2(chosen.mic_um.to_numpy(float)).mean(),
                    rtol=0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    state["mean"],
                    features[chosen.sequence_index].mean(axis=0, dtype=np.float64),
                    rtol=0,
                    atol=1e-10,
                )
                if state["arm"] != protocol["arm"] or not np.isfinite(state["loss_curve"]).all():
                    raise ValueError("Checkpoint settings or training loss differs")
                del state
                fit_manifest = json.loads((dest / "manifest.json").read_text())
                fit_records.append(
                    dict(
                        arm=name,
                        seed=seed,
                        fold=int(fold),
                        training_rows=int(train.sum()),
                        evaluation_rows=int(valid.sum()),
                        seconds=fit_manifest["seconds"],
                        fit_manifest=str(dest / "manifest.json"),
                        weights=str(dest / "weights.pt"),
                        weights_sha256=file_sha256(dest / "weights.pt"),
                        serialization_equal=True,
                        training_only_center_and_scaler=True,
                    )
                )
    external_split = json.loads((external / "split_manifest.json").read_text())
    by_id = {r["observation_id"]: r for r in external_split["assignments"]}
    folds_by_component = {}
    for r in rows.to_dict("records"):
        if by_id[r["observation_id"]]["component_id"] != r["component_id"]:
            raise ValueError("Frozen component changed")
        if (
            r["component_id"] in folds_by_component
            and folds_by_component[r["component_id"]] != r["homology_fold"]
        ):
            raise ValueError("Paper/homology component split across folds")
        folds_by_component[r["component_id"]] = r["homology_fold"]
    paired = json.loads((args.report / "paired_comparison.json").read_text())
    for record in paired:
        frame = pd.read_csv(args.report / f"paired-s{record['seed']}.csv.gz")
        repeated = paired_comparison(frame)
        for key in ["delta_macro_mae", "ci95", "common_exact_rows", "components", "species"]:
            if repeated[key] is None or record[key] is None:
                if repeated[key] != record[key]:
                    raise ValueError("Paired metric availability differs")
            else:
                np.testing.assert_allclose(repeated[key], record[key], rtol=0, atol=1e-12)
    status = json.loads((args.report / "evaluation_status.json").read_text())
    if status["primary_external"] != "not_executed_insufficient":
        raise ValueError("External scoring state changed")
    if (
        json.loads((external / "evaluation_protocol.json").read_text())["scoring_status"]
        != "not_started"
    ):
        raise ValueError("Source external protocol was mutated")
    prior = Path("work/mic_prediction/20260913-phase24-a")
    adoption = json.loads((prior / "report/adoption_report.json").read_text())
    candidate = adoption["recommended_research_candidate"]
    if candidate != "finetune8-w0.25":
        raise ValueError("Registered retained candidate changed")
    paths = [
        prior / "report/adoption_report.json",
        prior / "report/scenario_scores.csv",
        prior / "report/scenario_ranking.csv",
        prior / "review-r3/verification.json",
        prior / "handoff/manifest.json",
        prior / "selection/tops" / candidate / "manifest.json",
    ]
    selection = prior / "selection/tops" / candidate
    saved = json.loads((selection / "manifest.json").read_text())
    h = {str(selection / name): digest for name, digest in saved["artifacts_sha256"].items()}
    inputs.update(verify_hashes(h))
    inputs.update({str(p): file_sha256(p) for p in paths})
    retained_predictions = prior / "handoff/finetune8"
    inputs.update(checked(retained_predictions))
    inputs.update(checked(prior / "report"))
    for license_file in [Path("LICENSE"), Path("THIRD_PARTY_LICENSES.md")]:
        inputs[str(license_file)] = file_sha256(license_file)
    for p in retained_predictions.rglob("*"):
        if p.is_file():
            inputs[str(p)] = file_sha256(p)
    registry = Path("work/mic_prediction/20260914-papers-a/registry-v2")
    inputs.update(checked(registry))
    rights = json.loads((registry / "rights_manifest.json").read_text())["papers"]
    curation = Path("work/mic_prediction/20260914-papers-a/curation-v4")
    inputs.update(checked(curation))
    fresh_output(
        args.output, [args.prepared, args.trained, args.report, external, prior, registry, curation]
    )
    capture_execution(args.output)
    pd.DataFrame(fit_records).to_csv(args.output / "fit_audit.csv", index=False)
    pd.concat(coverage_frames).to_csv(args.output / "coverage.csv", index=False)
    data_uses = []
    for r in rows.to_dict("records"):
        data_uses.append(
            dict(
                observation_id=r["observation_id"],
                source=r["source"],
                role="legacy development control",
                original_arm=True,
                corrected_arm=r["observation_id"] not in corrections,
                excluded_by_correction=r["observation_id"] in corrections,
                used_for_training_original=bool(r["exact_regression"]),
                used_for_training_corrected=(
                    bool(r["exact_regression"]) and r["observation_id"] not in corrections
                ),
                used_for_shared_validation=r["observation_id"] not in corrections,
                disclosure="existing DB terms; original-paper licenses do not clear DB reuse",
                minimum_status="existing recorded source contract; no new certification",
                full_status="redistribution clearance not established by this run",
            )
        )
    lookup = {r["paper_id"]: r for r in rights}
    for r in read_jsonl(curation / "paper_observations.jsonl"):
        source = lookup[r["paper_id"]]
        data_uses.append(
            dict(
                observation_id=r["observation_id"],
                source=r["paper_id"],
                role="curation/chemistry evidence; diagnostic only",
                original_arm=False,
                corrected_arm=False,
                minimum_status="recorded authored-cell reuse scope",
                full_status=source["status"],
                disclosure=source["reuse_scope"],
                evidence=source["license_evidence"],
                source_sha256=r["source_sha256"],
                chemical_form=r["chemical_form"],
            )
        )
    write_jsonl(args.output / "data_use_ledger.jsonl", data_uses)
    write_json(
        args.output / "disclosure_manifest.json",
        dict(
            source_rights=rights,
            observations="data_use_ledger.jsonl",
            published=False,
            minimum="Recorded source scopes apply; private use is not new clearance",
            full="Cleared authored cells only; archives/DB/submission not certified",
            original_code="Repository and third-party licenses retain their scopes",
            reproduction=dict(
                config="configs/mic_correction_comparison.json",
                command="README correction workflow: prepare, smoke, train, report, handoff",
                external_labels_required_for_candidate_inference=False,
            ),
        ),
    )
    write_json(
        args.output / "equivalent_arms.json",
        dict(
            corrected_plus_new_training=dict(
                alias_of="corrected",
                new_observations=0,
                evidence="identical training IDs/settings; no separate expansion-effect estimate",
                predictions=[
                    str(args.trained / f"corrected-s{s}-oof.csv.gz") for s in protocol["seeds"]
                ],
            ),
            registered_fit_wall_seconds=sum(r["seconds"] for r in fit_records),
            actual_fits=len(fit_records),
        ),
    )
    write_json(
        args.output / "candidate_handoff.json",
        dict(
            candidate=candidate,
            recommendation_changed=False,
            adopted=False,
            policy="B1/L2/C0",
            development_protocol=str(args.prepared / "protocol.json"),
            comparison=str(args.report / "comparison.csv"),
            external_status=str(args.report / "evaluation_status.json"),
            retained_prediction_artifacts=str(retained_predictions),
            retained_selection_artifacts=str(selection),
            prior_joint_hc50_report=str(prior / "report/adoption_report.json"),
            limitations=[
                "external evaluation empty",
                "no new primary training data",
                "development OOF only",
                "chemical correction effect is not data expansion or independent efficacy",
            ],
            downstream_owner="competition-first adoption/integration and full generation/release",
            candidate_freeze="2026-09-24: report unresolved acquisition and evaluation",
        ),
    )
    comparison = pd.read_csv(args.report / "comparison.csv")
    summary = comparison[
        (comparison.scope == "development_oof") & (comparison.prediction_level == "all")
    ]
    lines = [
        "# MIC correction handoff",
        "",
        f"Retained research recommendation: `{candidate}`. Adopted policy: B1/L2/C0.",
        "",
        "The primary external collection has zero eligible rows. External scoring was not run.",
        "The added-data arm equals corrected: no new training observations.",
        "",
        "| Data arm | Seed | Development macro MAE |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {r.arm} | {int(r.seed)} | {r.macro_mae:.6f} |" for r in summary.itertuples()]
    lines += [
        "",
        "Development change: corrected minus original. Negative favors correction.",
    ]
    lines += [
        (
            f"- Seed {r['seed']}: delta {r['delta_macro_mae']:.6f}; CI {r['ci95']}; "
            f"common exact rows {r['common_exact_rows']}."
        )
        for r in paired
    ]
    lines += [
        "",
        "This correction comparison retains the registered pool recommendation.",
        "Original labels, models and OOF are intact. No new candidate or pool export.",
        "",
        (
            f"Comparison: `{args.report / 'comparison.csv'}`; "
            "predictions and weight hashes: `fit_audit.csv`."
        ),
        f"Existing pool predictions: `{retained_predictions}`; Top/library/ranking: `{selection}`.",
        f"Existing HC50 joint/scenario evidence: `{prior / 'report/adoption_report.json'}`.",
        "",
        "Recorded Minimum/Full scopes: `data_use_ledger.jsonl`, `disclosure_manifest.json`.",
        "Authored-cell clearance does not cover archives, DB annotations or the submission.",
        "",
        "At the 2026-09-24 freeze, report unresolved paper expansion and external evaluation.",
        "Competition-first owns adoption, integration, full generation and release.",
        "Missing external validation means insufficient evidence, not no effect.",
    ]
    (args.output / "handoff_report.md").write_text("\n".join(lines) + "\n")
    write_json(
        args.output / "verification.json",
        dict(
            fits=len(fit_records),
            seeds=protocol["seeds"],
            folds=5,
            common_evaluation_rows=len(protocol["comparison_rows"]),
            correction_ids=len(corrections),
            split_isolation=True,
            paired_metrics_replayed=True,
            verified_input_paths=len(inputs),
            retained_candidate_artifacts_verified=True,
            primary_external_rows=0,
            external_scoring="not_executed_insufficient",
            rights_status_counts=dict(Counter(r["status"] for r in rights)),
            data_use_rows=len(data_uses),
            new_model_adapter="not_applicable_no_new_candidate",
        ),
    )
    finish(args.output, inputs, started)
    print((args.output / "verification.json").read_text())


if __name__ == "__main__":
    main()
