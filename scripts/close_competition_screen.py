"""Publish a new closeout snapshot for verified frozen screens, retaining all prior evidence."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from robust_apex_qd.features.embeddings import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    old = args.prior
    out = args.output
    verification = json.loads((args.verification / "completion_verification.json").read_text())
    assert verification["passed"]
    for name, digest in verification["checked_sha256"].items():
        assert file_sha256(Path(name)) == digest, name
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root / "phase4/assets", out / "assets")
    comparison = pd.read_csv(root / "phase4-evaluation/report/generator_comparison.csv")
    decisions = pd.read_csv(root / "phase4-evaluation/report/screen_decisions.csv")
    runs = pd.read_csv(root / "phase4-evaluation/collect/runs.csv")
    registry = pd.read_csv(old / "experiment_registry.csv").fillna("")
    registry["cost_seconds"] = pd.to_numeric(registry.cost_seconds).astype(float)
    # Retain old roots explicitly; current records are a new registry snapshot.
    registry["output"] = registry.output.map(lambda p: str(old / p) if p else "")
    family_patterns = {
        "E09": "paired|steps|length",
        "E11": "hydramp",
        "E12": "designer|prompt",
        "E13": "ampgen",
        "E14": "deepamp",
        "E15": "evodiff",
    }
    for key, pattern in family_patterns.items():
        p = out / "assets" / key / "asset_manifest.json"
        m = json.loads(p.read_text())
        cohort = comparison[comparison.run.str.contains(pattern, regex=True)]
        screen = decisions[decisions.run.isin(cohort.run)]
        if key == "E09":
            cohort = comparison[comparison["mode"].eq("legacy-diffusion")]
            screen = decisions[decisions.run.isin(cohort.run)]
        m.update(
            status="screen_complete",
            comparison=str(root / "phase4-evaluation/report/generator_comparison.csv"),
            screen_decisions=str(root / "phase4-evaluation/report/screen_decisions.csv"),
            runs=sorted(set(cohort.run)),
            pareto_runs=sorted(set(screen[screen.pareto].run)),
            screen_decision=(
                "retain computational frontier and viable family representatives "
                "for later expansion; no measured-superiority claim"
            ),
        )
        logs = []
        for name in m["runs"]:
            local = root / (name + ".log")
            tmp = Path("/tmp") / ("competition-" + name + ".log")
            if local.exists():
                logs.append(str(local))
            elif tmp.exists():
                logs.append(str(out / "execution_logs" / tmp.name))
        m["execution_logs"] = logs
        m["evaluation_log"] = str(root / "generator-report.log")
        m["run_manifests"] = {
            str(row.run): str(row.manifest)
            for row in runs[runs.run.isin(cohort.run)].itertuples()
            if pd.notna(row.manifest)
        }
        m["cost_note"] = (
            "run wall times include reused legacy screens; new resource spending is "
            "recorded separately in budget.json"
        )
        if key == "E13":
            m["downstream_generated_manifest"] = str(
                root / "phase4/ampgen-downstream-generated/manifest.json"
            )
        if key == "E15":
            m.update(
                reproduced_prodcarl=False,
                optimization_manifest=str(root / "phase4/evodiff-optimization/manifest.json"),
                matched_seed_table=str(root / "phase4-evaluation/report/paired_comparisons.csv"),
                upstream_limitation=(
                    "public tree lacks aligned generator checkpoint; own reward-"
                    "weighted likelihood implemented independently"
                ),
            )
        m["environment_locks"] = {str(p): file_sha256(p) for p in (root / "envs").glob("*/uv.lock")}
        p.write_text(json.dumps(m, indent=2) + "\n")
        mask = registry.id.eq(key)
        registry.loc[mask, "status"] = "screen_complete"
        registry.loc[mask, "output"] = str(root / "phase4-evaluation/report")
        registry.loc[mask, "next_action"] = (
            "Compare computational frontier and family representatives in expanded pools"
        )
        registry.loc[mask, "cost_seconds"] = float(
            runs[runs.run.isin(cohort.run)].seconds.fillna(0).sum()
        )
    p = out / "assets/E16/asset_manifest.json"
    m = json.loads(p.read_text())
    m.update(
        status="structure_complete_membrane_partial_blocker",
        structure_manifest=str(root / "phase4/structure-r2/manifest.json"),
        execution_logs=[str(root / "structure-r2.log"), str(root / "membrane-r2.log")],
        membrane_manifest=str(root / "phase4/membrane-diagnostic-r2/manifest.json"),
        screen_decision=(
            "diagnostic only: one pure POPE2ps trajectory; control construction "
            "fails; no activity/binding/free-energy ranking"
        ),
    )
    p.write_text(json.dumps(m, indent=2) + "\n")
    mask = registry.id.eq("E16")
    registry.loc[mask, "status"] = "structure_complete_membrane_partial_blocker"
    registry.loc[mask, "output"] = str(root / "phase4/structure-r2")
    registry.loc[mask, "next_action"] = (
        "retain construction/short trajectory diagnostic; no equilibrated membrane "
        "activity estimate"
    )
    for key in ["E02", "E06", "E08", "E17"]:
        mask = registry.id.eq(key)
        registry.loc[mask, "status"] = "screen_complete"
        registry.loc[mask, "output"] = str(root / "phase3-report")
        registry.loc[mask, "next_action"] = (
            "Retain best predictor families and complementary Top candidates"
        )
    registry.to_csv(out / "experiment_registry.csv", index=False)
    logs = out / "execution_logs"
    logs.mkdir(exist_ok=False)
    for p in Path("/tmp").glob("competition-*.log"):
        shutil.copy2(p, logs / p.name)
    executed = out / "executed_sources"
    executed.mkdir(exist_ok=True)
    for name in [
        "competition_gpu_queue.py",
        "competition_preflight_report_queue.py",
        "report_competition_membrane_failure.py",
        "check_competition_membrane_inputs.py",
        "check_competition_ampgen_reference.py",
        "complete_competition_asset_inputs.py",
        "competition_gpu_retry_queue.py",
        "fix_competition_execution.py",
        "verify_competition_phase3.py",
        "competition_screen_queue.py",
        "competition_evaluation_queue.py",
        "competition_membrane_queue.py",
        "competition_gpu_queue_serial.py",
        "prepare_reward.py",
        "report_consensus_auxiliary.py",
        "report_competition_component_diagnostics.py",
        "report_competition_deepamp_pairs.py",
        "prepare_competition_hydramp_pairs.py",
        "report_competition_hydramp_pairs.py",
        "competition_progress.py",
        "audit_competition_predictors.py",
        "diagnose_hemopi_batch.py",
        "legacy_ensemble_sweep.py",
        "ampgen_downstream_smoke.py",
        "resume_msa_ranges.py",
        "download_msa_ranges.py",
    ]:
        p = Path("/tmp") / name
        if p.exists():
            shutil.copy2(p, executed / name)
    for p in list(Path("scripts").glob("*competition*.py")) + list(
        Path("src/robust_apex_qd/research").glob("competition*.py")
    ):
        dest = executed / (file_sha256(p) + "-" + p.name)
        shutil.copy2(p, dest)
    for p in (root / "envs").glob("*/uv.lock"):
        shutil.copy2(p, executed / (p.parent.name + "-uv.lock"))
    folds = pd.read_csv(args.verification / "fold_verification.csv")
    refits = pd.read_csv(args.verification / "refit_verification.csv")
    encoder_seconds = sum(
        json.loads((root / "phase3-r3" / n / "manifest.json").read_text())["seconds"]
        for n in ["esm8", "esm650"]
    )
    gpu_new = sum(
        json.loads(p.read_text()).get("seconds", 0)
        for p in (root / "phase4").glob("*/run_manifest.json")
        if json.loads(p.read_text()).get("device") == "cuda"
    )
    for name in [
        "evodiff-optimization",
        "structure-r2",
        "ampgen-downstream-generated",
        "membrane-diagnostic-r2",
    ]:
        gpu_new += json.loads((root / "phase4" / name / "manifest.json").read_text())["seconds"]
    upper = float(folds.seconds.sum() + refits.seconds.sum() + encoder_seconds + gpu_new)
    budget = json.loads((old / "budget.json").read_text())
    budget.update(
        prior_run=str(old),
        new_gpu_stage_wall_upper_bound_seconds=upper,
        new_gpu_stage_wall_upper_bound_hours=upper / 3600,
        used_gpu_hours=budget["used_gpu_hours"] + upper / 3600,
        measurement_note=(
            "Conservative stage wall time includes CPU work and artifact I/O in "
            "predictor folds/refits; not kernel time. Multi-fit CUDA peaks are "
            "cumulative process high-water marks. Failed stages/setup/downloads are"
            " separately logged."
        ),
        phase34_verified=True,
    )
    budget["remaining_gpu_hours"] = (
        sum(budget["allocated_gpu_hours"].values()) - budget["used_gpu_hours"]
    )
    cpu_records: list[dict[str, Any]] = []
    for p in root.glob("phase4-evaluation*/**/stage_manifest.json"):
        m = json.loads(p.read_text())
        cpu_records.append(
            dict(
                stage=str(p.parent.relative_to(root)),
                seconds=m["seconds"],
                kind="CPU evaluation stage wall",
            )
        )
    for p in (root / "phase4").glob("*/run_manifest.json"):
        m = json.loads(p.read_text())
        if m.get("device") == "cpu" or p.parent.name.startswith("hydramp-"):
            cpu_records.append(
                dict(
                    stage=str(p.parent.relative_to(root)),
                    seconds=m["seconds"],
                    kind="CPU generator wall",
                )
            )
    for name in [
        "phase4-deepamp-conditioning",
        "phase4-deepamp-edited",
        "phase4-hydramp-edited",
        "phase4-ampgen-first-screen",
    ]:
        p = root / name / "apex_manifest.json"
        m = json.loads(p.read_text())
        cpu_records.append(
            dict(stage=name, seconds=m["runtime_seconds"], kind="CPU paired-control APEX wall")
        )
    p = root / "phase4-ampgen-reference-cpu/parity.json"
    m = json.loads(p.read_text())
    cpu_records.append(
        dict(
            stage="phase4-ampgen-reference-cpu",
            seconds=m["seconds"],
            kind="CPU reference parity wall",
        )
    )
    cpu_records.append(
        dict(
            stage="phase3-report",
            seconds=next(
                m["seconds"]
                for m in json.loads((root / "gpu_queue_status.json").read_text())["records"]
                if m["stage"] == "model-report"
            ),
            kind="CPU report wall upper bound",
        )
    )
    pd.DataFrame(cpu_records).to_csv(out / "cpu_stage_costs.csv", index=False)
    anchor = root / "phase3/protocol.json"
    budget.update(
        cpu_stage_wall_seconds_sum=sum(m["seconds"] for m in cpu_records),
        cpu_stage_costs=str(out / "cpu_stage_costs.csv"),
        wall_seconds_since_initial_model_protocol=max(
            p.stat().st_mtime for p in (root / "phase4-evaluation").glob("*/stage_manifest.json")
        )
        - anchor.stat().st_mtime,
        timing_anchor=str(anchor),
        timing_anchor_mtime=anchor.stat().st_mtime,
        cpu_measurement=(
            "Stage wall durations, not process CPU seconds; concurrent sums must "
            "not be called end-to-end runtime. The wall anchor excludes earlier "
            "instruction/setup time. Uninstrumented tiny audits remain in execution"
            " logs."
        ),
    )
    (out / "budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    (out / "execution_manifest.json").write_text(
        json.dumps(
            dict(
                source_and_log_sha256={
                    str(p): file_sha256(p)
                    for base in [logs, executed]
                    for p in base.iterdir()
                    if p.is_file()
                },
                verification=str(args.verification / "completion_verification.json"),
                asset_manifest_sha256={
                    str(p): file_sha256(p) for p in (out / "assets").glob("*/asset_manifest.json")
                },
                registry_sha256=file_sha256(out / "experiment_registry.csv"),
                budget_sha256=file_sha256(out / "budget.json"),
                cpu_stage_costs_sha256=file_sha256(out / "cpu_stage_costs.csv"),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        dict(
            registry_rows=len(registry),
            upper_gpu_hours=upper / 3600,
            logs=len(list(logs.iterdir())),
        )
    )


if __name__ == "__main__":
    main()
