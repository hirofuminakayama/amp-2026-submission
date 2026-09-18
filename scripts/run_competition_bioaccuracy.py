"""Audit and rescore frozen predictions using molecule-balanced MIC and HC50 evidence."""

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from robust_apex_qd.evaluation.readiness import fresh_output, verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.bioaccuracy import (
    EndpointObservation,
    chemistry_key,
    dbaasp_hemolysis,
    joint_hit_bounds,
    mic_observation,
    molecular_labels,
    peptide_metrics,
    qmap_hc50,
)
from robust_apex_qd.research.biofeatures import BomanMode, feature_contract, research_features
from robust_apex_qd.research.biosplits import extend_identity, shared_folds
from robust_apex_qd.research.data import CANONICAL
from robust_apex_qd.research.mic_data import global_identity


class BioaccuracyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    esm8_checkpoint: Path
    prior_models: Path
    qmap: Path
    source_registration: Path
    metadata: Path
    mic_prepare: Path
    scale_root: Path
    frozen_pool: Path
    seeds: list[int]
    mic_threshold_um: float = Field(gt=0)
    selectivity_ratios: list[float]
    primary_ratio: float = Field(gt=0)
    outer_folds: int = Field(ge=2)
    inner_folds: int = Field(ge=2)
    identity_threshold: float = Field(gt=0, lt=1)
    cpu_threads: int = Field(ge=1)
    ridge_alphas: list[float]
    shared_gpu_hours: dict[str, float]
    candidate_freeze_jst: str
    method_freeze_jst: str
    deadline_jst: str


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def checked_manifest(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text())
    expected = {}
    for name, digest in manifest["artifacts_sha256"].items():
        target = path.parent / name
        if not target.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("Manifest artifact escapes its directory")
        expected[str(target)] = digest
    verify_hashes(expected)
    return {str(path): file_sha256(path), **expected}


def read_observations(path: Path) -> list[EndpointObservation]:
    with path.open() as stream:
        return [EndpointObservation.model_validate_json(line) for line in stream]


def archive_sources(output: Path, sources: list[Path]) -> dict[str, str]:
    inputs = {}
    for source in sources:
        relative = source.resolve().relative_to(Path.cwd().resolve())
        destination = output / "executed_sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        inputs[str(source)] = file_sha256(destination)
    return inputs


def finish_stage(output: Path, inputs: dict[str, str], started: float) -> None:
    verify_hashes(inputs)
    write_json(
        output / "manifest.json",
        dict(
            inputs_sha256=inputs,
            seconds=time.monotonic() - started,
            artifacts_sha256={
                str(p.relative_to(output)): file_sha256(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and p != output / "manifest.json"
            },
        ),
    )


def inventory(config: BioaccuracyConfig, output: Path, root: Path) -> dict[str, str]:
    del root
    inputs = checked_manifest(config.prior_models / "prepare/manifest.json")
    fits = []
    for path in sorted((config.prior_models / "fits").glob("*/fit_manifest.json")):
        manifest = json.loads(path.read_text())
        oof = path.parent / "oof.csv"
        expected = manifest["artifacts_sha256"]["oof.csv"]
        verify_hashes({str(oof): expected})
        inputs.update({str(path): file_sha256(path), str(oof): expected})
        fits.append(
            dict(
                id=path.parent.name,
                oof=str(oof),
                sha256=expected,
                arm=manifest["arm"],
                summary=manifest["summary"],
            )
        )
    queues = []
    for path in sorted(config.scale_root.glob("*-queue-r4/status.json")):
        status = json.loads(path.read_text())
        queues.append(
            dict(
                path=str(path),
                status=status["status"],
                current=status.get("current"),
                completed=len(status.get("records", [])),
            )
        )
    write_json(
        output / "baseline_inventory.json",
        dict(
            fits=fits,
            queues=queues,
            mic_prepare_complete=(config.mic_prepare / "manifest.json").exists(),
            frozen_pool=str(config.frozen_pool),
            adopted="B1/L2/C0",
            evidence_scope="Saved development OOF; new joint pipeline has not been evaluated",
        ),
    )
    write_json(
        output / "budget.json",
        dict(
            gpu_hours=config.shared_gpu_hours,
            total=sum(config.shared_gpu_hours.values()),
            concurrent_gpu_jobs=1,
            accounting="Shared future ceilings, not measured usage or additional MIC allocation",
            deadline_jst=config.deadline_jst,
            deadline_source="prior official API audit; recheck before release",
        ),
    )
    pd.DataFrame(
        [
            dict(
                owner="MIC research",
                produces="MIC nested fits and common split",
                requires="HC50 sequence union before joint fits",
                state="handoff_pending",
            ),
            dict(
                owner="competition scale",
                produces="completed pools, predictions, manifests",
                requires="source snapshot unchanged during jobs",
                state="running_snapshot",
            ),
            dict(
                owner="biological accuracy",
                produces="molecule metrics, HC50, joint selection",
                requires="completed verified MIC and scale artifacts",
                state="in_progress",
            ),
        ]
    ).to_csv(output / "handoff_matrix.csv", index=False)
    return inputs


def prepare(config: BioaccuracyConfig, output: Path, root: Path) -> dict[str, str]:
    del root
    inputs = checked_manifest(config.prior_models / "prepare/manifest.json")
    registration = json.loads(config.source_registration.read_text())
    entry = next(r for r in registration["sources"] if r["path"] == "qmap_hf/dbaasp.json")
    verify_hashes({str(config.qmap): entry["sha256"]})
    inputs[str(config.qmap)] = entry["sha256"]
    inputs[str(config.source_registration)] = file_sha256(config.source_registration)
    source = json.loads(config.qmap.read_text())
    source_index = {int(r["id"]): r for r in source}
    prior = pd.read_json(config.prior_models / "prepare/rows.jsonl", lines=True)
    observations = [mic_observation(r) for r in prior.to_dict("records")]
    exclusions = []
    for raw in source:
        if raw.get("hemolytic_hc50") is None:
            continue
        eligible = (
            8 <= len(raw["sequence"]) <= 50
            and not set(raw["sequence"]) - CANONICAL
            and not raw.get("nterminal")
            and not raw.get("cterminal")
            and not raw.get("bonds")
        )
        if not eligible:
            exclusions.append(
                dict(
                    source_id=raw["id"],
                    source="qmap",
                    reason="outside_reported_submission_chemistry",
                )
            )
            continue
        row = qmap_hc50(raw)
        if row is not None:
            observations.append(row)
    metadata_manifest = config.metadata / "metadata_manifest.json"
    meta = json.loads(metadata_manifest.read_text())
    inputs[str(metadata_manifest)] = file_sha256(metadata_manifest)
    eligible_ids = {int(r.source_id) for r in observations if r.endpoint == "consensus_hc50"}
    for saved in meta["records"]:
        if saved["status"] != "fetched" or int(saved["id"]) not in eligible_ids:
            continue
        path = config.metadata / f"{saved['id']}.json"
        verify_hashes({str(path): saved["sha256"]})
        inputs[str(path)] = saved["sha256"]
        peptide = json.loads(path.read_text())
        for assay in peptide.get("hemoliticCytotoxicActivities") or []:
            row, reason = dbaasp_hemolysis(peptide, assay, source_index[int(saved["id"])])
            if row is not None:
                observations.append(row)
            else:
                exclusions.append(dict(source_id=saved["id"], source="dbaasp", reason=reason))
    with (output / "endpoint_observations.jsonl").open("w") as stream:
        for row in observations:
            stream.write(row.model_dump_json() + "\n")
    hc50 = [r for r in observations if r.endpoint.endswith("hc50")]
    with (output / "hc50_observations.jsonl").open("w") as stream:
        for row in hc50:
            stream.write(row.model_dump_json() + "\n")
    labels = molecular_labels(observations, threshold=config.mic_threshold_um)
    labels.to_json(output / "peptide_target_labels.jsonl", orient="records", lines=True)
    molecular_labels(
        observations, endpoint="consensus_mic", threshold=config.mic_threshold_um
    ).to_json(output / "consensus_target_labels.jsonl", orient="records", lines=True)
    pd.DataFrame(exclusions).to_csv(output / "exclusions.csv", index=False)
    mapping = pd.DataFrame(
        [
            dict(
                observation_id=r.observation_id,
                sequence=r.sequence,
                molecule_id=chemistry_key(r),
                species=r.species,
                endpoint=r.endpoint,
                source_id=r.source_id,
                chemistry_support=json.dumps(r.chemistry_support, sort_keys=True),
            )
            for r in observations
        ]
    )
    mapping.to_csv(output / "observation_mapping.csv.gz", index=False)
    mapping.groupby(["endpoint", "chemistry_support"]).agg(
        observations=("observation_id", "size"), molecules=("molecule_id", "nunique")
    ).reset_index().to_csv(output / "chemistry_audit.csv", index=False)
    sequences = sorted({r.sequence for r in observations})
    write_json(output / "endpoint_sequences.json", sequences)
    write_json(
        output / "metric_contract.json",
        dict(
            schema_version=2,
            unit="molecule chemical profile x species; one selection vote",
            mic_threshold_um=config.mic_threshold_um,
            ks=[10, 25, 100],
            fraction=0.2,
            ambiguity="min/max across distinct reported assays",
            missing="unmeasured excluded; prediction coverage explicit; insufficient k is null",
            primary_endpoint="measured_mic",
            consensus="separate evidence cohort",
            chemistry="unknown profiles never join a known chemistry profile",
            storage="validated JSONL",
            legacy="observation_top20pct retained unchanged",
            development_only=True,
        ),
    )
    write_json(
        output / "data_inventory.json",
        dict(
            endpoints=dict(Counter(r.endpoint for r in observations)),
            sequences=len(sequences),
            molecular_targets=len(labels),
            qmap_hc50_raw=sum(r.get("hemolytic_hc50") is not None for r in source),
            measured_hc50_records=sum(r.endpoint == "measured_hc50" for r in observations),
            raw_acquisition="verified metadata only; unobserved RBC and exposure remain unknown",
            joint_fit_state="requires endpoint-union split; old MIC OOF is not joint OOF",
        ),
    )
    return inputs


def rescore(config: BioaccuracyConfig, output: Path, root: Path) -> dict[str, str]:
    inputs = checked_manifest(root / "prepare/manifest.json")
    labels = pd.read_json(root / "prepare/peptide_target_labels.jsonl", lines=True)
    mapping = pd.read_csv(root / "prepare/observation_mapping.csv.gz")
    mapping = mapping[mapping.endpoint == "measured_mic"]
    all_metrics, summaries = [], []
    for path in sorted((config.prior_models / "fits").glob("*/fit_manifest.json")):
        manifest = json.loads(path.read_text())
        oof = path.parent / "oof.csv"
        inputs[str(path)] = file_sha256(path)
        inputs[str(oof)] = manifest["artifacts_sha256"]["oof.csv"]
        verify_hashes({str(oof): inputs[str(oof)]})
        rows = mapping.merge(
            pd.read_csv(oof)[["observation_id", "species_prediction"]],
            on="observation_id",
            how="left",
            validate="one_to_one",
        )
        grouped = rows.groupby(["molecule_id", "species"]).species_prediction
        spread = grouped.max() - grouped.min()
        if (spread > 1e-5).any():
            raise ValueError("Species prediction changes across assays of the same molecule")
        predictions = grouped.mean().rename("prediction").reset_index()
        metrics = peptide_metrics(labels, predictions)
        for item in metrics:
            all_metrics.append(dict(model=path.parent.name, **item))
        top = [m for m in metrics if m["metric"] == "top20pct" and m["selected"]]
        arm = manifest["arm"]
        summaries.append(
            dict(
                model=path.parent.name,
                arm=arm["id"],
                family=arm["family"],
                seed=int(path.parent.name.rsplit("-s", 1)[1]),
                observation_top20pct=manifest["summary"]["macro_top20"],
                peptide_top20_lower=float(np.mean([m["precision_lower"] for m in top]))
                if top
                else None,
                peptide_top20_upper=float(np.mean([m["precision_upper"] for m in top]))
                if top
                else None,
                species=len(top),
                coverage=float(np.mean([m["coverage"] for m in top])) if top else None,
            )
        )
    pd.DataFrame(all_metrics).to_csv(output / "oof_metrics_peptide.csv", index=False)
    frame = pd.DataFrame(summaries)
    frame.to_csv(output / "model_metrics.csv", index=False)
    grouped = (
        frame.groupby(["family", "arm"])
        .agg(
            seeds=("seed", lambda s: json.dumps(sorted(s.tolist()))),
            seed_count=("seed", "nunique"),
            observation_top20pct=("observation_top20pct", "mean"),
            peptide_top20_lower=("peptide_top20_lower", "mean"),
            peptide_top20_upper=("peptide_top20_upper", "mean"),
            coverage=("coverage", "min"),
        )
        .reset_index()
    )
    grouped["balanced_seeds"] = grouped.seeds.eq(json.dumps(sorted(config.seeds)))
    grouped.to_csv(output / "arm_metrics.csv", index=False)
    eligible = grouped[grouped.balanced_seeds].sort_values(
        ["peptide_top20_lower", "peptide_top20_upper", "arm"], ascending=[False, False, True]
    )
    (output / "winner_change_report.md").write_text(
        "# Molecular OOF reaggregation\n\n"
        "Saved public-development OOF, not a new independent evaluation.\n"
        "Conflicting assays contribute bounds; consensus MIC is excluded. "
        "Species are equally weighted.\n"
        "Only the registered equal-seed cohorts enter the following comparison.\n\n"
        + eligible.to_csv(index=False)
        + "\nThis screening ranking is not adoption or measured generated-peptide success.\n"
    )
    return inputs


def features(config: BioaccuracyConfig, output: Path, root: Path) -> dict[str, str]:
    del config
    inputs = checked_manifest(root / "prepare/manifest.json")
    sequences = json.loads((root / "prepare/endpoint_sequences.json").read_text())
    contracts = []
    arms: list[tuple[BomanMode, bool, bool]] = [
        ("none", False, False),
        ("legacy", False, False),
        ("standard", False, False),
        ("both", False, False),
        ("standard", True, False),
        ("standard", False, True),
    ]
    for boman, local, interactions in arms:
        contract = feature_contract(boman, local=local, interactions=interactions)
        name = f"{boman}-local{int(local)}-interactions{int(interactions)}"
        frame = pd.DataFrame(
            [
                research_features(s, boman=boman, local=local, interactions=interactions)
                for s in sequences
            ]
        )
        if list(frame) != contract["names"] or not np.isfinite(frame.to_numpy()).all():
            raise ValueError("Feature contract mismatch")
        np.save(output / f"{name}.npy", frame.to_numpy(np.float64))
        contracts.append(dict(id=name, **contract))
    write_json(output / "feature_contracts.json", contracts)
    write_json(output / "sequences.json", sequences)
    return inputs


def joint_benchmark(config: BioaccuracyConfig, output: Path, root: Path) -> dict[str, str]:
    inputs = checked_manifest(root / "prepare/manifest.json")
    observations = read_observations(root / "prepare/endpoint_observations.jsonl")
    hc: dict[str, list[EndpointObservation]] = {}
    for row in observations:
        if row.endpoint.endswith("hc50"):
            hc.setdefault(chemistry_key(row), []).append(row)
    pairs = []
    for mic in observations:
        if not mic.endpoint.endswith("mic"):
            continue
        for hemolytic in hc.get(chemistry_key(mic), []):
            for ratio in config.selectivity_ratios:
                bounds = joint_hit_bounds(
                    mic, hemolytic, ratio=ratio, threshold=config.mic_threshold_um
                )
                if bounds is not None:
                    pairs.append(
                        dict(
                            molecule_id=chemistry_key(mic),
                            sequence=mic.sequence,
                            species=mic.species,
                            mic_id=mic.observation_id,
                            hc50_id=hemolytic.observation_id,
                            evidence=f"{mic.endpoint}+{hemolytic.endpoint}",
                            ratio=ratio,
                            rbc_species=hemolytic.rbc_species or "unknown",
                            hit_lower=bounds[0],
                            hit_upper=bounds[1],
                            condition_pairing="same chemical profile; joint assay unverified",
                        )
                    )
    frame = pd.DataFrame(pairs)
    frame.to_csv(output / "paired_mic_hc50.csv.gz", index=False)
    molecular = (
        frame.groupby(["molecule_id", "species", "evidence", "ratio", "rbc_species"])
        .agg(
            hit_lower=("hit_lower", "min"),
            hit_upper=("hit_upper", "max"),
            assay_pairs=("mic_id", "size"),
        )
        .reset_index()
    )
    molecular.to_csv(output / "molecular_joint_labels.csv", index=False)
    molecular.groupby(["species", "evidence", "ratio", "rbc_species"]).agg(
        molecules=("molecule_id", "nunique"),
        joint_lower=("hit_lower", "mean"),
        joint_upper=("hit_upper", "mean"),
    ).reset_index().to_csv(output / "joint_endpoint_metrics.csv", index=False)
    write_json(
        output / "coverage.json",
        dict(
            paired_molecules=frame.molecule_id.nunique(),
            pairs=len(frame),
            scope="observed label evidence inventory; no selection or prediction evaluation",
            measured_hc50_molecules=len(
                {chemistry_key(r) for r in observations if r.endpoint == "measured_hc50"}
            ),
        ),
    )
    return inputs


def split(config: BioaccuracyConfig, output: Path, root: Path) -> dict[str, str]:
    inputs = checked_manifest(root / "prepare/manifest.json")
    inputs.update(checked_manifest(config.mic_prepare / "manifest.json"))
    source = Path("src/robust_apex_qd/research/mic_data.py")
    inputs[str(source)] = file_sha256(source)
    old_protocol = json.loads((config.mic_prepare / "protocol.json").read_text())
    if old_protocol["identity_threshold"] != config.identity_threshold:
        raise ValueError("MIC and endpoint protocols disagree on identity threshold")
    sequences = json.loads((root / "prepare/endpoint_sequences.json").read_text())
    old_sequences = json.loads((config.mic_prepare / "sequences.json").read_text())
    old_matrix = np.load(config.mic_prepare / "identity.npy", allow_pickle=False)
    matrix = extend_identity(old_sequences, old_matrix, sequences, global_identity)
    folds = shared_folds(
        sequences,
        matrix,
        threshold=config.identity_threshold,
        outer_count=config.outer_folds,
        inner_count=config.inner_folds,
        seed=config.seeds[0],
    )
    np.save(output / "identity.npy", matrix)
    write_json(output / "sequences.json", sequences)
    write_json(output / "split_manifest.json", folds)
    rows = pd.read_json(config.mic_prepare / "rows.jsonl", lines=True)
    old_assignments = rows.homology_fold.copy()
    rows["homology_group"] = rows.sequence.map(folds["groups"])
    rows["homology_fold"] = rows.sequence.map(folds["outer"])
    rows.to_json(output / "rows.jsonl", orient="records", lines=True)
    assignments = np.array([folds["outer"][s] for s in sequences])
    distances = []
    for i, sequence in enumerate(sequences):
        other = assignments != assignments[i]
        distances.append(
            dict(
                sequence=sequence,
                fold=int(assignments[i]),
                max_train_identity=float(matrix[i, other].max()) if other.any() else None,
            )
        )
    pd.DataFrame(distances).to_csv(output / "cross_split_identity.csv", index=False)
    write_json(
        output / "mic_handoff.json",
        dict(
            prepared=str(output),
            rows="MIC rows retain original feature sequence_index",
            sequences=len(sequences),
            new_sequences=len(sequences) - len(old_sequences),
            changed_mic_observations=int((rows.homology_fold != old_assignments).sum()),
            old_fits_reusable=False,
            required_action="refit MIC and HC50 with this same prepared split",
            command=f"uv run --frozen python scripts/train_mic_models.py --prepared {output} "
            "--output work/competition_bioaccuracy/new-shared-mic",
            previous_mic_runs="retained as MIC-only development experiments",
        ),
    )
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/competition_bioaccuracy.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["inventory", "prepare", "rescore", "features", "joint", "split"],
        required=True,
    )
    args = parser.parse_args()
    config = BioaccuracyConfig.model_validate_json(args.config.read_text())
    output = args.output / args.stage
    fresh_output(
        output,
        [
            config.prior_models,
            config.qmap.parent,
            config.metadata,
            config.mic_prepare,
            config.scale_root,
            config.frozen_pool,
        ],
    )
    started = time.monotonic()
    sources = [Path(__file__), *Path("src/robust_apex_qd/research").glob("bio*.py")]
    inputs = archive_sources(output, [args.config, Path("uv.lock"), *sources])
    write_json(output / "protocol.json", config.model_dump(mode="json"))
    write_json(
        output / "execution.json",
        dict(
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            sources_sha256=inputs,
            stage=args.stage,
            biological_claim="development evidence only; no new molecules experimentally tested",
        ),
    )
    functions = dict(
        inventory=inventory,
        prepare=prepare,
        rescore=rescore,
        features=features,
        joint=joint_benchmark,
        split=split,
    )
    inputs.update(functions[args.stage](config, output, args.output))
    finish_stage(output, inputs, started)
    print(
        json.dumps(dict(stage=args.stage, output=str(output), seconds=time.monotonic() - started))
    )


if __name__ == "__main__":
    main()
