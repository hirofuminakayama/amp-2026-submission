"""Audit complete expanded-pool artifacts and count shared generation costs once."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_competition_models import write_json

from robust_apex_qd.evaluation.readiness import verify_hashes
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.validation.compliance import require_valid_submission


def generation_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    for record in records:
        if not record["newly_generated"]:
            continue
        key = record["source_path"]
        fields = {
            name: record[name]
            for name in ["input_rows", "generation_wall_seconds", "paused_seconds"]
        }
        if key in sources and sources[key] != fields:
            raise ValueError("Shared generation identity has inconsistent cost or count")
        if not 0 <= fields["paused_seconds"] <= fields["generation_wall_seconds"]:
            raise ValueError("Invalid recorded generation pause")
        sources[key] = fields
    wall = sum(r["generation_wall_seconds"] for r in sources.values())
    paused = sum(r["paused_seconds"] for r in sources.values())
    return dict(
        new_raw_count=sum(r["input_rows"] for r in sources.values()),
        generation_wall_seconds=wall,
        paused_seconds=paused,
        active_generation_wall_seconds=wall - paused,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/competition_scale.json"))
    parser.add_argument(
        "--donor-config", type=Path, default=Path("configs/competition_scale_donors.json")
    )
    parser.add_argument("--generation-config", type=Path, action="append", required=True)
    parser.add_argument("--pause-record", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh audit output directory")
    config = json.loads(args.config.read_text())
    donor_config = json.loads(args.donor_config.read_text())
    configs = [json.loads(p.read_text()) for p in args.generation_config]
    inputs = {
        str(p): file_sha256(p) for p in [args.config, args.donor_config, *args.generation_config]
    }
    verified: dict[str, str] = {}

    def verify_manifest(path: Path) -> dict[str, Any]:
        manifest = json.loads(path.read_text())
        hashes = {
            str(path.parent / name): digest for name, digest in manifest["artifacts_sha256"].items()
        }
        verify_hashes(hashes)
        verified.update(hashes)
        verified[str(path)] = file_sha256(path)
        return manifest

    paused = 0.0
    for path in args.pause_record:
        record = json.loads(path.read_text())
        if record["status"] != "DDIM resumed":
            raise ValueError("Generation pause has not been closed by a resume")
        paused += record["paused_seconds"]
        inputs[str(path)] = file_sha256(path)
    verify_manifest(args.comparison / "manifest.json")
    locked = json.loads((args.comparison / "comparison_set.json").read_text())
    verify_hashes(locked["inputs_sha256"])
    table = pd.read_csv(args.comparison / "full_candidate_comparison.csv").set_index("id")
    if set(table.index) != set(locked["candidates"]) or not table.index.is_unique:
        raise ValueError("Comparison candidate coverage differs from frozen artifact set")
    if not table.mean_rank.notna().all():
        raise ValueError("Missing scenario recommendation scores")
    spec = importlib.util.spec_from_file_location(
        "official_validator", "scripts/verify_submission.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Official validator unavailable")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    challenge = set(read_fasta_sequences(Path(config["challenge"])))
    pools = set()
    for candidate, item in locked["candidates"].items():
        path = Path(item["path"])
        if file_sha256(path / "manifest.json") != item["manifest_sha256"]:
            raise ValueError("Frozen candidate manifest changed")
        verify_manifest(path / "manifest.json")
        require_valid_submission(
            path, challenge, library_size=config["size"], top_k=config["top_k"]
        )
        full = official._verify_sequences(path / "library.fasta")
        if len(full) != config["size"]:
            raise ValueError("Wrong complete library count")
        official._verify_top(path / "top.fasta", full, config["top_k"])
        official._verify_no_overlap(full, challenge)
        official._veritfy_max_simularity(set(read_fasta_sequences(path / "top.fasta")), challenge)
        pools.add(path.parent.parent)
        if table.loc[candidate, "pool"] != path.parent.parent.name:
            raise ValueError("Candidate pool provenance changed")
    if not set(config["pools"]) <= {p.name for p in pools}:
        raise ValueError("A registered primary pool has no complete candidate")
    records, stage_costs = [], []
    all_sources = {**config["sources"], **donor_config["sources"]}
    source_costs = {}
    for name, source in all_sources.items():
        path = Path(source["path"])
        digest = file_sha256(path)
        if source.get("sha256") and digest != source["sha256"]:
            raise ValueError("Registered source changed")
        rows = pd.read_csv(path)
        if source.get("runs"):
            rows = rows[rows.run.isin(source["runs"])]
        if len(rows) != source["count"]:
            raise ValueError("Registered source count is incomplete")
        inputs[str(path)] = digest
        generation = verify_manifest(Path(source["manifest"])) if source.get("manifest") else None
        if generation is not None:
            verify_hashes(generation["input_sha256"])
            inputs.update(generation["input_sha256"])
        if generation is not None and (
            generation["config"] not in configs or generation["job"]["count"] != len(rows)
        ):
            raise ValueError("Executed generation differs from registered configuration")
        source_costs[name] = dict(
            source_path=str(path),
            source_sha256=digest,
            input_rows=len(rows),
            input_scope=source.get("input_scope", "raw generation attempts"),
            newly_generated=generation is not None,
            generation_wall_seconds=generation["runtime_seconds"] if generation else 0.0,
            paused_seconds=paused if name == "ddim" else 0.0,
            prior_generation_cost="sunk; excluded from new generation total"
            if generation is None
            else "",
        )
    for root in sorted(pools | {args.root / "donors"}):
        pool = pd.read_csv(root / "prepare/candidates.csv.gz")
        if pool.sequence.duplicated().any() or pool.sequence.isin(challenge).any():
            raise ValueError("Pool contains duplicate or exact-reference sequences")
        n = len(pool)
        if np.load(root / "features/candidate_embeddings.npy", mmap_mode="r").shape != (n, 320):
            raise ValueError("Embedding coverage differs from pool")
        for family in ["physchem", "linear8", "linear650", "mlp8", "finetune8"]:
            prediction = np.load(root / "models" / f"{family}.npz")
            if prediction["species"].shape != (n, 7) or prediction["strain"].shape != (n, 11):
                raise ValueError("Predictor coverage differs from pool")
        stages = sorted(root.glob("*/manifest.json"))
        shared = 0.0
        for path in stages:
            manifest = verify_manifest(path)
            seconds = manifest.get("seconds")
            if seconds is None:
                shared = max(shared, manifest.get("shared_build_seconds", 0.0))
            else:
                stage_costs.append(dict(pool=root.name, stage=path.parent.name, seconds=seconds))
        if shared:
            stage_costs.append(dict(pool=root.name, stage="shared-derived-build", seconds=shared))
        inventory = pd.read_csv(root / "prepare/pool_inventory.csv")
        if int(inventory.additional_unique.sum()) != n:
            raise ValueError("Source additions do not add up to the actual pool size")
        for row in inventory.to_dict("records"):
            records.append(dict(pool=root.name, **row, **source_costs[row["source"]]))
    totals = generation_totals(records)
    totals["processing_stage_wall_seconds"] = sum(row["seconds"] for row in stage_costs)
    totals["time_interpretation"] = (
        "Stage wall times include CPU and I/O, can overlap, and are not GPU-utilization hours. "
        "Shared generation is counted once; recorded DDIM pauses are excluded from "
        "active generation wall time."
    )
    args.output.mkdir(parents=True)
    pd.DataFrame(records).to_csv(args.output / "pool_inventory.csv", index=False)
    pd.DataFrame(stage_costs).to_csv(args.output / "stage_costs.csv", index=False)
    write_json(args.output / "budget.json", totals)
    write_json(
        args.output / "validation_report.json",
        dict(
            status="passed",
            candidates=len(locked["candidates"]),
            pools=len(pools),
            source_sha256=file_sha256(Path(__file__)),
            inputs_sha256=inputs,
            verified_sha256=verified,
            validators=["local", "vendored official"],
            scope="Actual expanded research artifacts; not new-method generation reproducibility",
        ),
    )
    write_json(
        args.output / "manifest.json",
        dict(
            source_sha256=file_sha256(Path(__file__)),
            artifacts_sha256={p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file()},
        ),
    )
    print(json.dumps(dict(status="passed", candidates=len(locked["candidates"]), **totals)))


if __name__ == "__main__":
    main()
