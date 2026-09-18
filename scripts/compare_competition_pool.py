"""Select full libraries and ranked candidates from a registered, scored pool."""

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_competition_models import write_json
from threadpoolctl import threadpool_limits

from robust_apex_qd.evaluation.developability import evaluate_developability
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.generation.sampler import LengthPolicy, build_length_quotas
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.ranking.objectives import percentile_score
from robust_apex_qd.research.exploration import ExplorationConstraints, select_exploration_top
from robust_apex_qd.research.scale import reference_mix
from robust_apex_qd.selection.library import LibraryCandidate, select_library
from robust_apex_qd.selection.top import maximum_levenshtein_ratio, maximum_local_similarity
from robust_apex_qd.validation.compliance import require_valid_submission


def save_fasta(frame: pd.DataFrame, path: Path) -> None:
    write_fasta(
        [FastaRecord(i, s) for i, s in zip(frame.candidate_id, frame.sequence, strict=True)], path
    )


def libraries(config: dict[str, Any], name: str, root: Path, output: Path) -> None:
    pool = pd.read_csv(root / "models/pool.csv.gz")
    quotas = build_length_quotas(
        config["size"], LengthPolicy.EMPIRICAL_TEMPERED, 10, 40, Path(config["reference"]), 0.75
    )
    candidates = [
        LibraryCandidate.model_validate({k: r[k] for k in LibraryCandidate.model_fields})
        for r in pool.to_dict("records")
    ]
    chosen = select_library(candidates, variant="L2", size=config["size"], target_quotas=quotas)
    l2 = pool.set_index("sequence").loc[[r.sequence for r in chosen.selected]].reset_index()
    reference = pd.read_csv(root / "features/reference_clusters.csv")
    variants = {
        "L2": l2,
        "mix0.75": reference_mix(pool, l2, reference, 0.75),
        "Lref": reference_mix(pool, l2, reference, 1.0),
    }
    quality = percentile_score(
        pool.physchem_ood.to_numpy(), higher_is_better=False
    ) + percentile_score(pool.embedding_ood.to_numpy(), higher_is_better=False)
    variants["lowOOD"] = (
        pool.assign(quality=quality)
        .sort_values(["quality", "raw_order", "sequence"], ascending=[False, True, True])
        .head(config["size"])
    )
    for key, frame in variants.items():
        path = output / f"{key}.fasta"
        save_fasta(frame, path)
        if name == "baseline":
            previous = Path(config["prior_selection"]) / "prepare/libraries" / path.name
            if read_fasta_sequences(path) != read_fasta_sequences(previous):
                raise ValueError(f"Reselected baseline library differs: {key}")


def requests(config: dict[str, Any], rankers: list[str] | None = None) -> list[dict[str, str]]:
    rankers = config["rankers"] if rankers is None else rankers
    if not rankers or len(set(rankers)) != len(rankers):
        raise ValueError("Require a nonempty unique ranker list")
    result = [
        dict(
            id=f"library-{name}", library=name, ranker="B1", constraint="current", factor="library"
        )
        for name in config["library_variants"]
    ]
    result += [
        dict(
            id=f"rank-{ranker}", library="L2", ranker=ranker, constraint="current", factor="ranker"
        )
        for ranker in rankers
        if ranker != "B1"
    ]
    result += [
        dict(
            id=f"constraint-{name}", library="L2", ranker="B1", constraint=name, factor="constraint"
        )
        for name in config["constraint_variants"]
        if name != "current"
    ]
    return result


def tops(
    config: dict[str, Any], name: str, root: Path, output: Path, specs: list[dict[str, str]]
) -> None:
    pool = pd.read_csv(root / "models/pool.csv.gz")
    exploration = json.loads(Path("configs/competition_exploration.json").read_text())
    prior = Path(config["prior_selection"])
    cache = json.loads((prior / "tops/similarity_cache.json").read_text())
    challenge_refs = tuple(read_fasta_sequences(Path(config["challenge"])))
    known_refs = tuple(read_fasta_sequences(Path(config["reference"])))
    module_spec = importlib.util.spec_from_file_location(
        "official_validator", "scripts/verify_submission.py"
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("Official validator unavailable")
    official = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(official)

    def challenge(sequence: str) -> float:
        if sequence not in cache["challenge"]:
            cache["challenge"][sequence] = maximum_levenshtein_ratio(sequence, challenge_refs)
        return cache["challenge"][sequence]

    def known(sequence: str) -> float:
        if sequence not in cache["known"]:
            cache["known"][sequence] = maximum_local_similarity(sequence, known_refs)
        return cache["known"][sequence]

    results = []
    for request in specs:
        started = time.monotonic()
        dest = output / request["id"]
        dest.mkdir()
        path = root / "libraries" / f"{request['library']}.fasta"
        seqs = read_fasta_sequences(path)
        frame = pool[pool.sequence.isin(seqs)].copy()
        if len(frame) != config["size"]:
            raise ValueError("Library is not covered by aligned pool predictions")
        frame["score"] = percentile_score(frame[request["ranker"]].to_numpy())
        constraints = ExplorationConstraints.model_validate(
            exploration["constraints"][request["constraint"]]
        )
        result: dict[str, Any]
        try:
            top = select_exploration_top(
                frame, "score", constraints, config["top_k"], challenge, known
            )
        except ValueError as error:
            if not str(error).startswith("infeasible:"):
                raise
            result = dict(**request, status="infeasible", reason=str(error))
        else:
            top["rank"] = np.arange(1, len(top) + 1)
            top["challenge_similarity"] = top.sequence.map(challenge)
            top["dev_pass"] = top.sequence.map(
                lambda s: evaluate_developability(s).hard_filter_pass
            )
            top.to_csv(dest / "ranking.csv", index=False)
            save_fasta(top, dest / "top.fasta")
            (dest / "library.fasta").write_bytes(path.read_bytes())
            full = official._verify_sequences(dest / "library.fasta")
            if len(full) != config["size"]:
                raise ValueError("Wrong library count")
            official._verify_top(dest / "top.fasta", full, config["top_k"])
            official._verify_no_overlap(full, set(challenge_refs))
            official._veritfy_max_simularity(set(top.sequence), set(challenge_refs))
            if (
                name == "baseline"
                and request["id"] == "library-L2"
                and top.sequence.tolist()
                != read_fasta_sequences(Path(config["baseline"]) / "top.fasta")
            ):
                raise ValueError("B1/L2 control differs from the frozen ranked Top")
            pd.DataFrame(
                {
                    "rank": top["rank"],
                    "candidate_id": top.candidate_id,
                    "sequence": top.sequence,
                    "final_score": top.score,
                }
            ).to_csv(dest / "ranking.tsv", sep="\t", index=False)
            write_json(
                dest / "validation_report.json",
                dict(
                    valid=True,
                    scope="official artifact checks, not generation reproducibility",
                    validator_sha256=file_sha256(Path("scripts/verify_submission.py")),
                ),
            )
            result = dict(
                **request,
                status="complete",
                apex_activity=float(top.species.mean()),
                apex_weak_species=float(top.worst3.mean()),
                safety=float(top.dev_pass.mean()),
                diversity=int(top.embedding_cluster.nunique()),
                **{
                    f"{family}_mean_log2": float(top[f"{family}_mean_log2"].mean())
                    for family in ["physchem", "linear8", "linear650", "mlp8", "finetune8"]
                },
            )
        result["runtime_seconds"] = time.monotonic() - started
        result["pool"] = name
        write_json(
            dest / "manifest.json",
            dict(
                **result,
                schema_version=1,
                library_count=config["size"] if result["status"] == "complete" else 0,
                top_count=config["top_k"] if result["status"] == "complete" else 0,
                manual_intervention=False,
                artifacts_sha256={p.name: file_sha256(p) for p in dest.iterdir() if p.is_file()},
            ),
        )
        if result["status"] == "complete":
            require_valid_submission(
                dest, set(challenge_refs), library_size=config["size"], top_k=config["top_k"]
            )
        results.append(result)
        print(f"{name}/{request['id']}: {result['status']}", flush=True)
    pd.DataFrame(results).to_csv(output / "candidate_comparison.csv", index=False)
    write_json(output / "similarity_cache.json", cache)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/competition_scale.json"))
    parser.add_argument("--pool", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["libraries", "tops", "compound"], required=True)
    parser.add_argument("--compound-specs", type=Path)
    parser.add_argument("--rankers-config", type=Path)
    parser.add_argument("--output-stage")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.root / args.pool
    if args.pool not in config["pools"]:
        raise ValueError("Unregistered pool")
    for stage in ["prepare", "features", "apex", "models"]:
        previous = root / stage
        manifest = json.loads((previous / "manifest.json").read_text())
        if manifest["config"] != config:
            raise ValueError("Pool configuration changed")
        for name, digest in manifest["artifacts_sha256"].items():
            if file_sha256(previous / name) != digest:
                raise ValueError("Scored pool artifact changed")
    output = root / (args.output_stage or args.stage)
    output.mkdir(exist_ok=False)
    source_hash = file_sha256(Path(__file__))
    (output / "executed_source.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    with threadpool_limits(limits=1):
        if args.stage == "libraries":
            libraries(config, args.pool, root, output)
        else:
            rankers = json.loads(args.rankers_config.read_text()) if args.rankers_config else None
            specs = requests(config, rankers)
            if args.stage == "compound":
                if args.compound_specs is None:
                    raise ValueError("Compound selection requires the earlier scored selection")
                specs = json.loads(args.compound_specs.read_text())
            tops(config, args.pool, root, output, specs)
    write_json(
        output / "manifest.json",
        dict(
            seconds=time.monotonic() - started,
            source_sha256=source_hash,
            config=config,
            selection_inputs_sha256={
                str(p): file_sha256(p)
                for p in [args.rankers_config, args.compound_specs]
                if p is not None
            },
            artifacts_sha256={
                str(p.relative_to(output)): file_sha256(p) for p in output.rglob("*") if p.is_file()
            },
        ),
    )


if __name__ == "__main__":
    main()
