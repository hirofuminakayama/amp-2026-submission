"""Build supply-aware full libraries from scored diffusion and public-generator pools."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_competition_pool import FAMILIES
from run_competition_models import write_json
from threadpoolctl import threadpool_limits

from robust_apex_qd.evaluation.readiness import verify_hashes
from robust_apex_qd.features.embeddings import compute_embedding_diagnostics, file_sha256
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.ranking.objectives import percentile_score
from robust_apex_qd.research.scale import mix_library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donors", type=Path, required=True)
    parser.add_argument(
        "--base-library", choices=["Lref", "mix0.75", "L2", "lowOOD"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_hashes = {}
    for source in [args.base, args.donors]:
        for stage in ["prepare", "features", "apex", "models"]:
            manifest_path = source / stage / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            verify_hashes(
                {str(source / stage / k): v for k, v in manifest["artifacts_sha256"].items()}
            )
            input_hashes[str(manifest_path)] = file_sha256(manifest_path)
    config = json.loads((args.base / "models/manifest.json").read_text())["config"]
    base = pd.read_csv(args.base / "models/pool.csv.gz")
    donors = pd.read_csv(args.donors / "models/pool.csv.gz")
    mask = ~donors.sequence.isin(base.sequence)
    donor_rows = donors[mask].copy()
    donor_rows["raw_order"] = np.arange(len(donor_rows)) + int(base.raw_order.max()) + 1
    pool = pd.concat([base, donor_rows], ignore_index=True)
    if pool.sequence.duplicated().any():
        raise ValueError("Mixture pool contains duplicate sequences")
    args.output.mkdir(parents=True, exist_ok=False)
    source_hash = file_sha256(Path(__file__))
    started = time.monotonic()
    directories = ["prepare", "features", "apex", "models", "libraries"]
    for stage in directories:
        (args.output / stage).mkdir()
    vectors = np.concatenate(
        [
            np.load(args.base / "features/candidate_embeddings.npy"),
            np.load(args.donors / "features/candidate_embeddings.npy")[mask],
        ]
    )
    references = np.load(args.base / "features/reference_embeddings.npy")
    with threadpool_limits(limits=1):
        diagnostics = compute_embedding_diagnostics(
            vectors,
            references,
            seed=42,
            pca_components=64,
            cluster_count=512,
            pca_reference_subset=10000,
            thread_count=1,
        )
    pool["embedding_cluster"] = diagnostics.cluster
    pool["embedding_ood"] = diagnostics.embedding_ood
    pool["rankmean"] = np.mean(
        [
            percentile_score(pool.species.to_numpy()),
            *[percentile_score(-pool[f"{name}_mean_log2"].to_numpy()) for name in FAMILIES],
        ],
        axis=0,
    )
    np.save(args.output / "features/candidate_embeddings.npy", vectors)
    np.save(args.output / "features/reference_embeddings.npy", references)
    prepare = args.output / "prepare"
    pool[["candidate_id", "sequence", "raw_order", "length", "valid", "rejection_reason"]].to_csv(
        prepare / "candidates.csv.gz", index=False
    )
    write_fasta(
        [FastaRecord(i, s) for i, s in zip(pool.candidate_id, pool.sequence, strict=True)],
        prepare / "sequences.fasta",
    )
    inventory = pd.concat(
        [
            pd.read_csv(args.base / "prepare/pool_inventory.csv").assign(
                input_basis="raw diffusion records"
            ),
            pd.read_csv(args.donors / "prepare/pool_inventory.csv").assign(
                input_basis="valid emitted screen records; raw/internal filters in generator report"
            ),
        ],
        ignore_index=True,
    )
    if args.base.name == "baseline":
        refits = Path(config["refits"])
        original = pd.read_csv(refits / "pool_sequences.csv")
        index = {s: i for i, s in enumerate(original.sequence)}
        base650 = np.load(refits / "esm650.npy")[[index[s] for s in base.sequence]]
    else:
        base650 = np.load(args.base / "models/esm650.npy")
    np.save(
        args.output / "models/esm650.npy",
        np.concatenate([base650, np.load(args.donors / "models/esm650.npy")[mask]]),
    )
    for name in FAMILIES:
        left = np.load(args.base / "models" / f"{name}.npz")
        right = np.load(args.donors / "models" / f"{name}.npz")
        np.savez(
            args.output / "models" / f"{name}.npz",
            **{key: np.concatenate([left[key], right[key][mask]]) for key in ["species", "strain"]},
        )
    pool.to_csv(args.output / "models/pool.csv.gz", index=False)
    pool.to_csv(args.output / "apex/pool.csv.gz", index=False)
    np.save(
        args.output / "apex/tensor.npy",
        np.concatenate(
            [np.load(args.base / "apex/tensor.npy"), np.load(args.donors / "apex/tensor.npy")[mask]]
        ),
    )
    base_sequences = read_fasta_sequences(args.base / "libraries" / f"{args.base_library}.fasta")
    indexed = pool.set_index("sequence", drop=False)
    selected_base = indexed.loc[base_sequences].reset_index(drop=True)
    origin = pd.read_csv(args.donors / "prepare/source_rows.csv.gz")
    seen = set(base.sequence)
    for family in pd.read_csv(args.donors / "prepare/pool_inventory.csv").source:
        supplied = set(origin[origin.source == family].sequence) & set(pool.sequence)
        inventory.loc[inventory.source.eq(family), "additional_unique"] = len(supplied - seen)
        seen.update(supplied)
    if int(inventory.additional_unique.sum()) != len(pool):
        raise ValueError("Mixture source accounting differs from actual unique pool")
    inventory.to_csv(prepare / "pool_inventory.csv", index=False)
    library_names = ["L2", args.base_library]
    library_names = list(dict.fromkeys(library_names))
    for name in library_names:
        (args.output / "libraries" / f"{name}.fasta").write_bytes(
            (args.base / "libraries" / f"{name}.fasta").read_bytes()
        )
    canonical = {
        frozenset(read_fasta_sequences(args.output / "libraries" / f"{name}.fasta")): name
        for name in library_names
    }
    mixtures = []
    for family in sorted(origin.source.unique()):
        available = donor_rows[donor_rows.sequence.isin(origin[origin.source == family].sequence)]
        available = indexed.loc[available.sequence].reset_index(drop=True)
        for fraction in config["mixture_fractions"]:
            library, counts = mix_library(
                selected_base, available, size=config["size"], fraction=fraction
            )
            name = f"mix-{family}-{fraction}"
            identity = frozenset(library.sequence)
            if identity not in canonical:
                canonical[identity] = name
                library_names.append(name)
            write_fasta(
                [
                    FastaRecord(i, s)
                    for i, s in zip(library.candidate_id, library.sequence, strict=True)
                ],
                args.output / "libraries" / f"{name}.fasta",
            )
            mixtures.append(
                dict(
                    library=name,
                    canonical_library=canonical[identity],
                    family=family,
                    base_library=args.base_library,
                    **counts,
                )
            )
    pd.DataFrame(mixtures).to_csv(args.output / "libraries/mixture_inventory.csv", index=False)
    config = {
        **config,
        "pools": {args.output.name: [args.base.name, "donors"]},
        "library_variants": library_names,
    }
    write_json(args.output / "config.json", config)
    for stage in directories:
        directory = args.output / stage
        (directory / "executed_source.py").write_bytes(Path(__file__).read_bytes())
        write_json(
            directory / "manifest.json",
            dict(
                config=config,
                source_sha256=source_hash,
                source_pool_manifests=input_hashes,
                scope=(
                    "derived sequence-aligned pool; reused predictions and "
                    "recomputed mixture cluster labels"
                ),
                seconds=None,
                shared_build_seconds=time.monotonic() - started,
                time_accounting="one parent build across five derived views; do not sum",
                artifacts_sha256={
                    p.name: file_sha256(p) for p in directory.iterdir() if p.is_file()
                },
            ),
        )


if __name__ == "__main__":
    main()
