"""Run the frozen library and ranker from inference assets and generated sequences."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from compare_competition_pool import libraries
from process_competition_pool import apex, features
from run_research_models import load_esm
from threadpoolctl import threadpool_limits

from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.features.physchem import compute_features
from robust_apex_qd.io.fasta import FastaRecord, read_fasta_sequences, write_fasta
from robust_apex_qd.ranking.adopted import consensus_rank, select_adopted
from robust_apex_qd.ranking.predictor_bundle import predict_bundle
from robust_apex_qd.research.scale import merge_pool_sources

FAMILIES = ("physchem", "linear8", "linear650", "mlp8", "finetune8")


def infer_models(root: Path, assets: Path, device: str) -> None:
    destination = root / "models"
    destination.mkdir()
    pool = pd.read_csv(root / "apex/pool.csv.gz")
    sequences = pool.sequence.tolist()
    small = np.load(root / "features/candidate_embeddings.npy")
    predictions = {}
    hashes = {}
    for family in FAMILIES:
        if family == "physchem":
            vectors = np.asarray(
                [
                    [
                        *compute_features(s).values(),
                        *[s.count(a) / len(s) for a in "ACDEFGHIKLMNPQRSTVWY"],
                    ]
                    for s in sequences
                ],
                dtype=np.float32,
            )
        elif family == "linear650":
            checkpoint = assets / "esm2_t33_650M_UR50D.pt"
            hashes[str(checkpoint)] = file_sha256(checkpoint)
            model, alphabet = load_esm(checkpoint)
            model.eval().to(device)
            values = []
            with torch.no_grad():
                for start in range(0, len(sequences), 64):
                    batch = sequences[start : start + 64]
                    _, _, tokens = alphabet.get_batch_converter()(
                        [(str(i), sequence) for i, sequence in enumerate(batch)]
                    )
                    representation = model(tokens.to(device), repr_layers=[33])["representations"][
                        33
                    ]
                    values.extend(
                        representation[i, 1 : len(s) + 1].mean(0).cpu().numpy()
                        for i, s in enumerate(batch)
                    )
            vectors = np.asarray(values)
            np.save(destination / "esm650.npy", vectors)
            del model, representation
            gc.collect()
            torch.cuda.empty_cache()
        else:
            vectors = small
        species, strain = predict_bundle(assets / family, vectors, sequences, device=device)
        predictions[family] = species
        np.savez(destination / f"{family}.npz", species=species, strain=strain)
        pool[f"{family}_mean_log2"] = species.mean(1)
        for path in (assets / family).iterdir():
            hashes[str(path)] = file_sha256(path)
        print(f"Inference {family}: {len(pool)} sequences", flush=True)
    pool["rankmean"] = consensus_rank(pool, predictions)
    pool.to_csv(destination / "pool.csv.gz", index=False)
    (destination / "assets_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


def select(
    root: Path, output: Path, size: int, top_k: int, reference: Path, challenge: Path
) -> None:
    config = dict(size=size, reference=str(reference))
    (root / "libraries").mkdir()
    libraries(config, "deployed", root, root / "libraries")
    pool = pd.read_csv(root / "models/pool.csv.gz")
    library = read_fasta_sequences(root / "libraries/Lref.fasta")
    top = select_adopted(
        pool,
        library,
        top_k,
        set(read_fasta_sequences(challenge)),
        set(read_fasta_sequences(reference)),
    )
    (output / "library.fasta").write_bytes((root / "libraries/Lref.fasta").read_bytes())
    write_fasta(
        [FastaRecord(i, s) for i, s in zip(top.candidate_id, top.sequence, strict=True)],
        output / "top.fasta",
    )
    pd.DataFrame(
        dict(
            rank=np.arange(1, len(top) + 1),
            candidate_id=top.candidate_id,
            sequence=top.sequence,
            final_score=top.score,
        )
    ).to_csv(output / "ranking.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--stage", choices=["all", "models", "select"], default="all")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    config = dict(reference=str(args.reference), challenge=str(args.challenge))
    if args.stage == "all":
        if args.raw is None:
            parser.error("--raw is required for all stages")
        raw = pd.read_csv(args.raw)
        pool, provenance, inventory = merge_pool_sources(
            {"ddim": raw.sequence.tolist()}, set(read_fasta_sequences(args.challenge))
        )
        if len(pool) < args.size:
            raise ValueError("Insufficient generated unique candidates")
        prepared = args.root / "prepare"
        prepared.mkdir(parents=True)
        pool.to_csv(prepared / "candidates.csv.gz", index=False)
        provenance.to_csv(prepared / "source_rows.csv.gz", index=False)
        inventory.to_csv(prepared / "pool_inventory.csv", index=False)
        write_fasta(
            [FastaRecord(i, s) for i, s in zip(pool.candidate_id, pool.sequence, strict=True)],
            prepared / "sequences.fasta",
        )
        for stage in ["features", "apex"]:
            path = args.root / stage
            path.mkdir()
            if stage == "features":
                features(config, "deployed", args.root, path, args.device)
            else:
                apex(config, "deployed", args.root, path)
    if args.stage in ["all", "models"]:
        infer_models(args.root, Path(settings["inference"]["predictor_assets"]), args.device)
    if args.stage in ["all", "select"]:
        select(args.root, args.output, args.size, args.top_k, args.reference, args.challenge)


if __name__ == "__main__":
    torch.set_num_threads(4)
    with threadpool_limits(1):
        main()
