"""Compare registered libraries in a second frozen ESM2 representation on shared subsets."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_research_models import load_esm
from scipy.linalg import sqrtm
from threadpoolctl import threadpool_limits

from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.research.selection import keyed_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.selection / "protocol.json").read_text())
    checkpoint = Path(
        "work/measured_activity_research/model-assets-20260911/esm2_t33_650M_UR50D.pt"
    )
    seed, size = config["seeds"][0], config["subset_sizes"][0]
    samples = {
        p.stem: keyed_subset(read_fasta_sequences(p), size, seed)
        for p in sorted((args.selection / "prepare/libraries").glob("*.fasta"))
    }
    references = list(
        dict.fromkeys(
            s
            for s in read_fasta_sequences(Path("data/training/training.fasta"))
            if 8 <= len(s) <= 50
        )
    )
    samples["reference"] = keyed_subset(references, size, seed)
    sequences = sorted(set().union(*map(set, samples.values())))
    protocol = dict(
        model="esm2_t33_650M_UR50D",
        checkpoint_sha256=file_sha256(checkpoint),
        seed=seed,
        subset_size=size,
        batch_size=8,
        pooling="mean residues excluding BOS/EOS/pad",
        purpose=(
            "single seed representation sensitivity, not selection objective; no ESM-C equivalence"
        ),
        samples=samples,
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    started = time.monotonic()
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model, alphabet = load_esm(checkpoint)
    model.eval().cuda()
    convert = alphabet.get_batch_converter()
    features = []
    for start in range(0, len(sequences), 8):
        batch = sequences[start : start + 8]
        _, _, tokens = convert([(str(i), s) for i, s in enumerate(batch)])
        with torch.no_grad():
            tensor = model(tokens.cuda(), repr_layers=[model.num_layers])["representations"][
                model.num_layers
            ]
        features.extend(
            tensor[i, 1 : len(s) + 1].mean(0).cpu().numpy() for i, s in enumerate(batch)
        )
        if start % 256 == 0:
            print(f"ESM650 {start + len(batch)}/{len(sequences)}", flush=True)
    array = np.asarray(features, dtype=np.float32)
    np.save(args.output / "embeddings.npy", array, allow_pickle=False)
    pd.DataFrame({"sequence": sequences}).to_csv(args.output / "sequences.csv", index=False)
    gpu_seconds = time.monotonic() - started
    index = {s: i for i, s in enumerate(sequences)}
    reference = array[[index[s] for s in samples["reference"]]].astype(float)
    # Full 1280-dimensional covariance FBD; no projection fitted to candidate labels.
    mu = reference.mean(0)
    cov = np.cov(reference, rowvar=False)
    results = []
    with threadpool_limits(limits=4):
        for name, subset in samples.items():
            if name == "reference":
                continue
            values = array[[index[s] for s in subset]].astype(float)
            covariance = np.cov(values, rowvar=False)
            root = sqrtm(cov @ covariance)
            distance = np.sum((mu - values.mean(0)) ** 2) + np.trace(
                cov + covariance - 2 * root.real
            )
            results.append(
                dict(
                    variant=name,
                    seed=seed,
                    subset_size=size,
                    fbd_esm650=float(max(0, distance)),
                    sqrt_imaginary_max=float(np.abs(root.imag).max())
                    if np.iscomplexobj(root)
                    else 0.0,
                )
            )
            print(f"ESM650 {name}: FBD={distance:.4f}", flush=True)
    pd.DataFrame(results).to_csv(args.output / "representation_comparison.csv", index=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                gpu_seconds=gpu_seconds,
                total_seconds=time.monotonic() - started,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                code_sha256=file_sha256(Path(__file__)),
                artifacts_sha256={
                    p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file()
                },
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
