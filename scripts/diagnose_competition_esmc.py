"""Built with ESM: isolated, pinned ESM-C representation sensitivity for saved subsets."""

import argparse
import hashlib
import json
import time
from importlib import import_module, metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import sqrtm
from threadpoolctl import threadpool_limits


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if metadata.version("esm") != "3.2.1":
        raise ValueError("Require the separately locked esm 3.2.1 environment")
    expected = "323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea"
    if sha256(args.weights) != expected:
        raise ValueError("ESM-C 300M weights differ from the pinned upstream object")
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.subsets.read_text())
    samples = source["samples"]
    sequences = sorted(set().union(*map(set, samples.values())))
    protocol = dict(
        model="esmc_300m_2024_12",
        sdk="esm==3.2.1",
        weight_sha256=expected,
        revision="7f10b20ae75017b2dbc884070e03434515709a8d",
        seed=source["seed"],
        subset_size=source["subset_size"],
        samples=samples,
        batch_size=8,
        pooling="mean residue embeddings excluding BOS/EOS/pad",
        attention="torch attention; flash-attn disabled",
        dimensions=960,
        attribution="Built with ESM; ESM-C 300M under Cambrian Open License Agreement",
        purpose="representation sensitivity only; no model training or submission integration",
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    started = time.monotonic()
    torch.manual_seed(source["seed"])
    torch.set_num_threads(4)
    tokenizer = import_module("esm.tokenization").get_esmc_model_tokenizers()
    model = import_module("esm.models.esmc").ESMC(
        d_model=960,
        n_heads=15,
        n_layers=30,
        tokenizer=tokenizer,
        use_flash_attn=False,
    )
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval().cuda()
    features = []
    for start in range(0, len(sequences), 8):
        batch = sequences[start : start + 8]
        tokens = model._tokenize(batch)
        with torch.no_grad():
            tensor = model(sequence_tokens=tokens).embeddings
        features.extend(
            tensor[i, 1 : len(s) + 1].mean(0).cpu().numpy() for i, s in enumerate(batch)
        )
        if start % 256 == 0:
            print(f"ESM-C {start + len(batch)}/{len(sequences)}", flush=True)
    array = np.asarray(features, dtype=np.float32)
    if array.shape != (len(sequences), 960) or not np.isfinite(array).all():
        raise ValueError("ESM-C embedding shape/finite contract failed")
    np.save(args.output / "embeddings.npy", array, allow_pickle=False)
    pd.DataFrame({"sequence": sequences}).to_csv(args.output / "sequences.csv", index=False)
    gpu_seconds = time.monotonic() - started
    index = {s: i for i, s in enumerate(sequences)}
    reference = array[[index[s] for s in samples["reference"]]].astype(float)
    mu, cov = reference.mean(0), np.cov(reference, rowvar=False)
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
                    seed=source["seed"],
                    subset_size=source["subset_size"],
                    fbd_esmc300=float(max(0, distance)),
                )
            )
            print(f"ESM-C {name} FBD={distance:.4f}", flush=True)
    pd.DataFrame(results).to_csv(args.output / "representation_comparison.csv", index=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                gpu_seconds=gpu_seconds,
                total_seconds=time.monotonic() - started,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                code_sha256=sha256(Path(__file__)),
                input_protocol_sha256=sha256(args.subsets),
                artifacts_sha256={p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
