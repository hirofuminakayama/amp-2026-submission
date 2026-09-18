"""Run the pinned HydrAMP starter in its own locked Python environment."""

import argparse
import hashlib
import importlib
import json
import os
import resource
import time
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"
    started = time.monotonic()
    inputs = [args.starter / "uv.lock", Path(__file__)]
    inputs.extend(p for p in (args.starter / "checkpoint").rglob("*") if p.is_file())
    protocol = {
        "seed": args.seed,
        "count": args.count,
        "min_length": 10,
        "max_length": 25,
        "filter_out": True,
        "n_attempts": 1,
        "softmax": True,
        "input_sha256": {str(p): sha256(p) for p in inputs},
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    HydrAMPGenerator = importlib.import_module("amp.inference.inference").HydrAMPGenerator

    generator = HydrAMPGenerator(
        model_path=str(args.starter / "checkpoint/model"),
        decomposer_path=str(args.starter / "checkpoint/pca_decomposer.joblib"),
        softmax=True,
    )
    selected = []
    raw = []
    rounds = []
    for round_index in range(20):
        count = args.count - len(selected)
        if count <= 0:
            break
        batch = generator.unconstrained_generation(
            mode="amp",
            n_target=count,
            seed=args.seed + round_index,
            filter_out=True,
            properties=True,
            n_attempts=1,
        )
        rounds.append({"seed": args.seed + round_index, "requested": count, "returned": len(batch)})
        for item in batch:
            sequence = str(item["sequence"])
            raw.append({"sequence": sequence, "amp": float(item["amp"]), "mic": float(item["mic"])})
            if 10 <= len(sequence) <= 25 and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
                selected.append(sequence)
        print(f"round={round_index} retained={len(selected)}", flush=True)
    (args.output / "raw.json").write_text(json.dumps(raw, indent=2) + "\n")
    if len(selected) != args.count:
        raise ValueError("Did not obtain requested length-valid sample within twenty rounds")
    (args.output / "raw.fasta").write_text(
        "".join(f">seq{i}\n{s}\n" for i, s in enumerate(selected))
    )
    manifest = {
        **protocol,
        "runtime_seconds": time.monotonic() - started,
        "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_cuda_bytes": 0,
        "rounds": rounds,
        "raw_emitted_count": len(raw),
        "raw_decoder_denominator_available": False,
        "artifacts_sha256": {p.name: sha256(p) for p in args.output.iterdir()},
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"seconds": manifest["runtime_seconds"], "count": len(selected)}))


if __name__ == "__main__":
    main()
