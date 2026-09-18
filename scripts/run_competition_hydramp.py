"""Exercise published HydrAMP analogue conditioning without inventing rejected outputs."""

import argparse
import hashlib
import importlib
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        MPLBACKEND="Agg",
        TF_CPP_MIN_LOG_LEVEL="3",
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="2",
        OMP_NUM_THREADS="2",
    )
    sequences = list(
        dict.fromkeys(
            line.strip()
            for line in args.input.read_text().splitlines()
            if line and not line.startswith(">")
        )
    )
    sequences = [s for s in sequences if 15 <= len(s) <= 25][: args.count]
    if len(sequences) != args.count:
        raise ValueError("Insufficient unique conditional inputs")
    protocol = dict(
        seed=args.seed,
        count=args.count,
        mode="analogue",
        temperature=5.0,
        attempts=1,
        filtering_criteria="discovery",
        min_length=10,
        max_length=25,
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        input_path=str(args.input),
        device="cpu",
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    start = time.monotonic()
    cls = importlib.import_module("amp.inference.inference").HydrAMPGenerator
    generator = cls(
        model_path=str(args.starter / "checkpoint/model"),
        decomposer_path=str(args.starter / "checkpoint/pca_decomposer.joblib"),
        softmax=True,
    )
    result = generator.analogue_generation(
        sequences=sequences, seed=args.seed, filtering_criteria="discovery", n_attempts=1, temp=5.0
    )
    (args.output / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "run_manifest.json").write_text(
        json.dumps(
            dict(
                **protocol,
                seconds=time.monotonic() - start,
                returned_inputs=len(result),
                raw_decoder_denominator_available=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                artifacts_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in args.output.iterdir()
                    if p.is_file()
                },
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
