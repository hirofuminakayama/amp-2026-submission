"""Run fixed small generator comparisons without changing submission generation."""

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch

from ampdiffusion_starter_kit.generate import _decode, load_model, set_seed
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.generation.sampler import (
    LengthPolicy,
    build_length_quotas,
    generate_candidates,
    set_process_determinism,
    write_candidates_csv,
)
from robust_apex_qd.io.fasta import FastaRecord, write_fasta
from robust_apex_qd.research.generation import ddim_sample


def jobs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {
        "smoke": {
            "count": 128,
            "temperature": 0.75,
            "steps": 1000,
            "seed": 42,
            "min_length": 20,
            "max_length": 20,
        }
    }
    for seed in config["seeds"]:
        for temperature in config["temperatures"]:
            result[f"length-t{temperature}-s{seed}"] = {
                "count": config["length_count"],
                "temperature": temperature,
                "steps": 1000,
                "seed": seed,
                "min_length": config["min_length"],
                "max_length": config["max_length"],
            }
        for steps in config["sampling_steps"]:
            result[f"steps-{steps}-s{seed}"] = {
                "count": config["sampler_count"],
                "temperature": 0.75,
                "steps": steps,
                "seed": seed,
                "min_length": config["min_length"],
                "max_length": config["max_length"],
            }
        result[f"paired-s{seed}"] = {
            "count": 1000,
            "temperature": 0.75,
            "steps": 1000,
            "seed": seed,
            "min_length": config["min_length"],
            "max_length": config["paired_max_length"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research_generation.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    job = jobs(config)[args.job]
    output = args.output / args.job
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inputs = [
        args.config,
        Path(config["checkpoint"]),
        Path(config["reference"]),
        Path(config["challenge"]),
        Path("uv.lock"),
        Path(__file__),
        Path("src/robust_apex_qd/research/generation.py"),
        Path("src/ampdiffusion_starter_kit/model.py"),
        Path("src/ampdiffusion_starter_kit/generate.py"),
        Path("src/robust_apex_qd/generation/sampler.py"),
    ]
    protocol = {
        "config": config,
        "job": job,
        "input_sha256": {str(p): file_sha256(p) for p in inputs},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if not torch.cuda.is_available():
        raise RuntimeError("Official generator requires CUDA")
    set_process_determinism(job["seed"], config["cpu_threads"])
    torch.cuda.reset_peak_memory_stats()
    model, esm, _, indices = load_model(Path(config["checkpoint"]), torch.device("cuda"))
    quotas = build_length_quotas(
        job["count"],
        LengthPolicy.EMPIRICAL_TEMPERED,
        job["min_length"],
        job["max_length"],
        Path(config["reference"]),
        job["temperature"],
    )
    batch_times = []

    def backend(design_length: int, batch_size: int, round_seed: int) -> list[str]:
        set_seed(round_seed)
        start = time.monotonic()
        if job["steps"] == 1000:
            sampled = model.sample(batch_size=batch_size, design_len=design_length + 2)
        else:
            sampled = ddim_sample(model, batch_size, design_length + 2, job["steps"])
        sequences = _decode(esm, sampled, indices, design_length)
        torch.cuda.synchronize()
        batch_times.append(
            {
                "length": design_length,
                "count": batch_size,
                "seed": round_seed,
                "seconds": time.monotonic() - start,
            }
        )
        print(f"Batch {len(batch_times)}: {batch_size} x {design_length}", flush=True)
        return sequences

    candidates = generate_candidates(
        job["count"], config["batch_size"], job["seed"], quotas, backend=backend
    )
    write_candidates_csv(candidates, output / "candidates.csv.gz")
    write_fasta([FastaRecord(c.candidate_id, c.sequence) for c in candidates], output / "raw.fasta")
    manifest = {
        **protocol,
        "runtime_seconds": time.monotonic() - started,
        "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "quotas": quotas,
        "batches": batch_times,
        "artifacts_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"job": args.job, "seconds": manifest["runtime_seconds"]}))


if __name__ == "__main__":
    main()
