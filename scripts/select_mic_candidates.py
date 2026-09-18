"""Pass saved MIC predictions through the existing full-library and Top validators."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from compare_competition_pool import tops
from run_mic_research import checked_manifest, finish_stage, write_json
from threadpoolctl import threadpool_limits

from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("work/competition_exploration/20260913-scale/processed/baseline"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    config_path = Path("configs/competition_scale.json")
    config = json.loads(config_path.read_text())
    fresh_output(args.output, [args.handoff, args.pool])
    inputs = checked_manifest(args.handoff / "manifest.json")
    inputs.update(checked_manifest(args.pool / "models/manifest.json"))
    inputs[str(config_path)] = file_sha256(config_path)
    pool = pd.read_csv(args.pool / "models/pool.csv.gz")
    names = json.loads((args.handoff / "handoff.json").read_text())["selected"]
    requests = [
        dict(id="library-L2", library="L2", ranker="B1", constraint="current", factor="control")
    ]
    for name in names:
        inputs.update(checked_manifest(args.handoff / name / "manifest.json"))
        for path in sorted((args.handoff / name).glob("ranking-*.csv.gz")):
            frame = pd.read_csv(path)
            if frame.sequence.duplicated().any() or set(frame.sequence) != set(pool.sequence):
                raise ValueError("Rankings must exactly cover the saved pool")
            method = path.name.removeprefix("ranking-").removesuffix(".csv.gz")
            column = f"{name}-{method}"
            pool[column] = -pool.sequence.map(frame.set_index("sequence").score)
            inputs[str(path)] = file_sha256(path)
            requests.append(
                dict(
                    id=column,
                    library="L2",
                    ranker=column,
                    constraint="current",
                    factor="MIC predictor",
                )
            )
    (args.output / "models").mkdir()
    pool.to_csv(args.output / "models/pool.csv.gz", index=False)
    (args.output / "libraries").mkdir()
    source = Path(config["prior_selection"]) / "prepare/libraries/L2.fasta"
    inputs[str(source)] = file_sha256(source)
    (args.output / "libraries/L2.fasta").write_bytes(source.read_bytes())
    write_json(args.output / "protocol.json", dict(config=config, requests=requests))
    destination = args.output / "tops"
    destination.mkdir()
    with threadpool_limits(2):
        tops(config, "baseline", args.output, destination, requests)
    control = read_fasta_sequences(destination / "library-L2/top.fasta")
    for name in names:
        if read_fasta_sequences(destination / f"{name}-w1/top.fasta") != control:
            raise ValueError("APEX-only handoff did not reproduce the validated control")
    finish_stage(
        args.output,
        inputs,
        start,
        adopted=False,
        scope="existing L2 library and ranked Tops; not full generation reproducibility",
    )


if __name__ == "__main__":
    main()
