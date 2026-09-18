"""Measure frozen HemoPI2 order/chunk sensitivity on labeled HC50 molecules."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_competition_bioaccuracy import (
    archive_sources,
    checked_manifest,
    finish_stage,
    read_observations,
    write_json,
)
from train_competition_hc50 import regression_metrics

from robust_apex_qd.evaluation.oracles import find_hemopi2_script, run_hemopi2
from robust_apex_qd.evaluation.readiness import fresh_output
from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.io.fasta import read_fasta_sequences
from robust_apex_qd.research.bioaccuracy import chemistry_key


def run_context(
    oracle: Path,
    sequences: list[str],
    *,
    batch_size: int,
    output: Path,
) -> dict[str, float]:
    output.mkdir(parents=True)
    values = {}
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        ids = {f"c{i + start}": s for i, s in enumerate(batch)}
        write_json(output / f"batch-{start}.json", ids)
        predictions = run_hemopi2(oracle, ids)
        for value in predictions.values():
            values[value.sequence] = value.hc50_u_m
    pd.DataFrame([dict(sequence=s, hc50_um=v) for s, v in values.items()]).to_csv(
        output / "predictions.csv", index=False
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, default=Path("work/oracles/hemopi2"))
    parser.add_argument("--tops", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    root, oracle = args.root, args.oracle
    output = root / "hc50-oracle"
    fresh_output(output, [oracle, root / "prepare"])
    start = time.monotonic()
    inputs = checked_manifest(root / "prepare/manifest.json")
    inputs.update(
        archive_sources(
            output,
            [
                Path(__file__),
                Path("scripts/train_competition_hc50.py"),
                Path("scripts/run_competition_bioaccuracy.py"),
            ],
        )
    )
    script = find_hemopi2_script(oracle)
    if script is None:
        raise ValueError("Registered HemoPI2 environment is missing")
    for path in [
        Path(__file__),
        Path("scripts/train_competition_hc50.py"),
        script,
        oracle / "manifest.json",
    ]:
        inputs[str(path)] = file_sha256(path)
    rows = read_observations(root / "prepare/hc50_observations.jsonl")
    sequences = sorted({r.sequence for r in rows if len(r.sequence) <= 40})
    top_sequences = {}
    for path in args.tops:
        inputs[str(path)] = file_sha256(path)
        top_sequences[str(path)] = read_fasta_sequences(path)
    sequences = sorted(
        set(sequences) | {s for seqs in top_sequences.values() for s in seqs if len(s) <= 40}
    )
    extra = [
        s
        for s in json.loads((root / "prepare/endpoint_sequences.json").read_text())
        if s not in set(sequences) and len(s) <= 40
    ][:8]
    contexts: list[tuple[str, list[str], int]] = [
        ("ordered", sequences, 1000),
        ("reversed", list(reversed(sequences)), 1000),
        ("added", sequences + extra, 1000),
        ("chunks100", sequences, 100),
    ]
    write_json(
        output / "protocol.json",
        dict(
            sequences=sequences,
            extra=extra,
            contexts=[c[0] for c in contexts],
            excluded_over40=len({r.sequence for r in rows if len(r.sequence) > 40}),
            oracle_overlap="unknown pretrained training overlap; not independent OOF",
            runtime_policy="unmodified pinned extractor/model; exact batch context retained",
        ),
    )
    predictions = {}
    for name, ordered, batch_size in contexts:
        predictions[name] = run_context(
            oracle, ordered, batch_size=batch_size, output=output / name
        )
        print(json.dumps(dict(context=name, predicted=len(predictions[name]))), flush=True)
    baseline = np.array([predictions["ordered"][s] for s in sequences])
    base_top = set(sorted(sequences, key=lambda s: (-predictions["ordered"][s], s))[:100])
    sensitivity: list[dict[str, Any]] = []
    for name, mapping in predictions.items():
        values = np.array([mapping[s] for s in sequences])
        top = set(sorted(sequences, key=lambda s: (-mapping[s], s))[:100])
        sensitivity.append(
            dict(
                context=name,
                sequences=len(sequences),
                max_abs_um=float(abs(values - baseline).max()),
                mean_abs_um=float(abs(values - baseline).mean()),
                spearman=float(pd.Series(values).corr(pd.Series(baseline), method="spearman")),
                top100_overlap=len(top & base_top),
                crosses100=int(((values >= 100) != (baseline >= 100)).sum()),
                crosses128=int(((values >= 128) != (baseline >= 128)).sum()),
            )
        )
    pd.DataFrame(sensitivity).to_csv(output / "hemopi_batch_sensitivity.csv", index=False)
    evaluation = []
    for row in rows:
        if row.sequence not in predictions["ordered"]:
            continue
        for name, mapping in predictions.items():
            evaluation.append(
                dict(
                    context=name,
                    molecule_id=chemistry_key(row),
                    endpoint=row.endpoint,
                    observation_id=row.observation_id,
                    sequence=row.sequence,
                    relation=row.relation,
                    observed_um=row.value_um,
                    hc50_um=mapping[row.sequence],
                    rbc_species=row.rbc_species,
                )
            )
    frame = pd.DataFrame(evaluation)
    frame.to_csv(output / "hc50_oracle_predictions.csv", index=False)
    metrics = []
    for (name, endpoint), cohort in frame[frame.relation == "="].groupby(["context", "endpoint"]):
        metrics.append(
            dict(
                context=name,
                endpoint=endpoint,
                **regression_metrics(
                    np.log2(cohort.observed_um.to_numpy()), np.log2(cohort.hc50_um.to_numpy())
                ),
            )
        )
    pd.DataFrame(metrics).to_csv(output / "hc50_oracle_metrics.csv", index=False)
    candidate_metrics = []
    for path, seqs in top_sequences.items():
        eligible = [s for s in seqs if s in predictions["ordered"]]
        for context, mapping in predictions.items():
            values = np.array([mapping[s] for s in eligible])
            candidate_metrics.append(
                dict(
                    top=path,
                    context=context,
                    total=len(seqs),
                    predicted=len(eligible),
                    median_hc50_um=float(np.median(values)) if len(values) else None,
                    fraction_ge100=float((values >= 100).mean()) if len(values) else None,
                )
            )
    pd.DataFrame(candidate_metrics).to_csv(output / "candidate_hc50_diagnostics.csv", index=False)
    finish_stage(output, inputs, start)


if __name__ == "__main__":
    main()
