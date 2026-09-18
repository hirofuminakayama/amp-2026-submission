"""Choose MIC and HC50 families using inner scores, then combine untouched outer results."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from run_competition_bioaccuracy import archive_sources, checked_manifest, finish_stage, write_json

from robust_apex_qd.evaluation.readiness import fresh_output


def choose_family(audits: dict[str, dict[str, Any]]) -> str:
    choices = []
    for family, audit in audits.items():
        best = sorted(audit["candidates"], key=lambda r: (-r["inner_score"], r["arm"]))[0]
        if best["arm"] != audit["selected_arm"]:
            raise ValueError("Saved HC50 choice differs from inner-score ranking")
        choices.append((float(best["inner_score"]), family))
    return sorted(choices, key=lambda r: (-r[0], r[1]))[0][1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "adaptive-nested"
    fresh_output(output, [args.root / "prepare"])
    start = time.monotonic()
    inputs = archive_sources(
        output, [Path(__file__), Path("scripts/run_competition_bioaccuracy.py")]
    )
    results, audit_rows = [], []
    for seed in [42, 43, 44]:
        sources = dict(
            linear8=args.root / "nested-selection",
            **{
                "esm8-exact": args.root
                / ("nested-esm8-exact" if seed == 42 else f"nested-esm8-exact-s{seed}")
            },
        )
        audits, metrics = {}, {}
        for family, source in sources.items():
            inputs.update(checked_manifest(source / "manifest.json"))
            audits[family] = {
                a["fold"]: a for a in json.loads((source / "selection_audit.json").read_text())
            }
            metrics[family] = pd.read_csv(source / "nested_selection_results.csv")
        for fold in range(5):
            candidates = {family: values[fold] for family, values in audits.items()}
            # Only inner-score dictionaries enter this decision.
            family = choose_family(candidates)
            frame = metrics[family]
            frame = frame[frame.fold.eq(fold)].copy()
            frame["mic_training_seed"], frame["chosen_mic_family"] = seed, family
            results.append(frame)
            audit_rows.append(
                dict(
                    mic_training_seed=seed,
                    fold=fold,
                    selected_mic_family=family,
                    selected_hc50=candidates[family]["selected_arm"],
                    inner_choices={
                        f: max(r["inner_score"] for r in a["candidates"])
                        for f, a in candidates.items()
                    },
                    source=str(sources[family]),
                )
            )
    frame = pd.concat(results, ignore_index=True)
    frame.to_csv(output / "nested_selection_results.csv", index=False)
    write_json(output / "selection_audit.json", audit_rows)
    write_json(
        output / "protocol.json",
        dict(
            families=["linear8", "esm8-exact"],
            training_seeds=[42, 43, 44],
            choice="inner molecular lower-bound joint top20%; ties family id",
            linear_seed="deterministic shared ridge fits reused across neural training repetitions",
            development_reuse=True,
            outer_labels_select_settings=False,
            limitations="candidate families were screened earlier on public development data",
        ),
    )
    primary = frame[
        (frame.evidence == "measured_mic+measured_hc50")
        & frame.rbc_species.eq("human")
        & frame.ratio.eq(8)
        & frame.supported
    ]
    primary.groupby(["selector", "metric", "species"]).agg(
        lower=("lower", "mean"), upper=("upper", "mean"), assessed_rows=("lower", "size")
    ).reset_index().groupby(["selector", "metric"]).agg(
        lower=("lower", "mean"),
        upper=("upper", "mean"),
        assessed_rows=("assessed_rows", "sum"),
        species=("species", "nunique"),
    ).reset_index().to_csv(output / "selector_comparison.csv", index=False)
    finish_stage(output, inputs, start)


if __name__ == "__main__":
    main()
