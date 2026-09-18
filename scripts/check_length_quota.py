import argparse
import csv
import gzip
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from robust_apex_qd.generation.sampler import LengthPolicy, build_length_quotas

ROOT = Path(__file__).resolve().parents[1]


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Config root must be a mapping")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Check candidate metadata length quotas")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = _config(arguments.config)
    generation = config["generation"]
    references = config["references"]
    training_fasta = ROOT / references["known_amp_fasta"]
    expected = build_length_quotas(
        int(generation["raw_pool_size"]),
        LengthPolicy(generation["length_policy"]),
        int(generation["min_generation_length"]),
        int(generation["max_generation_length"]),
        training_fasta,
        float(generation["length_temperature"]),
    )
    with gzip.open(arguments.metadata, "rt", newline="") as file:
        actual = Counter(int(row["design_length"]) for row in csv.DictReader(file))
    if dict(sorted(actual.items())) != expected:
        print(f"Quota mismatch\nexpected={expected}\nactual={dict(sorted(actual.items()))}")
        return 1
    print(f"Quota match: policy={generation['length_policy']} total={sum(actual.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
