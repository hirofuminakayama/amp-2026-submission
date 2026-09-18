import argparse
import csv
import gzip
import math
from collections.abc import Sequence
from pathlib import Path

from robust_apex_qd.features.physchem import FEATURE_NAMES

NUMERIC_COLUMNS = (*FEATURE_NAMES, "physchem_ood", "soft_penalty")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reject non-finite physicochemical features")
    parser.add_argument("features", type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    row_count = 0
    with gzip.open(options.features.resolve(), "rt", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not set(NUMERIC_COLUMNS) <= set(reader.fieldnames):
            raise ValueError("Feature table is missing required numeric columns")
        for row in reader:
            row_count += 1
            for name in NUMERIC_COLUMNS:
                if not math.isfinite(float(row[name])):
                    raise ValueError(
                        f"Non-finite {name} at candidate_id={row.get('candidate_id', '')}"
                    )
    if row_count == 0:
        raise ValueError("Feature table is empty")
    print(f"All {len(NUMERIC_COLUMNS)} required features are finite for {row_count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
