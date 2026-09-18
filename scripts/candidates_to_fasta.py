import argparse
import csv
import gzip
from collections.abc import Sequence
from pathlib import Path

from robust_apex_qd.io.fasta import FastaRecord, write_fasta


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ordered candidate metadata to FASTA")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    records: list[FastaRecord] = []
    with gzip.open(options.input.resolve(), "rt", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"candidate_id", "sequence"} <= set(reader.fieldnames):
            raise ValueError("Candidate table must contain candidate_id and sequence")
        for row in reader:
            records.append(FastaRecord(row["candidate_id"], row["sequence"]))
    write_fasta(records, options.output.resolve())
    print(f"Wrote {len(records)} ordered candidates to {options.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
