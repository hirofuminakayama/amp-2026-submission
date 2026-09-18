import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from robust_apex_qd.features.physchem import (
    fit_reference,
    write_candidate_features,
    write_reference,
)
from robust_apex_qd.io.fasta import read_fasta
from robust_apex_qd.validation.compliance import challenge_valid_records

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute physicochemical candidate features")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-fasta",
        type=Path,
        default=ROOT / "data/training/training.fasta",
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        default=ROOT / "work/physchem_reference.json",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    reference_path = options.reference_fasta.resolve()
    valid_reference = challenge_valid_records(read_fasta(reference_path))
    reference = fit_reference(
        [record.sequence for record in valid_reference],
        reference_sha256=_sha256(reference_path),
    )
    write_reference(reference, options.reference_output.resolve())
    row_count = write_candidate_features(
        options.input.resolve(),
        options.output.resolve(),
        reference,
    )
    print(f"Wrote {row_count} candidate rows to {options.output.resolve()} using {reference_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
