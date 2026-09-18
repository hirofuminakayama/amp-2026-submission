import argparse
from pathlib import Path

import verify_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the vendored official validator to an existing output."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--antibacterial-fasta", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library_fasta = args.output_dir / "library.fasta"
    top_fasta = args.output_dir / "top.fasta"

    full_sequences = verify_submission._verify_sequences(library_fasta)
    verify_submission._verify_top(
        top_fasta,
        full_sequences,
        verify_submission.TOP_SIZE,
    )
    _, antibacterial_sequences = verify_submission._read_fasta(args.antibacterial_fasta)
    antibacterial_set = set(antibacterial_sequences)
    verify_submission._verify_no_overlap(full_sequences, antibacterial_set)
    _, top_sequences = verify_submission._read_fasta(top_fasta)
    verify_submission._veritfy_max_simularity(
        set(top_sequences),
        antibacterial_set,
    )
    print(f"Official validator checks passed for {args.output_dir}")


if __name__ == "__main__":
    main()
