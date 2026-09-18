import argparse
import csv
import gzip
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from robust_apex_qd.features.embeddings import file_sha256, row_mapping_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate ESM2 embedding row alignment")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    identifiers: list[str] = []
    sequences: list[str] = []
    with gzip.open(options.candidates.resolve(), "rt", newline="") as file:
        for row in csv.DictReader(file):
            identifiers.append(row["candidate_id"])
            sequences.append(row["sequence"])
    manifest = json.loads(options.manifest.resolve().read_text())
    embeddings = np.load(options.embeddings.resolve(), mmap_mode="r")
    expected_shape = (len(identifiers), int(manifest["embedding_dimension"]))
    if embeddings.shape != expected_shape:
        raise ValueError(f"Embedding shape {embeddings.shape} != {expected_shape}")
    if embeddings.dtype != np.float32:
        raise ValueError(f"Embedding dtype {embeddings.dtype} != float32")
    observed_mapping = row_mapping_sha256(identifiers, sequences)
    if observed_mapping != manifest["candidate_row_mapping_sha256"]:
        raise ValueError("Candidate row mapping checksum differs from the manifest")
    observed_checksum = file_sha256(options.embeddings.resolve())
    if observed_checksum != manifest["candidate_embeddings_sha256"]:
        raise ValueError("Candidate embedding checksum differs from the manifest")
    if not np.isfinite(embeddings).all():
        raise ValueError("Candidate embeddings contain NaN or infinity")
    print(f"Embedding alignment passed for {len(identifiers)} candidate rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
