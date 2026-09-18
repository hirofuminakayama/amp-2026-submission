"""Artifact identity and shared coverage for a frozen adoption comparison."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_apex_qd.io.fasta import read_fasta_sequences


def candidate_identity(directory: Path) -> str:
    library = read_fasta_sequences(directory / "library.fasta")
    top = read_fasta_sequences(directory / "top.fasta")
    if len(set(library)) != len(library) or len(set(top)) != len(top):
        raise ValueError("Candidate artifacts must have unique sequences")
    if not library or not top or not set(top).issubset(library):
        raise ValueError("Require a nonempty library containing the ranked Top")
    payload = json.dumps([sorted(library), top], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def common_coverage(frame: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    omitted = [name for name in metrics if not np.isfinite(frame[name]).all()]
    for name in omitted:
        result[name] = np.nan
    return result, omitted
