"""Published-table MIC parsing and aligned normalized residue similarity."""

import re
from functools import lru_cache
from typing import Any

import numpy as np
from Bio import pairwise2
from Bio.Align import substitution_matrices


def matches_reviewed_assay(assay: dict[str, Any], contract: dict[str, Any]) -> bool:
    return (
        (assay.get("medium") or {}).get("name") == contract["medium"]
        and assay.get("cfu") == contract["cfu"]
        and assay.get("note", "") in contract["notes"]
        and all(not assay.get(field) for field in ["ph", "ionicStrength", "saltType"])
    )


def paper_mic(cell: str) -> tuple[str, float]:
    match = re.fullmatch(r"\s*(>?)[\d.]+/([\d.]+)\s*", cell)
    if match is None:
        raise ValueError("Expected published mass/molar MIC cell")
    value = float(match[2])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Positive finite molar concentration required")
    return match[1] or "=", value


@lru_cache(maxsize=1)
def normalized_residues() -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    matrix = substitution_matrices.load("BLOSUM62")
    alphabet = [*matrix.alphabet[:20], "*"]
    raw = np.array([[matrix[a, b] for b in alphabet] for a in alphabet])
    scaled = (raw - raw.min(0)) / (raw.max(0) - raw.min(0))
    symmetric = (scaled + scaled.T) / 2
    return (
        {
            (a, b): float(symmetric[i, j])
            for i, a in enumerate(alphabet)
            for j, b in enumerate(alphabet)
        },
        {(a, b): float(matrix[a, b]) for a in alphabet for b in alphabet},
    )


@lru_cache(maxsize=100000)
def cliff_similarity(left: str, right: str) -> float:
    """First local alignment, including padded terminal columns; canonical direction."""
    left, right = sorted((left, right))
    normalized, raw = normalized_residues()
    local_align = pairwise2.align.alignment_function("localds")
    alignments = local_align(left, right, raw, -11, -1, one_alignment_only=True)
    if not alignments:
        return 0.0
    first = alignments[0]
    return float(
        np.mean(
            [
                normalized[a.replace("-", "*"), b.replace("-", "*")]
                for a, b in zip(first.seqA, first.seqB, strict=True)
            ]
        )
    )
