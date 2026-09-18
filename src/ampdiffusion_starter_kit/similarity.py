"""Local-alignment sequence similarity for the paper's computational filtering.

Torres et al. (*Cell Biomaterials* 2025) filter generated peptides by "sequence similarity
using local sequence alignment": a **novelty** cut (> 0.60 to any known AMP is excluded) and a
**diversity** cut (any pair > 0.40 keeps only the lower-MIC peptide). The paper cites its
alignment method but does not pin down the scoring matrix, gap penalties, or normalization, so
this module fixes one concrete, documented operationalization:

    Smith-Waterman local alignment (match +1, mismatch -1, gap open/extend -1); similarity is
    ``max(0, score) / max(len(a), len(b))`` — the matched region as a fraction of the longer
    sequence, in [0, 1] (1.0 for identical sequences). The paper's 0.60 / 0.40 thresholds are
    applied to this.

Normalizing by the *longer* sequence (match coverage), not the shorter, is deliberate:
antimicrobial peptides all share short cationic K/L/R stretches, so normalizing a local match
by the shorter sequence scores almost every de-novo peptide > 0.60 against the training set and
the novelty filter rejects everything. Coverage-of-the-longer measures whether a peptide
substantially overlaps a known AMP. If you have the paper's exact alignment parameters and
normalization, adjust the aligner and denominator below to match.
"""

from __future__ import annotations

from Bio import Align

_aligner = Align.PairwiseAligner()
_aligner.mode = "local"
_aligner.match_score = 1.0
_aligner.mismatch_score = -1.0
_aligner.open_gap_score = -1.0
_aligner.extend_gap_score = -1.0


def local_similarity(a: str, b: str) -> float:
    """Normalized Smith-Waterman local-alignment similarity between two peptides, in [0, 1]."""
    score = _aligner.score(a, b)
    return max(0.0, score) / max(len(a), len(b))
