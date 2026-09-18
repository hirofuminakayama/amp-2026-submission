from Bio import Align


def _make_aligner() -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -1.0
    return aligner


_ALIGNER = _make_aligner()


def local_similarity(first: str, second: str) -> float:
    """Return the starter-kit Smith-Waterman score normalized by longer length."""
    if not first or not second:
        return 0.0
    score = _ALIGNER.score(first, second)
    return max(0.0, float(score)) / max(len(first), len(second))
