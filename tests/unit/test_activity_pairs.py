import pytest

from robust_apex_qd.research.activity_pairs import (
    cliff_similarity,
    matches_reviewed_assay,
    paper_mic,
)


def test_paper_mic_preserves_censoring_and_unit_column() -> None:
    assert paper_mic("128/64.53") == ("=", 64.53)
    assert paper_mic(">128/64.53") == (">", 64.53)
    with pytest.raises(ValueError):
        paper_mic("unknown")


def test_cliff_similarity_uses_normalized_residue_scores() -> None:
    assert cliff_similarity("AAAA", "AAAA") == pytest.approx(1)
    # A->R substitution: mean of column-normalized (-1+4)/(4+4)
    # and (-1+4)/(5+4), averaged with four identical residues.
    assert cliff_similarity("WAAAW", "WAARW") == pytest.approx((4 + (3 / 8 + 3 / 9) / 2) / 5)
    assert cliff_similarity("WAAAW", "WAARW") == cliff_similarity("WAARW", "WAAAW")


def test_assay_check_does_not_merge_salt_variants_with_standard_table() -> None:
    contract = dict(medium="MHB", cfu="5E5", notes=["", "MRSA"])
    assay = dict(medium={"name": "MHB"}, cfu="5E5", note="MRSA")
    assert matches_reviewed_assay(assay, contract)
    for override in [
        dict(saltType="NaCl"),
        dict(note="serum"),
        dict(cfu="unknown"),
        dict(medium={"name": "TSB"}),
    ]:
        assert not matches_reviewed_assay({**assay, **override}, contract)
