from collections import Counter

import pytest
from pydantic import ValidationError

from robust_apex_qd.selection.library import (
    LibraryCandidate,
    redistribute_length_quotas,
    select_library,
)


def _candidate(index: int, length: int, cluster: int) -> LibraryCandidate:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return LibraryCandidate(
        candidate_id=f"cand_{index}",
        sequence=alphabet[index] + "A" * (length - 1),
        raw_order=index,
        length=length,
        embedding_cluster=cluster,
        physchem_ood=float(index % 3),
        embedding_ood=float(index % 2),
        valid=True,
    )


def test_l0_l1_l2_are_exact_and_repeat_deterministically() -> None:
    candidates = tuple(_candidate(index, 10 if index < 6 else 11, index % 3) for index in range(12))
    quotas = {10: 4, 11: 4}

    l0 = select_library(candidates, variant="L0", size=8, target_quotas=quotas)
    l1 = select_library(candidates, variant="L1", size=8, target_quotas=quotas)
    l2_first = select_library(candidates, variant="L2", size=8, target_quotas=quotas)
    l2_second = select_library(candidates, variant="L2", size=8, target_quotas=quotas)

    assert [candidate.raw_order for candidate in l0.selected] == list(range(8))
    assert Counter(candidate.length for candidate in l1.selected) == quotas
    assert Counter(candidate.length for candidate in l2_first.selected) == quotas
    assert l2_first == l2_second
    assert len({candidate.embedding_cluster for candidate in l2_first.selected}) == 3


def test_nearest_length_redistribution_uses_lower_length_tiebreak_and_cascades() -> None:
    result = redistribute_length_quotas(
        target_quotas={10: 2, 11: 2, 12: 2},
        availability={10: 3, 11: 0, 12: 3},
    )

    assert result.actual_quotas == {10: 3, 11: 0, 12: 3}
    assert result.movements == ((11, 10, 1), (11, 12, 1))
    assert sum(result.actual_quotas.values()) == 6


def test_library_input_rejects_top_quality_and_manual_fields() -> None:
    with pytest.raises(ValidationError):
        LibraryCandidate.model_validate(
            {
                **_candidate(0, 10, 0).model_dump(),
                "final_score": 1.0,
                "manual_choice": True,
            }
        )
