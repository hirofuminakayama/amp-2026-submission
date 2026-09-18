from collections import Counter
from pathlib import Path

import pytest

from robust_apex_qd.generation.sampler import (
    Candidate,
    LengthPolicy,
    SamplerOutOfMemoryError,
    build_length_quotas,
    generate_candidates,
    largest_remainder_quotas,
)


def test_largest_remainder_is_exact_and_stably_breaks_ties() -> None:
    quotas = largest_remainder_quotas(10, {10: 1.0, 11: 1.0, 12: 1.0})
    assert quotas == {10: 4, 11: 3, 12: 3}
    assert sum(quotas.values()) == 10


def test_uniform_and_empirical_tempered_quotas_are_exact(tmp_path: Path) -> None:
    training = tmp_path / "training.fasta"
    training.write_text(">a\nAAAAAAAAAA\n>b\nAAAAAAAAAAA\n>c\nCCCCCCCCCCC\n")
    uniform = build_length_quotas(31, LengthPolicy.UNIFORM, 10, 12, training, 0.75)
    empirical = build_length_quotas(31, LengthPolicy.EMPIRICAL_TEMPERED, 10, 12, training, 0.75)
    assert uniform == {10: 11, 11: 10, 12: 10}
    assert sum(empirical.values()) == 31
    assert empirical[11] > empirical[10] > empirical[12]


def test_same_seed_repeats_sequence_and_metadata_order() -> None:
    first = generate_candidates(64, 8, 42, {10: 32, 11: 32})
    second = generate_candidates(64, 8, 42, {10: 32, 11: 32})
    different = generate_candidates(64, 8, 43, {10: 32, 11: 32})
    assert first == second
    assert [row.sequence for row in first] != [row.sequence for row in different]
    assert [row.raw_order for row in first] == list(range(64))
    assert Counter(row.design_length for row in first) == {10: 32, 11: 32}


def test_metadata_schema_rejects_seed_round() -> None:
    assert Candidate.column_names() == (
        "candidate_id",
        "sequence",
        "raw_order",
        "round_index",
        "round_seed",
        "design_length",
        "length",
        "valid",
        "rejection_reason",
    )
    assert "seed_round" not in Candidate.column_names()


def test_oom_is_fail_fast_and_does_not_change_batch_size() -> None:
    calls: list[int] = []

    def failing_backend(design_length: int, batch_size: int, round_seed: int) -> list[str]:
        del design_length, round_seed
        calls.append(batch_size)
        raise SamplerOutOfMemoryError("simulated OOM")

    with pytest.raises(SamplerOutOfMemoryError, match="simulated OOM"):
        generate_candidates(16, 8, 42, {10: 16}, backend=failing_backend)
    assert calls == [8]
