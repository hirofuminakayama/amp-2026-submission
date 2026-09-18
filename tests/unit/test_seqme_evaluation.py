import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from robust_apex_qd.evaluation.seqme_eval import (
    CachedEmbeddingLookup,
    decide_library_adoption,
    fixed_subset_indices,
    validated_embedding_lookup,
)
from robust_apex_qd.features.embeddings import (
    ESM2_EMBEDDING_DIM,
    ESM2_MODEL_NAME,
    ESM2_MODEL_REVISION,
    file_sha256,
    row_mapping_sha256,
)


def test_fixed_subset_indices_repeat_and_change_with_seed() -> None:
    first = fixed_subset_indices(100, 12, seed=42)
    second = fixed_subset_indices(100, 12, seed=42)
    different = fixed_subset_indices(100, 12, seed=43)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 12
    assert len(set(first.tolist())) == 12
    assert not np.array_equal(first, different)


def test_cached_embedding_lookup_validates_sequence_alignment() -> None:
    lookup = CachedEmbeddingLookup(
        sequences=("AAAA", "CCCC"),
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        lookup(["CCCC", "AAAA"]),
        np.asarray([[3.0, 4.0], [1.0, 2.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="absent"):
        lookup(["DDDD"])
    with pytest.raises(ValueError, match="unique"):
        CachedEmbeddingLookup(
            sequences=("AAAA", "AAAA"),
            embeddings=np.ones((2, 2), dtype=np.float32),
        )


def test_pareto_rule_adopts_only_noninferior_two_metric_improvement() -> None:
    baseline = {
        "uniqueness": 1.0,
        "novelty": 1.0,
        "fbd": 10.0,
        "conformity": 0.80,
        "diversity": 0.70,
        "fkea": 10.0,
        "precision": 0.70,
    }
    improving = {
        **baseline,
        "fbd": 9.8,
        "diversity": 0.71,
        "fkea": 10.5,
    }
    regressing = {
        **improving,
        "fbd": 10.6,
    }

    adopted = decide_library_adoption({"L0": baseline, "L1": improving, "L2": regressing})

    assert adopted["adopted_variant"] == "L1"
    assert adopted["variants"]["L1"]["eligible"]
    assert not adopted["variants"]["L2"]["eligible"]


def test_pareto_rule_accepts_requested_variant_subset() -> None:
    baseline = {
        "uniqueness": 1.0,
        "novelty": 1.0,
        "fbd": 10.0,
        "conformity": 0.80,
        "diversity": 0.70,
        "fkea": 10.0,
        "precision": 0.70,
    }
    improving = {**baseline, "fbd": 9.8, "diversity": 0.71, "fkea": 10.5}

    adopted = decide_library_adoption({"L0": baseline, "L2": improving})

    assert adopted["adopted_variant"] == "L2"
    assert set(adopted["variants"]) == {"L0", "L2"}


def test_cached_embedding_manifest_rejects_stale_row_mapping(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv.gz"
    with gzip.open(candidates, "wt", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("candidate_id", "sequence"))
        writer.writeheader()
        writer.writerow({"candidate_id": "cand_1", "sequence": "ACDEFGHI"})
    reference = tmp_path / "reference.fasta"
    reference.write_text(">ref_1\nKLMNPQRS\n")
    candidate_embeddings = tmp_path / "candidate.npy"
    reference_embeddings = tmp_path / "reference.npy"
    np.save(candidate_embeddings, np.ones((1, ESM2_EMBEDDING_DIM), dtype=np.float32))
    np.save(reference_embeddings, np.zeros((1, ESM2_EMBEDDING_DIM), dtype=np.float32))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_name": ESM2_MODEL_NAME,
                "model_revision": ESM2_MODEL_REVISION,
                "embedding_dtype": "float32",
                "embedding_dimension": ESM2_EMBEDDING_DIM,
                "candidate_count": 1,
                "reference_count": 1,
                "candidate_input_sha256": file_sha256(candidates),
                "reference_input_sha256": file_sha256(reference),
                "candidate_row_mapping_sha256": row_mapping_sha256(["cand_1"], ["STALESEQ"]),
                "reference_row_mapping_sha256": row_mapping_sha256(["ref_1"], ["KLMNPQRS"]),
                "candidate_embeddings_sha256": file_sha256(candidate_embeddings),
                "reference_embeddings_sha256": file_sha256(reference_embeddings),
            }
        )
    )

    with pytest.raises(ValueError, match="row mapping"):
        validated_embedding_lookup(
            candidates_path=candidates,
            candidate_embeddings_path=candidate_embeddings,
            reference_fasta_path=reference,
            reference_embeddings_path=reference_embeddings,
            embedding_manifest_path=manifest,
        )
