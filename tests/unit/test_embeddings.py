import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from robust_apex_qd.features.embeddings import (
    EmbeddingDiagnostics,
    compute_embedding_diagnostics,
    file_sha256,
    mean_pool_token_representations,
    row_mapping_sha256,
    validate_reusable_embeddings,
    write_embedding_memmap,
)


def test_mean_pool_excludes_bos_eos_and_padding() -> None:
    representations = np.asarray(
        [
            [
                [100.0, 100.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [200.0, 200.0],
                [300.0, 300.0],
            ]
        ],
        dtype=np.float32,
    )

    pooled = mean_pool_token_representations(representations, [2])

    np.testing.assert_array_equal(pooled, np.asarray([[2.0, 3.0]], dtype=np.float32))


def test_embedding_memmap_preserves_sequence_row_alignment(tmp_path: Path) -> None:
    sequences = ["AAAA", "CCCC", "DDDD"]

    def fake_encoder(sequences: list[str]) -> np.ndarray:
        return np.asarray(
            [[len(sequence), index, ord(sequence[0])] for index, sequence in enumerate(sequences)],
            dtype=np.float32,
        )

    path = tmp_path / "embeddings.npy"
    shape = write_embedding_memmap(sequences, path, 2, fake_encoder, embedding_dim=3)
    embeddings = np.load(path, mmap_mode="r")

    assert shape == (3, 3)
    assert embeddings.shape == (3, 3)
    assert embeddings.dtype == np.float32
    np.testing.assert_array_equal(
        embeddings,
        np.asarray([[4, 0, 65], [4, 1, 67], [4, 0, 68]], dtype=np.float32),
    )


def test_pca_knn_ood_and_clustering_repeat_exactly() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(96, 24)).astype(np.float32)
    candidates = rng.normal(loc=0.2, size=(80, 24)).astype(np.float32)

    first = compute_embedding_diagnostics(
        candidates,
        reference,
        seed=42,
        pca_components=8,
        cluster_count=8,
        pca_reference_subset=64,
        thread_count=1,
    )
    second = compute_embedding_diagnostics(
        candidates,
        reference,
        seed=42,
        pca_components=8,
        cluster_count=8,
        pca_reference_subset=64,
        thread_count=1,
    )

    assert isinstance(first, EmbeddingDiagnostics)
    np.testing.assert_array_equal(first.cluster, second.cluster)
    np.testing.assert_array_equal(first.embedding_ood, second.embedding_ood)
    assert first.pca_transform_sha256 == second.pca_transform_sha256
    assert len(set(first.cluster.tolist())) == 8
    assert all(math.isfinite(float(value)) for value in first.embedding_ood)


def test_row_mapping_hash_covers_id_order_and_sequence() -> None:
    identifiers = ["cand_2", "cand_1"]
    sequences = ["AAAA", "CCCC"]
    expected = hashlib.sha256(b"cand_2\0AAAA\ncand_1\0CCCC\n").hexdigest()

    assert row_mapping_sha256(identifiers, sequences) == expected


def test_reuse_rejects_changed_rows_and_embedding_values(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.npy"
    reference_path = tmp_path / "reference.npy"
    np.save(candidate_path, np.ones((2, 3), dtype=np.float32))
    np.save(reference_path, np.ones((1, 3), dtype=np.float32))
    manifest = {
        "schema_version": 2,
        "model_name": "test-model",
        "model_revision": "test-revision",
        "embedding_dtype": "float32",
        "embedding_dimension": 3,
        "candidate_count": 2,
        "reference_count": 1,
        "candidate_input_sha256": "candidate-input",
        "reference_input_sha256": "reference-input",
        "candidate_row_mapping_sha256": row_mapping_sha256(["c1", "c2"], ["AAAA", "CCCC"]),
        "reference_row_mapping_sha256": row_mapping_sha256(["r1"], ["DDDDDDDD"]),
        "candidate_embeddings_sha256": file_sha256(candidate_path),
        "reference_embeddings_sha256": file_sha256(reference_path),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    validate_reusable_embeddings(
        manifest_path=manifest_path,
        candidate_embeddings_path=candidate_path,
        reference_embeddings_path=reference_path,
        candidate_ids=["c1", "c2"],
        candidate_sequences=["AAAA", "CCCC"],
        reference_ids=["r1"],
        reference_sequences=["DDDDDDDD"],
        candidate_input_sha256="candidate-input",
        reference_input_sha256="reference-input",
        model_name="test-model",
        model_revision="test-revision",
        embedding_dimension=3,
    )

    with pytest.raises(ValueError, match="candidate row mapping"):
        validate_reusable_embeddings(
            manifest_path=manifest_path,
            candidate_embeddings_path=candidate_path,
            reference_embeddings_path=reference_path,
            candidate_ids=["c1", "c2"],
            candidate_sequences=["CCCC", "AAAA"],
            reference_ids=["r1"],
            reference_sequences=["DDDDDDDD"],
            candidate_input_sha256="candidate-input",
            reference_input_sha256="reference-input",
            model_name="test-model",
            model_revision="test-revision",
            embedding_dimension=3,
        )

    changed = np.load(candidate_path)
    changed[0, 0] = 2.0
    np.save(candidate_path, changed)
    with pytest.raises(ValueError, match="candidate embedding checksum"):
        validate_reusable_embeddings(
            manifest_path=manifest_path,
            candidate_embeddings_path=candidate_path,
            reference_embeddings_path=reference_path,
            candidate_ids=["c1", "c2"],
            candidate_sequences=["AAAA", "CCCC"],
            reference_ids=["r1"],
            reference_sequences=["DDDDDDDD"],
            candidate_input_sha256="candidate-input",
            reference_input_sha256="reference-input",
            model_name="test-model",
            model_revision="test-revision",
            embedding_dimension=3,
        )
