import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

ESM2_MODEL_NAME = "esm2_t6_8M_UR50D"
ESM2_MODEL_REVISION = "fair-esm-2.0.0"
ESM2_EMBEDDING_DIM = 320


class EmbeddingOutOfMemoryError(RuntimeError):
    """Raised without retrying or changing the requested ESM2 batch size."""


class EmbeddingBatchEncoder(Protocol):
    def __call__(self, sequences: list[str]) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class EmbeddingDiagnostics:
    embedding_ood: NDArray[np.float32]
    cluster: NDArray[np.int32]
    pca_transform_sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reusable_embeddings(
    *,
    manifest_path: Path,
    candidate_embeddings_path: Path,
    reference_embeddings_path: Path,
    candidate_ids: Sequence[str],
    candidate_sequences: Sequence[str],
    reference_ids: Sequence[str],
    reference_sequences: Sequence[str],
    candidate_input_sha256: str,
    reference_input_sha256: str,
    model_name: str,
    model_revision: str,
    embedding_dimension: int,
) -> None:
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot reuse embeddings without a valid manifest: {error}") from error
    expected_values: tuple[tuple[str, object], ...] = (
        ("model_name", model_name),
        ("model_revision", model_revision),
        ("embedding_dtype", "float32"),
        ("embedding_dimension", embedding_dimension),
        ("candidate_count", len(candidate_ids)),
        ("reference_count", len(reference_ids)),
        ("candidate_input_sha256", candidate_input_sha256),
        ("reference_input_sha256", reference_input_sha256),
        ("candidate_row_mapping_sha256", row_mapping_sha256(candidate_ids, candidate_sequences)),
        ("reference_row_mapping_sha256", row_mapping_sha256(reference_ids, reference_sequences)),
    )
    for field, expected in expected_values:
        if manifest.get(field) != expected:
            label = field.replace("_sha256", "").replace("_", " ")
            raise ValueError(f"Reusable embedding {label} differs from the current input")
    paths_and_fields = (
        (candidate_embeddings_path, "candidate_embeddings_sha256", "candidate"),
        (reference_embeddings_path, "reference_embeddings_sha256", "reference"),
    )
    for path, field, label in paths_and_fields:
        if not path.is_file() or file_sha256(path) != manifest.get(field):
            raise ValueError(f"Reusable {label} embedding checksum differs from the manifest")
        embeddings = np.load(path, mmap_mode="r")
        expected_rows = len(candidate_ids) if label == "candidate" else len(reference_ids)
        expected_shape = (expected_rows, embedding_dimension)
        if embeddings.shape != expected_shape:
            raise ValueError(f"Reusable {label} embedding shape differs from {expected_shape}")
        if embeddings.dtype != np.float32:
            raise ValueError(f"Reusable {label} embedding dtype differs from float32")


def mean_pool_token_representations(
    token_representations: NDArray[np.float32],
    sequence_lengths: Sequence[int],
) -> NDArray[np.float32]:
    if token_representations.ndim != 3:
        raise ValueError("Token representations must have shape [batch, tokens, dimensions]")
    if len(sequence_lengths) != token_representations.shape[0]:
        raise ValueError("Sequence lengths must align with the token representation rows")
    pooled = np.empty(
        (token_representations.shape[0], token_representations.shape[2]),
        dtype=np.float32,
    )
    for row_index, sequence_length in enumerate(sequence_lengths):
        if sequence_length <= 0 or sequence_length + 2 > token_representations.shape[1]:
            raise ValueError(f"Invalid ESM2 sequence length: {sequence_length}")
        pooled[row_index] = token_representations[
            row_index,
            1 : sequence_length + 1,
        ].mean(axis=0, dtype=np.float32)
    return pooled


class Esm2BatchEncoder:
    def __init__(self, device: str) -> None:
        import esm
        import torch

        self._torch = torch
        self._device = torch.device(device)
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self._model = model.eval().to(self._device)
        self._batch_converter = alphabet.get_batch_converter()

    def __call__(self, sequences: list[str]) -> NDArray[np.float32]:
        labels = [f"seq_{index}" for index in range(len(sequences))]
        _, _, tokens = self._batch_converter(list(zip(labels, sequences, strict=True)))
        try:
            with self._torch.no_grad():
                result = self._model(tokens.to(self._device), repr_layers=[6])
        except self._torch.cuda.OutOfMemoryError as error:
            raise EmbeddingOutOfMemoryError(
                f"CUDA OOM at ESM2 batch_size={len(sequences)}; batch size was not changed"
            ) from error
        token_representations = (
            result["representations"][6].detach().cpu().numpy().astype(np.float32, copy=False)
        )
        return mean_pool_token_representations(
            token_representations,
            [len(sequence) for sequence in sequences],
        )


def write_embedding_memmap(
    sequences: Sequence[str],
    output_path: Path,
    batch_size: int,
    encoder: EmbeddingBatchEncoder,
    *,
    embedding_dim: int = ESM2_EMBEDDING_DIM,
) -> tuple[int, int]:
    if not sequences:
        raise ValueError("At least one sequence is required")
    if batch_size <= 0:
        raise ValueError("Embedding batch size must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(sequences), embedding_dim)
    embeddings = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    for start in range(0, len(sequences), batch_size):
        batch = list(sequences[start : start + batch_size])
        encoded = np.asarray(encoder(batch), dtype=np.float32)
        if encoded.shape != (len(batch), embedding_dim):
            raise ValueError(
                f"Encoder returned shape {encoded.shape}; expected {(len(batch), embedding_dim)}"
            )
        embeddings[start : start + len(batch)] = encoded
    embeddings.flush()
    return shape


def row_mapping_sha256(identifiers: Sequence[str], sequences: Sequence[str]) -> str:
    if len(identifiers) != len(sequences):
        raise ValueError("Identifier and sequence rows must align")
    digest = hashlib.sha256()
    for identifier, sequence in zip(identifiers, sequences, strict=True):
        digest.update(identifier.encode())
        digest.update(b"\0")
        digest.update(sequence.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _array_sha256(*arrays: NDArray[np.floating]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def compute_embedding_diagnostics(
    candidate_embeddings: NDArray[np.floating],
    reference_embeddings: NDArray[np.floating],
    *,
    seed: int,
    pca_components: int,
    cluster_count: int,
    pca_reference_subset: int,
    thread_count: int,
) -> EmbeddingDiagnostics:
    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    references = np.asarray(reference_embeddings, dtype=np.float32)
    if candidates.ndim != 2 or references.ndim != 2:
        raise ValueError("Candidate and reference embeddings must be two-dimensional")
    if candidates.shape[1] != references.shape[1]:
        raise ValueError("Candidate and reference embedding dimensions differ")
    if not np.isfinite(candidates).all() or not np.isfinite(references).all():
        raise ValueError("Embeddings contain NaN or infinity")
    if cluster_count <= 0 or cluster_count > len(candidates):
        raise ValueError("cluster_count must be between 1 and the candidate count")
    subset_size = min(pca_reference_subset, len(references))
    if pca_components <= 0 or pca_components > min(subset_size, references.shape[1]):
        raise ValueError("pca_components exceeds the deterministic reference subset")
    if thread_count <= 0:
        raise ValueError("thread_count must be positive")

    rng = np.random.default_rng(seed)
    subset_indices = np.sort(rng.choice(len(references), size=subset_size, replace=False))
    reference_subset = references[subset_indices]
    with threadpool_limits(limits=thread_count):
        pca = PCA(n_components=pca_components, svd_solver="full")
        pca.fit(reference_subset)
        candidate_pca = pca.transform(candidates).astype(np.float32)
        reference_pca = pca.transform(references).astype(np.float32)

        neighbor_count = min(5, len(references))
        neighbors = NearestNeighbors(
            n_neighbors=neighbor_count,
            algorithm="brute",
            n_jobs=1,
        )
        neighbors.fit(reference_pca)
        distances, _ = neighbors.kneighbors(candidate_pca)
        embedding_ood = np.median(distances, axis=1).astype(np.float32)

        clustering = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=seed,
            n_init=10,
            batch_size=min(2048, max(256, len(candidates))),
            reassignment_ratio=0.0,
        )
        cluster = clustering.fit_predict(candidate_pca).astype(np.int32)

    return EmbeddingDiagnostics(
        embedding_ood=embedding_ood,
        cluster=cluster,
        pca_transform_sha256=_array_sha256(pca.components_, pca.mean_),
    )
