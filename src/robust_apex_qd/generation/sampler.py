import csv
import gzip
import io
import math
import os
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np

from robust_apex_qd.io.fasta import read_fasta_sequences

CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class LengthPolicy(str, Enum):
    UNIFORM = "uniform"
    EMPIRICAL_TEMPERED = "empirical_tempered"


class SamplerOutOfMemoryError(RuntimeError):
    """Raised without retrying or changing the requested batch size."""


class SamplingBackend(Protocol):
    def __call__(self, design_length: int, batch_size: int, round_seed: int) -> list[str]: ...


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    sequence: str
    raw_order: int
    round_index: int
    round_seed: int
    design_length: int
    length: int
    valid: bool
    rejection_reason: str

    @classmethod
    def column_names(cls) -> tuple[str, ...]:
        return (
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

    def as_row(self) -> dict[str, str | int | bool]:
        return {
            "candidate_id": self.candidate_id,
            "sequence": self.sequence,
            "raw_order": self.raw_order,
            "round_index": self.round_index,
            "round_seed": self.round_seed,
            "design_length": self.design_length,
            "length": self.length,
            "valid": self.valid,
            "rejection_reason": self.rejection_reason,
        }


def set_process_determinism(seed: int, cpu_threads: int = 8) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def largest_remainder_quotas(total: int, weights: dict[int, float]) -> dict[int, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights or any(weight < 0 or not math.isfinite(weight) for weight in weights.values()):
        raise ValueError("weights must be non-empty, finite, and non-negative")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("at least one weight must be positive")

    exact = {key: total * weight / weight_sum for key, weight in weights.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    priority = sorted(weights, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in priority[:remaining]:
        quotas[key] += 1
    return {key: quotas[key] for key in sorted(quotas)}


def build_length_quotas(
    total: int,
    policy: LengthPolicy,
    minimum: int,
    maximum: int,
    training_fasta: Path,
    temperature: float,
) -> dict[int, int]:
    if minimum > maximum:
        raise ValueError("minimum length cannot exceed maximum length")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    lengths = range(minimum, maximum + 1)
    if policy is LengthPolicy.UNIFORM:
        weights = {length: 1.0 for length in lengths}
        return largest_remainder_quotas(total, weights)
    if policy is not LengthPolicy.EMPIRICAL_TEMPERED:
        raise ValueError(f"Unsupported length policy: {policy}")

    canonical = set(CANONICAL_AMINO_ACIDS)
    observed = Counter(
        len(sequence)
        for sequence in read_fasta_sequences(training_fasta)
        if minimum <= len(sequence) <= maximum and set(sequence) <= canonical
    )
    weights = {length: (observed[length] + 1) ** temperature for length in lengths}
    return largest_remainder_quotas(total, weights)


def _deterministic_backend(design_length: int, batch_size: int, round_seed: int) -> list[str]:
    rng = random.Random(round_seed)
    return [
        "".join(rng.choice(CANONICAL_AMINO_ACIDS) for _ in range(design_length))
        for _ in range(batch_size)
    ]


def _batch_schedule(quotas: dict[int, int], batch_size: int, seed: int) -> list[tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches: list[tuple[int, int]] = []
    for design_length, quota in sorted(quotas.items()):
        remaining = quota
        while remaining:
            current = min(batch_size, remaining)
            batches.append((design_length, current))
            remaining -= current
    random.Random(seed).shuffle(batches)
    return batches


def generate_candidates(
    raw_pool_size: int,
    batch_size: int,
    seed: int,
    quotas: dict[int, int],
    *,
    backend: SamplingBackend | None = None,
) -> list[Candidate]:
    if sum(quotas.values()) != raw_pool_size:
        raise ValueError("Length quotas must sum to raw_pool_size")
    sample_batch = backend or _deterministic_backend
    candidates: list[Candidate] = []
    for round_index, (design_length, current_batch_size) in enumerate(
        _batch_schedule(quotas, batch_size, seed)
    ):
        round_seed = seed + round_index
        sequences = sample_batch(design_length, current_batch_size, round_seed)
        if len(sequences) != current_batch_size:
            raise ValueError(
                f"Sampler returned {len(sequences)} sequences for batch size {current_batch_size}"
            )
        for sequence in sequences:
            raw_order = len(candidates)
            valid = (
                len(sequence) == design_length
                and 8 <= len(sequence) <= 50
                and set(sequence) <= set(CANONICAL_AMINO_ACIDS)
            )
            candidates.append(
                Candidate(
                    candidate_id=f"cand_{raw_order + 1:06d}",
                    sequence=sequence,
                    raw_order=raw_order,
                    round_index=round_index,
                    round_seed=round_seed,
                    design_length=design_length,
                    length=len(sequence),
                    valid=valid,
                    rejection_reason="" if valid else "generation_invalid",
                )
            )
    return candidates


def make_official_backend(
    checkpoint: Path, device: str, *, sampling_steps: int = 1000
) -> SamplingBackend:
    import torch

    from ampdiffusion_starter_kit.generate import _decode, load_model, set_seed
    from robust_apex_qd.research.generation import ddim_sample

    if not 1 <= sampling_steps <= 1000:
        raise ValueError("Sampling steps must lie within the original 1000-step schedule")

    torch_device = torch.device(device)
    ema_model, esm2, _alphabet, standard_indices = load_model(checkpoint, torch_device)
    generated = 0
    batches = 0

    def sample_batch(design_length: int, batch_size: int, round_seed: int) -> list[str]:
        nonlocal generated, batches
        set_seed(round_seed)
        try:
            sampled = (
                ema_model.sample(batch_size=batch_size, design_len=design_length + 2)
                if sampling_steps == 1000
                else ddim_sample(ema_model, batch_size, design_length + 2, sampling_steps)
            )
        except torch.cuda.OutOfMemoryError as error:
            raise SamplerOutOfMemoryError(
                f"CUDA OOM at batch_size={batch_size}; batch size was not changed"
            ) from error
        decoded = _decode(esm2, sampled, standard_indices, design_length)
        generated += len(decoded)
        batches += 1
        if batches == 1 or batches % 50 == 0:
            print(f"Generated {generated} raw candidates in {batches} batches", flush=True)
        return decoded

    return sample_batch


def write_candidates_csv(candidates: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw_file,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as gzip_file,
        io.TextIOWrapper(gzip_file, encoding="utf-8", newline="") as text_file,
    ):
        writer = csv.DictWriter(text_file, fieldnames=Candidate.column_names())
        writer.writeheader()
        writer.writerows(candidate.as_row() for candidate in candidates)


GenerationBackend = Callable[[int, int, int], list[str]]
