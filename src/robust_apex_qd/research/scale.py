"""Sequence-keyed pool accounting and supply-aware library mixtures."""

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from robust_apex_qd.research.selection import bounded_quotas


def align_cached_rows(
    sequences: list[str], cached_sequences: list[str], cached: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(set(cached_sequences)) != len(cached_sequences):
        raise ValueError("Cached sequences must be unique")
    if len(cached) != len(cached_sequences):
        raise ValueError("Cached row count differs from sequence count")
    index = {sequence: i for i, sequence in enumerate(cached_sequences)}
    mask = np.asarray([sequence in index for sequence in sequences])
    values = np.full((len(sequences), *cached.shape[1:]), np.nan, dtype=cached.dtype)
    values[mask] = cached[[index[s] for s in sequences if s in index]]
    return values, mask


def reference_mix(
    pool: pd.DataFrame, l2: pd.DataFrame, reference: pd.DataFrame, fraction: float
) -> pd.DataFrame:
    if not 0 <= fraction <= 1 or pool.sequence.duplicated().any():
        raise ValueError("Require unique candidates and a convex reference mixture")
    selected = []
    for length, count in sorted(l2.length.value_counts().items()):
        available = pool[pool.length == length]
        capacity = available.embedding_cluster.value_counts().to_dict()
        original = l2[l2.length == length].embedding_cluster.value_counts(normalize=True).to_dict()
        target = (
            reference[reference.length == length].cluster.value_counts(normalize=True).to_dict()
        )
        weights = {
            c: (1 - fraction) * original.get(c, 0) + fraction * target.get(c, 0) for c in capacity
        }
        for cluster, quota in bounded_quotas(int(count), weights, capacity).items():
            selected.append(
                available[available.embedding_cluster == cluster]
                .sort_values(["physchem_ood", "embedding_ood", "raw_order", "sequence"])
                .head(quota)
            )
    result = pd.concat(selected).sort_values(["length", "raw_order", "sequence"])
    if len(result) != len(l2) or result.sequence.duplicated().any():
        raise ValueError("Reference allocation did not preserve the library size")
    return result


def merge_pool_sources(
    sources: dict[str, list[str]], references: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seen: set[str] = set()
    rows, provenance, inventory = [], [], []
    for source, sequences in sources.items():
        valid, unique, added = [], set(), 0
        for source_order, sequence in enumerate(sequences):
            digest = hashlib.sha256(sequence.encode()).hexdigest()
            reason = ""
            if not 8 <= len(sequence) <= 50 or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
                reason = "invalid"
            elif sequence in references:
                reason = "exact_reference_overlap"
            else:
                valid.append(sequence)
                unique.add(sequence)
                if sequence in seen:
                    reason = "duplicate"
            raw_order = len(provenance)
            provenance.append(
                dict(
                    source=source,
                    source_order=source_order,
                    raw_order=raw_order,
                    sequence=sequence,
                    sequence_sha256=digest,
                    reason=reason,
                )
            )
            if not reason:
                seen.add(sequence)
                added += 1
                rows.append(
                    dict(
                        candidate_id="sha256_" + digest,
                        sequence=sequence,
                        raw_order=raw_order,
                        length=len(sequence),
                        valid=True,
                        rejection_reason="",
                    )
                )
        inventory.append(
            dict(
                source=source,
                raw_count=len(sequences),
                valid_count=len(valid),
                unique_count=len(unique),
                additional_unique=added,
            )
        )
    if not rows:
        raise ValueError("No valid unique candidates")
    return pd.DataFrame(rows), pd.DataFrame(provenance), pd.DataFrame(inventory)


def mix_library(
    base: pd.DataFrame, donor: pd.DataFrame, *, size: int, fraction: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if size <= 0 or not 0 <= fraction <= 1:
        raise ValueError("Require a positive size and mixture fraction in [0, 1]")
    base = base.drop_duplicates("sequence")
    donor = donor[~donor.sequence.isin(base.sequence)].drop_duplicates("sequence")
    desired = round(size * fraction)
    selected = donor.head(desired)
    remainder = base.head(size - len(selected))
    if len(selected) + len(remainder) != size:
        raise ValueError("Insufficient distinct donor and base supply for requested library")
    return pd.concat([selected, remainder], ignore_index=True), dict(
        desired_fraction=fraction,
        desired_donor_count=desired,
        available_donor_count=len(donor),
        actual_donor_count=len(selected),
        actual_fraction=len(selected) / size,
        supply_limited=len(selected) < desired,
    )
