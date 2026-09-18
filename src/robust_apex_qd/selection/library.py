from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


class LibraryCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    sequence: str
    raw_order: int
    length: int
    embedding_cluster: int
    physchem_ood: float
    embedding_ood: float
    valid: bool


@dataclass(frozen=True)
class QuotaRedistribution:
    actual_quotas: dict[int, int]
    movements: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class LibrarySelectionResult:
    selected: tuple[LibraryCandidate, ...]
    variant: str
    target_quotas: dict[int, int]
    actual_quotas: dict[int, int]
    movements: tuple[tuple[int, int, int], ...]
    cluster_coverage: int


def redistribute_length_quotas(
    target_quotas: dict[int, int],
    availability: dict[int, int],
) -> QuotaRedistribution:
    if set(target_quotas) != set(availability):
        raise ValueError("Target quota and availability lengths must match")
    if any(value < 0 for value in (*target_quotas.values(), *availability.values())):
        raise ValueError("Length quotas and availability must be non-negative")
    actual = {
        length: min(target_quotas[length], availability[length]) for length in sorted(target_quotas)
    }
    movements: list[tuple[int, int, int]] = []
    for source in sorted(target_quotas):
        deficit = max(0, target_quotas[source] - actual[source])
        while deficit:
            destinations = [
                length for length in sorted(target_quotas) if availability[length] > actual[length]
            ]
            if not destinations:
                raise ValueError("Candidate availability cannot satisfy the requested library size")
            destination = min(destinations, key=lambda length: (abs(length - source), length))
            moved = min(deficit, availability[destination] - actual[destination])
            actual[destination] += moved
            movements.append((source, destination, moved))
            deficit -= moved
    return QuotaRedistribution(actual_quotas=actual, movements=tuple(movements))


def _quality_order(candidate: LibraryCandidate) -> tuple[float, float, int, str]:
    return (
        candidate.physchem_ood,
        candidate.embedding_ood,
        candidate.raw_order,
        candidate.sequence,
    )


def _round_robin_length(
    candidates: Sequence[LibraryCandidate],
    quota: int,
) -> list[LibraryCandidate]:
    by_cluster: dict[int, deque[LibraryCandidate]] = {}
    grouped: dict[int, list[LibraryCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.embedding_cluster].append(candidate)
    for cluster, rows in grouped.items():
        by_cluster[cluster] = deque(sorted(rows, key=_quality_order))
    selected: list[LibraryCandidate] = []
    cluster_order = sorted(by_cluster)
    while len(selected) < quota:
        added = False
        for cluster in cluster_order:
            if by_cluster[cluster]:
                selected.append(by_cluster[cluster].popleft())
                added = True
                if len(selected) == quota:
                    break
        if not added:
            raise RuntimeError("Cluster round-robin exhausted before its length quota")
    return selected


def select_library(
    candidates: Sequence[LibraryCandidate],
    *,
    variant: str,
    size: int,
    target_quotas: dict[int, int],
) -> LibrarySelectionResult:
    if variant not in {"L0", "L1", "L2"}:
        raise ValueError("Library variant must be L0, L1, or L2")
    if size <= 0 or sum(target_quotas.values()) != size:
        raise ValueError("Target length quotas must sum to the positive library size")
    valid = [candidate for candidate in candidates if candidate.valid]
    if len(valid) < size:
        raise ValueError("There are not enough valid candidates for the library")
    if variant == "L0":
        selected = sorted(valid, key=lambda candidate: candidate.raw_order)[:size]
        actual = dict(sorted(Counter(candidate.length for candidate in selected).items()))
        return LibrarySelectionResult(
            selected=tuple(selected),
            variant=variant,
            target_quotas=dict(sorted(target_quotas.items())),
            actual_quotas=actual,
            movements=(),
            cluster_coverage=len({candidate.embedding_cluster for candidate in selected}),
        )
    lengths = sorted(target_quotas)
    availability = Counter(candidate.length for candidate in valid)
    redistribution = redistribute_length_quotas(
        target_quotas,
        {length: availability[length] for length in lengths},
    )
    by_length: dict[int, list[LibraryCandidate]] = defaultdict(list)
    for candidate in valid:
        if candidate.length in target_quotas:
            by_length[candidate.length].append(candidate)
    selected = []
    for length in lengths:
        quota = redistribution.actual_quotas[length]
        rows = by_length[length]
        if variant == "L1":
            selected.extend(sorted(rows, key=_quality_order)[:quota])
        else:
            selected.extend(_round_robin_length(rows, quota))
    if len(selected) != size or len({candidate.sequence for candidate in selected}) != size:
        raise RuntimeError("Library selection did not produce the exact unique requested size")
    selected.sort(key=lambda candidate: (candidate.length, candidate.raw_order, candidate.sequence))
    return LibrarySelectionResult(
        selected=tuple(selected),
        variant=variant,
        target_quotas=dict(sorted(target_quotas.items())),
        actual_quotas=redistribution.actual_quotas,
        movements=redistribution.movements,
        cluster_coverage=len({candidate.embedding_cluster for candidate in selected}),
    )
