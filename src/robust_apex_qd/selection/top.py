from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import Levenshtein
from pydantic import BaseModel, ConfigDict, model_validator

from robust_apex_qd.validation.compliance import (
    SELECTION_CHALLENGE_SIMILARITY_MAX,
)
from robust_apex_qd.validation.similarity import local_similarity

ReferenceSimilarity = Callable[[str, Sequence[str]], float]
PairwiseSimilarity = Callable[[str, str], float]


class TopCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    sequence: str
    raw_order: int
    final_score: float
    embedding_cluster: int
    physchem_hard_reject: bool
    external_hard_reject: bool = False
    median_log2_mic: float


class TopSelectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    challenge_similarity_max: float = SELECTION_CHALLENGE_SIMILARITY_MAX
    known_similarity_max: float = 0.60
    prefilter_sizes: tuple[int, ...] = (5_000, 10_000, 10_000, 10_000)
    cluster_caps: tuple[int, ...] = (5, 5, 8, 8)
    pairwise_thresholds: tuple[float, ...] = (0.50, 0.50, 0.50, 0.55)

    @model_validator(mode="after")
    def validate_steps(self) -> "TopSelectionConfig":
        step_count = len(self.prefilter_sizes)
        if step_count == 0 or len(self.cluster_caps) != step_count:
            raise ValueError("Top selection fallback arrays must have the same non-zero length")
        if len(self.pairwise_thresholds) != step_count:
            raise ValueError("Top selection fallback arrays must have the same non-zero length")
        if self.challenge_similarity_max != SELECTION_CHALLENGE_SIMILARITY_MAX:
            raise ValueError("Challenge selection similarity is fixed at 0.78")
        if self.known_similarity_max != 0.60:
            raise ValueError("Known AMP similarity is fixed at 0.60")
        if any(size <= 0 for size in self.prefilter_sizes):
            raise ValueError("Prefilter sizes must be positive")
        if any(cap <= 0 for cap in self.cluster_caps):
            raise ValueError("Cluster caps must be positive")
        return self


def top_selection_config_from_mapping(
    ranking_config: Mapping[str, object],
) -> TopSelectionConfig:
    def numeric_value(key: str, default: int | float) -> int | float:
        value = ranking_config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"ranking.{key} must be numeric")
        return value

    raw_prefilter_sizes = ranking_config.get("prefilter_sizes", (5_000, 10_000))
    if not isinstance(raw_prefilter_sizes, (list, tuple)) or len(raw_prefilter_sizes) != 2:
        raise ValueError("ranking.prefilter_sizes must contain initial and expanded sizes")
    initial_prefilter, expanded_prefilter = (int(value) for value in raw_prefilter_sizes)
    initial_cluster_cap = int(numeric_value("max_per_embedding_cluster", 5))
    fallback_cluster_cap = int(numeric_value("fallback_cluster_cap", 8))
    initial_pairwise = float(numeric_value("pairwise_local_similarity_max", 0.50))
    fallback_pairwise = float(numeric_value("fallback_pairwise_local_similarity_max", 0.55))
    return TopSelectionConfig(
        challenge_similarity_max=float(numeric_value("selection_challenge_similarity_max", 0.78)),
        known_similarity_max=float(numeric_value("known_local_similarity_max", 0.60)),
        prefilter_sizes=(
            initial_prefilter,
            expanded_prefilter,
            expanded_prefilter,
            expanded_prefilter,
        ),
        cluster_caps=(
            initial_cluster_cap,
            initial_cluster_cap,
            fallback_cluster_cap,
            fallback_cluster_cap,
        ),
        pairwise_thresholds=(
            initial_pairwise,
            initial_pairwise,
            initial_pairwise,
            fallback_pairwise,
        ),
    )


@dataclass(frozen=True)
class SelectedTopRow:
    candidate: TopCandidate
    challenge_similarity: float
    known_similarity: float
    pairwise_similarity: float


@dataclass(frozen=True)
class TopSelectionResult:
    selected: tuple[SelectedTopRow, ...]
    relaxation_step: int
    prefilter_size: int
    cluster_cap: int
    pairwise_threshold: float
    rejection_counts: dict[str, int]


def maximum_levenshtein_ratio(sequence: str, references: Sequence[str]) -> float:
    best = 0.0
    for reference in references:
        theoretical_maximum = (
            2 * min(len(sequence), len(reference)) / (len(sequence) + len(reference))
        )
        if theoretical_maximum <= best:
            continue
        best = max(best, float(Levenshtein.ratio(sequence, reference)))
    return best


def _match_count_upper_bound(first: str, second: str) -> int:
    first_counts = Counter(first)
    second_counts = Counter(second)
    return sum(min(count, second_counts[amino_acid]) for amino_acid, count in first_counts.items())


def maximum_local_similarity(sequence: str, references: Sequence[str]) -> float:
    best = 0.0
    for reference in references:
        denominator = max(len(sequence), len(reference))
        length_upper_bound = min(len(sequence), len(reference)) / denominator
        if length_upper_bound <= best:
            continue
        match_upper_bound = _match_count_upper_bound(sequence, reference) / denominator
        if match_upper_bound <= best:
            continue
        best = max(best, local_similarity(sequence, reference))
    return best


def select_top_with_fallback(
    candidates: Sequence[TopCandidate],
    *,
    challenge_references: Sequence[str],
    known_references: Sequence[str],
    top_k: int,
    config: TopSelectionConfig,
    challenge_similarity: ReferenceSimilarity = maximum_levenshtein_ratio,
    known_similarity: ReferenceSimilarity = maximum_local_similarity,
    pairwise_similarity: PairwiseSimilarity = local_similarity,
) -> TopSelectionResult:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ordered = sorted(
        candidates,
        key=lambda candidate: (-candidate.final_score, candidate.raw_order, candidate.sequence),
    )
    challenge_cache: dict[str, float] = {}
    known_cache: dict[str, float] = {}
    last_counts: Counter[str] = Counter()
    for step, (prefilter_size, cluster_cap, pairwise_threshold) in enumerate(
        zip(
            config.prefilter_sizes,
            config.cluster_caps,
            config.pairwise_thresholds,
            strict=True,
        )
    ):
        selected: list[SelectedTopRow] = []
        cluster_counts: Counter[int] = Counter()
        rejection_counts: Counter[str] = Counter()
        for candidate in ordered[:prefilter_size]:
            if candidate.external_hard_reject:
                rejection_counts["external_hard_reject"] += 1
                continue
            if candidate.sequence not in challenge_cache:
                challenge_cache[candidate.sequence] = challenge_similarity(
                    candidate.sequence, challenge_references
                )
            challenge_value = challenge_cache[candidate.sequence]
            if challenge_value > config.challenge_similarity_max:
                rejection_counts["challenge_similarity"] += 1
                continue
            if candidate.sequence not in known_cache:
                known_cache[candidate.sequence] = known_similarity(
                    candidate.sequence, known_references
                )
            known_value = known_cache[candidate.sequence]
            if known_value > config.known_similarity_max:
                rejection_counts["known_similarity"] += 1
                continue
            pairwise_value = max(
                (
                    pairwise_similarity(candidate.sequence, row.candidate.sequence)
                    for row in selected
                ),
                default=0.0,
            )
            if pairwise_value > pairwise_threshold:
                rejection_counts["pairwise_similarity"] += 1
                continue
            if cluster_counts[candidate.embedding_cluster] >= cluster_cap:
                rejection_counts["cluster_cap"] += 1
                continue
            if candidate.physchem_hard_reject:
                rejection_counts["physchem_hard_reject"] += 1
                continue
            selected.append(
                SelectedTopRow(
                    candidate=candidate,
                    challenge_similarity=challenge_value,
                    known_similarity=known_value,
                    pairwise_similarity=pairwise_value,
                )
            )
            cluster_counts[candidate.embedding_cluster] += 1
            if len(selected) == top_k:
                return TopSelectionResult(
                    selected=tuple(selected),
                    relaxation_step=step,
                    prefilter_size=prefilter_size,
                    cluster_cap=cluster_cap,
                    pairwise_threshold=pairwise_threshold,
                    rejection_counts=dict(rejection_counts),
                )
        last_counts = rejection_counts
    raise RuntimeError(
        f"Only {sum(last_counts.values())} rejected candidates were classified; "
        f"the fixed fallback sequence could not collect {top_k} selections"
    )
