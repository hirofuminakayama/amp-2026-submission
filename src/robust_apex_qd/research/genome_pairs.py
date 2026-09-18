"""Peptide-genome features with explicit label resolution and isolated evaluation."""

import re
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from robust_apex_qd.research.prediction_mic import prediction_records


class GenomeMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target: str
    species: str
    status: Literal["exact_label", "species_reference", "unknown"]
    accession: str | None = None
    experimental_isolate_verified: Literal[False] = False

    @model_validator(mode="after")
    def validate_accession(self) -> "GenomeMapping":
        if (self.status == "unknown") != (self.accession is None):
            raise ValueError("Only unresolved genomes have no accession")
        if (
            self.accession is not None
            and re.fullmatch(r"GC[AF]_\d{9}\.\d+", self.accession) is None
        ):
            raise ValueError("A versioned assembly accession is required")
        return self


def dna_fourmers(contigs: Iterable[str]) -> np.ndarray:
    """L1 frequencies, reverse-complement invariant; never bridge contigs or ambiguity."""
    table = np.full(256, -1, dtype=np.int16)
    table[list(b"ACGT")] = np.arange(4)
    counts = np.zeros(256, dtype=np.int64)
    for contig in contigs:
        bases = table[np.frombuffer(contig.upper().encode("ascii"), dtype=np.uint8)]
        if len(bases) < 4:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(bases, 4)
        valid = windows[(windows >= 0).all(axis=1)]
        codes = valid @ np.array([64, 16, 4, 1])
        counts += np.bincount(codes, minlength=256)
    if counts.sum() == 0:
        raise ValueError("Genome contains no unambiguous DNA four-mers")
    codes = np.arange(256)
    reverse = sum((3 - (codes // 4**i) % 4) * 4 ** (3 - i) for i in range(4))
    counts = counts + counts[reverse]
    return (counts / counts.sum()).astype(np.float32)


def resolve_genome(target: str, species: str, config: dict[str, Any]) -> GenomeMapping:
    matches = [a["accession"] for a in config["assemblies"] if target in a["labels"]]
    if len(matches) > 1:
        raise ValueError("Ambiguous target label in genome registry")
    accession = matches[0] if matches else config["species_references"].get(species)
    status = "exact_label" if matches else "species_reference" if accession else "unknown"
    return GenomeMapping(target=target, species=species, status=status, accession=accession)


def pair_masks(
    rows: pd.DataFrame, regime: str, peptide_fold: int | None, genome_fold: int | None
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in {"peptide", "strain", "both"}:
        raise ValueError("Unknown evaluation regime")
    training = np.ones(len(rows), dtype=bool)
    validation = rows.primary.to_numpy(bool).copy()
    if regime in {"peptide", "both"}:
        if peptide_fold is None:
            raise ValueError("Peptide fold required")
        training &= rows.peptide_fold.to_numpy() != peptide_fold
        validation &= rows.peptide_fold.to_numpy() == peptide_fold
    if regime in {"strain", "both"}:
        if genome_fold is None:
            raise ValueError("Genome fold required")
        resolved = rows.genome_status.eq("exact_label").to_numpy() & (rows.genome_fold >= 0)
        training &= resolved & (rows.genome_fold.to_numpy() != genome_fold)
        validation &= resolved & (rows.genome_fold.to_numpy() == genome_fold)
    return np.asarray(training), np.asarray(validation)


def fit_categories(rows: pd.DataFrame, assay: bool) -> dict[str, list[str]]:
    columns = ["species", "medium", "cfu"] if assay else ["species"]
    return {c: sorted(rows[c].dropna().astype(str).unique().tolist()) for c in columns}


def pair_features(
    peptide: np.ndarray,
    rows: pd.DataFrame,
    genomes: dict[str, np.ndarray],
    categories: dict[str, list[str]],
    use_genome: bool,
    *,
    mask_assay: bool = False,
) -> np.ndarray:
    if len(peptide) != len(rows) or not np.isfinite(peptide).all():
        raise ValueError("Aligned finite peptide features required")
    parts = [peptide]
    for col, values in categories.items():
        observed = rows[col].copy()
        if mask_assay and col != "species":
            observed[:] = None
        text = observed.fillna("").astype(str).to_numpy()
        # Explicit unknown/missing indicator even when absent from the training fold.
        parts.extend(
            [(text[:, None] == np.array(values)[None, :]), ~np.isin(text, values)[:, None]]
        )
    if use_genome:
        accessions = rows.genome_accession.fillna("").tolist()
        available = np.array([a in genomes for a in accessions])
        parts.extend(
            [
                np.stack([genomes.get(a, np.zeros(256)) for a in accessions]),
                (~available)[:, None],
                rows.genome_status.eq("species_reference").to_numpy()[:, None],
            ]
        )
    return np.column_stack(parts).astype(np.float32)


def fallback_reason(mapping: GenomeMapping, genomes: dict[str, np.ndarray]) -> str:
    if mapping.accession is None or mapping.accession not in genomes:
        return "genome_missing"
    if mapping.status != "exact_label":
        return "species_reference_not_exact_strain"
    return ""


def genome_prediction_records(
    sequences: list[str],
    raw: np.ndarray,
    apex: np.ndarray,
    mappings: list[GenomeMapping],
    genomes: dict[str, np.ndarray],
    model_hash: str,
    apex_hash: str,
) -> pd.DataFrame:
    from robust_apex_qd.apex.ensemble import APEX_PATHOGENS

    if [m.target for m in mappings] != list(APEX_PATHOGENS):
        raise ValueError("Mappings must preserve the common strain order")
    if raw.shape != (len(sequences), 11) or not np.isfinite(raw).all():
        raise ValueError("Finite aligned pair predictions required")
    means = np.full((len(sequences), 18), np.nan)
    for head, mapping in enumerate(mappings):
        if not fallback_reason(mapping, genomes):
            means[:, head + 7] = raw[:, head]
    frame = prediction_records(
        sequences, means, np.full_like(means, np.nan), apex, model_hash, apex_hash
    )
    frame["genome_accession"] = ""
    frame["genome_status"] = ""
    frame["experimental_isolate_verified"] = False
    frame["assay"] = "not_conditioned"
    for mapping in mappings:
        mask = frame.target_id.eq(mapping.target)
        frame.loc[mask, "fallback_reason"] = fallback_reason(mapping, genomes)
        frame.loc[mask, "genome_accession"] = mapping.accession or ""
        frame.loc[mask, "genome_status"] = mapping.status
    return frame
