"""Explicit research feature schemas; legacy inference descriptors remain unchanged."""

import hashlib
import json
from typing import Any, Literal

import numpy as np

from robust_apex_qd.features.physchem import (
    CANONICAL_AMINO_ACIDS,
    EISENBERG_HYDROPHOBICITY,
    compute_features,
)

BomanMode = Literal["none", "legacy", "standard", "both"]
BOMAN_TRANSFER_ENERGY = {
    "L": -4.92,
    "I": -4.92,
    "V": -4.04,
    "F": -2.98,
    "M": -2.35,
    "W": -2.33,
    "A": -1.81,
    "C": -1.28,
    "G": -0.94,
    "Y": 0.14,
    "T": 2.57,
    "S": 3.40,
    "H": 4.66,
    "Q": 5.54,
    "K": 5.55,
    "N": 6.64,
    "E": 6.81,
    "D": 8.72,
    "R": 14.92,
    "P": 0.0,
}


def hydrophobic_moment(sequence: str, angle: int) -> float:
    energies = np.array([EISENBERG_HYDROPHOBICITY[a] for a in sequence])
    phase = np.arange(len(sequence)) * np.deg2rad(angle)
    return float(abs(np.sum(energies * np.exp(1j * phase))) / len(sequence))


def research_features(
    sequence: str,
    *,
    boman: BomanMode = "standard",
    local: bool = False,
    interactions: bool = False,
) -> dict[str, float]:
    if boman not in {"none", "legacy", "standard", "both"}:
        raise ValueError("Unknown Boman feature schema")
    features = compute_features(sequence)
    sequence = sequence.strip().upper()
    legacy = features.pop("boman_index")
    if boman in {"legacy", "both"}:
        features["legacy_solubility_mean"] = legacy
    if boman in {"standard", "both"}:
        features["boman_standard"] = sum(BOMAN_TRANSFER_ENERGY[a] for a in sequence) / len(sequence)
    for a in sorted(CANONICAL_AMINO_ACIDS):
        features[f"aac_{a}"] = sequence.count(a) / len(sequence)
    if local:
        for angle in [100, 180]:
            features[f"moment_{angle}_full"] = hydrophobic_moment(sequence, angle)
            for window in [8, 12]:
                size = min(window, len(sequence))
                values = [
                    hydrophobic_moment(sequence[i : i + size], angle)
                    for i in range(len(sequence) - size + 1)
                ]
                features[f"moment_{angle}_window{window}_max"] = max(values)
                features[f"moment_{angle}_window{window}_mean"] = float(np.mean(values))
    if interactions:
        features["gravy_squared"] = features["gravy"] ** 2
        features["charge_times_gravy"] = features["charge_ph_7_4"] * features["gravy"]
    return features


def feature_contract(
    boman: BomanMode = "standard",
    *,
    local: bool = False,
    interactions: bool = False,
) -> dict[str, Any]:
    contract = dict(
        schema_version=2,
        boman=boman,
        local=local,
        interactions=interactions,
        names=list(
            research_features(
                "ACDEFGHIKLMNPQRSTVWY", boman=boman, local=local, interactions=interactions
            )
        ),
        boman_units="kcal/mol",
        short_windows="min(window, sequence_length)",
        boman_energy_sha256=hashlib.sha256(
            json.dumps(BOMAN_TRANSFER_ENERGY, sort_keys=True).encode()
        ).hexdigest(),
    )
    contract["sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return contract
