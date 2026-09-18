import re
from collections import Counter

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pydantic import BaseModel, ConfigDict, Field

from robust_apex_qd.features.physchem import EISENBERG_HYDROPHOBICITY, compute_features

HYDROPHOBIC_RESIDUES = frozenset("AVILMFWYC")
CLEAVAGE_PRONE_RESIDUES = frozenset("KRFWYLM")


class DevelopabilityResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    sequence: str
    length: int
    developability_score: float = Field(ge=0, le=100)
    developability_class: str
    hard_filter_pass: bool
    hard_filter_reasons: tuple[str, ...]
    aggregation_risk: str
    hemolysis_risk: str
    proteolysis_risk: str
    spps_difficulty_score: float = Field(ge=0)
    spps_class: str
    synthesizability_score: float = Field(ge=0, le=100)
    solubility_aggregation_score: float = Field(ge=0, le=100)
    safety_proxy_score: float = Field(ge=0, le=100)


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _range_score(value: float, best_low: float, best_high: float, low: float, high: float) -> float:
    if best_low <= value <= best_high:
        return 100.0
    if value < best_low:
        return _clamp(100 * (value - low) / max(best_low - low, 1e-12))
    return _clamp(100 * (high - value) / max(high - best_high, 1e-12))


def _lower_is_better(value: float, full: float, zero: float) -> float:
    if value <= full:
        return 100.0
    return _clamp(100 * (zero - value) / max(zero - full, 1e-12))


def _higher_is_better(value: float, zero: float, full: float) -> float:
    if value >= full:
        return 100.0
    return _clamp(100 * (value - zero) / max(full - zero, 1e-12))


def _longest_clean_stretch(sequence: str) -> int:
    longest = 0
    current = 0
    previous = ""
    for residue in sequence:
        is_xp = residue == "P" and bool(previous)
        if residue in CLEAVAGE_PRONE_RESIDUES or is_xp:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
        previous = residue
    return longest


def _degradation_hotspot_count(sequence: str) -> int:
    patterns = (r"N[GSTN]", r"Q[GS]", r"D[GSDT]", r"DP", r"CC")
    count = sum(len(re.findall(pattern, sequence)) for pattern in patterns)
    if sequence[0] in "GPEQ":
        count += 1
    count += sum(sequence.count(residue) for residue in "MWCH")
    return count


def _spps_score(sequence: str, hydrophobic_fraction: float) -> int:
    length = len(sequence)
    score = 4 if length > 50 else 2 if length > 30 else 0
    score += 3 if re.search(r"[IVT]{4,}", sequence) else 0
    score += 3 if re.search(r"[AVILMFWYC]{6,}", sequence) else 0
    score += 2 if hydrophobic_fraction > 0.55 else 0
    cysteine_count = sequence.count("C")
    score += 2 if cysteine_count % 2 == 1 else 0
    score += 2 if cysteine_count >= 4 else 0
    score += 1 if sequence.count("H") / length > 0.20 else 0
    score += 1 if sequence.count("M") >= 2 else 0
    score += 1 if "DP" in sequence else 0
    score += 1 if re.search(r"Q{4,}|G{5,}|(?:GS){4,}", sequence) else 0
    return score


def evaluate_developability(sequence: str) -> DevelopabilityResult:
    normalized = sequence.strip().upper()
    features = compute_features(normalized)
    analysis = ProteinAnalysis(normalized)
    counts = Counter(normalized)
    length = len(normalized)
    hydrophobic_fraction = sum(counts[residue] for residue in HYDROPHOBIC_RESIDUES) / length
    eisenberg_mean = sum(EISENBERG_HYDROPHOBICITY[residue] for residue in normalized) / length
    instability = float(analysis.instability_index())
    amphiphilicity = features["hydrophobic_moment"]
    charge = features["charge_ph_7_4"]
    gravy = features["gravy"]
    aromaticity = features["aromaticity"]

    aggregation_conditions = sum(
        (
            hydrophobic_fraction >= 0.45 or gravy > 0.5,
            abs(charge) < 1,
            aromaticity >= 0.25,
            instability > 40,
        )
    )
    aggregation_risk = (
        "High" if aggregation_conditions >= 3 else "Medium" if aggregation_conditions else "Low"
    )
    hemolysis_conditions = sum(
        (charge > 6, amphiphilicity > 0.5, gravy > 0.5, hydrophobic_fraction > 0.5)
    )
    hemolysis_risk = (
        "High" if hemolysis_conditions >= 3 else "Medium" if hemolysis_conditions else "Low"
    )
    longest_clean = _longest_clean_stretch(normalized)
    cleavage_density = sum(counts[residue] for residue in CLEAVAGE_PRONE_RESIDUES) / length
    n_terminal_risk = length > 1 and normalized[1] in "PA"
    if longest_clean < 5 or cleavage_density > 0.45 or n_terminal_risk:
        proteolysis_risk = "High"
    elif longest_clean < 10 or cleavage_density > 0.25:
        proteolysis_risk = "Medium"
    else:
        proteolysis_risk = "Low"

    spps = _spps_score(normalized, hydrophobic_fraction)
    spps_class = "Difficult" if spps >= 7 else "Moderate" if spps >= 4 else "Favorable"
    hard_reasons: list[str] = []
    if aggregation_risk == "High":
        hard_reasons.append("high_aggregation_risk")
    if hemolysis_risk == "High":
        hard_reasons.append("high_hemolysis_risk")
    if spps >= 8:
        hard_reasons.append("spps_difficulty_at_least_8")
    if instability > 60 and longest_clean < 5:
        hard_reasons.append("severe_instability_and_proteolysis")
    if counts["C"] % 2 == 1:
        hard_reasons.append("odd_cysteine_count")

    physicochemical = (
        sum(
            (
                _range_score(length, 8, 30, 5, 50),
                _range_score(features["molecular_weight"], 700, 3500, 500, 5000),
                _range_score(charge, 0, 4, -1, 6),
                _range_score(features["isoelectric_point"], 5, 10, 4, 11),
                _range_score(gravy, -1.2, 0.2, -2.0, 0.8),
                _lower_is_better(aromaticity, 0.25, 0.40),
                _higher_is_better(features["shannon_entropy"], 1.2, 2.0),
            )
        )
        / 7
    )
    n_end_score = 100 if normalized[0] in "GAVPMST" else 35 if normalized[0] in "RKLFWYI" else 70
    risk_score = {"Low": 100.0, "Medium": 60.0, "High": 20.0}
    stability = (
        sum(
            (
                _lower_is_better(instability, 40, 60),
                n_end_score,
                _lower_is_better(_degradation_hotspot_count(normalized), 1, 6),
                risk_score[proteolysis_risk],
            )
        )
        / 4
    )
    solubility_conditions = sum(
        (gravy <= 0.5, eisenberg_mean <= 0.4, abs(charge) >= 1, instability < 40)
    )
    solubility_score = (
        100 if solubility_conditions >= 3 else 65 if solubility_conditions == 2 else 25
    )
    aggregation_score = {"Low": 100.0, "Medium": 60.0, "High": 15.0}[aggregation_risk]
    solubility_aggregation = (
        sum(
            (
                solubility_score,
                aggregation_score,
                _lower_is_better(hydrophobic_fraction, 0.45, 0.70),
                _range_score(amphiphilicity, 0.15, 0.45, 0.05, 0.65),
            )
        )
        / 4
    )
    synthesizability = (
        sum(
            (
                _lower_is_better(spps, 3, 8),
                100 if counts["C"] % 2 == 0 else 20,
                _range_score(length, 8, 30, 5, 60),
            )
        )
        / 3
    )
    excess_charge_score = 100 if charge <= 6 else _lower_is_better(charge - 6, 0, 4)
    safety = (
        sum(
            (
                {"Low": 100.0, "Medium": 60.0, "High": 15.0}[hemolysis_risk],
                excess_charge_score,
                _lower_is_better(aromaticity, 0.30, 0.50),
            )
        )
        / 3
    )
    developability = _clamp(
        0.25 * physicochemical
        + 0.25 * stability
        + 0.25 * solubility_aggregation
        + 0.15 * synthesizability
        + 0.10 * safety
    )
    developability_class = (
        "Excellent"
        if developability >= 80
        else "Good"
        if developability >= 65
        else "Moderate"
        if developability >= 50
        else "Low"
    )
    return DevelopabilityResult(
        sequence=normalized,
        length=length,
        developability_score=developability,
        developability_class=developability_class,
        hard_filter_pass=not hard_reasons,
        hard_filter_reasons=tuple(hard_reasons),
        aggregation_risk=aggregation_risk,
        hemolysis_risk=hemolysis_risk,
        proteolysis_risk=proteolysis_risk,
        spps_difficulty_score=float(spps),
        spps_class=spps_class,
        synthesizability_score=synthesizability,
        solubility_aggregation_score=solubility_aggregation,
        safety_proxy_score=safety,
    )
