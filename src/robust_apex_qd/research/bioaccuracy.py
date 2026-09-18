"""Molecule-balanced endpoint evidence, with explicit chemical and assay uncertainty."""

import hashlib
import json
import math
import re
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from robust_apex_qd.research.data import CANONICAL, parse_mic

Endpoint = Literal[
    "measured_mic", "consensus_mic", "measured_hc50", "consensus_hc50", "hemolysis_percent"
]
Relation = Literal["=", "<", "<=", ">", ">="]


class EndpointObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[2] = 2
    observation_id: str
    source: str
    source_id: str
    sequence: str
    endpoint: Endpoint
    target: str
    species: str
    target_level: Literal["strain", "species", "erythrocyte"] = "species"
    nterminal: str | None = None
    cterminal: str | None = None
    bonds: str | None = None
    stereochemistry: str | None = None
    chemistry_support: dict[str, str] = Field(
        default_factory=lambda: dict.fromkeys(
            ["nterminal", "cterminal", "bonds", "stereochemistry"], "unknown"
        )
    )
    value_um: float | None = Field(default=None, gt=0)
    relation: Relation | None = None
    hemolysis_percent: float | None = Field(default=None, ge=0, le=100)
    test_concentration_um: float | None = Field(default=None, gt=0)
    rbc_species: str | None = None
    exposure_time: str | None = None
    medium: str | None = None
    inoculum: str | None = None
    study: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_endpoint(self) -> "EndpointObservation":
        if not self.sequence or set(self.sequence) - CANONICAL:
            raise ValueError("Canonical sequence required; preserve excluded chemistry separately")
        if self.endpoint == "hemolysis_percent" and self.value_um is not None:
            raise ValueError("Hemolysis percent is not a MIC or HC50 concentration")
        if self.value_um is not None and self.relation is None:
            raise ValueError("Concentration requires its reported relation")
        if self.endpoint.startswith("consensus") and self.relation not in {None, "="}:
            raise ValueError("Consensus cannot recover raw measurement censoring")
        return self


def chemistry_key(row: EndpointObservation) -> str:
    content = {
        k: getattr(row, k)
        for k in ["sequence", "nterminal", "cterminal", "bonds", "stereochemistry"]
    }
    content["chemistry_support"] = row.chemistry_support
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def normalized_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if value == [] or value == {}:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "none", "nan", "null", "[]"} else text


def reported_chemistry(raw: dict[str, Any], *, reported_free: bool) -> dict[str, Any]:
    values = {k: normalized_text(raw.get(k)) for k in ["nterminal", "cterminal", "bonds"]}
    support = {
        k: "reported_modified" if v else "reported_free" if reported_free else "unknown"
        for k, v in values.items()
    }
    support["stereochemistry"] = "unknown"
    return dict(**values, chemistry_support=support)


def qmap_hc50(raw: dict[str, Any]) -> EndpointObservation | None:
    values = raw.get("hemolytic_hc50")
    if values is None:
        return None
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("Expected QMAP min/max/consensus HC50 triplet")
    return EndpointObservation(
        observation_id=f"qmap:{raw['id']}:hc50",
        source="qmap",
        source_id=str(raw["id"]),
        sequence=raw["sequence"],
        endpoint="consensus_hc50",
        target="erythrocytes_unspecified",
        species="erythrocytes_unspecified",
        target_level="erythrocyte",
        value_um=float(values[2]),
        relation="=",
        raw={"hemolytic_hc50": values},
        **reported_chemistry(raw, reported_free={"nterminal", "cterminal", "bonds"} <= raw.keys()),
    )


def mic_observation(raw: dict[str, Any]) -> EndpointObservation:
    measured = raw["objective"] == "measured_mic"
    concentration = raw.get("mic_um") if measured else raw.get("consensus_um")
    text = normalized_text(concentration)
    value = float(text) if text is not None else None
    return EndpointObservation(
        observation_id=raw["observation_id"],
        source=raw["source"],
        source_id=str(raw["source_id"]),
        sequence=raw["sequence"],
        endpoint="measured_mic" if measured else "consensus_mic",
        target=raw["target"],
        species=raw["species"],
        target_level=raw["target_level"],
        value_um=value,
        relation=raw.get("relation") if measured and value is not None else "=" if value else None,
        medium=normalized_text(raw.get("medium")),
        inoculum=normalized_text(raw.get("cfu")),
        study=normalized_text(raw.get("study")),
        raw={k: normalized_text(raw.get(k)) for k in ["raw_value", "raw_unit", "note"]},
        **reported_chemistry(raw, reported_free=raw["chemical_form"] == "reported_linear_free"),
    )


def dbaasp_hemolysis(
    peptide: dict[str, Any],
    assay: dict[str, Any],
    chemistry: dict[str, Any],
) -> tuple[EndpointObservation | None, str]:
    cell = (assay.get("targetCell") or {}).get("name", "")
    if "erythrocyt" not in cell.lower() and "red blood" not in cell.lower():
        return None, "not_erythrocytes"
    measure = str(assay.get("activityMeasureForLysisValue") or "").strip()
    hc50 = bool(re.fullmatch(r"(?:HC50|50\s*%\s*Hemolysis)", measure, re.IGNORECASE))
    percent_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%\s*Hemolysis", measure, re.IGNORECASE)
    if not hc50 and percent_match is None:
        return None, "not_explicit_hemolysis_endpoint"
    percent = float(percent_match[1]) if percent_match is not None else None
    if percent is not None and not 0 <= percent <= 100:
        return None, "invalid_hemolysis_percent"
    if peptide["sequence"] != chemistry["sequence"]:
        return None, "sequence_mismatch"
    if any(
        bool(peptide.get(a)) != bool(chemistry.get(b))
        for a, b in [
            ("nTerminus", "nterminal"),
            ("cTerminus", "cterminal"),
            ("intrachainBonds", "bonds"),
        ]
    ):
        return None, "chemical_metadata_conflict"
    unit = (assay.get("unit") or {}).get("name", "")
    if unit not in {"µM", "μM", "uM"}:
        return None, "unsupported_unit"
    relation, value = parse_mic(str(assay.get("concentration") or ""))
    if value is None:
        return None, "ambiguous_concentration"
    if not hc50 and relation != "=":
        return None, "nonexact_test_concentration"
    endpoint_tag = "hc50" if hc50 else "hemolysis"
    result = EndpointObservation.model_validate(
        dict(
            observation_id=f"dbaasp:{peptide['id']}:{endpoint_tag}:{assay['id']}",
            source="dbaasp",
            source_id=str(peptide["id"]),
            sequence=peptide["sequence"],
            endpoint="measured_hc50" if hc50 else "hemolysis_percent",
            target=cell,
            species=cell,
            target_level="erythrocyte",
            value_um=value if hc50 else None,
            relation=relation if hc50 else None,
            hemolysis_percent=percent if not hc50 else None,
            test_concentration_um=value if not hc50 else None,
            rbc_species="human" if "human" in cell.lower() else cell,
            raw=assay,
            **reported_chemistry(chemistry, reported_free=True),
        )
    )
    return result, "included"


def dbaasp_hc50(
    peptide: dict[str, Any], assay: dict[str, Any], chemistry: dict[str, Any]
) -> tuple[EndpointObservation | None, str]:
    result, reason = dbaasp_hemolysis(peptide, assay, chemistry)
    if result is not None and result.endpoint != "measured_hc50":
        return None, "not_explicit_hc50"
    return result, reason


def concentration_bounds(row: EndpointObservation) -> tuple[float, float, bool, bool] | None:
    if row.value_um is None or row.relation is None:
        return None
    value, relation = row.value_um, row.relation
    if relation == "=":
        return value, value, True, True
    if relation in {"<", "<="}:
        return 0.0, value, False, relation == "<="
    return value, math.inf, relation == ">=", False


def activity_bounds(row: EndpointObservation, threshold: float = 16) -> tuple[int, int] | None:
    bounds = concentration_bounds(row)
    if bounds is None:
        return None
    low, high, closed_low, _closed_high = bounds
    return int(high <= threshold), int(low < threshold or (low == threshold and closed_low))


def joint_hit_bounds(
    mic: EndpointObservation, hc50: EndpointObservation, *, ratio: float = 8, threshold: float = 16
) -> tuple[int, int] | None:
    if ratio <= 0 or threshold <= 0:
        raise ValueError("Positive activity threshold and selectivity ratio required")
    if mic.endpoint not in {"measured_mic", "consensus_mic"} or hc50.endpoint not in {
        "measured_hc50",
        "consensus_hc50",
    }:
        raise ValueError("Joint evidence requires MIC and HC50 endpoints")
    if chemistry_key(mic) != chemistry_key(hc50):
        return None
    m, h = concentration_bounds(mic), concentration_bounds(hc50)
    if m is None or h is None:
        return None
    ml, mu, ml_closed, _mu_closed = m
    hl, hu, _hl_closed, hu_closed = h
    possible_mic = ml < threshold or (ml == threshold and ml_closed)
    possible_ratio = hu > ratio * ml or (hu == ratio * ml and hu_closed and ml_closed)
    return int(mu <= threshold and hl >= ratio * mu), int(possible_mic and possible_ratio)


def assay_key(row: EndpointObservation) -> str:
    content = row.model_dump(exclude={"observation_id"})
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def molecular_labels(
    observations: Iterable[EndpointObservation],
    *,
    endpoint: Endpoint = "measured_mic",
    threshold: float = 16,
) -> pd.DataFrame:
    if endpoint not in {"measured_mic", "consensus_mic"}:
        raise ValueError("Activity labels require a MIC endpoint")
    groups: dict[tuple[str, str], dict[str, EndpointObservation]] = {}
    for row in observations:
        if row.endpoint != endpoint or activity_bounds(row, threshold) is None:
            continue
        groups.setdefault((chemistry_key(row), row.species), {})[assay_key(row)] = row
    records = []
    for (molecule, species), assays in groups.items():
        rows = list(assays.values())
        bounds = [activity_bounds(r, threshold) for r in rows]
        determined = [b for b in bounds if b is not None]
        records.append(
            dict(
                molecule_id=molecule,
                sequence=rows[0].sequence,
                species=species,
                endpoint=endpoint,
                hit_lower=min(b[0] for b in determined),
                hit_upper=max(b[1] for b in determined),
                assays=len(rows),
                targets=json.dumps(sorted({r.target for r in rows})),
                chemistry_support=json.dumps(rows[0].chemistry_support, sort_keys=True),
            )
        )
    columns = [
        "molecule_id",
        "sequence",
        "species",
        "endpoint",
        "hit_lower",
        "hit_upper",
        "assays",
        "targets",
        "chemistry_support",
    ]
    return (
        pd.DataFrame(records, columns=pd.Index(columns))
        .sort_values(["species", "sequence", "molecule_id"])
        .reset_index(drop=True)
    )


def peptide_metrics(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (10, 25, 100),
) -> list[dict[str, Any]]:
    keys = ["molecule_id", "species"]
    if any(k <= 0 for k in ks):
        raise ValueError("Positive k required")
    frame = labels.merge(
        predictions[[*keys, "prediction"]], on=keys, how="left", validate="one_to_one"
    )
    records = []
    for species, cohort in frame.groupby("species", sort=True):
        scored = cohort[np.isfinite(cohort.prediction)].sort_values(["prediction", "molecule_id"])
        available, predicted = len(cohort), len(scored)
        for name, k in [(f"p_at_{k}", k) for k in ks] + [("top20pct", math.ceil(available * 0.2))]:
            selected = scored.head(k) if predicted >= k else scored.head(0)
            known = selected[selected.hit_lower == selected.hit_upper]
            background = cohort[cohort.hit_lower == cohort.hit_upper]
            precision = float(known.hit_lower.mean()) if len(known) else None
            prevalence = float(background.hit_lower.mean()) if len(background) else None
            records.append(
                dict(
                    species=species,
                    metric=name,
                    k=k,
                    available=available,
                    predicted=predicted,
                    coverage=predicted / available,
                    selected=len(selected),
                    determined=len(known),
                    precision_lower=float(selected.hit_lower.mean()) if len(selected) else None,
                    precision_upper=float(selected.hit_upper.mean()) if len(selected) else None,
                    precision_determined=precision,
                    prevalence_determined=prevalence,
                    enrichment_determined=precision / prevalence
                    if precision is not None and prevalence is not None and prevalence > 0
                    else None,
                    reason=None if len(selected) else "insufficient_predicted_candidates",
                )
            )
    return records
