"""Strict assay linkage and cautious extraction of published modal MICs."""

import hashlib
from collections import defaultdict
from typing import Any

from robust_apex_qd.research.data import activity_label, parse_mic
from robust_apex_qd.research.metadata import publication_keys


def training_folds(
    assignments: list[dict[str, str]], n_folds: int = 5, seed: int = 42
) -> dict[str, int]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in assignments:
        if row["split"] == "train":
            groups[row["group"]].append(row["sequence"])
    if n_folds < 2 or len(groups) < n_folds:
        raise ValueError("Inner validation requires at least one training group per fold")
    order = sorted(
        groups,
        key=lambda group: (
            -len(groups[group]),
            hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
        ),
    )
    loads = [0] * n_folds
    result = {}
    for group in order:
        fold = min(range(n_folds), key=lambda index: (loads[index], index))
        for sequence in groups[group]:
            result[sequence] = fold
        loads[fold] += len(groups[group])
    return result


def modal_mic(value: str) -> tuple[float | None, int | None]:
    relation, number = parse_mic(value)
    if relation != "=":
        return None, None
    return number, activity_label(relation, number)


def match_activity(observation: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "no_exact_assay_match",
        "activity_ids": [],
        "reference_tokens": [],
        "publication_candidates": sorted(publication_keys(metadata.get("articles", []))),
        "assay_publication_verified": False,
    }
    if observation["sequence"] != metadata["sequence"]:
        result["status"] = "sequence_conflict"
        return result
    for activity in metadata.get("targetActivities", []):
        if (activity.get("activityMeasureGroup") or {}).get("name") != "MIC":
            continue
        fields = {
            "target": (activity.get("targetSpecies") or {}).get("name"),
            "raw_value": activity.get("concentration"),
            "raw_unit": (activity.get("unit") or {}).get("name"),
            "medium": (activity.get("medium") or {}).get("name"),
            "cfu": activity.get("cfu"),
            "note": activity.get("note"),
        }
        if all(
            str(observation.get(key) or "").strip() == str(value or "").strip()
            for key, value in fields.items()
        ):
            result["activity_ids"].append(activity["id"])
            result["reference_tokens"].append(str(activity.get("reference") or ""))
    if result["activity_ids"]:
        result["status"] = "matched_current_assay"
    return result
