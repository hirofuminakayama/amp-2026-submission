"""Registered generator screens count raw attempts separately from retained peptides."""

from typing import Any


def validate_screen(
    sequences: list[str], observed: dict[str, Any], expected: dict[str, Any]
) -> dict[str, int]:
    if any(observed.get(k) != value for k, value in expected.items()):
        raise ValueError("Generator protocol differs from registration")
    if len(sequences) != expected["count"]:
        raise ValueError("Raw count differs from registration")
    alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    valid = [
        s
        for s in sequences
        if set(s) <= alphabet and expected["min_length"] <= len(s) <= expected["max_length"]
    ]
    common = [s for s in valid if 15 <= len(s) <= 25]
    return dict(
        raw=len(sequences),
        valid=len(valid),
        unique=len(set(valid)),
        common_valid=len(common),
        common_unique=len(set(common)),
    )
