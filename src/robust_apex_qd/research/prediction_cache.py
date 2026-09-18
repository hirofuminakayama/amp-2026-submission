"""Immutable, sequence-keyed predictions with explicit dependency checks."""

import json
from pathlib import Path

import numpy as np

from robust_apex_qd.features.embeddings import file_sha256


def isolated_rows(
    sequences: list[str], groups: list[str], training: np.ndarray, validation: np.ndarray
) -> None:
    if not len(training) or not len(validation):
        raise ValueError("Training and validation must be nonempty")
    for labels in [sequences, groups]:
        if {labels[i] for i in training} & {labels[i] for i in validation}:
            raise ValueError("Training and validation overlap")


class PredictionCache:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, sequences: list[str], values: np.ndarray, dependencies: dict[str, str]) -> None:
        if len(set(sequences)) != len(sequences) or len(sequences) != len(values):
            raise ValueError("Unique sequence IDs must align with prediction rows")
        if not dependencies or values.ndim != 2 or np.isinf(values).any():
            raise ValueError("Prediction matrix and explicit dependencies required")
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "sequences.json").write_text(json.dumps(sequences) + "\n")
        np.save(self.path / "values.npy", values, allow_pickle=False)
        (self.path / "manifest.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    dependencies=dependencies,
                    artifacts_sha256={
                        name: file_sha256(self.path / name)
                        for name in ["sequences.json", "values.npy"]
                    },
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def read(self, sequences: list[str], dependencies: dict[str, str]) -> np.ndarray:
        manifest = json.loads((self.path / "manifest.json").read_text())
        if manifest["schema_version"] != 1 or manifest["dependencies"] != dependencies:
            raise ValueError("Prediction cache dependencies changed")
        for name, digest in manifest["artifacts_sha256"].items():
            if file_sha256(self.path / name) != digest:
                raise ValueError("Prediction cache artifact changed")
        stored = json.loads((self.path / "sequences.json").read_text())
        if len(set(sequences)) != len(sequences) or set(sequences) != set(stored):
            raise ValueError("Prediction cache sequence membership changed")
        index = {sequence: i for i, sequence in enumerate(stored)}
        return np.load(self.path / "values.npy", allow_pickle=False)[[index[s] for s in sequences]]
