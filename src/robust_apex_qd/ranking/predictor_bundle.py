"""Portable, inference-only bundles for fixed species and strain MIC predictors."""

import json
from pathlib import Path
from typing import Any, Literal

import esm
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.competition_models import STRAIN_SPECIES, predict_ridge_heads

DIMENSIONS = {"physchem": 33, "linear8": 320, "linear650": 1280, "mlp8": 320, "finetune8": 320}


class PredictorBundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    family: Literal["physchem", "linear8", "linear650", "mlp8", "finetune8"]
    feature_dimension: int
    unit: Literal["log2_uM"] = "log2_uM"
    width: int | None = Field(default=None, gt=0)
    training_parameters: dict[str, int | float]
    training_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts_sha256: dict[str, str]

    @field_validator("artifacts_sha256")
    @classmethod
    def relative_files(cls, values: dict[str, str]) -> dict[str, str]:
        for name, digest in values.items():
            if not name or Path(name).name != name or name in [".", ".."]:
                raise ValueError("Bundle files must be relative basenames")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid artifact SHA256")
        return values

    @model_validator(mode="after")
    def consistent(self) -> "PredictorBundleManifest":
        if self.feature_dimension != DIMENSIONS[self.family]:
            raise ValueError("Predictor family feature dimension differs")
        expected = {"arrays.npz"}
        if self.family in ["mlp8", "finetune8"]:
            expected.add("network.pt")
        if set(self.artifacts_sha256) != expected:
            raise ValueError("Unexpected bundle file contract")
        if self.family == "mlp8" and self.width is None:
            raise ValueError("MLP width is required")
        return self


def export_predictor_bundle(source: Path, output: Path) -> PredictorBundleManifest:
    metadata_path = source / "manifest.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["unit"] != "log2_uM":
        raise ValueError("Only measured MIC predictors can be exported")
    family = metadata["arm"]["family"]
    if family not in DIMENSIONS:
        raise ValueError("Unsupported predictor family")
    neural = family in ["mlp8", "finetune8"]
    weights = source / ("weights.pt" if neural else "weights.npz")
    digest = file_sha256(weights)
    if digest != metadata["artifacts_sha256"][weights.name]:
        raise ValueError("Refit weight hash differs")
    # Conversion consumes an explicitly chosen, verified local training checkpoint.
    state = (
        torch.load(weights, map_location="cpu", weights_only=False)
        if neural
        else dict(np.load(weights))
    )
    names = ["mean", "scale", "species_support", "strain_support"]
    names += ["center"] if neural else ["coef", "intercept", "support", "offsets"]
    arrays = {name: np.asarray(state[name]) for name in names}
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "arrays.npz", allow_pickle=False, **arrays)
    if neural:
        network = {"head": state["head"]}
        if family == "finetune8":
            network["encoder"] = state["encoder"]
        torch.save(network, output / "network.pt")
    manifest = PredictorBundleManifest(
        family=family,
        feature_dimension=DIMENSIONS[family],
        width=metadata["arm"].get("width"),
        training_parameters={
            k: v for k, v in metadata["arm"].items() if k in ["alpha", "epochs", "lr", "width"]
        },
        training_dataset_sha256=metadata["dataset_sha256"],
        source_weights_sha256=digest,
        source_manifest_sha256=file_sha256(metadata_path),
        artifacts_sha256={p.name: file_sha256(p) for p in output.iterdir()},
    )
    (output / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def load_bundle(bundle: Path) -> tuple[PredictorBundleManifest, dict[str, np.ndarray]]:
    manifest = PredictorBundleManifest.model_validate_json((bundle / "manifest.json").read_text())
    for name, digest in manifest.artifacts_sha256.items():
        if file_sha256(bundle / name) != digest:
            raise ValueError("Predictor bundle artifact hash differs")
    state = dict(np.load(bundle / "arrays.npz", allow_pickle=False))
    dimensions = manifest.feature_dimension
    shapes = {
        "mean": (dimensions,),
        "scale": (dimensions,),
        "species_support": (7,),
        "strain_support": (11,),
    }
    if manifest.family in ["mlp8", "finetune8"]:
        shapes["center"] = ()
    else:
        shapes.update(coef=(7, dimensions), intercept=(7,), support=(7,), offsets=(11,))
    if set(state) != set(shapes) or any(
        state[name].shape != shape for name, shape in shapes.items()
    ):
        raise ValueError("Predictor numeric state shape differs")
    if (
        any(not np.isfinite(value).all() for value in state.values())
        or not (state["scale"] > 0).all()
    ):
        raise ValueError("Predictor numeric state is not finite or has invalid scales")
    for name in ["species_support", "strain_support", *(["support"] if "support" in state else [])]:
        if state[name].dtype != np.bool_:
            raise ValueError("Predictor support masks must be boolean")
    return manifest, state


def predict_bundle(
    bundle: Path, features: np.ndarray, sequences: list[str], *, device: str
) -> tuple[np.ndarray, np.ndarray]:
    manifest, state = load_bundle(bundle)
    if features.shape != (len(sequences), manifest.feature_dimension) or not len(sequences):
        raise ValueError("Aligned nonempty sequences and feature rows are required")
    if not np.isfinite(features).all():
        raise ValueError("Predictor input features must be finite")
    if manifest.family in ["physchem", "linear8", "linear650"]:
        species = predict_ridge_heads(state, features)
        strains = species[:, STRAIN_SPECIES] + state["offsets"]
    else:
        network = torch.load(bundle / "network.pt", map_location="cpu", weights_only=True)
        encoder: Any = None
        if manifest.family == "finetune8":
            encoder = esm.ESM2(
                num_layers=6,
                embed_dim=320,
                attention_heads=20,
                alphabet="ESM-1b",
                token_dropout=True,
            )
            encoder.load_state_dict(network["encoder"])
            encoder.eval().to(device)
            head = torch.nn.Linear(320, 25)
        else:
            if manifest.width is None:
                raise ValueError("Missing MLP architecture")
            head = torch.nn.Sequential(
                torch.nn.Linear(320, manifest.width),
                torch.nn.ReLU(),
                torch.nn.Linear(manifest.width, 25),
            )
        head.load_state_dict(network["head"])
        head.eval().to(device)
        values = []
        with torch.no_grad():
            for start in range(0, len(sequences), 32):
                if encoder is not None:
                    batch = sequences[start : start + 32]
                    _, _, tokens = encoder.alphabet.get_batch_converter()(
                        [(str(i), s) for i, s in enumerate(batch)]
                    )
                    tokens = tokens.to(device)
                    rep = encoder(tokens, repr_layers=[6])["representations"][6]
                    mask = (tokens != 1) & (tokens != 0) & (tokens != 2)
                    x = (rep * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
                else:
                    x = torch.tensor(
                        (features[start : start + 32] - state["mean"]) / state["scale"],
                        dtype=torch.float32,
                        device=device,
                    )
                values.append(head(x).cpu().numpy())
        raw = np.concatenate(values)
        species = raw[:, :7] + float(state["center"])
        strains = species[:, STRAIN_SPECIES] + raw[:, 7:18]
    species[:, ~state["species_support"]] = np.nan
    strains[:, ~state["strain_support"]] = np.nan
    return species, strains
