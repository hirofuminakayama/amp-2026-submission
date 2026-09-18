"""Inference-only loading of numeric MIC/HC50 heads and empirical residuals."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.biofeatures import feature_contract, research_features
from robust_apex_qd.research.biomodels import HC50Bundle, predict_hc50
from robust_apex_qd.research.bioscenarios import marginal_joint_probability
from robust_apex_qd.research.competition_models import SPECIES, predict_ridge_heads
from robust_apex_qd.research.mic_models import predict_regressor


class BioBundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    schema_version: Literal[2]
    mic_family: Literal["linear8", "esm8-exact"]
    mic_alpha: float | None = Field(gt=0)
    hc50_arm: Literal[
        "median", "standard-exact", "standard-interval", "esm8-exact", "esm8-interval"
    ]
    hc50_selection: str
    hc50_votes: dict[str, int]
    feature_dimension: Literal[320]
    feature_name: Literal["esm2_t6_8M_UR50D mean residue"]
    hc50_feature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    species: list[str]
    units: Literal["log2_uM"]
    ratio: float = Field(gt=0)
    mic_threshold_um: Literal[16]
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reload_permutation_atol: float = Field(ge=0)
    artifact_sha256: dict[str, str]

    @model_validator(mode="after")
    def check_contract(self) -> "BioBundleManifest":
        if tuple(self.species) != SPECIES:
            raise ValueError("Biological species order mismatch")
        if (self.mic_family == "linear8") != (self.mic_alpha is not None):
            raise ValueError("Ridge alpha applies only to the linear MIC family")
        weights = "weights.npz" if self.mic_family == "linear8" else "weights.pt"
        if set(self.artifact_sha256) != {weights, "hc50.json", "residuals.npz"}:
            raise ValueError("Biological bundle requires exactly its three relative artifacts")
        if any(
            len(h) != 64 or set(h) - set("0123456789abcdef") for h in self.artifact_sha256.values()
        ):
            raise ValueError("Invalid artifact hash")
        return self


@dataclass(frozen=True)
class BioBundle:
    manifest: BioBundleManifest
    mic: dict[str, Any]
    hc50: HC50Bundle
    mic_residuals: list[np.ndarray]
    hc50_residuals: np.ndarray


def load_bio_bundle(path: Path) -> BioBundle:
    manifest = BioBundleManifest.model_validate_json((path / "bundle.json").read_text())
    for name, expected in manifest.artifact_sha256.items():
        if file_sha256(path / name) != expected:
            raise ValueError("Biological bundle artifact hash mismatch")
    if manifest.mic_family == "linear8":
        mic = dict(np.load(path / "weights.npz", allow_pickle=False))
        shapes = dict(
            mean=(320,),
            scale=(320,),
            coef=(7, 320),
            intercept=(7,),
            support=(7,),
            species_support=(7,),
            strain_support=(11,),
            offsets=(11,),
        )
        if set(mic) != set(shapes) or any(mic[k].shape != shape for k, shape in shapes.items()):
            raise ValueError("MIC numeric feature/head shape mismatch")
        if not all(np.isfinite(a).all() for a in mic.values()) or np.any(mic["scale"] <= 0):
            raise ValueError("Finite MIC state and positive preprocessing scale required")
        if any(mic[k].dtype != bool for k in ["support", "species_support", "strain_support"]):
            raise ValueError("MIC head support must use boolean arrays")
        if not mic["support"].all() or not mic["species_support"].all():
            raise ValueError("Seven supported species heads required; no implicit label imputation")
    else:
        mic = torch.load(path / "weights.pt", weights_only=True, map_location="cpu")
        if mic["dimension"] != 320 or mic["settings"]["heads"] != 18:
            raise ValueError("Native MIC feature/head shape mismatch")
        if mic["settings"]["family"] != manifest.mic_family:
            raise ValueError("Native MIC model family mismatch")
        if not set(range(7)) <= set(mic["supported_heads"]):
            raise ValueError("Seven supported species heads required")
        if (
            mic["feature_mean"].shape != (320,)
            or mic["feature_scale"].shape != (320,)
            or not torch.isfinite(mic["feature_mean"]).all()
            or not torch.isfinite(mic["feature_scale"]).all()
            or not (mic["feature_scale"] > 0).all()
            or not np.isfinite(mic["center"])
            or not all(torch.isfinite(t).all() for t in mic["state"].values())
        ):
            raise ValueError("Finite native MIC state and positive scaler required")
    hc50 = HC50Bundle.model_validate_json((path / "hc50.json").read_text())
    if hc50.endpoint != "measured_hc50" or hc50.feature_sha256 != manifest.hc50_feature_sha256:
        raise ValueError("HC50 endpoint/feature contract mismatch")
    if (
        not manifest.hc50_arm.startswith("esm8")
        and feature_contract()["sha256"] != hc50.feature_sha256
    ):
        raise ValueError("HC50 descriptor implementation differs from fitted feature contract")
    residuals = dict(np.load(path / "residuals.npz", allow_pickle=False))
    if set(residuals) != {"hc50", *[f"mic{i}" for i in range(7)]}:
        raise ValueError("One empirical residual array per endpoint required")
    if any(r.ndim != 1 or not len(r) or not np.isfinite(r).all() for r in residuals.values()):
        raise ValueError("Nonempty finite empirical residual arrays required")
    return BioBundle(
        manifest, mic, hc50, [residuals[f"mic{i}"] for i in range(7)], residuals["hc50"]
    )


def predict_bio_bundle(
    bundle: BioBundle, sequences: list[str], embeddings: np.ndarray, *, embedding_feature_name: str
) -> dict[str, np.ndarray]:
    if embedding_feature_name != bundle.manifest.feature_name:
        raise ValueError("Embedding feature identity mismatch")
    if embeddings.shape != (len(sequences), 320) or not np.isfinite(embeddings).all():
        raise ValueError("Aligned finite ESM8 feature matrix required")
    if not sequences or len(set(sequences)) != len(sequences):
        raise ValueError("Nonempty unique candidate sequences required")
    mic = (
        predict_ridge_heads(bundle.mic, embeddings)
        if bundle.manifest.mic_family == "linear8"
        else predict_regressor(bundle.mic, embeddings)[0][:, :7]
    )
    if bundle.manifest.hc50_arm.startswith("esm8"):
        features = embeddings
    else:
        features = np.array(
            [list(research_features(s, boman="standard").values()) for s in sequences]
        )
    hc50 = predict_hc50(bundle.hc50, features, bundle.manifest.hc50_feature_sha256)
    joint = marginal_joint_probability(
        mic, hc50, bundle.mic_residuals, bundle.hc50_residuals, ratio=bundle.manifest.ratio
    )
    return dict(mic_log2_um=mic, hc50_log2_um=hc50, joint_probability=joint)
