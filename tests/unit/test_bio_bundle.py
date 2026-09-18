import json
from pathlib import Path

import numpy as np
import pytest
import torch

from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.research.biobundle import load_bio_bundle, predict_bio_bundle
from robust_apex_qd.research.biofeatures import feature_contract
from robust_apex_qd.research.biomodels import HC50Bundle
from robust_apex_qd.research.competition_models import SPECIES
from robust_apex_qd.research.mic_models import fit_regressor


def fixture_bundle(path: Path) -> None:
    contract = feature_contract()
    np.savez(
        path / "weights.npz",
        mean=np.zeros(320),
        scale=np.ones(320),
        coef=np.zeros((7, 320)),
        intercept=np.full(7, 2.0),
        support=np.ones(7, dtype=bool),
        species_support=np.ones(7, dtype=bool),
        strain_support=np.ones(11, dtype=bool),
        offsets=np.zeros(11),
    )
    dimension = len(contract["names"])
    model = HC50Bundle(
        model="ridge",
        endpoint="measured_hc50",
        feature_sha256=contract["sha256"],
        mean=[0.0] * dimension,
        scale=[1.0] * dimension,
        coefficients=[0.0] * dimension,
        intercept=8.0,
        training_groups=["fixture"],
        alpha=1.0,
    )
    (path / "hc50.json").write_text(model.model_dump_json())
    np.savez(
        path / "residuals.npz",
        allow_pickle=False,
        hc50=np.zeros(3),
        **{f"mic{i}": np.zeros(3) for i in range(7)},
    )
    meta = dict(
        schema_version=2,
        mic_family="linear8",
        mic_alpha=1.0,
        hc50_arm="standard-exact",
        hc50_selection="fixture",
        hc50_votes={"standard-exact": 5},
        feature_dimension=320,
        feature_name="esm2_t6_8M_UR50D mean residue",
        hc50_feature_sha256=contract["sha256"],
        species=list(SPECIES),
        units="log2_uM",
        ratio=8,
        mic_threshold_um=16,
        split_sha256="a" * 64,
        reload_permutation_atol=1e-12,
        artifact_sha256={
            name: file_sha256(path / name) for name in ["weights.npz", "hc50.json", "residuals.npz"]
        },
    )
    (path / "bundle.json").write_text(json.dumps(meta))


def test_portable_bundle_predicts_without_training_labels_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    fixture_bundle(tmp_path)
    bundle = load_bio_bundle(tmp_path)
    result = predict_bio_bundle(
        bundle,
        ["ACDEFGHIK", "KLLKLLKLL"],
        np.zeros((2, 320)),
        embedding_feature_name="esm2_t6_8M_UR50D mean residue",
    )
    np.testing.assert_equal(result["mic_log2_um"], np.full((2, 7), 2.0))
    np.testing.assert_equal(result["joint_probability"], np.ones((2, 7)))
    with pytest.raises(ValueError, match="feature"):
        predict_bio_bundle(
            bundle, ["ACDEFGHIK"], np.zeros((1, 320)), embedding_feature_name="fine-tuned"
        )
    (tmp_path / "weights.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash"):
        load_bio_bundle(tmp_path)


def test_native_mic_head_bundle_reuses_the_existing_predictor(tmp_path: Path) -> None:
    fixture_bundle(tmp_path)
    settings = dict(
        device="cpu",
        width=32,
        heads=18,
        scale_floor=0.1,
        learning_rate=0.001,
        epochs=2,
        batch_size=256,
        loss="interval",
        family="esm8-exact",
    )
    state = fit_regressor(
        np.zeros((7, 320)), np.arange(7), np.arange(7.0), np.arange(7.0), settings, 42
    )
    torch.save(state, tmp_path / "weights.pt")
    manifest = json.loads((tmp_path / "bundle.json").read_text())
    manifest.update(mic_family="esm8-exact", mic_alpha=None)
    manifest["artifact_sha256"].pop("weights.npz")
    manifest["artifact_sha256"]["weights.pt"] = file_sha256(tmp_path / "weights.pt")
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    result = predict_bio_bundle(
        load_bio_bundle(tmp_path),
        ["ACDEFGHIK"],
        np.zeros((1, 320)),
        embedding_feature_name="esm2_t6_8M_UR50D mean residue",
    )
    assert result["mic_log2_um"].shape == (1, 7)
    assert np.isfinite(result["mic_log2_um"]).all()
