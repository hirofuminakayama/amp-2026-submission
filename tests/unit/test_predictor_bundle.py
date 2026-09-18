import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from robust_apex_qd.features.embeddings import file_sha256
from robust_apex_qd.ranking.predictor_bundle import export_predictor_bundle, predict_bundle


def test_ridge_bundle_is_portable_and_preserves_unsupported_targets(tmp_path: Path) -> None:
    source = tmp_path / "training-run"
    source.mkdir()
    state = dict(
        mean=np.zeros(33),
        scale=np.ones(33),
        coef=np.ones((7, 33)),
        intercept=np.arange(7, dtype=float),
        offsets=np.arange(11, dtype=float),
        support=np.ones(7, dtype=bool),
        species_support=np.array([True] * 6 + [False]),
        strain_support=np.array([True] * 10 + [False]),
    )
    np.savez(source / "weights.npz", allow_pickle=False, **state)
    (source / "manifest.json").write_text(
        json.dumps(
            dict(
                arm=dict(family="physchem", alpha=1.0),
                dataset_sha256="a" * 64,
                unit="log2_uM",
                train_ids=["a-training-label-id"],
                artifacts_sha256={"weights.npz": file_sha256(source / "weights.npz")},
            )
        )
    )
    bundle = tmp_path / "portable"
    export_predictor_bundle(source, bundle)
    shutil.rmtree(source)
    species, strains = predict_bundle(bundle, np.ones((2, 33)), ["A" * 10, "K" * 10], device="cpu")
    np.testing.assert_array_equal(species[:, :6], np.tile(33 + np.arange(6), (2, 1)))
    assert np.isnan(species[:, 6]).all() and np.isnan(strains[:, 10]).all()
    assert strains[0, 3] == 37.0
    metadata = (bundle / "manifest.json").read_text()
    assert "training-label" not in metadata and str(source) not in metadata
    with pytest.raises(ValueError, match="feature"):
        predict_bundle(bundle, np.ones((2, 32)), ["A" * 10, "K" * 10], device="cpu")


def test_network_bundle_uses_tensor_only_weights_and_checked_hashes(tmp_path: Path) -> None:
    source = tmp_path / "refit"
    source.mkdir()
    torch.manual_seed(42)
    head = torch.nn.Sequential(torch.nn.Linear(320, 4), torch.nn.ReLU(), torch.nn.Linear(4, 25))
    for p in head.parameters():
        p.data.zero_()
    head[2].bias.data.copy_(torch.arange(25))
    state = dict(
        mean=np.zeros(320),
        scale=np.ones(320),
        center=2.0,
        head=head.state_dict(),
        species_support=np.ones(7, dtype=bool),
        strain_support=np.ones(11, dtype=bool),
    )
    torch.save(state, source / "weights.pt")
    (source / "manifest.json").write_text(
        json.dumps(
            dict(
                arm=dict(family="mlp8", width=4),
                dataset_sha256="b" * 64,
                unit="log2_uM",
                artifacts_sha256={"weights.pt": file_sha256(source / "weights.pt")},
            )
        )
    )
    bundle = tmp_path / "bundle"
    export_predictor_bundle(source, bundle)
    torch.load(bundle / "network.pt", map_location="cpu", weights_only=True)
    a, b = predict_bundle(bundle, np.ones((1, 320)), ["A" * 10], device="cpu")
    np.testing.assert_array_equal(a[0], 2 + np.arange(7))
    assert b[0, 0] == 9.0 and b[0, 3] == 13.0
    with (bundle / "arrays.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        predict_bundle(bundle, np.ones((1, 320)), ["A" * 10], device="cpu")
