import hashlib
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from robust_apex_qd.validation.inference_assets import Asset, AssetManifest, prepare_assets


def entry(
    path: str, content: bytes, destination: Literal["repository", "torch_hub"] = "repository"
) -> Asset:
    return Asset(
        path=path,
        destination=destination,
        source=path,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_preparation_validates_entire_package_before_copy(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "weights").write_bytes(b"good")
    manifest = AssetManifest(assets=[entry("weights", b"good"), entry("missing", b"x")])
    with pytest.raises(FileNotFoundError):
        prepare_assets(manifest, package, tmp_path / "repo", tmp_path / "hub")
    assert not (tmp_path / "repo/weights").exists()


def test_preparation_copies_to_separate_roots_and_rejects_corruption(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "weights").write_bytes(b"good")
    (package / "encoder").write_bytes(b"esm")
    manifest = AssetManifest(
        assets=[entry("weights", b"good"), entry("encoder", b"esm", "torch_hub")]
    )
    repo, hub = tmp_path / "repo", tmp_path / "hub"
    assert prepare_assets(manifest, package, repo, hub) == {"copied": 2, "verified": 2}
    assert (hub / "encoder").read_bytes() == b"esm"
    assert prepare_assets(manifest, package, repo, hub) == {"copied": 0, "verified": 2}
    (repo / "weights").write_bytes(b"bad")
    with pytest.raises(ValueError, match=r"hash|size"):
        prepare_assets(manifest, package, repo, hub)
    assert (repo / "weights").read_bytes() == b"bad"


def test_manifest_rejects_traversal_duplicates_and_symlink_escape(tmp_path: Path) -> None:
    for path in ["../outside", "/absolute", "a/../outside"]:
        with pytest.raises(ValidationError):
            entry(path, b"x")
    with pytest.raises(ValidationError):
        AssetManifest(assets=[entry("same", b"x"), entry("same", b"x")])
    package = tmp_path / "package"
    package.mkdir()
    (package / "file").write_bytes(b"x")
    outside = tmp_path / "outside"
    outside.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "subdir").symlink_to(outside, target_is_directory=True)
    asset = entry("subdir/file", b"x").model_copy(update={"source": "file"})
    with pytest.raises(ValueError, match="escapes"):
        prepare_assets(AssetManifest(assets=[asset]), package, repo, tmp_path / "hub")
    assert not (outside / "file").exists()


def test_encoder_download_is_verified_and_cached(tmp_path: Path) -> None:
    from robust_apex_qd.validation.inference_assets import PublicEncoder, ensure_public_encoders

    encoder = PublicEncoder(
        **entry("esm.pt", b"esm").model_dump(),
        url="https://dl.fbaipublicfiles.com/fair-esm/models/test.pt",
    )
    calls = []

    def fetch(url: str, target: Path, size: int) -> None:
        calls.append(url)
        assert size == 3
        target.write_bytes(b"esm")

    repo, hub = tmp_path / "repo", tmp_path / "hub"
    ensure_public_encoders([encoder], repo, hub, fetch=fetch)
    ensure_public_encoders([encoder], repo, hub, fetch=fetch)
    assert len(calls) == 1
    (repo / "esm.pt").write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash"):
        ensure_public_encoders([encoder], repo, hub, fetch=fetch)
    assert len(calls) == 1


def test_bad_encoder_download_leaves_no_checkpoint(tmp_path: Path) -> None:
    from robust_apex_qd.validation.inference_assets import PublicEncoder, ensure_public_encoders

    encoder = PublicEncoder(
        **entry("esm.pt", b"esm").model_dump(),
        url="https://dl.fbaipublicfiles.com/fair-esm/models/test.pt",
    )

    def fetch(url: str, target: Path, size: int) -> None:
        target.write_bytes(b"bad")

    with pytest.raises(ValueError, match="hash"):
        ensure_public_encoders([encoder], tmp_path / "repo", tmp_path / "hub", fetch=fetch)
    assert not list((tmp_path / "repo").iterdir())
    with pytest.raises(ValidationError):
        encoder.model_validate({**encoder.model_dump(), "url": "https://example.com/model"})
