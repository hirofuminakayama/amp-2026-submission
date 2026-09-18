"""Verify and install a frozen inference package without replacing existing assets."""

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robust_apex_qd.features.embeddings import file_sha256


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    source: str
    destination: Literal["repository", "torch_hub"]
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path", "source")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError("Asset paths must be relative and contained")
        return value


class AssetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    assets: list[Asset] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_destinations(self) -> "AssetManifest":
        keys = {(asset.destination, str(Path(asset.path))) for asset in self.assets}
        if len(keys) != len(self.assets):
            raise ValueError("Duplicate asset destination")
        return self


class PublicEncoder(Asset):
    url: str

    @field_validator("url")
    @classmethod
    def official_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "dl.fbaipublicfiles.com"
            or not parsed.path.startswith("/fair-esm/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Encoder URL must identify the official FAIR ESM distribution")
        return value


def contained(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes root: {relative}")
    return path


def verify_asset(path: Path, asset: Asset) -> None:
    if path.stat().st_size != asset.size_bytes or file_sha256(path) != asset.sha256:
        raise ValueError(f"Asset size or hash differs: {path}")


def prepare_assets(
    manifest: AssetManifest, package: Path, repository: Path, torch_hub: Path
) -> dict[str, int]:
    roots = {"repository": repository, "torch_hub": torch_hub}
    pending = []
    for asset in manifest.assets:
        source = contained(package, asset.source)
        destination = contained(roots[asset.destination], asset.path)
        verify_asset(source, asset)
        if destination.exists():
            verify_asset(destination, asset)
        else:
            pending.append((asset, source, destination))
    for asset, source, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            scratch = Path(temporary.name)
        try:
            shutil.copyfile(source, scratch)
            verify_asset(scratch, asset)
            # A concurrent installation must not be overwritten.
            os.link(scratch, destination)
        finally:
            scratch.unlink(missing_ok=True)
    return {"copied": len(pending), "verified": len(manifest.assets)}


def download_encoder(url: str, target: Path, expected_bytes: int) -> None:
    count = 0
    with urlopen(url, timeout=60) as response, target.open("wb") as output:
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > expected_bytes:
                raise ValueError("Encoder download exceeds registered size")
            output.write(block)


def ensure_public_encoders(
    encoders: list[PublicEncoder],
    repository: Path,
    torch_hub: Path,
    *,
    fetch: Callable[[str, Path, int], None] = download_encoder,
) -> None:
    roots = {"repository": repository, "torch_hub": torch_hub}
    AssetManifest(assets=list(encoders))
    for encoder in encoders:
        target = contained(roots[encoder.destination], encoder.path)
        if target.exists():
            verify_asset(target, encoder)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            scratch = Path(temporary.name)
        try:
            print(f"Downloading verified encoder: {target.name}", flush=True)
            fetch(encoder.url, scratch, encoder.size_bytes)
            verify_asset(scratch, encoder)
            os.link(scratch, target)
        finally:
            scratch.unlink(missing_ok=True)


def prepare_default_encoders(repository: Path) -> None:
    import torch

    config = json.loads((repository / "configs/inference_encoders.json").read_text())
    encoders = [PublicEncoder.model_validate(item) for item in config]
    ensure_public_encoders(encoders, repository, Path(torch.hub.get_dir()))
