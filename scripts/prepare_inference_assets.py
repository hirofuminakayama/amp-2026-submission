"""Install a verified inference package into a checkout and an explicit torch hub."""

import argparse
import json
from pathlib import Path

from robust_apex_qd.validation.inference_assets import AssetManifest, prepare_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    args = parser.parse_args()
    manifest = AssetManifest.model_validate_json(args.manifest.read_text())
    result = prepare_assets(manifest, args.package, args.repository, args.torch_home / "hub")
    print(json.dumps(result, sort_keys=True))
    print(f"Run generation with TORCH_HOME={args.torch_home.resolve()}")


if __name__ == "__main__":
    main()
