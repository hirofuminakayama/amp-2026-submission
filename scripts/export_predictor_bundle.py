"""Export a verified local refit as a portable inference-only predictor bundle."""

import argparse
from pathlib import Path

from robust_apex_qd.ranking.predictor_bundle import export_predictor_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_predictor_bundle(args.source, args.output)
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
