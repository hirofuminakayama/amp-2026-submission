import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from robust_apex_qd.apex.ensemble import load_prediction_archive


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare ensemble mean with official APEX CSV")
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-6)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    archive = load_prediction_archive(options.ensemble.resolve())
    official = pd.read_csv(options.official.resolve(), index_col=0)
    if tuple(str(value) for value in official.index) != archive.sequences:
        raise ValueError("Official APEX sequence order differs from the ensemble archive")
    if tuple(official.columns) != archive.pathogens:
        raise ValueError("Official APEX pathogen order differs from the ensemble archive")
    observed = archive.mic_u_m.mean(axis=1, dtype=np.float32)
    expected = official.to_numpy(dtype=np.float32)
    np.testing.assert_allclose(observed, expected, rtol=options.rtol, atol=options.atol)
    maximum_difference = float(np.max(np.abs(observed - expected)))
    print(
        f"APEX official mean matched for {len(archive.sequences)} sequences; "
        f"max_abs_difference={maximum_difference:.10g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
