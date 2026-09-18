"""APEX activity scorer for ranking the AMP-Diffusion top-100.

``apex_mean_mic`` — mean predicted MIC across the 11 clinical-pathogen panel from the APEX
ensemble (Wan/de la Fuente), the activity signal used to rank candidates in the AMP-Diffusion
selection pipeline (Torres et al., *Cell Biomaterials* 2025). We use the **APEX-pathogen**
release (gitlab ``machine-biology-group-public/apex-pathogen``), which is the model that
produced the paper's reported MICs — it reproduces the supplementary values exactly, whereas
the broad de-extinction APEX (``machine-biology-group-public/apex``) predicts on a different
absolute scale. APEX runs in its own isolated ``uv`` environment (``apex/``) invoked as a
subprocess, so it never conflicts with the kit's ESM2 stack. Lower = more potent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

# The 11 clinical pathogens AMP-Diffusion was benchmarked against (Cell Biomaterials Fig. 1/2).
# APEX-pathogen outputs exactly these columns; we average them for a broad-spectrum potency
# score. Names match the paper's supplementary (mmc3) columns.
APEX_PANEL = [
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
]


def apex_mean_mic(
    seqs: list[str], apex_dir: Path, workdir: Path, panel: list[str] = APEX_PANEL
) -> dict[str, float]:
    """Mean APEX-predicted MIC (uM) across the 11-pathogen panel for each sequence.

    Runs the vendored APEX-pathogen ensemble in its isolated ``uv`` environment via subprocess
    (``apex/APEX_predict.py`` → a per-sequence 11-pathogen matrix), then averages the columns.
    Deterministic: a fixed set of 8 models evaluated in eval mode.
    """
    from ampdiffusion_starter_kit.generate import _write_fasta

    workdir.mkdir(parents=True, exist_ok=True)
    fasta = (workdir / "apex_candidates.fasta").resolve()
    out = (workdir / "apex_pred.csv").resolve()
    _write_fasta(seqs, fasta)  # headers are ignored by APEX_predict.py (it indexes by sequence)

    # The isolated apex/ uv project syncs on this first call automatically. APEX_predict.py
    # auto-detects the device (CPU here, since the isolated env pins the CPU torch build).
    subprocess.run(
        ["uv", "run", "python", "APEX_predict.py", "-i", str(fasta), "-o", str(out)],
        cwd=str(apex_dir),
        check=True,
    )
    df = pd.read_csv(out, index_col=0)  # index = sequence, columns = the 11 pathogens

    missing = [c for c in panel if c not in df.columns]
    if missing:
        raise RuntimeError(f"APEX output missing panel columns: {missing}")

    means = df[panel].mean(axis=1)
    return {str(seq): float(mic) for seq, mic in means.items()}
