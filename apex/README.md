# apex/ — vendored APEX-pathogen scorer

Third-party code, **not** part of the starter kit's own model. This is a vendored copy of
**APEX-pathogen** ([`machine-biology-group-public/apex-pathogen`](https://gitlab.com/machine-biology-group-public/apex-pathogen),
Wan / de la Fuente lab; [*Nat. Microbiol.* 2025](https://www.nature.com/articles/s41564-025-02061-0)),
the APEX release specialized for the 11 clinical pathogens. It is the
model that produced the AMP-Diffusion paper's reported MICs and reproduces the supplementary
values exactly — the broad "de-extinction" APEX (`.../apex`) is a different model on a
different MIC scale and is **not** interchangeable here (see the main README).

It predicts per-species MIC (µM) against the 11-pathogen panel with an 8-model ensemble, and is
run as an **isolated `uv` project** (its own `pyproject.toml`/`uv.lock`, CPU torch) invoked as a
subprocess from [`../src/ampdiffusion_starter_kit/scoring.py`](../src/ampdiffusion_starter_kit/scoring.py),
so its dependencies never co-resolve with the kit's ESM2 stack. The environment syncs on first
call.

Contents:
- `APEX_models.py`, `utils.py`, `aaindex1.csv` — model and featurization (upstream).
- `APEX_pathogen_models/` — the 8 pretrained ensemble weights (Git LFS).
- `APEX_predict.py` — upstream inference script, lightly wrapped for the kit (`-i`/`-o` paths,
  auto device detection, CPU-safe model loading); `utils.py` is trimmed to the inference-time
  functions (dropping training-only scipy/scikit-learn imports).

**License:** MIT — see [`LICENSE`](LICENSE) (© 2025 Fangping Wan).

The adopted root DDIM/Lref/rankmean route uses APEX_predict_ensemble.py with the root
locked Python environment; the separate uv project described above is for the legacy route.
