# Robust APEX-QD submission

Deterministic AMP-Diffusion generation with 250-step DDIM, a 120,000-candidate pool,
a 50,000-member Lref library, and rankmean consensus from APEX and five MIC predictors.
This dedicated distribution repository is prepared for Full submission. Eligibility and
Kaggle submission remain pending.

## Run

Requirements: Linux, Git LFS, uv, Python 3.10, a compatible CUDA GPU, and internet for the first
encoder download. The locked PyTorch build is CUDA 12.8; the reference machine is an RTX 5060.
Allow approximately 4 hours per full generation on that machine, plus download/setup time.

```bash
git clone https://github.com/hirofuminakayama/amp-2026-submission.git
cd amp-2026-submission
git lfs install --local
git lfs pull
uv sync --frozen
uv run --frozen generate
uv run --frozen verify-local
uv run --frozen python scripts/verify_existing_output.py --output-dir generate --antibacterial-fasta data/antibacterial.fasta
```

Until public release, cloning requires owner authentication. Generator, APEX and five MIC
model bundles are distributed here. Three public ESM encoder files are downloaded from fixed
FAIR URLs and verified against `configs/inference_encoders.json`. `TORCH_HOME` optionally
selects the encoder cache; use an absolute path.

Defaults in `configs/final.yaml` define the model, seed, sampler, thresholds and ranking.
No training-only measurements or private research directories are needed for inference.
An incompatible GPU, corrupt model or insufficient memory is an error; the method and batch
size are not silently changed.

Outputs are `generate/library.fasta`, `generate/top.fasta` (rank order), `generate/ranking.tsv`
and `generate/manifest.json`. Sequences are intended as linear peptides with free termini.
They are computational candidates, not experimentally validated treatments.

## Reproduce twice

```bash
uv run --frozen python scripts/verify_submission.py https://github.com/hirofuminakayama/amp-2026-submission.git --branch submission-v1 --dir /tmp/amp-submission-verification --antibacterial-fasta data/antibacterial.fasta
```

Use a new clone directory for every verification. The verifier executes the default entrypoint
twice and compares outputs. Release review additionally records asset and ranking hashes.
`source_provenance.json` records exact source/model identities without importing the development
repository's Git history. It is not itself proof of completed remote validation.

## Method, data and license

See [abstract](abstract.md), [disclosure](disclosure.md), [data sources](DATA_SOURCES.md),
[training disclosure](training_disclosure.json), and [third-party notices](THIRD_PARTY_LICENSES.md).
The five MIC bundles share 5,036 exact public observations. Public partitions were reused for
development; no independent holdout or complete chemistry annotation is claimed.
The supplied APEX model is used under the official starter provenance and retained MIT notice.
Known training information and unresolved overlap are disclosed; a fresh organizer permission
request is not a prerequisite. Final eligibility is determined by the organizers.

Root code and our five predictor bundles are MIT licensed. Upstream notices and dataset terms
remain applicable. Additional raw training tables and research outputs are not redistributed.
Supporting research utilities are retained for import compatibility and tests. Research-only
commands require separately obtained inputs and are not required for the inference workflow.

## Checks

```bash
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen ty check
CUDA_VISIBLE_DEVICES= uv run --frozen pytest -q
```
