"""AMP Challenge 2027 baseline generator (AMP-Diffusion).

Produces a default-settings 50,000-sequence library and a ranked top-100 with the published
AMP-Diffusion latent-diffusion model (Torres et al., *Cell Biomaterials* 2025; Chen et al.,
bioRxiv 2023). Output goes to ``generate_broad_spectrum/``:

    generate_broad_spectrum/library.fasta   full 50,000-sequence library
    generate_broad_spectrum/top.fasta       top-100 ranked candidates

AMP-Diffusion is a competition baseline (excluded from rankings) and is strain-agnostic, so
only the broad-spectrum entry point is wired. Generation samples Gaussian noise, denoises it
into ESM2 embeddings with the diffusion model, and decodes them to sequences via the ESM2
language-model head. Only the challenge's hard sequence rules are enforced on the library
(20 canonical residues, length 8-50, uniqueness, no identity with the antibacterial
reference). The top-100 reproduces the paper's computational filtering — APEX activity
(preferring MIC <= 64 uM), <0.60 local-alignment similarity to known AMPs (novelty), and a
>0.40 pairwise diversity cut — plus the challenge's hard <80% novelty guard.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

CATEGORY = "generate_broad_spectrum"
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoint" / "model.pt"
ANTIBACTERIAL_FASTA = ROOT / "data" / "antibacterial.fasta"
TRAINING_FASTA = ROOT / "data" / "training" / "training.fasta"
APEX_DIR = ROOT / "apex"

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_AA_SET = set(STANDARD_AA)
MIN_LENGTH = 8
MAX_LENGTH = 50
SIMILARITY_THRESHOLD = 0.8

# Paper's computational-filtering thresholds (Torres et al., Cell Biomaterials 2025).
MIC_THRESHOLD = 64.0  # (1) activity: keep APEX mean MIC <= 64 uM
NOVELTY_SIMILARITY = 0.60  # (2) exclude local-alignment similarity > 0.60 to known AMPs
DIVERSITY_SIMILARITY = 0.40  # (3) among survivors, > 0.40 pair -> keep the lower-MIC peptide

# AMP-Diffusion samples peptides in this length regime (paper §5.1: 15-40 aa; the released
# 8M checkpoint decodes 42 token positions). We sample lengths uniformly here.
GEN_MIN_LEN = 10
GEN_MAX_LEN = 40
EMBED_DIM = 320
PEP_MAX_LEN = 42


def _read_fasta_sequences(path: Path) -> list[str]:
    sequences, parts = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
                parts = []
        else:
            parts.append(line.upper())
    if parts:
        sequences.append("".join(parts))
    return sequences


def _write_fasta(sequences: list[str], path: Path) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">seq{i}\n{seq}\n")


def _is_valid(seq: str) -> bool:
    return MIN_LENGTH <= len(seq) <= MAX_LENGTH and not (set(seq) - STANDARD_AA_SET)


def set_seed(seed: int) -> None:
    """Seed every RNG and force deterministic kernels so repeated runs are byte-identical."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(checkpoint: Path, device):
    """Load ESM2-8M and the AMP-Diffusion EMA model from ``checkpoint``."""
    import esm
    from ema_pytorch import EMA

    from ampdiffusion_starter_kit.model import Denoise_Transformer, GaussianDiffusion1D

    esm2, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm2.eval().to(device)

    model = Denoise_Transformer(esm_layers=esm2.layers, embed_dim=EMBED_DIM, pep_max_len=PEP_MAX_LEN)
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=PEP_MAX_LEN,
        timesteps=1000,
        objective="pred_x0",
        loss_type="l2",
        auto_normalize=False,
        embed_dim=EMBED_DIM,
        self_condition=False,
        device=device,
    ).to(device)

    data = torch.load(checkpoint, map_location=device, weights_only=False)
    ema = EMA(diffusion, beta=0.999, update_every=10)
    ema.to(device)
    ema.load_state_dict(data["ema"])
    ema.ema_model.eval()

    std_idxs = [alphabet.get_idx(aa) for aa in STANDARD_AA]
    return ema.ema_model, esm2, alphabet, std_idxs


@torch.no_grad()
def _decode(esm2, sampled, std_idxs, design_len: int) -> list[str]:
    """Decode diffusion-sampled embeddings [B, PEP_MAX_LEN, EMBED_DIM] to peptide strings."""
    seqs = []
    for i in range(sampled.shape[0]):
        logits = esm2.lm_head(sampled[i, :])
        aa_logits = logits[:, std_idxs]
        pred = torch.argmax(aa_logits, dim=1)
        s = "".join(STANDARD_AA[j] for j in pred.tolist())[:design_len]
        seqs.append(s)
    return seqs


def generate_library(
    ema_model,
    esm2,
    std_idxs,
    n_sequences: int,
    seed: int,
    references: set[str],
    batch_size: int,
    device,
) -> list[str]:
    """Accumulate ``n_sequences`` unique, valid, novel-vs-reference sequences.

    Each round re-seeds deterministically, samples a batch at a length drawn from
    ``[GEN_MIN_LEN, GEN_MAX_LEN]``, decodes, and keeps sequences that are valid (8-50 aa,
    canonical), unique, and not identical to any antibacterial reference. Insertion order is
    preserved, so the whole procedure is reproducible for a fixed seed.
    """
    collected: dict[str, None] = {}
    length_rng = np.random.default_rng(seed)
    round_idx = 0
    while len(collected) < n_sequences:
        set_seed(seed + round_idx)
        design_len = int(length_rng.integers(GEN_MIN_LEN, GEN_MAX_LEN + 1))
        sampled = ema_model.sample(batch_size=batch_size, design_len=design_len + 2)
        for seq in _decode(esm2, sampled, std_idxs, design_len):
            if seq in collected or seq in references or not _is_valid(seq):
                continue
            collected[seq] = None
            if len(collected) == n_sequences:
                break
        round_idx += 1
        print(f"  round {round_idx} (len {design_len}): {len(collected)}/{n_sequences} collected")
    return list(collected)


def select_top(
    library: list[str],
    top_k: int,
    references: set[str],
    known_amps: list[str],
    workdir: Path,
    mic_threshold: float = MIC_THRESHOLD,
    novelty_threshold: float = NOVELTY_SIMILARITY,
    diversity_threshold: float = DIVERSITY_SIMILARITY,
    apex_mic: dict[str, float] | None = None,
    challenge_similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[str]:
    """Select the top-``top_k`` via the paper's computational filtering (Torres et al. 2025).

    Three criteria over the library — activity, novelty, diversity — plus the challenge's hard
    novelty guard:

      1. *Activity*: prefer peptides with APEX mean MIC <= ``mic_threshold`` (64 uM), taken in
         ascending-MIC order. If novelty/diversity leave fewer than ``top_k`` within the cut,
         the remaining slots are filled by the next-most-potent peptides beyond it.
      2. *Novelty*: drop any peptide with local-alignment similarity > ``novelty_threshold``
         (0.60) to any known AMP (``known_amps``, the DRAMP/APD3/DBAASP training set).
      3. *Diversity*: walk survivors in ascending MIC and accept a peptide only if its local-
         alignment similarity to every already-accepted peptide is <= ``diversity_threshold``
         (0.40) — i.e. of any > 0.40 pair, only the lower-MIC peptide is kept.

    A final guard enforces the challenge's <80% Levenshtein rule vs ``references``. Returns the
    ``top_k`` lowest-MIC peptides that pass all filters. (Diversity is checked before novelty
    only for speed — the accepted set is identical either way, since blocking peptides are
    always novel.)
    """
    import Levenshtein

    from ampdiffusion_starter_kit.similarity import local_similarity

    if apex_mic is None:
        from ampdiffusion_starter_kit.scoring import apex_mean_mic

        print(f"Scoring {len(library)} sequences with APEX ...")
        apex_mic = apex_mean_mic(library, APEX_DIR, workdir)

    ranked = sorted((s for s in library if s in apex_mic), key=lambda s: apex_mic[s])
    n_active = sum(apex_mic[s] <= mic_threshold for s in ranked)
    print(f"(1) activity (MIC <= {mic_threshold}): {n_active}/{len(ranked)} scored peptides")

    def novel_vs_known(seq: str) -> bool:
        return all(local_similarity(seq, k) <= novelty_threshold for k in known_amps)

    def passes_challenge(seq: str) -> bool:
        return all(
            Levenshtein.ratio(seq, r) <= challenge_similarity_threshold for r in references
        )

    # Walk candidates most-potent-first, applying diversity + novelty. The paper's MIC <= 64
    # activity cut usually leaves fewer than top_k after novelty/diversity, so we fill any
    # remaining slots from the next-most-potent peptides beyond the cut (still potency-ordered,
    # novelty and diversity kept strict) and report how many of the top_k met the cut.
    top: list[str] = []
    for seq in ranked:
        # (3) diversity first — cheap (<= len(top) comparisons) and rejects near-duplicates.
        if any(local_similarity(seq, t) > diversity_threshold for t in top):
            continue
        # challenge hard rule (Levenshtein vs the challenge reference set).
        if not passes_challenge(seq):
            continue
        # (2) novelty vs known AMPs — expensive (local alignment vs the whole training set).
        if not novel_vs_known(seq):
            continue
        top.append(seq)
        if len(top) == top_k:
            break
    if len(top) < top_k:
        raise RuntimeError(
            f"Only {len(top)} of {top_k} peptides passed novelty+diversity across the whole "
            f"{len(library)}-peptide library; increase --n-sequences."
        )
    n_within = sum(apex_mic[s] <= mic_threshold for s in top)
    print(
        f"    collected {top_k}: {n_within} within MIC <= {mic_threshold}, "
        f"{top_k - n_within} filled by potency (max MIC {apex_mic[top[-1]]:.1f})"
    )
    return top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sequences", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--antibacterial-fasta", default=str(ANTIBACTERIAL_FASTA))
    parser.add_argument("--known-amps-fasta", default=str(TRAINING_FASTA))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for path in (args.checkpoint, args.antibacterial_fasta, args.known_amps_fasta):
        if not Path(path).exists():
            sys.exit(f"ERROR: required path not found (cwd: {os.getcwd()}): {path}")

    device = torch.device(args.device)
    set_seed(args.seed)

    references = set(_read_fasta_sequences(Path(args.antibacterial_fasta)))
    known_amps = _read_fasta_sequences(Path(args.known_amps_fasta))
    ema_model, esm2, alphabet, std_idxs = load_model(Path(args.checkpoint), device)

    out_dir = Path(CATEGORY)
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "_work"

    print(f"Generating {args.n_sequences} sequences (seed={args.seed}, device={device})")
    library = generate_library(
        ema_model, esm2, std_idxs, args.n_sequences, args.seed, references, args.batch_size, device
    )
    _write_fasta(library, out_dir / "library.fasta")
    print(f"Wrote {len(library)} sequences -> {out_dir / 'library.fasta'}")

    top = select_top(library, args.top_k, references, known_amps, workdir)
    _write_fasta(top, out_dir / "top.fasta")
    print(f"Wrote top {len(top)} sequences -> {out_dir / 'top.fasta'}")
