"""A bounded EvoDiff SFT and reward-weighted masked-likelihood experiment."""

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_competition_generator import sha

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robust_apex_qd.features.physchem import compute_features
from robust_apex_qd.research.competition_models import predict_ridge_heads


def rewards(sequences: list[str], state: dict) -> np.ndarray:
    features = []
    for sequence in sequences:
        props = compute_features(sequence)
        features.append(
            [*props.values(), *[sequence.count(a) / len(sequence) for a in "ACDEFGHIKLMNPQRSTVWY"]]
        )
    return -np.nanmean(predict_ridge_heads(state, np.asarray(features)), axis=1)


def encode_batch(tokenizer: Any, sequences: list[str], device: str) -> torch.Tensor:
    tokens = torch.full(
        (len(sequences), max(map(len, sequences))),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    for i, sequence in enumerate(sequences):
        encoded = tokenizer.tokenize([sequence])
        if len(encoded) != len(sequence):
            raise ValueError("Tokenizer did not preserve every residue")
        tokens[i, : len(sequence)] = torch.tensor(encoded, device=device)
    return tokens


def update(
    model: torch.nn.Module,
    tokenizer: Any,
    sequences: list[str],
    weights: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
) -> list[float]:
    rng = np.random.default_rng(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _epoch in range(epochs):
        total = 0.0
        order = rng.permutation(len(sequences))
        for start in range(0, len(order), 16):
            ids = order[start : start + 16]
            batch = [sequences[i] for i in ids]
            tokens = encode_batch(tokenizer, batch, "cuda")
            target = tokens.clone()
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            for i, s in enumerate(batch):
                positions = rng.choice(len(s), max(1, int(0.3 * len(s))), replace=False)
                mask[i, positions] = True
            tokens[mask] = tokenizer.mask_id
            logits = model(
                tokens,
                torch.zeros(len(batch), device="cuda", dtype=torch.long),
                input_mask=(target != tokenizer.pad_id).unsqueeze(-1),
            )
            per_token = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), target, reduction="none"
            )
            per_sequence = (per_token * mask).sum(1) / mask.sum(1)
            loss = (
                per_sequence * torch.tensor(weights[ids], device="cuda", dtype=torch.float32)
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        losses.append(total / len(sequences))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reward_root = Path("work/competition_exploration/20260912-b/phase4/reward")
    reward_state = dict(np.load(reward_root / "weights.npz"))
    data = [
        s
        for s in (reward_root / "sft.fasta").read_text().splitlines()
        if s and not s.startswith(">") and 15 <= len(s) <= 35
    ]
    protocol = dict(
        seed=42,
        iterations=5,
        count=1000,
        sft_epochs=3,
        sft_lr=0.0001,
        update_epochs=1,
        update_lr=0.00001,
        elite_fraction=0.3,
        min_length=15,
        max_length=35,
        method="own reward-weighted masked likelihood, not reproduced ProDCARL/PPO",
        reward_model="new physchem species MIC heads; negative mean log2 uM",
        evaluation_only_models=["APEX", "HemoPI2 HC50"],
        input_sha256={
            str(p): sha(p)
            for p in [Path(__file__), reward_root / "weights.npz", reward_root / "sft.fasta"]
        },
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    model, _, tokenizer, _ = importlib.import_module("evodiff.pretrained").OA_DM_38M()
    model.cuda()
    sample = importlib.import_module("evodiff.generate").generate_oaardm
    started = time.monotonic()
    losses = update(model, tokenizer, data, np.ones(len(data)), 42, 3, 0.0001)
    torch.save(model.state_dict(), args.output / "sft.pt")
    records: list[dict[str, Any]] = [dict(stage="sft", rows=len(data), losses=losses)]
    for iteration in range(6):
        path = args.output / f"iteration{iteration}"
        path.mkdir()
        seed = 42 + iteration
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model.eval()
        sequences = []
        while len(sequences) < 1000:
            _, batch = sample(
                model,
                tokenizer,
                random.randint(15, 35),
                batch_size=min(32, 1000 - len(sequences)),
                device="cuda",
            )
            sequences.extend(batch)
        score = rewards(sequences, reward_state)
        (path / "raw.fasta").write_text("".join(f">seq{i}\n{s}\n" for i, s in enumerate(sequences)))
        np.save(path / "reward.npy", score)
        records.append(
            dict(
                iteration=iteration,
                seed=seed,
                count=len(sequences),
                reward_mean=float(score.mean()),
                reward_p90=float(np.quantile(score, 0.9)),
            )
        )
        if iteration < 5:
            selected = np.argsort(-score, kind="stable")[:300]
            standardized = (score[selected] - score[selected].mean()) / (
                score[selected].std() + 1e-6
            )
            weights = np.exp(np.clip(standardized, -2, 2))
            weights /= weights.mean()
            records[-1]["losses"] = update(
                model, tokenizer, [sequences[i] for i in selected], weights, seed, 1, 0.00001
            )
            torch.save(model.state_dict(), args.output / f"rl{iteration + 1}.pt")
        (path / "iteration.json").write_text(json.dumps(records[-1], indent=2) + "\n")
        print(records[-1], flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                **protocol,
                seconds=time.monotonic() - started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                records=records,
                artifacts_sha256={
                    str(p.relative_to(args.output)): sha(p)
                    for p in args.output.rglob("*")
                    if p.is_file()
                },
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
