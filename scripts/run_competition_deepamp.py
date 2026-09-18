"""Public deepAMP general-checkpoint masked completion and published SVM inference."""

import argparse
import hashlib
import importlib
import json
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from run_competition_generator import sha


def conditioning_inputs(sequences: list[str], count: int, mode: str) -> list[str]:
    unique = list(dict.fromkeys(sequences))
    if mode == "common":
        unique = sorted(
            [s for s in unique if 15 <= len(s) <= 25],
            key=lambda s: (hashlib.sha256(f"42:{s}".encode()).digest(), s),
        )
    chosen = unique[:count]
    if len(chosen) != count:
        raise ValueError("Insufficient conditioning inputs")
    return chosen


def complete_masked_positions(
    tokens: torch.Tensor, positions: torch.Tensor, predictions: torch.Tensor
) -> torch.Tensor:
    # MaskCollate pads unused prediction slots with zero, the CLS position.
    rows = torch.arange(tokens.shape[0], device=tokens.device)[:, None].expand_as(positions)
    valid = positions != 0
    completed = tokens.clone()
    completed[rows[valid], positions[valid]] = predictions[valid]
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditioning", choices=["prefix", "common"], default="prefix")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path("work/competition_exploration/20260912-b/assets/deepAMP").resolve()
    weights = source.parent / "weights/deepAMP-general.pkl"
    svm_path = source.parent / "weights/deepAMP-predict.pkl"
    input_path = Path("work/phase11-clean-0f2f600/full_run_1/library.fasta")
    inputs = conditioning_inputs(
        [s for s in input_path.read_text().splitlines() if s and not s.startswith(">")],
        args.count,
        args.conditioning,
    )
    (args.output / "conditioning.fasta").write_text(
        "".join(f">input{i}\n{s}\n" for i, s in enumerate(inputs))
    )
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "code"))
    config_cls = importlib.import_module("configer").BertConfig
    model_cls = importlib.import_module("src.model").AMPBERT
    collater_cls = importlib.import_module("src.DataProcessTools").MaskCollate
    vocab = list((source / "vocab/Protein.vocab").read_text().strip())
    stoi = {s: i for i, s in enumerate(vocab)}
    itos = dict(enumerate(vocab))
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    protocol = dict(
        seed=args.seed,
        count=args.count,
        mode="general_masked_completion",
        conditioning=args.conditioning,
        conditioning_selection_seed=42 if args.conditioning == "common" else None,
        mask_ratio=0.3,
        max_pred=4,
        min_length=10,
        max_length=40,
        device="cpu",
        internal_filter=False,
        reproduction=(
            "published general checkpoint and MaskCollate, "
            "one pass per input; no AOM/POM checkpoint"
        ),
        input_sha256={
            str(p): sha(p)
            for p in [
                weights,
                svm_path,
                input_path,
                Path(__file__),
                source / "src/model.py",
                source / "src/DataProcessTools.py",
                source / "code/score_list.py",
            ]
        },
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    start = time.monotonic()
    model = model_cls(
        config_cls(
            len(vocab),
            64,
            n_embd=32,
            n_layers=12,
            n_heads=12,
            d_model=512,
            d_ff=1024,
            d_k=64,
            d_v=64,
        )
    )
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    dataset = SimpleNamespace(stoi=stoi, itos=itos, vocab_size=len(vocab), block_size=64)
    collater = collater_cls(dataset, mask_ratio=0.3, max_pred=4, mask=True, sample=False)
    sequences = []
    with torch.no_grad():
        for start_i in range(0, len(inputs), 32):
            raw = [
                [stoi["$"], *[stoi[c] for c in s], stoi["&"]]
                for s in inputs[start_i : start_i + 32]
            ]
            x, _, positions = collater.mask_collate(raw)
            logits, _ = model(x, positions)
            x = complete_masked_positions(x, positions, logits.argmax(-1))
            sequences.extend(
                "".join(itos[int(i)] for i in row if itos[int(i)] not in "#$&*") for row in x
            )
    # The pinned public classifier is sklearn SVC, not a torch checkpoint or MIC regressor.
    with svm_path.open("rb") as f:
        svm = pickle.load(f)
    encoder = importlib.import_module("score_list").s2t(str(source / "src/RECM-position.txt"))
    scores = svm.predict_proba(np.array([encoder.embed_RECM_position(s) for s in sequences]))
    pd.DataFrame(dict(sequence=sequences, svm_probability=scores[:, 1])).to_csv(
        args.output / "predictions.csv", index=False
    )
    (args.output / "raw.fasta").write_text(
        "".join(f">seq{i}\n{s}\n" for i, s in enumerate(sequences))
    )
    (args.output / "run_manifest.json").write_text(
        json.dumps(
            dict(
                **protocol,
                seconds=time.monotonic() - start,
                raw_count=len(sequences),
                changed=sum(a != b for a, b in zip(inputs, sequences, strict=True)),
                classifier_classes=svm.classes_.tolist(),
                classifier_unit="class probability, not MIC",
                artifacts_sha256={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
