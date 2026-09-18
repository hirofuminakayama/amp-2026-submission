"""Check a shared ESM2-3B stem against AMPGen examples before scoring new samples."""

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from run_competition_generator import module_at, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path("work/competition_exploration/20260912-b")
    source = Path("work/measured_activity_research/generation-assets/AMPGen")
    assets = root / "assets/esmfold"
    examples = source / "data/example/output/sequences.fasta"
    generated = root / "phase4/ampgen-s42/raw_sequences.json"
    reference_sequences = [s for s in examples.read_text().splitlines() if not s.startswith(">")]
    new_sequences = list(
        dict.fromkeys(
            s
            for s in json.loads(generated.read_text())
            if 15 <= len(s) <= 35 and set(s) <= set("ACDEFGHIKLMNPQRSTVWY")
        )
    )[:32]
    sequences = reference_sequences + new_sequences
    protocol = dict(
        scope="ESMFold frozen ESM2-3B stem -> AMPGen public scalers/LSTM, float16 embedding",
        seed=42,
        reference_count=len(reference_sequences),
        generated_count=len(new_sequences),
        generated_filter="canonical amino acids, native length15-35, first32 unique",
        pooling="layer36 mean residues, excluding BOS/EOS/padding",
        parity_minimum_cosine=0.999,
        parity_maximum_rmse=0.02,
        unit="published logMIC target; base, concentration unit and exact strain not verified",
        input_sha256={
            str(p): sha(p)
            for p in [
                Path(__file__),
                examples,
                generated,
                assets / "config.json",
                assets / "pytorch_model.bin",
            ]
        },
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    transformers = importlib.import_module("transformers")
    accelerate = importlib.import_module("accelerate")
    cfg = transformers.EsmConfig.from_json_file(assets / "config.json")
    with accelerate.init_empty_weights():
        model = transformers.EsmModel(cfg, add_pooling_layer=False)
    archive = torch.load(
        assets / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True
    )
    state = {k.removeprefix("esm."): v for k, v in archive.items() if k.startswith("esm.")}
    if not torch.equal(state.pop("embeddings.position_ids"), model.embeddings.position_ids):
        raise ValueError("ESM position buffer differs")
    model.contact_head = torch.nn.Identity()
    model.load_state_dict(state, assign=True, strict=True)
    model.to(device="cuda", dtype=torch.float16).eval()
    vocabulary = {token: i for i, token in enumerate(cfg.vocab_list)}
    vectors = []
    for sequence in sequences:
        ids = [vocabulary["<cls>"], *[vocabulary[a] for a in sequence], vocabulary["<eos>"]]
        tokens = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            out = model(tokens, attention_mask=tokens.ne(vocabulary["<pad>"]))
        vectors.append(out.last_hidden_state[0, 1:-1].float().mean(0).cpu().numpy())
    vectors = np.asarray(vectors)
    np.save(args.output / "embeddings.npy", vectors)
    expected = np.asarray(
        [
            torch.load(
                source / f"data/example/output/embeddings/{i + 1}.pt",
                map_location="cpu",
                weights_only=True,
            )["mean_representations"][36].numpy()
            for i in range(len(reference_sequences))
        ]
    )
    observed = vectors[: len(expected)]
    cosine = (observed * expected).sum(1) / (
        np.linalg.norm(observed, axis=1) * np.linalg.norm(expected, axis=1)
    )
    rmse = np.sqrt(((observed - expected) ** 2).mean(1))
    parity = dict(
        cosine=cosine.tolist(),
        rmse=rmse.tolist(),
        passed=bool(cosine.min() >= 0.999 and rmse.max() <= 0.02),
    )
    (args.output / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    if not parity["passed"]:
        raise ValueError(
            "Reference embedding parity failed; do not claim published scorer reproduction"
        )
    pandas = importlib.import_module("pandas")
    frame = pandas.DataFrame(dict(Sequence=sequences))
    frame.to_csv(args.output / "input.csv", index=False)
    inputs = pandas.concat([frame, pandas.DataFrame(vectors)], axis=1)
    scorer = module_at(source / "MIC_scorer/scorer.py")
    scorer_inputs = [source / "MIC_scorer/scorer.py"]
    for name, checkpoint in [
        ("ecoli", "2ecoli_best_model_checkpoint.pth"),
        ("stpa", "1stpa_best_model_checkpoint.pth"),
    ]:
        scaler = source / f"MIC_scorer/Scorer_model/{name}scaler.pkl"
        weights = source / f"MIC_scorer/Scorer_model/{checkpoint}"
        scorer_inputs += [scaler, weights]
        scorer.get_predicted_mic(
            inputs,
            str(args.output / "input.csv"),
            str(scaler),
            str(weights),
            str(args.output / f"{name}.csv"),
            "cpu",
        )
        predicted = pandas.read_csv(args.output / f"{name}.csv")
        if not np.isfinite(predicted["Predicted Values"]).all():
            raise ValueError("Nonfinite scorer predictions")
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                **protocol,
                seconds=time.monotonic() - started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                parity=parity,
                scorer_sha256={str(p): sha(p) for p in scorer_inputs},
                artifacts_sha256={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
