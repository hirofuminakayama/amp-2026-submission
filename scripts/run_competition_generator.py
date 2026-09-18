"""Run bounded public generator inference in its pinned isolated environment."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robust_apex_qd.research.competition_generators import validate_screen


def module_at(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load public module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=["designer", "prompt", "ampgen", "evodiff"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "cpu":
        # The published Designer function selects device via this capability check.
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    old = Path("work/measured_activity_research/generation-assets")
    protocol = dict(
        seed=args.seed,
        count=args.count,
        mode=args.family,
        min_length=10 if args.family in ["designer", "prompt"] else 15,
        max_length=32 if args.family == "designer" else 34 if args.family == "prompt" else 35,
        device=args.device,
        batch_size=32 if args.family in ["designer", "prompt", "evodiff"] else 1,
        internal_filter=False,
        common_length=[15, 25],
        weights=str(args.weights) if args.weights else None,
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    inputs = [Path(__file__), Path("src/robust_apex_qd/research/competition_generators.py")]
    start = time.monotonic()
    sequences = []
    if args.family in ["designer", "prompt"]:
        transformer = importlib.import_module("transformers")
        source = old / "AMP-Designer"
        sys.path.insert(0, str(source.resolve()))
        weights = args.weights or old / "amp-designer-weights"
        tokenizer = transformer.BertTokenizer(vocab_file=str(source / "voc/vocab.txt"))
        model = transformer.GPT2LMHeadModel.from_pretrained(weights, local_files_only=True)
        generator = module_at(
            source
            / ("AMP_GPT_generator.py" if args.family == "designer" else "AMP_prompt_generator.py")
        )
        if args.family == "prompt":
            state = torch.load(
                weights / "pytorch_model.bin", map_location="cpu", weights_only=False
            )
            embedding = module_at(source / "soft_prompt_embedding.py").SoftEmbedding(
                model.get_input_embeddings(), n_tokens=10, initialize_from_vocab=True
            )
            embedding.learned_embedding.data = state["transformer.wte.learned_embedding"]
            embedding.wte.weight.data = state["transformer.wte.wte.weight"]
            model.set_input_embeddings(embedding)
        inputs.extend(
            [
                weights / "config.json",
                weights / "pytorch_model.bin",
                source / "voc/vocab.txt",
                source
                / (
                    "AMP_GPT_generator.py"
                    if args.family == "designer"
                    else "AMP_prompt_generator.py"
                ),
            ]
        )
        with torch.no_grad():
            for offset in range(0, args.count, 32):
                count = min(32, args.count - offset)
                samples = (
                    generator.predict(model, tokenizer, batch_size=count)
                    if args.family == "designer"
                    else generator.predict(
                        argparse.Namespace(top_k=0, top_p=1.0), model, tokenizer, batch_size=count
                    )
                )
                sequences.extend(generator.decode(x) for x in samples)
    else:
        pretrained = importlib.import_module("evodiff.pretrained")
        if args.family == "ampgen":
            model, _, tokenizer, _ = pretrained.MSA_OA_DM_MAXSUB()
            sample = importlib.import_module("evodiff.generate_msa").generate_query_oadm_msa_simple
            msa = old / "AMPGen/data/example/msa_files/example_1944.a3m"
            inputs.append(msa)
            depth = min(64, sum(line.startswith(">") for line in msa.read_text().splitlines()) - 1)
        else:
            model, _, tokenizer, _ = pretrained.OA_DM_38M()
            sample = importlib.import_module("evodiff.generate").generate_oaardm
            if args.weights:
                model.load_state_dict(
                    torch.load(args.weights, map_location="cpu", weights_only=True)
                )
                inputs.append(args.weights)
        model.eval().to(args.device)
        while len(sequences) < args.count:
            length = random.randint(15, 35)
            if args.family == "ampgen":
                _, batch = sample(
                    str(msa),
                    model,
                    tokenizer,
                    depth,
                    length,
                    device=args.device,
                    selection_type="MaxHamming",
                )
                sequences.append(batch[0][0].replace("!", "").replace("-", ""))
            else:
                _, batch = sample(
                    model,
                    tokenizer,
                    length,
                    batch_size=min(32, args.count - len(sequences)),
                    device=args.device,
                )
                sequences.extend(batch)
            print(f"{args.family} {len(sequences)}/{args.count}", flush=True)
    validation = validate_screen(sequences, protocol, protocol)
    (args.output / "raw.fasta").write_text(
        "".join(f">seq{i}\n{s}\n" for i, s in enumerate(sequences))
    )
    (args.output / "raw_sequences.json").write_text(json.dumps(sequences) + "\n")
    manifest = dict(
        **protocol,
        validation=validation,
        seconds=time.monotonic() - start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
        input_sha256={str(p): sha(p) for p in inputs},
        artifacts_sha256={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
    )
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(dict(count=len(sequences), seconds=manifest["seconds"], validation=validation))
    )


if __name__ == "__main__":
    main()
