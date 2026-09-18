"""Bounded ESMFold structures for candidates that differ between frozen rankings."""

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_competition_generator import sha


class FoldingStem(torch.nn.Module):
    """Run language representations on one device and return them to CPU folding."""

    def __init__(self, stem: torch.nn.Module, device: str) -> None:
        super().__init__()
        self.stem = stem.to(device)
        self.device = device

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_hidden_states: bool
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        output = self.stem(
            input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=output_hidden_states,
        )
        return {"hidden_states": tuple(value.cpu() for value in output["hidden_states"])}


def load_fold_model(assets: Path) -> tuple[Any, dict[str, Any]]:
    transformers = importlib.import_module("transformers")
    accelerate = importlib.import_module("accelerate")
    cfg = transformers.EsmConfig.from_json_file(assets / "config.json")
    cfg.esmfold_config.trunk.max_recycles = 1
    with accelerate.init_empty_weights():
        model = transformers.EsmForProteinFolding(cfg)
    state = torch.load(
        assets / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True
    )
    # This checkpoint predates nonpersistent position_ids. Check the buffer before dropping it.
    position_ids = state.pop("esm.embeddings.position_ids")
    if not torch.equal(position_ids, model.esm.embeddings.position_ids):
        raise ValueError("Stored position IDs differ from the configured buffer")
    # The unused contact-regression head is absent from the folding checkpoint.
    model.esm.contact_head = torch.nn.Identity()
    model.load_state_dict(state, assign=True, strict=True)
    audit = dict(
        position_ids_equal=True,
        contact_head="unused by folding; absent from checkpoint",
        strict_load=True,
        half_parameter_bytes=2 * sum(p.numel() for p in model.parameters()),
        meta_buffers=[n for n, b in model.named_buffers() if b.device.type == "meta"],
    )
    return model, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = Path("work/competition_exploration/20260912-a/phase2/tops")

    def read(path: Path) -> list[str]:
        return [s for s in path.read_text().splitlines() if s and not s.startswith(">")]

    control = read(base / "library-L2/top.fasta")
    alternative = read(base / "rank-species/top.fasta")
    sequences = [s for s in alternative if s not in set(control)][:10] + [
        s for s in control if s not in set(alternative)
    ][:10]
    assets = Path("work/competition_exploration/20260912-b/assets/esmfold")
    weights = assets / "pytorch_model.bin"
    config = assets / "config.json"
    protocol = dict(
        count=len(sequences),
        sequences=sequences,
        seed=42,
        model="facebook/esmfold_v1",
        dtype="ESM float16 CUDA, folding trunk float32 CPU",
        recycles=1,
        chunk_size=8,
        confidence_unit="pLDDT0-100 and PDB B-factor; pinned Transformers raw0-1 scaled by100",
        scope="single-chain structure pilot, no membrane or MD",
        membrane_condition=None,
        force_field=None,
        gpu_budget_seconds=7200,
        input_sha256={
            str(p): sha(p)
            for p in [
                weights,
                config,
                Path(__file__),
                base / "library-L2/top.fasta",
                base / "rank-species/top.fasta",
            ]
        },
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    started = time.monotonic()
    model, compatibility = load_fold_model(assets)
    (args.output / "compatibility.json").write_text(json.dumps(compatibility, indent=2) + "\n")
    model.esm = FoldingStem(model.esm.half(), "cuda")
    model.eval()
    model.trunk.set_chunk_size(8)
    torch.cuda.reset_peak_memory_stats()
    results = []
    for index, sequence in enumerate(sequences):
        if time.monotonic() - started > 7200:
            raise TimeoutError("Structure pilot GPU budget exceeded")
        start = time.monotonic()
        with torch.no_grad():
            output = model.infer(sequence)
        raw_confidence = output["plddt"]
        if (
            not torch.isfinite(raw_confidence).all()
            or not ((raw_confidence >= 0) & (raw_confidence <= 1)).all()
        ):
            raise ValueError("Unexpected raw confidence scale")
        output["plddt"] = 100 * raw_confidence
        pdb = model.output_to_pdb(output)[0]
        (args.output / f"{index:02}.pdb").write_text(pdb)
        confidence = output["plddt"].float().cpu().numpy()
        exists = output["atom37_atom_exists"].bool().cpu().numpy()
        confidence = confidence[exists]
        coordinates = output["positions"].float().cpu().numpy()
        if not np.isfinite(confidence).all() or not np.isfinite(coordinates).all():
            raise ValueError("Nonfinite structure prediction")
        result = dict(
            index=index,
            sequence=sequence,
            cohort="alternative" if index < 10 else "control",
            plddt_mean=float(confidence.mean()),
            seconds=time.monotonic() - start,
        )
        results.append(result)
        (args.output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
        print(result, flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                **protocol,
                seconds=time.monotonic() - started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                results=results,
                interpretation=(
                    "pLDDT is structural confidence, not activity, "
                    "folding stability or membrane affinity"
                ),
                artifacts_sha256={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
