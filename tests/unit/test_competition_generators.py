from pathlib import Path

import pytest

from robust_apex_qd.research.competition_generators import validate_screen


def test_manifest_hashes_exclude_the_manifest_being_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    artifact_hashes = importlib.import_module("evaluate_competition_generators").artifact_hashes

    (tmp_path / "predictions.csv").write_text("value\n1\n")
    manifest = tmp_path / "manifest.json"
    first = artifact_hashes(tmp_path, manifest)
    manifest.write_text("old manifest")
    (tmp_path / "stage_manifest.json").write_text("old enclosing stage record")
    assert artifact_hashes(tmp_path, manifest) == first
    assert set(first) == {"predictions.csv"}


def test_generator_registration_checks_seed_and_count() -> None:
    expected = dict(seed=42, count=2, min_length=10, max_length=32, mode="base")
    observed = {**expected, "seed": 99}
    with pytest.raises(ValueError, match="protocol"):
        validate_screen(["AK" * 5] * 2, observed, expected)
    with pytest.raises(ValueError, match="count"):
        validate_screen(["AK" * 5], expected, expected)


def test_raw_and_filtered_denominators_stay_separate() -> None:
    p = dict(seed=42, count=4, min_length=10, max_length=32, mode="base")
    result = validate_screen(["AK" * 8, "AK" * 8, "A" * 33, "AX" * 5], p, p)
    assert result == dict(raw=4, valid=2, unique=1, common_valid=2, common_unique=1)


def test_reward_training_tokenizes_entire_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    module = importlib.import_module("optimize_competition_evodiff")

    class Tokenizer:
        pad_id = 0

        def tokenize(self, batch: list[str]) -> list[int]:
            return [ord(c) for c in batch[0]]

    result = module.encode_batch(Tokenizer(), ["ACD", "EF"], "cpu")
    assert result.tolist() == [[65, 67, 68], [69, 70, 0]]


def test_pareto_screen_keeps_tradeoffs_but_does_not_reward_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    from pathlib import Path

    import numpy as np

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    nondominated = importlib.import_module("evaluate_competition_generators").nondominated

    values = np.array([[0.8, 3.0], [0.7, 2.0], [0.8, 2.0], [np.nan, 9.0], [0.8, 3.0]])
    assert nondominated(values).tolist() == [True, False, False, False, True]


def test_membrane_placement_preserves_structure_and_starts_above_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    from pathlib import Path

    import numpy as np

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    translate = importlib.import_module("diagnose_competition_membrane").surface_coordinates
    original = np.array([[1.0, 2.0, -1.0], [2.0, 4.0, 1.0], [3.0, 0.0, 0.0]])
    moved = translate(original)
    np.testing.assert_allclose(moved[:, :2].mean(0), 0, atol=1e-12)
    assert moved[:, 2].min() == 2.0
    np.testing.assert_allclose(moved[1:] - moved[0], original[1:] - original[0])


def test_parallel_safety_batches_preserve_sequence_mapping_and_reject_wrong_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import importlib
    from pathlib import Path
    from types import SimpleNamespace

    import pandas as pd

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    module = importlib.import_module("evaluate_competition_generators")
    calls = []

    def fake(_oracle: Path, sequences: dict[str, str]) -> dict[str, SimpleNamespace]:
        calls.append(sequences)
        return {key: SimpleNamespace(hc50_u_m=float(len(seq))) for key, seq in sequences.items()}

    monkeypatch.setattr(module, "run_hemopi2", fake)
    output = tmp_path
    sequences = ["A" * 10, "K" * 11, "L" * 12]
    result = module.hemopi_batches(sequences, output, batch_size=2)
    assert result == {s: float(len(s)) for s in sequences}
    assert len(calls) == 2
    pd.DataFrame(dict(sequence=["G" * 10, "K" * 11], hc50=[10.0, 11.0])).to_csv(
        output / "batch00000.csv", index=False
    )
    with pytest.raises(ValueError, match="batch mismatch"):
        module.hemopi_batches(sequences, output, batch_size=2)


def test_folding_stem_preserves_hidden_states_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    import torch

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    module = importlib.import_module("diagnose_competition_structure")

    class Stem(torch.nn.Module):
        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_hidden_states: bool
        ) -> dict[str, tuple[torch.Tensor, ...]]:
            assert output_hidden_states
            return {"hidden_states": (input_ids * attention_mask, input_ids + 1)}

    inputs = torch.tensor([[1, 2]])
    actual = module.FoldingStem(Stem(), "cpu")(inputs, torch.ones_like(inputs), True)
    assert all(t.device.type == "cpu" for t in actual["hidden_states"])
    assert torch.equal(actual["hidden_states"][0], inputs)
    assert torch.equal(actual["hidden_states"][1], inputs + 1)


def test_optimization_comparison_rejects_device_or_seed_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    validate_optimization_pair = importlib.import_module(
        "evaluate_competition_generators"
    ).validate_optimization_pair

    control = dict(
        seed=42, device="cuda", requested=1000, min_length=15, max_length=35, batch_size=32
    )
    validate_optimization_pair(control, dict(control))
    with pytest.raises(ValueError, match="device"):
        validate_optimization_pair(control, {**control, "device": "cpu"})
    with pytest.raises(ValueError, match="seed"):
        validate_optimization_pair(control, {**control, "seed": 43})
    with pytest.raises(ValueError, match="batch_size"):
        validate_optimization_pair(control, {**control, "batch_size": 16})


def test_deepamp_common_conditioning_is_order_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    select = importlib.import_module("run_competition_deepamp").conditioning_inputs
    inputs = ["A" * 10, "K" * 15, "L" * 20, "R" * 25, "W" * 30]
    selected = select(inputs, 3, "common")
    assert set(selected) == {"K" * 15, "L" * 20, "R" * 25}
    assert selected == select(list(reversed(inputs)), 3, "common")
    with pytest.raises(ValueError, match="Insufficient"):
        select(inputs, 4, "common")


def test_deepamp_completion_preserves_padding_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    import torch

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    complete = importlib.import_module("run_competition_deepamp").complete_masked_positions
    tokens = torch.tensor([[1, 5, 5, 3, 2, 0], [1, 5, 5, 5, 2, 0]])
    positions = torch.tensor([[1, 2, 0], [1, 2, 3]])
    predictions = torch.tensor([[6, 7, 8], [6, 7, 8]])
    actual = complete(tokens, positions, predictions)
    assert actual.tolist() == [[1, 6, 7, 3, 2, 0], [1, 6, 7, 8, 2, 0]]
    assert tokens[0, 1].item() == 5


def test_membrane_lipid_selection_uses_openmm_template_residue_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    module = importlib.import_module("diagnose_competition_membrane")
    atoms = [
        SimpleNamespace(
            index=i, residue=SimpleNamespace(name=name), element=SimpleNamespace(symbol=symbol)
        )
        for i, (name, symbol) in enumerate([("ALA", "C"), ("POP", "P"), ("POP", "H"), ("HOH", "O")])
    ]
    assert module.lipid_heavy_atoms(atoms) == [1]
    with pytest.raises(ValueError, match="lipid"):
        module.lipid_heavy_atoms(atoms[:1])
