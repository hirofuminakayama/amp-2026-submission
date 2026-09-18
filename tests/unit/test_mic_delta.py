from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from robust_apex_qd.research.mic_delta import (
    DeltaObservation,
    DeltaPair,
    build_delta_pairs,
    delta_huber_loss,
)


def observation(name: str, sequence: str, mic: float, scaffold: str = "g") -> DeltaObservation:
    return DeltaObservation(
        observation_id=name,
        sequence=sequence,
        scaffold_id=scaffold,
        target_id="strain:fixture",
        chemical_profile="linear-L-free",
        study_id="fixture-study",
        comparable_assay_id="fixture-assay",
        verification_evidence="synthetic test evidence",
        mic_um=mic,
        objective="measured_mic",
        relation="=",
        assay_publication_verified=True,
    )


def test_pairs_keep_direction_and_serialize(tmp_path: Path) -> None:
    rows = [observation("a", "AAAA", 16), observation("b", "AAAK", 4)]
    partition = {"AAAA": "train", "AAAK": "train"}
    pair = build_delta_pairs(rows, [("a", "b")], partition, "train")[0]
    reverse = build_delta_pairs(rows, [("b", "a")], partition, "train")[0]
    assert pair.delta_log2_um == -2
    assert reverse.delta_log2_um == 2
    path = tmp_path / "pairs.jsonl"
    path.write_text(pair.model_dump_json() + "\n")
    assert DeltaPair.model_validate_json(path.read_text()) == pair


@pytest.mark.parametrize(
    "field,value",
    [
        ("mic_um", 0),
        ("mic_um", float("nan")),
        ("mic_um", float("inf")),
        ("relation", ">"),
        ("objective", "qmap_consensus"),
        ("assay_publication_verified", False),
        ("verification_evidence", " "),
    ],
)
def test_only_verified_exact_measurements_are_accepted(field: str, value: object) -> None:
    raw = observation("a", "AAAA", 4).model_dump()
    raw[field] = value
    with pytest.raises(ValidationError):
        DeltaObservation.model_validate(raw)


@pytest.mark.parametrize(
    "field", ["target_id", "chemical_profile", "study_id", "comparable_assay_id", "scaffold_id"]
)
def test_incomparable_pairs_are_rejected(field: str) -> None:
    a = observation("a", "AAAA", 4)
    raw = observation("b", "AAAK", 8).model_dump()
    raw[field] = "different"
    b = DeltaObservation.model_validate(raw)
    with pytest.raises(ValueError, match="comparable"):
        build_delta_pairs([a, b], [("a", "b")], {"AAAA": "train", "AAAK": "train"}, "train")


def test_split_and_duplicate_guards() -> None:
    rows = [observation("a", "AAAA", 4), observation("b", "AAAK", 8)]
    with pytest.raises(ValueError, match="partition"):
        build_delta_pairs(rows, [("a", "b")], {"AAAA": "train", "AAAK": "valid"}, "train")
    with pytest.raises(ValueError, match="partition"):
        build_delta_pairs(rows, [("a", "b")], {"AAAA": "valid", "AAAK": "valid"}, "train")
    with pytest.raises(ValueError, match="partition"):
        build_delta_pairs(rows, [], {"AAAA": "train"}, "train")
    with pytest.raises(ValueError, match="Duplicate"):
        build_delta_pairs(rows + rows[:1], [], {"AAAA": "train", "AAAK": "train"}, "train")
    with pytest.raises(ValueError, match="Duplicate"):
        build_delta_pairs(
            rows, [("a", "b"), ("b", "a")], {"AAAA": "train", "AAAK": "train"}, "train"
        )
    with pytest.raises(ValueError, match="Unknown"):
        build_delta_pairs(rows, [("a", "missing")], {"AAAA": "train", "AAAK": "train"}, "train")


def test_scaffold_leak_is_rejected_even_for_unused_endpoint() -> None:
    rows = [observation("a", "AAAA", 4), observation("b", "AAAK", 8), observation("c", "AAAR", 16)]
    with pytest.raises(ValueError, match="partition"):
        build_delta_pairs(
            rows, [("a", "b")], {"AAAA": "train", "AAAK": "train", "AAAR": "valid"}, "train"
        )


def test_same_sequence_replicates_are_not_mutation_pairs() -> None:
    rows = [observation("a", "AAAA", 4), observation("b", "AAAA", 8)]
    with pytest.raises(ValueError, match="same sequence"):
        build_delta_pairs(rows, [("a", "b")], {"AAAA": "train"}, "train")


def test_loss_weights_scaffolds_equally_and_aligns_ids() -> None:
    rows = [
        observation("a", "AAAA", 1),
        observation("b", "AAAK", 1),
        observation("c", "AAAR", 1),
        observation("d", "KKKK", 1, "h"),
        observation("e", "KKKA", 1, "h"),
    ]
    pairs = build_delta_pairs(
        rows, [("a", "b"), ("a", "c"), ("d", "e")], {r.sequence: "train" for r in rows}, "train"
    )
    pred = torch.tensor([0.0, 1.0, 1.0, 0.0, 3.0], requires_grad=True)
    loss = delta_huber_loss(pred, [r.observation_id for r in rows], pairs)
    assert loss.item() == pytest.approx(1.5)  # mean(mean(.5, .5), 2.5)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert delta_huber_loss(pred.flip(0), [r.observation_id for r in rows][::-1], pairs) == loss
    assert delta_huber_loss(pred + 100, [r.observation_id for r in rows], pairs) == loss
    with pytest.raises(ValueError, match="prediction"):
        delta_huber_loss(pred, ["a"] * 5, pairs)
    with pytest.raises(ValueError, match="prediction"):
        delta_huber_loss(pred[:1], ["a"], pairs)


def test_empty_loss_and_identical_predictions_have_zero_gradient() -> None:
    pred = torch.tensor([3.0, 3.0], requires_grad=True)
    empty = delta_huber_loss(pred, ["a", "b"], [])
    empty.backward()
    assert empty.item() == 0
    assert pred.grad is not None and pred.grad.tolist() == [0.0, 0.0]
    rows = [observation("a", "AAAA", 1), observation("b", "AAAK", 1)]
    pairs = build_delta_pairs(rows, [("a", "b")], {r.sequence: "train" for r in rows}, "train")
    assert delta_huber_loss(pred, ["a", "b"], pairs).item() == 0


def test_auxiliary_loss_updates_tiny_model_and_round_trips(tmp_path: Path) -> None:
    rows = [observation("a", "AAAA", 16), observation("b", "AAAK", 4)]
    pairs = build_delta_pairs(rows, [("a", "b")], {r.sequence: "train" for r in rows}, "train")
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    inputs = torch.tensor([[0.0], [1.0]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = delta_huber_loss(model(inputs).flatten(), ["a", "b"], pairs)
    before.backward()
    optimizer.step()
    after = delta_huber_loss(model(inputs).flatten(), ["a", "b"], pairs)
    assert after.item() < before.item()
    checkpoint = tmp_path / "weights.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = torch.nn.Linear(1, 1, bias=False)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(model(inputs), reloaded(inputs))


def test_loss_rejects_mixed_partitions_and_conflicting_measurements() -> None:
    rows = [observation("a", "AAAA", 4), observation("b", "AAAK", 8), observation("c", "AAAR", 16)]
    pairs = build_delta_pairs(
        rows, [("a", "b"), ("a", "c")], {r.sequence: "train" for r in rows}, "train"
    )
    raw = pairs[1].model_dump()
    raw["partition"] = "valid"
    with pytest.raises(ValueError, match="partition"):
        delta_huber_loss(torch.zeros(3), ["a", "b", "c"], [pairs[0], DeltaPair.model_validate(raw)])
    raw = pairs[1].model_dump()
    raw["left"]["mic_um"] = 32
    with pytest.raises(ValueError, match="Conflicting"):
        delta_huber_loss(torch.zeros(3), ["a", "b", "c"], [pairs[0], DeltaPair.model_validate(raw)])
