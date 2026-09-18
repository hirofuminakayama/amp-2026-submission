import pytest
import torch

from robust_apex_qd.research.generation import ddim_sample, timestep_pairs


def test_timestep_pairs_keep_training_schedule() -> None:
    pairs = timestep_pairs(1000, 250)
    assert len(pairs) == 250
    assert pairs[0][0] == 999
    assert pairs[-1][1] == -1
    assert all(a > b for a, b in pairs)
    with pytest.raises(ValueError):
        timestep_pairs(1000, 1001)


def test_ddim_passes_length_without_clipping_or_self_condition_mixup() -> None:
    class Model:
        num_timesteps = 1000
        seq_length = 6
        embed_dim = 2
        self_condition = False
        betas = torch.ones(1000)
        alphas_cumprod = torch.linspace(0.999, 0.001, 1000)

        def __init__(self) -> None:
            self.calls: list[int] = []

        def model_predictions(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            *,
            design_len: int,
            x_self_cond: torch.Tensor | None,
            clip_x_start: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert design_len == 4
            assert x_self_cond is None
            assert clip_x_start is False
            self.calls.append(int(t[0]))
            return torch.zeros_like(x), torch.full_like(x, 2.0)

        def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model = Model()
    result = ddim_sample(model, 1, 4, 250)
    assert result.shape == (1, 6, 2)
    assert len(model.calls) == 250
    assert torch.all(result == 2)


def test_frozen_ridge_inference_preserves_feature_and_strain_layout() -> None:
    import numpy as np

    from robust_apex_qd.research.generation_evaluation import frozen_ridge_predict

    # One feature: shared coefficient 2, strain-1 intercept 3, interaction 4.
    coef = np.zeros(23)
    coef[0], coef[2], coef[13] = 2, 3, 4
    state = {"coef": coef, "intercept": np.array(5), "mean": np.array([1]), "scale": np.array([2])}
    result = frozen_ridge_predict(np.array([[3.0], [5.0]]), 1, state)
    np.testing.assert_allclose(result, [14, 20])


def test_ddim_eta_zero_update_keeps_schedule_buffers() -> None:
    class Model:
        num_timesteps = 4
        seq_length = 1
        embed_dim = 1
        self_condition = True
        betas = torch.tensor([0.1, 0.2, 0.3, 0.4])
        alphas_cumprod = torch.tensor([0.9, 0.6, 0.3, 0.1])

        def __init__(self) -> None:
            self.initial = torch.empty(0)
            self.calls = 0

        def model_predictions(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            *,
            design_len: int,
            x_self_cond: torch.Tensor | None,
            clip_x_start: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if self.calls == 0:
                self.initial = x.clone()
                assert x_self_cond is None
                assert int(t[0]) == 3
            else:
                assert x_self_cond is not None
                torch.testing.assert_close(x_self_cond, self.initial / 2)
                assert int(t[0]) == 1
            self.calls += 1
            return x / 3, x / 2

        def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model = Model()
    before = model.alphas_cumprod.clone()
    result = ddim_sample(model, 1, 1, 2)
    expected = model.initial * (before[1].sqrt() / 2 + (1 - before[1]).sqrt() / 3) / 2
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(model.alphas_cumprod, before)
