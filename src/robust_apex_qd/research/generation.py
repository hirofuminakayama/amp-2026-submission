"""Research-only DDIM sampling with the checkpoint's original noise schedule."""

from itertools import pairwise
from typing import Any

import torch


def timestep_pairs(training_steps: int, sampling_steps: int) -> list[tuple[int, int]]:
    if not 1 <= sampling_steps <= training_steps:
        raise ValueError("Sampling steps must lie within the training schedule")
    times = torch.linspace(-1, training_steps - 1, sampling_steps + 1).int().tolist()[::-1]
    return list(pairwise(times))


@torch.no_grad()
def ddim_sample(model: Any, batch_size: int, design_len: int, steps: int) -> torch.Tensor:
    device = model.betas.device
    x = torch.randn((batch_size, model.seq_length, model.embed_dim), device=device)
    start = None
    for time, following in timestep_pairs(model.num_timesteps, steps):
        times = torch.full((batch_size,), time, device=device, dtype=torch.long)
        noise, start = model.model_predictions(
            x,
            times,
            design_len=design_len,
            x_self_cond=start if model.self_condition else None,
            clip_x_start=False,
        )
        if following < 0:
            x = start
        else:
            alpha_next = model.alphas_cumprod[following]
            x = start * alpha_next.sqrt() + noise * (1 - alpha_next).sqrt()
    return model.unnormalize(x)
