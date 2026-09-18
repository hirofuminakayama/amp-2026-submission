"""Small masked MIC/HC50 heads with training-only endpoint and feature scaling."""

from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


class JointEndpointHead(nn.Module):
    def __init__(self, dimension: int, shared: bool) -> None:
        super().__init__()
        self.shared = shared
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dimension, 32), nn.ReLU(), nn.Linear(32, 8 if shared else 1)
                )
                for _ in range(1 if shared else 8)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.shared:
            return self.heads[0](features)
        return torch.cat([head(features) for head in self.heads], dim=1)


def fit_joint_head(
    features: np.ndarray,
    sequence_rows: np.ndarray,
    heads: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    *,
    shared: bool,
    seed: int,
    epochs: int = 80,
) -> dict[str, Any]:
    if set(groups[training]) & set(groups[validation]):
        raise ValueError("Endpoint group crossing")
    if not len(training) or epochs < 1 or not np.isfinite(features).all():
        raise ValueError("Finite features and nonempty training required")
    mask = np.isin(sequence_rows, training)
    seq, target, lo, hi = sequence_rows[mask], heads[mask], lower[mask], upper[mask]
    if np.any(lo > hi) or np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError("Valid observed endpoint bounds required")
    if np.any(~np.isfinite(lo) & ~np.isfinite(hi)):
        raise ValueError("Missing endpoints must be absent, not fabricated labels")
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(features[training])
    x = torch.tensor(scaler.transform(features[training]), dtype=torch.float32)
    lookup = {int(i): j for j, i in enumerate(training)}
    seq_tensor = torch.tensor([lookup[int(s)] for s in seq], dtype=torch.long)
    target_tensor = torch.tensor(target, dtype=torch.long)
    center, scale, weights = np.zeros(8), np.ones(8), np.zeros(len(seq))
    support = np.zeros(8, dtype=bool)
    for head in range(8):
        selected = target == head
        if not selected.any():
            continue
        support[head] = True
        exact = selected & (lo == hi)
        observed = (
            lo[exact]
            if exact.any()
            else np.where(np.isfinite(lo[selected]), lo[selected], hi[selected])
        )
        center[head] = np.median(observed)
        scale[head] = max(float(np.std(observed)), 1.0)
        _, inverse, counts = np.unique(
            groups[seq[selected]], return_inverse=True, return_counts=True
        )
        w = 1.0 / counts[inverse]
        weights[selected] = w / w.sum()
    # MIC species together and HC50 receive equal endpoint mass.
    mic_heads = int(support[:7].sum())
    weights[target < 7] /= max(mic_heads, 1)
    weights /= weights.sum()
    low = torch.tensor((lo - center[target]) / scale[target], dtype=torch.float32)
    high = torch.tensor((hi - center[target]) / scale[target], dtype=torch.float32)
    weight = torch.tensor(weights, dtype=torch.float32)
    net = JointEndpointHead(features.shape[1], shared)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.01)
    losses = []
    for _ in range(epochs):
        values = net(x)[seq_tensor, target_tensor]
        loss = (
            weight * (torch.relu(low - values).square() + torch.relu(values - high).square())
        ).sum()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite joint endpoint loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return dict(
        schema_version=2,
        state=net.state_dict(),
        dimension=features.shape[1],
        shared=shared,
        mean=scaler.mean_,
        scale=scaler.scale_,
        target_center=center,
        target_scale=scale,
        supported=support,
        seed=seed,
        epochs=epochs,
        loss_curve=losses,
        parameter_count=sum(p.numel() for p in net.parameters()),
        training_groups=sorted(set(groups[training])),
    )


def predict_joint_head(bundle: dict[str, Any], features: np.ndarray) -> np.ndarray:
    if (
        features.ndim != 2
        or features.shape[1] != bundle["dimension"]
        or not np.isfinite(features).all()
    ):
        raise ValueError("Finite matching endpoint features required")
    net = JointEndpointHead(bundle["dimension"], bundle["shared"])
    net.load_state_dict(bundle["state"])
    net.eval()
    with torch.no_grad():
        result = net(
            torch.tensor((features - bundle["mean"]) / bundle["scale"], dtype=torch.float32)
        ).numpy()
    result = result * bundle["target_scale"] + bundle["target_center"]
    result[:, ~bundle["supported"]] = np.nan
    return result
