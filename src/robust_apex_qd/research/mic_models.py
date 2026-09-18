"""Point and censored-distribution MIC regression with fold-local preprocessing."""

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from robust_apex_qd.research.competition_models import STRAIN_SPECIES
from robust_apex_qd.research.mic_delta import DeltaPair, delta_huber_loss


class MICRegressor(nn.Module):
    def __init__(self, dimension: int, width: int, heads: int, scale_floor: float) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(dimension, width), nn.ReLU())
        self.output = nn.Linear(width, 2 * heads)
        self.heads = heads
        self.scale_floor = scale_floor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.output(self.hidden(x))
        means = values[:, : self.heads]
        if self.heads == 18:
            means = torch.cat((means[:, :7], means[:, 7:] + means[:, STRAIN_SPECIES]), dim=1)
        return means, nn.functional.softplus(values[:, self.heads :]) + self.scale_floor


def censored_normal_nll(
    mean: torch.Tensor, scale: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    if torch.any(lower > upper) or torch.any(torch.isnan(lower) | torch.isnan(upper)):
        raise ValueError("Invalid observation bounds")
    if torch.any(~torch.isfinite(mean) | ~torch.isfinite(scale) | (scale <= 0)):
        raise ValueError("Finite mean and positive scale required")
    if torch.any(torch.isneginf(lower) & torch.isposinf(upper)):
        raise ValueError("Observation requires at least one bound")
    # Slice before arithmetic so unused infinite boundaries never enter autograd.
    result = torch.empty_like(mean)
    exact = lower == upper
    result[exact] = (
        0.5 * ((lower[exact] - mean[exact]) / scale[exact]).square()
        + scale[exact].log()
        + 0.5 * math.log(2 * math.pi)
    )
    left = torch.isneginf(lower)
    right = torch.isposinf(upper)
    result[left] = -torch.special.log_ndtr((upper[left] - mean[left]) / scale[left])
    result[right] = -torch.special.log_ndtr((mean[right] - lower[right]) / scale[right])
    interval = ~(exact | left | right)
    lo = (lower[interval] - mean[interval]) / scale[interval]
    hi = (upper[interval] - mean[interval]) / scale[interval]
    # Subtract survival probabilities in the right tail, CDFs elsewhere.
    flip = lo > 0
    a = torch.special.log_ndtr(torch.where(flip, -lo, hi))
    b = torch.special.log_ndtr(torch.where(flip, -hi, lo))
    result[interval] = -(a + torch.log(-torch.expm1(b - a)))
    return result


def mic_metrics(rows: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(prediction)
    exact = finite & rows.exact_regression.to_numpy(bool)
    truth = np.log2(rows.loc[exact, "mic_um"].to_numpy(float))
    p = prediction[exact]
    error = p - truth
    correlated = len(p) >= 2 and np.std(truth) > 0 and np.std(p) > 0
    return dict(
        rows=len(rows),
        predicted_rows=int(finite.sum()),
        exact_rows=int(exact.sum()),
        mae=float(np.abs(error).mean()) if len(error) else None,
        rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
        within1=float((np.abs(error) <= 1).mean()) if len(error) else None,
        within2=float((np.abs(error) <= 2).mean()) if len(error) else None,
        pcc=float(np.corrcoef(p, truth)[0, 1]) if correlated else None,
        spearman=float(pd.Series(p).rank().corr(pd.Series(truth).rank())) if correlated else None,
    )


def fit_regressor(
    features: np.ndarray,
    heads: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: dict[str, Any],
    seed: int,
    *,
    pairs: list[DeltaPair] | None = None,
    observation_ids: list[str] | None = None,
    delta_weight: float = 0.0,
) -> dict[str, Any]:
    if not math.isfinite(delta_weight) or delta_weight < 0:
        raise ValueError("Delta weight must be finite and nonnegative")
    pairs = pairs or []
    pair_indices: list[int] = []
    pair_ids: list[str] = []
    if pairs:
        if observation_ids is None or len(observation_ids) != len(features):
            raise ValueError("Pair training requires aligned observation IDs")
        delta_huber_loss(torch.zeros(len(features)), observation_ids, pairs)
        index = {identifier: i for i, identifier in enumerate(observation_ids)}
        pair_ids = sorted({r.observation_id for p in pairs for r in (p.left, p.right)})
        pair_indices = [index[identifier] for identifier in pair_ids]
        for p in pairs:
            for r in (p.left, p.right):
                i = index[r.observation_id]
                if lower[i] != upper[i] or not np.isclose(lower[i], math.log2(r.mic_um)):
                    raise ValueError("Pair endpoint does not match exact training label")
            if heads[index[p.left.observation_id]] != heads[index[p.right.observation_id]]:
                raise ValueError("Pair endpoints require the same target head")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaler = StandardScaler().fit(features)
    device = settings["device"]
    x = torch.tensor(scaler.transform(features), dtype=torch.float32, device=device)
    exact = lower == upper
    center = float(np.median(lower[exact])) if exact.any() else 4.0
    lo = torch.tensor(lower - center, dtype=torch.float64, device=device)
    hi = torch.tensor(upper - center, dtype=torch.float64, device=device)
    target = torch.tensor(heads, dtype=torch.long, device=device)
    net = MICRegressor(
        features.shape[1], settings["width"], settings["heads"], settings["scale_floor"]
    )
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=settings["learning_rate"], weight_decay=0.01)
    history = []
    for _epoch in range(settings["epochs"]):
        total = 0.0
        for batch in np.array_split(
            rng.permutation(len(x)), max(1, math.ceil(len(x) / settings["batch_size"]))
        ):
            means, scales = net(x[batch])
            b = torch.arange(len(batch), device=device)
            mean, scale = means[b, target[batch]].double(), scales[b, target[batch]].double()
            if settings["loss"] == "normal":
                loss = censored_normal_nll(mean, scale, lo[batch], hi[batch]).mean()
            else:
                loss = (
                    torch.relu(lo[batch] - mean).square() + torch.relu(mean - hi[batch]).square()
                ).mean()
            if pairs and delta_weight:
                # Every optimizer step sees all supplied pairs. Scaffold weights therefore
                # do not depend on the composition or size of the absolute-loss minibatch.
                pair_mean, _ = net(x[pair_indices])
                selected = pair_mean[
                    torch.arange(len(pair_indices), device=device), target[pair_indices]
                ]
                loss = loss + delta_weight * delta_huber_loss(selected, pair_ids, pairs)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite MIC training loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        history.append(total / len(x))
    return dict(
        state={k: v.detach().cpu() for k, v in net.state_dict().items()},
        feature_mean=torch.tensor(scaler.mean_),
        feature_scale=torch.tensor(scaler.scale_),
        center=center,
        settings=settings,
        dimension=features.shape[1],
        loss_curve=history,
        supported_heads=sorted(
            set(heads.tolist()) | {int(STRAIN_SPECIES[h - 7]) for h in heads if h >= 7}
        )
        if settings["heads"] == 18
        else sorted(set(heads.tolist())),
        seed=seed,
    )


def predict_regressor(
    bundle: dict[str, Any], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    settings = bundle["settings"]
    net = MICRegressor(
        bundle["dimension"], settings["width"], settings["heads"], settings["scale_floor"]
    )
    net.load_state_dict(bundle["state"])
    net.eval()
    x = (features - bundle["feature_mean"].numpy()) / bundle["feature_scale"].numpy()
    with torch.no_grad():
        mean, scale = net(torch.tensor(x, dtype=torch.float32))
    means, scales = mean.numpy() + bundle["center"], scale.numpy()
    unsupported = sorted(set(range(settings["heads"])) - set(bundle["supported_heads"]))
    means[:, unsupported] = np.nan
    scales[:, unsupported] = np.nan
    if settings["loss"] != "normal":
        scales[:] = np.nan
    return means, scales
