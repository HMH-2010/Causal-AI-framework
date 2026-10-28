"""DeepSurv, weighted partial likelihood, and absolute survival prediction."""

import numpy as np
import torch
from scipy.special import logsumexp
from torch import nn


class DeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.7, batch_norm=True):
        super().__init__()
        layers = []
        for width in hidden_dims:
            layers.append(nn.Linear(input_dim, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            input_dim = width
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def weighted_cox_loss(scores, times, events, weights):
    """Weight event terms and risk sets; use Breslow handling of tied times."""
    order = torch.argsort(times)
    scores, times = scores[order], times[order]
    event_weights = events[order] * weights[order]
    log_risk = torch.logcumsumexp(
        (scores + torch.log(weights[order])).flip(0), dim=0
    ).flip(0)
    _, counts = torch.unique_consecutive(times, return_counts=True)
    starts = counts.cumsum(0) - counts
    log_risk = torch.repeat_interleave(log_risk[starts], counts)
    return -(event_weights * (scores - log_risk)).sum() / event_weights.sum().clamp_min(1e-8)


def fit_baseline(times, events, scores, weights):
    """Weighted Breslow cumulative hazard, stored in log space."""
    event_times = np.unique(times[events == 1])
    increments = []
    for t in event_times:
        at_event = (times == t) & (events == 1)
        at_risk = times >= t
        increments.append(
            np.log(weights[at_event].sum())
            - logsumexp(scores[at_risk] + np.log(weights[at_risk]))
        )
    return event_times, np.logaddexp.accumulate(increments)


def survival_at_times(scores, event_times, log_cumulative_hazard, time_points):
    index = np.searchsorted(event_times, time_points, side="right") - 1
    log_hazard = np.full(len(time_points), -np.inf)
    observed = index >= 0
    log_hazard[observed] = log_cumulative_hazard[index[observed]]
    with np.errstate(over="ignore"):
        return np.exp(-np.exp(scores[:, None] + log_hazard[None, :]))
