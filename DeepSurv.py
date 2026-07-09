import numpy as np
import torch
import torch.nn as nn


class DeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.7, batch_norm=True):
        super().__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def weighted_cox_loss(scores, times, events, weights=None):
    order = torch.argsort(times)
    scores = scores[order]
    events = events[order].float()

    if weights is None:
        weights = torch.ones_like(events)
    else:
        weights = weights[order].float()

    score_shift = scores.max().detach()
    weighted_exp = weights * torch.exp(scores - score_shift)

    risk_sum = torch.flip(
        torch.cumsum(
            torch.flip(weighted_exp, dims=[0]),
            dim=0,
        ),
        dims=[0],
    )

    log_risk_sum = torch.log(risk_sum + 1e-12) + score_shift
    event_weights = events * weights

    return -(
        (scores - log_risk_sum) * event_weights
    ).sum() / (event_weights.sum() + 1e-8)


def estimate_weighted_baseline_survival(times, events, scores, weights=None):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if weights is None:
        weights = np.ones(len(times), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(times)
        & np.isfinite(events)
        & np.isfinite(scores)
        & np.isfinite(weights)
        & (times > 0)
        & (weights > 0)
    )

    times = times[valid]
    events = events[valid]
    scores = scores[valid]
    weights = weights[valid]

    event_times = np.sort(np.unique(times[events == 1]))

    if len(event_times) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    exp_scores = np.exp(np.clip(scores, -50, 50))
    cumulative_hazard = []
    hazard = 0.0

    for time_point in event_times:
        event_mask = (times == time_point) & (events == 1)
        risk_mask = times >= time_point

        weighted_events = weights[event_mask].sum()
        weighted_risk = (
            weights[risk_mask] * exp_scores[risk_mask]
        ).sum()

        if weighted_risk <= 0 or not np.isfinite(weighted_risk):
            raise ValueError("Invalid weighted risk-set denominator.")

        hazard += weighted_events / weighted_risk
        cumulative_hazard.append(hazard)

    baseline_survival = np.exp(-np.asarray(cumulative_hazard, dtype=float))
    baseline_survival = np.minimum.accumulate(baseline_survival)

    return event_times, np.clip(baseline_survival, 0.0, 1.0)


def baseline_survival_at_times(event_times, baseline_survival, time_points):
    event_times = np.asarray(event_times, dtype=float)
    baseline_survival = np.asarray(baseline_survival, dtype=float)
    time_points = np.asarray(time_points, dtype=float)

    output = np.ones(len(time_points), dtype=float)

    if len(event_times) == 0:
        return output

    for i, time_point in enumerate(time_points):
        idx = np.searchsorted(event_times, time_point, side="right") - 1
        if idx >= 0:
            output[i] = baseline_survival[idx]

    return np.clip(output, 0.0, 1.0)


def predict_survival(scores, event_times, baseline_survival, time_points):
    scores = np.asarray(scores, dtype=float)

    baseline_at_times = baseline_survival_at_times(
        event_times,
        baseline_survival,
        time_points,
    )

    relative_risk = np.exp(np.clip(scores, -50, 50))

    survival = np.power(
        baseline_at_times.reshape(1, -1),
        relative_risk.reshape(-1, 1),
    )

    if np.any(~np.isfinite(survival)):
        raise ValueError("Predicted survival contains non-finite values.")

    return np.clip(survival, 0.0, 1.0)
