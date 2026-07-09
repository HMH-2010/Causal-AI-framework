import random
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from config import (
    HIDDEN_DIMS,
    DROPOUT,
    BATCH_NORM,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    N_FOLDS,
    RANDOM_STATE,
)
from model import (
    DeepSurv,
    weighted_cox_loss,
    estimate_weighted_baseline_survival,
    predict_survival,
)


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    X,
    times,
    events,
    weights,
    input_dim,
    device,
    epochs=EPOCHS,
):
    model = DeepSurv(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        batch_norm=BATCH_NORM,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    time_tensor = torch.as_tensor(times, dtype=torch.float32, device=device)
    event_tensor = torch.as_tensor(events, dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        scores = model(X_tensor)

        loss = weighted_cox_loss(
            scores,
            time_tensor,
            event_tensor,
            weight_tensor,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model


def predict_scores(model, X, device):
    model.eval()

    with torch.no_grad():
        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=device,
        )
        return model(X_tensor).cpu().numpy()


def cross_fit(
    X,
    times,
    events,
    weights,
    time_points,
    device,
):
    n_samples = len(X)

    oof_scores = np.full(n_samples, np.nan, dtype=float)
    oof_survival = np.full(
        (n_samples, len(time_points)),
        np.nan,
        dtype=float,
    )
    fold_ids = np.full(n_samples, -1, dtype=int)

    folds = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold_id, (train_idx, val_idx) in enumerate(
        folds.split(X, events),
        start=1,
    ):
        model = train_model(
            X=X[train_idx],
            times=times[train_idx],
            events=events[train_idx],
            weights=weights[train_idx],
            input_dim=X.shape[1],
            device=device,
        )

        train_scores = predict_scores(
            model,
            X[train_idx],
            device,
        )

        val_scores = predict_scores(
            model,
            X[val_idx],
            device,
        )

        event_times, baseline_survival = (
            estimate_weighted_baseline_survival(
                times[train_idx],
                events[train_idx],
                train_scores,
                weights[train_idx],
            )
        )

        val_survival = predict_survival(
            val_scores,
            event_times,
            baseline_survival,
            time_points,
        )

        oof_scores[val_idx] = val_scores
        oof_survival[val_idx] = val_survival
        fold_ids[val_idx] = fold_id

    if np.any(~np.isfinite(oof_scores)):
        raise ValueError("Incomplete OOF risk predictions.")

    if np.any(~np.isfinite(oof_survival)):
        raise ValueError("Incomplete OOF survival predictions.")

    return oof_scores, oof_survival, fold_ids
