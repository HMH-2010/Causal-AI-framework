"""Fold-local preprocessing and nested training of two treatment-specific models."""

import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.model_selection import ParameterSampler, StratifiedKFold

import config as cfg
from model import DeepSurv, fit_baseline, weighted_cox_loss
from predict import predict_pair, predict_scores, transform


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cindex(times, events, scores):
    """Unweighted concordance for strictly ordered follow-up times (inner tuning)."""
    concordant = comparable = 0.0
    for i in np.flatnonzero(events):
        later = scores[times > times[i]]
        comparable += len(later)
        concordant += np.sum(scores[i] > later) + 0.5 * np.sum(scores[i] == later)
    if comparable == 0:
        raise ValueError("An inner validation fold has no comparable survival pairs.")
    return concordant / comparable


def cox_score_pvalues(x, times, events, weights):
    """Weighted univariate score tests at beta=0, with tied risk sets grouped."""
    weights = weights / weights.mean()
    order = np.argsort(-times, kind="stable")
    x, times, events, weights = x[order], times[order], events[order], weights[order]
    risk_w = 0.0
    risk_x = np.zeros(x.shape[1])
    risk_x2 = np.zeros(x.shape[1])
    score = np.zeros(x.shape[1])
    information = np.zeros(x.shape[1])
    boundaries = np.r_[0, np.flatnonzero(np.diff(times)) + 1, len(times)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        gx, gw = x[start:end], weights[start:end]
        risk_w += gw.sum()
        risk_x += (gx * gw[:, None]).sum(axis=0)
        risk_x2 += (gx ** 2 * gw[:, None]).sum(axis=0)
        event_mask = events[start:end] == 1
        ew = gw[event_mask]
        mean = risk_x / risk_w
        score += (gx[event_mask] * ew[:, None]).sum(axis=0) - ew.sum() * mean
        information += ew.sum() * np.maximum(risk_x2 / risk_w - mean ** 2, 0)
    z = np.divide(score, np.sqrt(information), out=np.zeros_like(score), where=information > 1e-12)
    return 2 * norm.sf(np.abs(z))


def fit_preprocessing(frame, feature_columns, threshold):
    """Select image features; impute and scale using this training arm only."""
    image_columns = [c for c in feature_columns if c.startswith("img_")]
    clinical_columns = [c for c in feature_columns if not c.startswith("img_")]
    image = frame[image_columns].to_numpy(dtype=float)
    image_fill = np.nanmedian(image, axis=0)
    image_fill = np.nan_to_num(image_fill, nan=0.0)
    image = np.where(np.isnan(image), image_fill, image)
    usable = np.flatnonzero(image.var(axis=0) > cfg.VARIANCE_THRESHOLD)
    if len(usable) == 0:
        raise ValueError("No imaging features remain after variance filtering.")
    pvalues = cox_score_pvalues(
        image[:, usable], frame.time.to_numpy(), frame.event.to_numpy(), frame.weight.to_numpy()
    )
    order = np.argsort(pvalues, kind="stable")
    ranked = usable[order]
    selected = ranked[pvalues[order] < threshold]
    if len(selected) < cfg.MIN_IMAGE_FEATURES:
        selected = ranked[:cfg.MIN_IMAGE_FEATURES]
    selected = selected[:cfg.MAX_IMAGE_FEATURES]
    if len(selected) > 1:
        correlation = np.corrcoef(image[:, selected], rowvar=False)
        keep = np.ones(len(selected), dtype=bool)
        for j in range(1, len(selected)):
            keep[j] = not np.any(np.abs(correlation[j, :j][keep[:j]]) > cfg.CORRELATION_THRESHOLD)
        selected = selected[keep]
    columns = [image_columns[i] for i in selected] + clinical_columns
    values = frame[columns].to_numpy(dtype=float)
    fill = np.nanmedian(values, axis=0)
    for j, column in enumerate(columns):
        if column.startswith("cat_"):
            modes = frame[column].mode()
            fill[j] = modes.iloc[0] if len(modes) else 0.0
    fill = np.nan_to_num(fill, nan=0.0)
    values = np.where(np.isnan(values), fill, values)
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    for j, column in enumerate(columns):
        if column.startswith("cat_"):
            mean[j], scale[j] = 0.0, 1.0
    return {"columns": columns, "fill": fill.tolist(), "mean": mean.tolist(), "scale": scale.tolist()}


def network_settings(parameters):
    return {
        "hidden_dims": tuple(parameters["hidden_dims"]),
        "dropout": parameters["dropout"], "batch_norm": cfg.BATCH_NORM,
    }


def fit_network(frame, preprocessing, device, seed, epochs, parameters, validation=None):
    set_seed(seed)
    x = torch.as_tensor(transform(frame, preprocessing), device=device)
    times = torch.as_tensor(frame.time.to_numpy(dtype=np.float32), device=device)
    events = torch.as_tensor(frame.event.to_numpy(dtype=np.float32), device=device)
    weights = frame.weight.to_numpy(dtype=np.float32)
    weights = torch.as_tensor(weights / weights.mean(), device=device)
    model = DeepSurv(x.shape[1], **network_settings(parameters)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"]
    )
    if validation is not None:
        xv = transform(validation, preprocessing)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg.LR_FACTOR,
            patience=cfg.LR_PATIENCE, min_lr=cfg.MIN_LR,
        )
    rng = np.random.default_rng(seed)
    best_score, best_epoch, stale = -np.inf, 1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        indices = rng.permutation(len(frame))
        batch_size = parameters["batch_size"]
        batches = [indices[i:i + batch_size] for i in range(0, len(frame), batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            last = batches.pop()
            batches[-1] = np.concatenate([batches[-1], last])
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_cox_loss(model(x[batch]), times[batch], events[batch], weights[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if validation is not None:
            score = cindex(validation.time.to_numpy(), validation.event.to_numpy(), predict_scores(model, xv, device))
            scheduler.step(score)
            if score > best_score + 1e-8:
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
                if stale >= cfg.PATIENCE:
                    break
    return model, best_score, best_epoch


def select_configuration(frame, feature_columns, device, seed):
    folds = list(StratifiedKFold(cfg.INNER_FOLDS, shuffle=True, random_state=seed).split(frame, frame.event))
    candidates = list(ParameterSampler(
        cfg.SEARCH_SPACE, n_iter=cfg.SEARCH_ITERATIONS, random_state=cfg.SEARCH_SEED,
    ))
    results = []
    for candidate, parameters in enumerate(candidates):
        scores, epochs = [], []
        for fold, (train, valid) in enumerate(folds):
            inner_train, inner_valid = frame.iloc[train], frame.iloc[valid]
            preprocessing = fit_preprocessing(inner_train, feature_columns, parameters["p_threshold"])
            _, score, epoch = fit_network(
                inner_train, preprocessing, device, seed + 100 * (fold + 1),
                cfg.MAX_EPOCHS, parameters, inner_valid,
            )
            scores.append(float(score))
            epochs.append(int(epoch))
        results.append({
            "parameters": parameters, "mean_cindex": float(np.mean(scores)),
            "fold_cindices": scores, "fold_best_epochs": epochs,
            "epochs": int(np.median(epochs)),
        })
        print(f"Inner candidate {candidate + 1}/{len(candidates)}: C-index={np.mean(scores):.4f}", flush=True)
    # Ties retain the first configuration in the reproducibly sampled order.
    best = max(results, key=lambda result: result["mean_cindex"])
    return best["parameters"], best["epochs"], results


def fit_arm(frame, feature_columns, device, seed):
    parameters, epochs, search_results = select_configuration(frame, feature_columns, device, seed)
    preprocessing = fit_preprocessing(frame, feature_columns, parameters["p_threshold"])
    model, _, _ = fit_network(frame, preprocessing, device, seed + 90000, epochs, parameters)
    scores = predict_scores(model, transform(frame, preprocessing), device)
    event_times, log_hazard = fit_baseline(
        frame.time.to_numpy(), frame.event.to_numpy(), scores, frame.weight.to_numpy()
    )
    return {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "network": network_settings(parameters), "preprocessing": preprocessing,
        "time_points": list(cfg.TIME_POINTS), "event_times": event_times.tolist(),
        "log_cumulative_hazard": log_hazard.tolist(),
        "selected_p": parameters["p_threshold"], "epochs": epochs,
        "training_config": parameters, "search_seed": cfg.SEARCH_SEED,
        "search_results": search_results,
    }


def train_framework(frame, output_dir, device):
    feature_columns = [c for c in frame.columns if c.startswith(("img_", "cont_", "cat_"))]
    if not frame.dataset.isin([1, 2]).all() or not frame.treatment.isin(cfg.TREATMENTS).all():
        raise ValueError("Use dataset 1/2 and treatment surgery/neoadjuvant.")
    if frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
        raise ValueError("patient_id must be present and unique across both datasets.")
    if not frame.event.isin([0, 1]).all() or not (frame.time > 0).all() or not (frame.weight > 0).all():
        raise ValueError("Use event 0/1, positive time, and positive sample weights.")
    if not np.isfinite(frame[["time", "weight"]].to_numpy()).all() or np.isinf(frame[feature_columns].to_numpy()).any():
        raise ValueError("Time and weights must be finite; use empty cells for missing features.")
    development = frame.loc[frame.dataset == 1].reset_index(drop=True)
    external = frame.loc[frame.dataset == 2].reset_index(drop=True)
    if development.empty or external.empty:
        raise ValueError("Provide both a development dataset and an external dataset.")
    strata = development.treatment + "_" + development.event.astype(str)
    folds = StratifiedKFold(cfg.OUTER_FOLDS, shuffle=True, random_state=cfg.SEED)
    outputs = []
    for fold, (train, valid) in enumerate(folds.split(development, strata), start=1):
        outer_train = development.iloc[train]
        bundles = {
            arm: fit_arm(outer_train.loc[outer_train.treatment == arm], feature_columns, device, cfg.SEED + fold * 1000 + a * 100)
            for a, arm in enumerate(cfg.TREATMENTS)
        }
        result = predict_pair(bundles, development.iloc[valid], device)
        result["fold"] = fold
        result["_row"] = valid
        outputs.append(result)
        print(f"Completed outer fold {fold}/{cfg.OUTER_FOLDS}", flush=True)
    pd.concat(outputs).sort_values("_row").drop(columns="_row").to_csv(output_dir / "oof_predictions.csv", index=False)
    final_bundles = {
        arm: fit_arm(development.loc[development.treatment == arm], feature_columns, device, cfg.SEED + 10000 + a * 100)
        for a, arm in enumerate(cfg.TREATMENTS)
    }
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for arm, bundle in final_bundles.items():
        torch.save(bundle, model_dir / f"{arm}.pt")
    predict_pair(final_bundles, external, device).to_csv(output_dir / "external_predictions.csv", index=False)
