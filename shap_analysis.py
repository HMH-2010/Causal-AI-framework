from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor

from config import (
    HIDDEN_DIMS,
    DROPOUT,
    BATCH_NORM,
    RANDOM_STATE,
    TIME_POINTS,
)
from model import DeepSurv


DATA_PATH = "data.csv"
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("shap_results")

BACKGROUND_SIZE = 200
EXPLAIN_SIZE = 800
TOP_N = 20

META_COLUMNS = {
    "patient_id",
    "dataset",
    "treatment",
    "time",
    "event",
    "weight",
}


class RiskWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).reshape(-1, 1)


def sample_rows(X, max_n, seed):
    if len(X) <= max_n:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx]


def normalize_shap_values(values):
    if isinstance(values, list):
        values = values[0]

    values = np.asarray(values)

    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]

    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]

    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")

    return values


def save_shap(values, X, feature_names, prefix):
    mean_abs = np.abs(values).mean(axis=0)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    importance.head(TOP_N).to_csv(
        OUTPUT_DIR / f"{prefix}_top_features.csv",
        index=False,
    )

    pd.DataFrame(
        values,
        columns=feature_names,
    ).to_csv(
        OUTPUT_DIR / f"{prefix}_shap_values.csv",
        index=False,
    )

    pd.DataFrame(
        X,
        columns=feature_names,
    ).to_csv(
        OUTPUT_DIR / f"{prefix}_features.csv",
        index=False,
    )


def load_model(path, input_dim, device):
    model = DeepSurv(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        batch_norm=BATCH_NORM,
    ).to(device)

    state = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(state)
    model.eval()

    return model


def deepsurv_shap(
    model,
    X,
    feature_names,
    prefix,
    device,
    seed,
):
    background = sample_rows(
        X,
        BACKGROUND_SIZE,
        seed,
    )

    X_explain = sample_rows(
        X,
        EXPLAIN_SIZE,
        seed + 1,
    )

    wrapper = RiskWrapper(model).to(device)
    wrapper.eval()

    background_tensor = torch.as_tensor(
        background,
        dtype=torch.float32,
        device=device,
    )

    explain_tensor = torch.as_tensor(
        X_explain,
        dtype=torch.float32,
        device=device,
    )

    explainer = shap.GradientExplainer(
        wrapper,
        background_tensor,
    )

    values = explainer.shap_values(
        explain_tensor,
    )

    values = normalize_shap_values(values)

    save_shap(
        values,
        X_explain,
        feature_names,
        prefix,
    )


def ite_surrogate_shap(
    X,
    y,
    feature_names,
    prefix,
    seed,
):
    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(X, y)

    X_explain = sample_rows(
        X,
        EXPLAIN_SIZE,
        seed + 1,
    )

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_explain)
    values = normalize_shap_values(values)

    save_shap(
        values,
        X_explain,
        feature_names,
        prefix,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data = pd.read_csv(DATA_PATH)

    feature_names = [
        column
        for column in data.columns
        if column not in META_COLUMNS
    ]

    X = data[
        feature_names
    ].to_numpy(
        dtype=np.float32
    )

    surgery_model = load_model(
        RESULTS_DIR / "surgery_model.pt",
        X.shape[1],
        device,
    )

    neoadjuvant_model = load_model(
        RESULTS_DIR / "neoadjuvant_model.pt",
        X.shape[1],
        device,
    )

    deepsurv_shap(
        surgery_model,
        X,
        feature_names,
        "surgery_risk",
        device,
        RANDOM_STATE,
    )

    deepsurv_shap(
        neoadjuvant_model,
        X,
        feature_names,
        "neoadjuvant_risk",
        device,
        RANDOM_STATE + 100,
    )

    ite = pd.read_csv(
        RESULTS_DIR / "individual_treatment_effects.csv"
    )

    merged = data[
        ["patient_id"] + feature_names
    ].merge(
        ite,
        on="patient_id",
        how="inner",
    )

    X_ite = merged[
        feature_names
    ].to_numpy(
        dtype=np.float32
    )

    for time_point in TIME_POINTS:
        target = f"ITE_{time_point}m"

        if target not in merged.columns:
            continue

        y = merged[
            target
        ].to_numpy(
            dtype=float
        )

        valid = np.isfinite(y)

        ite_surrogate_shap(
            X_ite[valid],
            y[valid],
            feature_names,
            target,
            RANDOM_STATE + int(time_point),
        )


if __name__ == "__main__":
    main()
