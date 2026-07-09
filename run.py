from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    TIME_POINTS,
    TREATMENTS,
    RANDOM_STATE,
)
from training import (
    set_seed,
    cross_fit,
    train_model,
    predict_scores,
)
from model import estimate_weighted_baseline_survival
from counterfactual import (
    predict_potential_survival,
    compute_ite,
)


DATA_PATH = "data.csv"
OUTPUT_DIR = "results"

META_COLUMNS = {
    "patient_id",
    "dataset",
    "treatment",
    "time",
    "event",
    "weight",
}


def validate_data(data):
    required = {
        "patient_id",
        "dataset",
        "treatment",
        "time",
        "event",
        "weight",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if not set(data["dataset"].unique()).issubset({1, 2}):
        raise ValueError("dataset must contain only 1 or 2.")

    if not set(data["treatment"].unique()).issubset(set(TREATMENTS)):
        raise ValueError(
            f"treatment must contain only {TREATMENTS}."
        )

    if data["patient_id"].duplicated().any():
        raise ValueError("patient_id must be unique.")

    if data[["time", "event", "weight"]].isna().any().any():
        raise ValueError("time, event, and weight must not contain missing values.")

    if (data["time"] <= 0).any():
        raise ValueError("time must be positive.")

    if not set(data["event"].unique()).issubset({0, 1}):
        raise ValueError("event must contain only 0 or 1.")

    if (data["weight"] <= 0).any():
        raise ValueError("weight must be positive.")


def main():
    set_seed(RANDOM_STATE)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    validate_data(data)

    feature_columns = [
        column
        for column in data.columns
        if column not in META_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No feature columns found.")

    if data[feature_columns].isna().any().any():
        raise ValueError("Feature columns must not contain missing values.")

    dataset_1 = data[data["dataset"] == 1].copy()
    dataset_2 = data[data["dataset"] == 2].copy()

    if len(dataset_1) == 0 or len(dataset_2) == 0:
        raise ValueError("Both Dataset 1 and Dataset 2 are required.")

    fitted_models = {}
    oof_outputs = []

    for treatment in TREATMENTS:
        train_data = dataset_1[
            dataset_1["treatment"] == treatment
        ].copy()

        if len(train_data) == 0:
            raise ValueError(
                f"No Dataset 1 samples for treatment: {treatment}"
            )

        X = train_data[
            feature_columns
        ].to_numpy(dtype=np.float32)

        times = train_data[
            "time"
        ].to_numpy(dtype=np.float32)

        events = train_data[
            "event"
        ].to_numpy(dtype=np.int32)

        weights = train_data[
            "weight"
        ].to_numpy(dtype=np.float32)

        weights = weights / weights.mean()

        oof_scores, oof_survival, fold_ids = cross_fit(
            X=X,
            times=times,
            events=events,
            weights=weights,
            time_points=TIME_POINTS,
            device=device,
        )

        oof_output = train_data[
            [
                "patient_id",
                "dataset",
                "treatment",
                "time",
                "event",
            ]
        ].copy()

        oof_output["fold"] = fold_ids
        oof_output["risk_score"] = oof_scores

        for index, time_point in enumerate(TIME_POINTS):
            oof_output[
                f"S_{treatment}_{time_point}m"
            ] = oof_survival[:, index]

        oof_outputs.append(oof_output)

        model = train_model(
            X=X,
            times=times,
            events=events,
            weights=weights,
            input_dim=X.shape[1],
            device=device,
        )

        train_scores = predict_scores(
            model,
            X,
            device,
        )

        event_times, baseline_survival = (
            estimate_weighted_baseline_survival(
                times,
                events,
                train_scores,
                weights,
            )
        )

        fitted_models[treatment] = {
            "model": model,
            "event_times": event_times,
            "baseline_survival": baseline_survival,
        }

        torch.save(
            model.state_dict(),
            output_dir / f"{treatment}_model.pt",
        )

    pd.concat(
        oof_outputs,
        ignore_index=True,
    ).to_csv(
        output_dir / "dataset1_oof_predictions.csv",
        index=False,
    )

    X_all = data[
        feature_columns
    ].to_numpy(dtype=np.float32)

    risk_surgery, survival_surgery = (
        predict_potential_survival(
            model=fitted_models["surgery"]["model"],
            X=X_all,
            event_times=fitted_models["surgery"]["event_times"],
            baseline_survival=fitted_models["surgery"]["baseline_survival"],
            time_points=TIME_POINTS,
            device=device,
        )
    )

    risk_neoadjuvant, survival_neoadjuvant = (
        predict_potential_survival(
            model=fitted_models["neoadjuvant"]["model"],
            X=X_all,
            event_times=fitted_models["neoadjuvant"]["event_times"],
            baseline_survival=fitted_models["neoadjuvant"]["baseline_survival"],
            time_points=TIME_POINTS,
            device=device,
        )
    )

    ite = compute_ite(
        survival_surgery,
        survival_neoadjuvant,
    )

    output = data[
        [
            "patient_id",
            "dataset",
            "treatment",
            "time",
            "event",
        ]
    ].copy()

    output["risk_score_surgery"] = risk_surgery
    output["risk_score_neoadjuvant"] = risk_neoadjuvant

    for index, time_point in enumerate(TIME_POINTS):
        output[
            f"S_surgery_{time_point}m"
        ] = survival_surgery[:, index]

        output[
            f"S_neoadjuvant_{time_point}m"
        ] = survival_neoadjuvant[:, index]

        output[
            f"ITE_{time_point}m"
        ] = ite[:, index]

    output.to_csv(
        output_dir / "individual_treatment_effects.csv",
        index=False,
    )

    dataset_2_output = output[
        output["dataset"] == 2
    ].copy()

    dataset_2_output.to_csv(
        output_dir / "dataset2_predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
