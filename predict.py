"""Load local model bundles and predict paired potential survival and ITE."""

import numpy as np
import torch

from model import DeepSurv, survival_at_times


def transform(frame, preprocessing):
    values = frame[preprocessing["columns"]].to_numpy(dtype=np.float32)
    values = np.where(np.isfinite(values), values, preprocessing["fill"])
    return ((values - preprocessing["mean"]) / preprocessing["scale"]).astype(np.float32)


def predict_scores(model, x, device):
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            model(torch.as_tensor(x[i:i + 2048], device=device)).cpu().numpy()
            for i in range(0, len(x), 2048)
        ]).astype(float)


def predict_arm(bundle, frame, device):
    model = DeepSurv(len(bundle["preprocessing"]["columns"]), **bundle["network"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    x = transform(frame, bundle["preprocessing"])
    scores = predict_scores(model, x, device)
    return survival_at_times(
        scores, np.asarray(bundle["event_times"]),
        np.asarray(bundle["log_cumulative_hazard"]), bundle["time_points"],
    )


def predict_pair(bundles, frame, device):
    times = bundles["surgery"]["time_points"]
    if times != bundles["neoadjuvant"]["time_points"]:
        raise ValueError("The two model bundles must use the same prediction times.")
    surgery = predict_arm(bundles["surgery"], frame, device)
    neo = predict_arm(bundles["neoadjuvant"], frame, device)
    output = frame[["patient_id"]].reset_index(drop=True).copy()
    for j, t in enumerate(times):
        output[f"S_surgery_{t}m"] = surgery[:, j]
        output[f"S_neoadjuvant_{t}m"] = neo[:, j]
        output[f"ITE_{t}m"] = neo[:, j] - surgery[:, j]
    return output


def load_bundles(model_dir):
    return {
        treatment: torch.load(model_dir / f"{treatment}.pt", map_location="cpu", weights_only=True)
        for treatment in ("surgery", "neoadjuvant")
    }
