"""Train the core framework or apply models downloaded from an external host."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from predict import load_bundles, predict_pair
from training import train_framework


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "predict"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    frame = pd.read_csv(args.data, dtype={"patient_id": str})
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.mode == "train":
        train_framework(frame, args.output, device)
    else:
        predict_pair(load_bundles(args.models), frame, device).to_csv(
            args.output / "predictions.csv", index=False
        )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
