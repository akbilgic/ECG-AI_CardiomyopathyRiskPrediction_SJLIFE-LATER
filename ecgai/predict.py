"""Command-line inference on a synthetic or locally supplied ECG."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .preprocessing import prepare_model_array
from .synthetic import make_synthetic_ecg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="Run a mathematical demo; no patient data needed.")
    source.add_argument("--input", type=Path, help="One local numerical .npy ECG (pickle disabled).")
    parser.add_argument("--output", type=Path, help="Optional JSON result path; existing files are not overwritten.")
    args = parser.parse_args()
    if args.synthetic:
        ecg = prepare_model_array(make_synthetic_ecg())
    else:
        if args.input.suffix.lower() != ".npy":
            parser.error("--input must be one numerical .npy file.")
        try:
            array = np.load(args.input, allow_pickle=False)
            ecg = prepare_model_array(array)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    if args.output and args.output.exists():
        parser.error("The output already exists; choose a new filename.")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from .model import load_model, predict_scores
    score = float(predict_scores(load_model(), ecg)[0])
    result = {
        "ecgai_score": score,
        "score_type": "uncalibrated_sigmoid",
        "synthetic_example": bool(args.synthetic),
        "interpretation": "Research score; not an absolute five-year risk estimate or a clinical decision.",
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
