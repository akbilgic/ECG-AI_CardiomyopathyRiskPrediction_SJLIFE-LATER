"""Generate a mathematical test waveform, with no patient-derived information.

Run from the repository root:
    python -m examples.generate_synthetic_ecg --output examples/synthetic_ecg.npy

The output is a prepared (5000, 12) array for a software smoke test. It is not a
physiologically validated simulation, and its model score has no clinical meaning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ecgai.synthetic import make_synthetic_ecg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/synthetic_ecg.npy"))
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npy":
        parser.error("--output must end in .npy")
    if args.output.exists():
        parser.error("The output file already exists; choose a new --output path.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, make_synthetic_ecg(), allow_pickle=False)
    print("Saved a synthetic (5000, 12) float32 test waveform; no patient data were used.")


if __name__ == "__main__":
    main()
