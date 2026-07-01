"""Audit GestureSlide training data for class balance and group leakage risk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_pipeline import load_training_dataset, print_dataset_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit GestureSlide training data")
    parser.add_argument("data_dir", nargs="?", default="training_data",
                        help="Directory containing session_*.json files")
    args = parser.parse_args()

    dataset = load_training_dataset(args.data_dir)
    print_dataset_audit(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
