"""Evaluate geometric LEFT/RIGHT pointing rules on saved training features.

This does not retrain a model and does not need a camera. It reads existing
session_*.json files, reconstructs the 21x3 MediaPipe feature coordinates and
finger-state bits, then measures how often the geometry rule agrees with labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gesture_model import GESTURE_NAMES, GESTURE_TO_ID  # noqa: E402

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8

TARGET_LABELS = {"LEFT_POINT", "RIGHT_POINT"}


def iter_samples(data_dir: Path):
    for path in sorted(data_dir.rglob("session_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] skip {path}: {exc}")
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            features = item.get("features")
            label = item.get("label")
            label_id = item.get("label_id")
            if label is None and label_id is not None:
                try:
                    label = GESTURE_NAMES[int(label_id)]
                except Exception:
                    label = str(label_id)
            if isinstance(features, list) and len(features) >= 68 and label:
                yield path, label, features


def geometry_direction(features: list[float], min_length: float, ratio: float) -> str:
    """Return LEFT_POINT / RIGHT_POINT / OTHER from saved 69-D features."""
    # First 63 values are 21 relative landmarks (x,y,z). Values 63:68 are
    # finger-state bits: thumb, index, middle, ring, pinky.
    finger = [bool(round(float(v))) for v in features[63:68]]
    if len(finger) < 5:
        return "OTHER"

    thumb, index, middle, ring, pinky = finger[:5]
    if not index or middle or ring or pinky:
        return "OTHER"

    coords = features[:63]
    def xy(idx: int) -> tuple[float, float]:
        base = idx * 3
        return float(coords[base]), float(coords[base + 1])

    mcp_x, mcp_y = xy(INDEX_MCP)
    pip_x, pip_y = xy(INDEX_PIP)
    tip_x, tip_y = xy(INDEX_TIP)

    dx = tip_x - pip_x
    dy = tip_y - pip_y
    total_dx = tip_x - mcp_x
    total_dy = tip_y - mcp_y
    length = (total_dx ** 2 + total_dy ** 2) ** 0.5
    if length < min_length:
        return "OTHER"

    if abs(dx) > abs(dy) * ratio:
        return "LEFT_POINT" if dx < 0 else "RIGHT_POINT"
    return "OTHER"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate geometric pointing rule on saved GestureSlide data")
    parser.add_argument("data", nargs="?", default="training_data", help="Directory containing session_*.json")
    parser.add_argument("--min-length", type=float, default=0.08)
    parser.add_argument("--ratio", type=float, default=1.30)
    parser.add_argument("--show-errors", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"[error] data directory not found: {data_dir}")
        return 1

    total = 0
    label_counts = Counter()
    pred_counts = Counter()
    confusion = defaultdict(Counter)
    target_total = 0
    target_correct = 0
    false_positive = 0
    missed_target = 0
    errors = []

    for path, label, features in iter_samples(data_dir):
        total += 1
        pred = geometry_direction(features, args.min_length, args.ratio)
        label_counts[label] += 1
        pred_counts[pred] += 1
        confusion[label][pred] += 1

        is_target = label in TARGET_LABELS
        is_pred_target = pred in TARGET_LABELS
        if is_target:
            target_total += 1
            if pred == label:
                target_correct += 1
            else:
                missed_target += 1
                if len(errors) < args.show_errors:
                    errors.append((path.name, label, pred))
        elif is_pred_target:
            false_positive += 1
            if len(errors) < args.show_errors:
                errors.append((path.name, label, pred))

    print("Geometry direction evaluation")
    print("=" * 72)
    print(f"data:        {data_dir}")
    print(f"samples:     {total}")
    print(f"min_length:  {args.min_length:.3f}")
    print(f"ratio:       {args.ratio:.2f}")
    print()

    if target_total:
        recall = target_correct / target_total
    else:
        recall = 0.0
    non_target_total = total - target_total
    fp_rate = false_positive / non_target_total if non_target_total > 0 else 0.0

    print("Core numbers")
    print("-" * 72)
    print(f"LEFT/RIGHT total:       {target_total}")
    print(f"LEFT/RIGHT correct:     {target_correct}")
    print(f"LEFT/RIGHT recall:      {recall:.4f}")
    print(f"missed LEFT/RIGHT:      {missed_target}")
    print(f"false positive commands:{false_positive}")
    print(f"false positive rate:    {fp_rate:.4f}")
    print()

    print("Per-class geometry outputs")
    print("-" * 72)
    print(f"{'label':<18} {'n':>6} {'LEFT':>8} {'RIGHT':>8} {'OTHER':>8}")
    for label, n in label_counts.most_common():
        row = confusion[label]
        print(f"{label:<18} {n:>6} {row['LEFT_POINT']:>8} {row['RIGHT_POINT']:>8} {row['OTHER']:>8}")

    if errors:
        print()
        print("Example errors")
        print("-" * 72)
        for filename, label, pred in errors:
            print(f"{filename:<36} true={label:<14} geom={pred}")

    print()
    print("Threshold tips")
    print("-" * 72)
    print("If LEFT/RIGHT recall is low, try:      --ratio 1.15")
    print("If false positives are high, try:      --ratio 1.45")
    print("If short/unclear fingers trigger, try: --min-length 0.10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
