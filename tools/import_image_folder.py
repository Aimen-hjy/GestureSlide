"""Import a lightweight image-folder dataset into GestureSlide training JSON.

Expected layout:
  dataset_root/
    rock/*.png
    paper/*.png
    scissors/*.png

Example:
  python tools/import_image_folder.py \
    --dataset-root datasets/rps \
    --map rock=FIST paper=OPEN_PALM scissors=PEACE_UP \
    --output-dir training_data/imported
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from gesture_model import GESTURE_TO_ID, extract_features  # noqa: E402
from hand_detector import HandDetector  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def parse_mapping(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid mapping '{value}', expected folder=GESTURE")
        folder, gesture = value.split("=", 1)
        folder = folder.strip()
        gesture = gesture.strip().upper()
        if not folder or not gesture:
            raise ValueError(f"Invalid mapping '{value}', expected folder=GESTURE")
        if gesture not in GESTURE_TO_ID:
            raise ValueError(f"Unknown GestureSlide label '{gesture}'")
        mapping[folder] = gesture
    return mapping


def iter_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in IMAGE_SUFFIXES:
        paths.extend(folder.rglob(f"*{suffix}"))
        paths.extend(folder.rglob(f"*{suffix.upper()}"))
    return sorted(set(paths))


def make_hand_data(detector: HandDetector, frame):
    hand_data = detector.detect_hand(frame)
    if hand_data is None:
        return None
    landmarks = hand_data["landmarks"]
    hand_data["finger_states"] = detector.get_finger_states(landmarks)
    hand_data["thumb_index_dist"] = detector.get_thumb_index_distance(landmarks)
    return hand_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Import image-folder hand dataset into GestureSlide format")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--map", nargs="+", required=True,
                        help="Folder-to-label mapping, e.g. rock=FIST paper=OPEN_PALM")
    parser.add_argument("--output-dir", default="training_data/imported")
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--source-name", default="image_folder")
    parser.add_argument("--group-by", choices=("folder", "parent", "image"), default="parent",
                        help="Grouping key for leakage-aware split")
    parser.add_argument("--no-flip-horizontal", action="store_true")
    parser.add_argument("--min-detection-confidence", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    mapping = parse_mapping(args.map)
    rng = random.Random(args.seed)

    config.STATIC_IMAGE_MODE = True
    config.MAX_NUM_HANDS = 1
    config.MIN_DETECTION_CONFIDENCE = args.min_detection_confidence

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = HandDetector()
    samples = []
    stats: Counter = Counter()
    try:
        for folder_name, label_name in mapping.items():
            folder = root / folder_name
            if not folder.exists():
                stats[f"missing_folder:{folder_name}"] += 1
                continue

            image_paths = iter_images(folder)
            rng.shuffle(image_paths)
            accepted = 0
            for image_path in image_paths:
                if args.max_per_class is not None and accepted >= args.max_per_class:
                    break
                frame = cv2.imread(str(image_path))
                if frame is None:
                    stats[f"bad_image:{folder_name}"] += 1
                    continue
                if not args.no_flip_horizontal:
                    frame = cv2.flip(frame, 1)

                hand_data = make_hand_data(detector, frame)
                if hand_data is None:
                    stats[f"no_hand:{folder_name}"] += 1
                    continue

                if args.group_by == "folder":
                    group = folder_name
                elif args.group_by == "image":
                    group = image_path.stem
                else:
                    group = image_path.parent.name

                label_id = GESTURE_TO_ID[label_name]
                features = extract_features(hand_data)
                samples.append({
                    "features": features.tolist(),
                    "label": label_name,
                    "label_id": label_id,
                    "source": args.source_name,
                    "source_class": folder_name,
                    "source_image": image_path.name,
                    "source_path": str(image_path),
                    "source_group": f"{args.source_name}:{group}",
                    "flipped_horizontal": not args.no_flip_horizontal,
                })
                accepted += 1
                stats[f"accepted:{label_name}"] += 1

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"session_{args.source_name}_{stamp}.json"
        out_path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(samples)} samples → {out_path}")
        print("\nImport stats:")
        for key, value in sorted(stats.items()):
            print(f"  {key:<32} {value}")
        return 0
    finally:
        detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
