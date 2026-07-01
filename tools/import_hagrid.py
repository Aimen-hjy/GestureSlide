"""Import local HaGRID/HaGRIDv2 images into GestureSlide training JSON.

The script does not download data. Download/unzip HaGRID first so that you have:
  hagrid_dataset/<gesture>/*.jpg
  hagrid_annotations/<split>/<gesture>.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from gesture_model import GESTURE_TO_ID, extract_features  # noqa: E402
from hand_detector import HandDetector  # noqa: E402

DIRECT_CLASS_MAP = {
    "fist": "FIST",
    "palm": "OPEN_PALM",
    "stop": "OPEN_PALM",
    "like": "THUMB_UP",
    "peace": "PEACE_UP",
    "two_up": "PEACE_UP",
    "peace_inverted": "PEACE_DOWN",
    "two_up_inverted": "PEACE_DOWN",
    "three": "THREE_FINGERS",
    "three2": "THREE_FINGERS",
    "three3": "THREE_FINGERS",
    "no_gesture": "NONE",
}
DIRECTION_CLASSES = {"point", "one"}
DEFAULT_TARGETS = sorted(set(DIRECT_CLASS_MAP) | DIRECTION_CLASSES)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_annotation_items(data: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    """Support common dict/list annotation layouts."""
    if isinstance(data, dict):
        for image_id, meta in data.items():
            if isinstance(meta, dict):
                yield str(image_id), meta
            elif isinstance(meta, list):
                yield str(image_id), {"objects": meta}
    elif isinstance(data, list):
        for idx, meta in enumerate(data):
            if isinstance(meta, dict):
                image_id = (meta.get("image_id") or meta.get("file_name") or
                            meta.get("name") or meta.get("id") or str(idx))
                yield str(image_id), meta


def _annotation_class(meta: dict[str, Any], fallback: str) -> str:
    label = meta.get("label") or meta.get("gesture") or meta.get("category")
    labels = meta.get("labels") or meta.get("gestures")
    if isinstance(labels, list) and labels:
        label = labels[0]
    return str(label or fallback)


def _annotation_user_id(meta: dict[str, Any], split: str, image_id: str) -> str:
    value = (meta.get("user_id") or meta.get("subject_id") or
             meta.get("person_id") or meta.get("user") or meta.get("worker_id"))
    return str(value) if value is not None else f"unknown_user:{split}:{image_id[:2]}"


def _find_image_path(dataset_root: Path, class_name: str, image_id: str) -> Path | None:
    class_dir = dataset_root / class_name
    candidate = class_dir / image_id
    if candidate.exists():
        return candidate
    stem = Path(image_id).stem
    for suffix in IMAGE_SUFFIXES:
        candidate = class_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _classify_point_direction(landmarks: list[dict[str, float]], min_length: float = 0.08) -> str | None:
    mcp = landmarks[HandDetector.INDEX_MCP]
    pip = landmarks[HandDetector.INDEX_PIP]
    tip = landmarks[HandDetector.INDEX_TIP]
    dx = float(tip["x"] - pip["x"])
    dy = float(tip["y"] - pip["y"])
    total_dx = float(tip["x"] - mcp["x"])
    total_dy = float(tip["y"] - mcp["y"])
    length = (total_dx ** 2 + total_dy ** 2) ** 0.5
    if length < min_length:
        return None
    if abs(dx) > abs(dy) * 1.3:
        return "LEFT_POINT" if dx < 0 else "RIGHT_POINT"
    if abs(dy) > abs(dx) * 1.3:
        return "POINT_INDEX" if dy < 0 else "DOWN_POINT"
    return None


def _make_hand_data(detector: HandDetector, frame):
    hand_data = detector.detect_hand(frame)
    if hand_data is None:
        return None
    landmarks = hand_data["landmarks"]
    hand_data["finger_states"] = detector.get_finger_states(landmarks)
    hand_data["thumb_index_dist"] = detector.get_thumb_index_distance(landmarks)
    return hand_data


def _import_split(detector: HandDetector, dataset_root: Path, annotations_root: Path,
                  split: str, targets: list[str], max_per_class: int | None,
                  include_direction_classes: bool, flip_horizontal: bool,
                  seed: int) -> tuple[list[dict[str, Any]], Counter]:
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    stats: Counter = Counter()

    for source_class in targets:
        if source_class in DIRECTION_CLASSES and not include_direction_classes:
            continue
        annotation_path = annotations_root / split / f"{source_class}.json"
        if not annotation_path.exists():
            stats[f"missing_annotation:{source_class}"] += 1
            continue

        items = list(_iter_annotation_items(_load_json(annotation_path)))
        rng.shuffle(items)
        accepted_for_source = 0

        for image_id, meta in items:
            if max_per_class is not None and accepted_for_source >= max_per_class:
                break
            source_label = _annotation_class(meta, source_class)
            image_path = _find_image_path(dataset_root, source_class, image_id)
            if image_path is None:
                stats[f"missing_image:{source_class}"] += 1
                continue
            frame = cv2.imread(str(image_path))
            if frame is None:
                stats[f"bad_image:{source_class}"] += 1
                continue
            if flip_horizontal:
                frame = cv2.flip(frame, 1)

            hand_data = _make_hand_data(detector, frame)
            if hand_data is None:
                stats[f"no_hand:{source_class}"] += 1
                continue

            if source_class in DIRECTION_CLASSES:
                label_name = _classify_point_direction(hand_data["landmarks"])
                if label_name is None:
                    stats[f"ambiguous_direction:{source_class}"] += 1
                    continue
            else:
                label_name = DIRECT_CLASS_MAP.get(source_label) or DIRECT_CLASS_MAP.get(source_class)
                if label_name is None:
                    stats[f"unmapped:{source_class}"] += 1
                    continue

            label_id = GESTURE_TO_ID[label_name]
            user_id = _annotation_user_id(meta, split, image_id)
            features = extract_features(hand_data)
            samples.append({
                "features": features.tolist(),
                "label": label_name,
                "label_id": label_id,
                "source": "hagrid",
                "source_dataset": "HaGRID/HaGRIDv2",
                "source_class": source_class,
                "source_label": source_label,
                "source_split": split,
                "source_image": str(image_id),
                "source_user_id": user_id,
                "source_group": f"hagrid:{user_id}",
                "flipped_horizontal": bool(flip_horizontal),
            })
            stats[f"accepted:{label_name}"] += 1
            accepted_for_source += 1
    return samples, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Import HaGRID images into GestureSlide training JSON")
    parser.add_argument("--dataset-root", required=True, help="Path to hagrid_dataset")
    parser.add_argument("--annotations-root", required=True, help="Path to hagrid_annotations")
    parser.add_argument("--output-dir", default="training_data/imported")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--include-direction-classes", action="store_true",
                        help="Import point/one and auto-label direction")
    parser.add_argument("--no-flip-horizontal", action="store_true")
    parser.add_argument("--min-detection-confidence", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # The existing HandDetector constructor reads config.*. Override before creating it.
    config.STATIC_IMAGE_MODE = True
    config.MAX_NUM_HANDS = 1
    config.MIN_DETECTION_CONFIDENCE = args.min_detection_confidence

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = HandDetector()
    try:
        all_stats = Counter()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for split in args.splits:
            samples, stats = _import_split(
                detector=detector,
                dataset_root=Path(args.dataset_root),
                annotations_root=Path(args.annotations_root),
                split=split,
                targets=args.targets,
                max_per_class=args.max_per_class,
                include_direction_classes=args.include_direction_classes,
                flip_horizontal=not args.no_flip_horizontal,
                seed=args.seed,
            )
            all_stats.update(stats)
            out_path = out_dir / f"session_hagrid_{split}_{stamp}.json"
            out_path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
            print(f"Saved {len(samples)} samples → {out_path}")

        print("\nImport stats:")
        for key, value in sorted(all_stats.items()):
            print(f"  {key:<36} {value}")
        return 0
    finally:
        detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
