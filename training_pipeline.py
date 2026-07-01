"""Training data loading, audit, and leakage-aware training helpers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from gesture_model import GESTURE_NAMES, ID_TO_GESTURE, FEATURE_DIM, COORD_FEATURE_DIM


@dataclass(frozen=True)
class TrainingDataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    files: np.ndarray
    sources: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])


def _session_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    files = sorted(root.rglob("session_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No session_*.json files found under {root}. "
            "Run main.py --collect or tools/import_hagrid.py first."
        )
    return files


def load_training_dataset(data_dir: str | Path) -> TrainingDataset:
    """Load session JSON files. Old files are grouped by filename by default."""
    xs: list[np.ndarray] = []
    ys: list[int] = []
    groups: list[str] = []
    files_out: list[str] = []
    sources: list[str] = []

    for path in _session_files(data_dir):
        samples = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(samples, list):
            raise ValueError(f"{path}: top-level JSON value must be a list")

        file_group = path.stem
        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"{path}: sample {idx} must be an object")
            try:
                features = np.asarray(sample["features"], dtype=np.float64)
                label_id = int(sample["label_id"])
                label_name = str(sample["label"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid sample {idx}: {exc}") from exc

            if features.shape != (FEATURE_DIM,):
                raise ValueError(
                    f"{path}: sample {idx} has shape {features.shape}; "
                    f"expected ({FEATURE_DIM},)"
                )
            if not np.all(np.isfinite(features)):
                raise ValueError(f"{path}: sample {idx} contains NaN or Inf")
            if label_id not in ID_TO_GESTURE:
                raise ValueError(f"{path}: sample {idx} has unknown label_id {label_id}")
            if ID_TO_GESTURE[label_id] != label_name:
                raise ValueError(
                    f"{path}: sample {idx} label mismatch: id {label_id} means "
                    f"{ID_TO_GESTURE[label_id]}, label is {label_name}"
                )

            xs.append(features)
            ys.append(label_id)
            groups.append(str(sample.get("source_group") or sample.get("group")
                              or sample.get("session_id") or file_group))
            files_out.append(path.name)
            sources.append(str(sample.get("source") or "local_camera"))

    if not xs:
        raise ValueError(f"No samples found under {data_dir}")
    return TrainingDataset(
        X=np.vstack(xs).astype(np.float64),
        y=np.asarray(ys, dtype=np.int32),
        groups=np.asarray(groups, dtype=object),
        files=np.asarray(files_out, dtype=object),
        sources=np.asarray(sources, dtype=object),
    )


def print_dataset_audit(dataset: TrainingDataset) -> None:
    counts = Counter(int(v) for v in dataset.y)
    group_counts = Counter(str(v) for v in dataset.groups)
    source_counts = Counter(str(v) for v in dataset.sources)

    print(f"Loaded {dataset.n_samples} samples")
    print(f"Files: {len(set(dataset.files))}  Groups: {len(group_counts)}")
    print("\nClass counts:")
    for class_id, name in enumerate(GESTURE_NAMES):
        count = counts.get(class_id, 0)
        suffix = "  [missing]" if count == 0 else ""
        print(f"  {name:<14} {count:>6}{suffix}")

    print("\nSource counts:")
    for source, count in source_counts.most_common():
        print(f"  {source:<20} {count:>6}")

    print("\nWarnings:")
    warned = False
    missing = [GESTURE_NAMES[i] for i in range(len(GESTURE_NAMES)) if counts.get(i, 0) == 0]
    if missing:
        warned = True
        print(f"  - Missing classes: {', '.join(missing)}")

    present = [v for v in counts.values() if v > 0]
    if present and max(present) / min(present) > 3:
        warned = True
        print("  - Class imbalance is high (>3x between largest and smallest class)")

    by_class: dict[int, set[str]] = defaultdict(set)
    for y, g in zip(dataset.y, dataset.groups):
        by_class[int(y)].add(str(g))
    weak = [GESTURE_NAMES[i] for i, gs in by_class.items() if len(gs) < 2]
    if weak:
        warned = True
        print("  - These classes have <2 groups, so group holdout is weak: " + ", ".join(weak))
    if not warned:
        print("  - No obvious structural issues found")


def _scale(X_train: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[:, :COORD_FEATURE_DIM] = scaler.fit_transform(X_train[:, :COORD_FEATURE_DIM])
    X_test_scaled[:, :COORD_FEATURE_DIM] = scaler.transform(X_test[:, :COORD_FEATURE_DIM])
    return X_train_scaled, X_test_scaled, scaler


def _random_split(dataset: TrainingDataset, test_size: float, seed: int):
    counts = Counter(int(v) for v in dataset.y)
    stratify = dataset.y if all(v >= 2 for v in counts.values()) else None
    if stratify is None:
        print("[warn] Some classes have fewer than 2 samples; split is not stratified")
    return train_test_split(np.arange(dataset.n_samples), test_size=test_size,
                            stratify=stratify, random_state=seed)


def _build_group_maps(dataset: TrainingDataset):
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    group_to_labels: dict[str, set[int]] = defaultdict(set)
    label_to_groups: dict[int, set[str]] = defaultdict(set)
    for idx, (label, group) in enumerate(zip(dataset.y, dataset.groups)):
        label_id = int(label)
        group_name = str(group)
        group_to_indices[group_name].append(idx)
        group_to_labels[group_name].add(label_id)
        label_to_groups[label_id].add(group_name)
    return group_to_indices, group_to_labels, label_to_groups


def _group_split(dataset: TrainingDataset, test_size: float, seed: int):
    """Class-aware group holdout split.

    The first version used a purely random group split, which could leave many
    classes out of the test set when each session contains only one gesture.
    This heuristic first holds out at least one group for every class that has
    2+ groups, while keeping classes with only one group in train.
    """
    group_to_indices, group_to_labels, label_to_groups = _build_group_maps(dataset)
    all_groups = sorted(group_to_indices)
    if len(all_groups) < 2:
        print("[warn] Fewer than 2 groups; falling back to random split")
        return _random_split(dataset, test_size, seed)

    rng = np.random.RandomState(seed)
    target_test_samples = max(1, int(round(dataset.n_samples * test_size)))
    selected: set[str] = set()

    def selected_count_for_label(label: int) -> int:
        return sum(1 for g in selected if label in group_to_labels[g])

    def can_add(group: str) -> bool:
        # Never put the only group for a class into test; that class would have
        # no training examples and model.fit would not learn it.
        for label in group_to_labels[group]:
            if len(label_to_groups[label]) - selected_count_for_label(label) <= 1:
                return False
        return True

    def group_size(group: str) -> int:
        return len(group_to_indices[group])

    eligible_labels = [label for label, groups in label_to_groups.items() if len(groups) >= 2]
    single_group_labels = [label for label, groups in label_to_groups.items() if len(groups) < 2]
    if single_group_labels:
        names = ", ".join(ID_TO_GESTURE.get(label, str(label)) for label in sorted(single_group_labels))
        print(f"[warn] These classes have only one group and will be kept in train only: {names}")

    # First pass: make the test set cover as many classes as possible.
    for label in sorted(eligible_labels, key=lambda x: len(label_to_groups[x])):
        if any(label in group_to_labels[g] for g in selected):
            continue
        candidates = [g for g in label_to_groups[label] if g not in selected and can_add(g)]
        if not candidates:
            continue
        candidates.sort(key=lambda g: (abs((sum(group_size(x) for x in selected) + group_size(g)) - target_test_samples),
                                       group_size(g), rng.random()))
        selected.add(candidates[0])

    # Second pass: fill toward the requested test size without removing any
    # class entirely from train.
    while sum(group_size(g) for g in selected) < target_test_samples:
        candidates = [g for g in all_groups if g not in selected and can_add(g)]
        if not candidates:
            break
        current = sum(group_size(g) for g in selected)
        candidates.sort(key=lambda g: (abs((current + group_size(g)) - target_test_samples),
                                       group_size(g), rng.random()))
        selected.add(candidates[0])

    if not selected:
        print("[warn] Could not build a safe group split; falling back to random split")
        return _random_split(dataset, test_size, seed)

    test_idx = np.array(sorted(i for g in selected for i in group_to_indices[g]), dtype=np.int64)
    train_idx = np.array(sorted(set(range(dataset.n_samples)) - set(test_idx.tolist())), dtype=np.int64)

    if len(train_idx) == 0 or len(test_idx) == 0:
        print("[warn] Empty train/test split; falling back to random split")
        return _random_split(dataset, test_size, seed)
    return train_idx, test_idx


def _print_class_counts(title: str, y: np.ndarray) -> None:
    counts = Counter(int(v) for v in y)
    print(title)
    for label in sorted(counts):
        print(f"  {ID_TO_GESTURE.get(label, str(label)):<14} {counts[label]:>6}")


def _print_eval(y_test: np.ndarray, y_pred: np.ndarray, model: MLPClassifier) -> None:
    # Report labels that are actually present in y_test or were predicted. This
    # keeps absent classes from cluttering the report while still surfacing false
    # positives for classes absent from the test split.
    labels = np.asarray(sorted(set(int(v) for v in y_test) | set(int(v) for v in y_pred)), dtype=np.int32)
    names = [ID_TO_GESTURE.get(int(label), str(label)) for label in labels]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels))


def train_real_dataset(data_dir: str | Path = "training_data",
                       test_size: float = 0.2,
                       model_path: str = "gesture_model.joblib",
                       scaler_path: str = "gesture_scaler.joblib",
                       seed: int = 42,
                       split_strategy: str = "group") -> dict:
    dataset = load_training_dataset(data_dir)
    print_dataset_audit(dataset)

    if split_strategy == "group":
        print("\nSplit strategy: class-aware group holdout (recommended)")
        train_idx, test_idx = _group_split(dataset, test_size, seed)
    elif split_strategy == "random":
        print("\nSplit strategy: random frame split (optimistic)")
        train_idx, test_idx = _random_split(dataset, test_size, seed)
    else:
        raise ValueError("split_strategy must be 'group' or 'random'")

    X_train, X_test = dataset.X[train_idx], dataset.X[test_idx]
    y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]
    X_train_scaled, X_test_scaled, scaler = _scale(X_train, X_test)

    print(f"\nTrain samples: {len(train_idx)}  Test samples: {len(test_idx)}")
    print(f"Train groups: {len(set(dataset.groups[train_idx]))}  Test groups: {len(set(dataset.groups[test_idx]))}")
    _print_class_counts("\nTrain class counts:", y_train)
    _print_class_counts("\nTest class counts:", y_test)

    missing_in_test = sorted(set(int(v) for v in dataset.y) - set(int(v) for v in y_test))
    if missing_in_test:
        names = ", ".join(ID_TO_GESTURE.get(label, str(label)) for label in missing_in_test)
        print(f"[warn] Test split does not contain: {names}")

    print("\nTraining MLP (hidden: 128,64)...")
    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                          max_iter=500, early_stopping=True, validation_fraction=0.1,
                          random_state=seed, verbose=True)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'=' * 55}\n  Test Accuracy: {acc:.4f}\n{'=' * 55}")
    _print_eval(y_test, y_pred, model)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    return {"accuracy": float(acc), "model_path": model_path,
            "scaler_path": scaler_path, "n_samples": dataset.n_samples,
            "split_strategy": split_strategy}
