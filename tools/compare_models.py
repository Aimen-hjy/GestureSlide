"""Compare lightweight classifiers and save the best GestureSlide model.

This script keeps the project lightweight: all models consume the same 69-D
MediaPipe feature vector, so it is suitable for a classroom PPT-control demo.
It also writes per-class metrics and a confusion matrix for data-driven
collection decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gesture_model import ID_TO_GESTURE  # noqa: E402
from training_pipeline import (  # noqa: E402
    _augment_training_set,
    _group_split,
    _random_split,
    _scale,
    load_training_dataset,
    print_dataset_audit,
)


@dataclass
class ModelResult:
    name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    train_seconds: float
    n_train: int
    n_test: int


def _label_name(label: int) -> str:
    return ID_TO_GESTURE.get(int(label), str(label))


def build_model(name: str, seed: int):
    """Return a probability-capable lightweight classifier."""
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
    if name == "svm":
        return SVC(C=8.0, kernel="rbf", gamma="scale", probability=True,
                   class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.07,
            l2_regularization=0.02,
            random_state=seed,
        )
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=7, weights="distance")
    raise ValueError(f"Unknown model: {name}")


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> tuple[ModelResult, np.ndarray]:
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started
    pred = model.predict(X_test)
    result = ModelResult(
        name=name,
        accuracy=float(accuracy_score(y_test, pred)),
        macro_f1=float(f1_score(y_test, pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        train_seconds=float(train_seconds),
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
    )
    return result, pred


def per_class_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    labels = sorted(set(map(int, y_true)) | set(map(int, y_pred)))
    names = [_label_name(label) for label in labels]
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )


def write_class_metrics_csv(path: Path, reports: dict[str, dict]) -> None:
    rows = []
    for model_name, report in reports.items():
        for class_name, metrics in report.items():
            if not isinstance(metrics, dict) or class_name in {"accuracy", "macro avg", "weighted avg"}:
                continue
            rows.append({
                "model": model_name,
                "class": class_name,
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1_score": metrics.get("f1-score", 0.0),
                "support": metrics.get("support", 0),
            })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "class", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    labels = sorted(set(map(int, y_true)) | set(map(int, y_pred)))
    names = [_label_name(label) for label in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *map(int, row)])


def print_results(results: list[ModelResult]) -> None:
    print("\nModel comparison")
    print("=" * 78)
    print(f"{'model':<16} {'accuracy':>10} {'macro_f1':>10} {'weighted_f1':>12} {'train_s':>10}")
    print("-" * 78)
    for r in sorted(results, key=lambda x: (x.macro_f1, x.accuracy), reverse=True):
        print(f"{r.name:<16} {r.accuracy:>10.4f} {r.macro_f1:>10.4f} {r.weighted_f1:>12.4f} {r.train_seconds:>10.2f}")


def print_low_class_scores(report: dict, max_rows: int = 5) -> None:
    class_rows = []
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and class_name not in {"macro avg", "weighted avg"}:
            class_rows.append((class_name, float(metrics.get("f1-score", 0.0)), int(metrics.get("support", 0))))
    class_rows.sort(key=lambda x: (x[1], x[2]))
    print("\nLowest-F1 classes for the best model")
    print("=" * 52)
    print(f"{'class':<18} {'f1':>8} {'support':>10}")
    print("-" * 52)
    for class_name, f1, support in class_rows[:max_rows]:
        print(f"{class_name:<18} {f1:>8.4f} {support:>10}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare lightweight GestureSlide classifiers")
    parser.add_argument("--data", default="training_data/", help="Directory containing session_*.json")
    parser.add_argument("--models", nargs="+",
                        default=["mlp", "svm", "random_forest", "extra_trees", "hgb"],
                        choices=["mlp", "svm", "random_forest", "extra_trees", "hgb", "knn"],
                        help="Models to compare")
    parser.add_argument("--split-strategy", choices=("group", "random"), default="group")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--augment", action="store_true",
                        help="Augment training split before model comparison")
    parser.add_argument("--augment-factor", type=int, default=1)
    parser.add_argument("--balance-target", type=int, default=0)
    parser.add_argument("--metric", choices=("macro_f1", "accuracy", "weighted_f1"), default="macro_f1")
    parser.add_argument("--output", default="gesture_model.joblib")
    parser.add_argument("--scaler", default="gesture_scaler.joblib")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_training_dataset(args.data)
    print_dataset_audit(dataset)

    if args.split_strategy == "group":
        train_idx, test_idx = _group_split(dataset, args.test_size, args.seed)
    else:
        train_idx, test_idx = _random_split(dataset, args.test_size, args.seed)

    X_train, X_test = dataset.X[train_idx], dataset.X[test_idx]
    y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]

    if args.augment:
        X_train, y_train = _augment_training_set(
            X_train, y_train,
            augment_factor=args.augment_factor,
            balance_target=args.balance_target,
            seed=args.seed,
        )

    X_train_scaled, X_test_scaled, scaler = _scale(X_train, X_test)

    results: list[ModelResult] = []
    trained_models = {}
    predictions = {}
    class_reports = {}
    for name in args.models:
        print(f"\nTraining {name}...")
        model = build_model(name, args.seed)
        result, pred = evaluate_model(name, model, X_train_scaled, y_train, X_test_scaled, y_test)
        results.append(result)
        trained_models[name] = model
        predictions[name] = pred
        class_reports[name] = per_class_report_dict(y_test, pred)
        print(f"  accuracy={result.accuracy:.4f}, macro_f1={result.macro_f1:.4f}, train={result.train_seconds:.2f}s")

    print_results(results)
    best = max(results, key=lambda r: (getattr(r, args.metric), r.accuracy))
    best_model = trained_models[best.name]
    best_pred = predictions[best.name]

    joblib.dump(best_model, args.output)
    joblib.dump(scaler, args.scaler)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"model_comparison_{stamp}.json"
    metrics_csv_path = report_dir / f"class_metrics_{stamp}.csv"
    confusion_csv_path = report_dir / f"confusion_matrix_{best.name}_{stamp}.csv"

    report = {
        "best_model": best.name,
        "selection_metric": args.metric,
        "data": args.data,
        "split_strategy": args.split_strategy,
        "test_size": args.test_size,
        "augment": args.augment,
        "augment_factor": args.augment_factor,
        "balance_target": args.balance_target,
        "results": [asdict(r) for r in results],
        "class_reports": class_reports,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_class_metrics_csv(metrics_csv_path, class_reports)
    write_confusion_csv(confusion_csv_path, y_test, best_pred)

    print_low_class_scores(class_reports[best.name])
    print(f"\nBest model: {best.name} ({args.metric}={getattr(best, args.metric):.4f})")
    print(f"Saved model:  {args.output}")
    print(f"Saved scaler: {args.scaler}")
    print(f"Saved report: {report_path}")
    print(f"Saved class metrics: {metrics_csv_path}")
    print(f"Saved confusion matrix: {confusion_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
