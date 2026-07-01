"""Compare lightweight classifiers and save the best GestureSlide model.

This script keeps the project lightweight: all models consume the same 69-D
MediaPipe feature vector, so it is suitable for a classroom PPT-control demo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> ModelResult:
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started
    pred = model.predict(X_test)
    return ModelResult(
        name=name,
        accuracy=float(accuracy_score(y_test, pred)),
        macro_f1=float(f1_score(y_test, pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        train_seconds=float(train_seconds),
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
    )


def print_results(results: list[ModelResult]) -> None:
    print("\nModel comparison")
    print("=" * 78)
    print(f"{'model':<16} {'accuracy':>10} {'macro_f1':>10} {'weighted_f1':>12} {'train_s':>10}")
    print("-" * 78)
    for r in sorted(results, key=lambda x: (x.macro_f1, x.accuracy), reverse=True):
        print(f"{r.name:<16} {r.accuracy:>10.4f} {r.macro_f1:>10.4f} {r.weighted_f1:>12.4f} {r.train_seconds:>10.2f}")


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
    for name in args.models:
        print(f"\nTraining {name}...")
        model = build_model(name, args.seed)
        result = evaluate_model(name, model, X_train_scaled, y_train, X_test_scaled, y_test)
        results.append(result)
        trained_models[name] = model
        print(f"  accuracy={result.accuracy:.4f}, macro_f1={result.macro_f1:.4f}, train={result.train_seconds:.2f}s")

    print_results(results)
    best = max(results, key=lambda r: (getattr(r, args.metric), r.accuracy))
    best_model = trained_models[best.name]

    joblib.dump(best_model, args.output)
    joblib.dump(scaler, args.scaler)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"model_comparison_{stamp}.json"
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
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nBest model: {best.name} ({args.metric}={getattr(best, args.metric):.4f})")
    print(f"Saved model:  {args.output}")
    print(f"Saved scaler: {args.scaler}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
