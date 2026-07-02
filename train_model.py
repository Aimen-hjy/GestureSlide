"""
单模型训练入口。

最终演示模型推荐使用：
  bash tools/run_project_pipeline.sh

本脚本保留用于调试单个模型或兼容旧流程：
  python train_model.py --real training_data/ --augment --balance-target 800
  python train_model.py --real training_data/ --split-strategy group
  python train_model.py                         # 旧的合成数据调试模式
"""

import argparse
import sys

from gesture_model import GestureModel
from training_pipeline import train_real_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train a single GestureSlide ML classifier")
    parser.add_argument("--real", type=str, default=None,
                        help="Path to local training data directory")
    parser.add_argument("--mix", type=str, default=None,
                        help="Legacy mode: real data mixed with synthetic samples")
    parser.add_argument("--samples", type=int, default=1500,
                        help="Synthetic samples per class for legacy synthetic mode")
    parser.add_argument("--synth-per-class", type=int, default=500,
                        help="Synthetic samples per class in legacy mix mode")
    parser.add_argument("--noise", type=float, default=0.015,
                        help="Gaussian noise std for synthetic samples")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test split ratio")
    parser.add_argument("--split-strategy", choices=("group", "random"),
                        default="group",
                        help=("Split strategy for --real mode. 'group' keeps "
                              "whole sessions out of train set; 'random' is "
                              "faster but optimistic. Default: group"))
    parser.add_argument("--augment", action="store_true",
                        help="Apply lightweight feature augmentation to the training split")
    parser.add_argument("--augment-factor", type=int, default=1,
                        help="Number of augmented variants per training sample when --augment is used")
    parser.add_argument("--balance-target", type=int, default=0,
                        help="Oversample weak classes up to this training count using augmentation; 0 disables")
    parser.add_argument("--output", type=str, default="gesture_model.joblib",
                        help="Output model path")
    parser.add_argument("--scaler", type=str, default="gesture_scaler.joblib",
                        help="Output scaler path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    if args.mix:
        print("Mode: Legacy mixed data (real + synthetic)")
        print(f"  Real data: {args.mix}")
        print(f"  Synthetic per class: {args.synth_per_class}")
        print("  Note: final demo training should use tools/run_project_pipeline.sh")
        result = GestureModel.train_from_mixed(
            data_dir=args.mix,
            synth_per_class=args.synth_per_class,
            noise_std=args.noise,
            test_size=args.test_size,
            model_path=args.output,
            scaler_path=args.scaler,
            seed=args.seed,
        )
    elif args.real:
        print("Mode: Local real data")
        print(f"  Data dir: {args.real}")
        print(f"  Split strategy: {args.split_strategy}")
        print(f"  Augmentation: {'on' if args.augment else 'off'}")
        result = train_real_dataset(
            data_dir=args.real,
            test_size=args.test_size,
            model_path=args.output,
            scaler_path=args.scaler,
            seed=args.seed,
            split_strategy=args.split_strategy,
            augment=args.augment,
            augment_factor=args.augment_factor,
            balance_target=args.balance_target,
        )
    else:
        print("Mode: Legacy synthetic data")
        print("  Note: this mode is kept for quick debugging, not for final demo training.")
        print(f"  Samples per class: {args.samples}")
        result = GestureModel.train_from_synthetic(
            n_per_class=args.samples,
            noise_std=args.noise,
            test_size=args.test_size,
            model_path=args.output,
            scaler_path=args.scaler,
            seed=args.seed,
        )

    print(f"\nDone! {result['n_samples']} samples → "
          f"{result['accuracy']:.2%} accuracy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
