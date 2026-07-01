"""
ML手势分类器训练脚本

支持三种训练模式:
  python train_model.py                                  # 合成数据 (默认)
  python train_model.py --real training_data/            # 真实/导入数据
  python train_model.py --real training_data/ --augment  # 真实数据 + 轻量特征增强
  python train_model.py --mix training_data/             # 真实 + 合成数据混合
"""

import argparse
import sys

from gesture_model import GestureModel
from training_pipeline import train_real_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train ML gesture classifier")
    parser.add_argument("--real", type=str, default=None,
                        help="Path to real/imported training data directory")
    parser.add_argument("--mix", type=str, default=None,
                        help="Path to real data, mixed with synthetic data")
    parser.add_argument("--samples", type=int, default=1500,
                        help="Synthetic samples per class (default: 1500)")
    parser.add_argument("--synth-per-class", type=int, default=500,
                        help="Synthetic samples per class in mix mode (default: 500)")
    parser.add_argument("--noise", type=float, default=0.015,
                        help="Gaussian noise std (default: 0.015)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test split ratio (default: 0.2)")
    parser.add_argument("--split-strategy", choices=("group", "random"),
                        default="group",
                        help=("Split strategy for --real mode. 'group' keeps "
                              "whole sessions/source users out of train set; "
                              "'random' is faster but optimistic. Default: group"))
    parser.add_argument("--augment", action="store_true",
                        help="Apply lightweight pose-preserving feature augmentation to the training split")
    parser.add_argument("--augment-factor", type=int, default=1,
                        help="Number of augmented variants per training sample when --augment is used")
    parser.add_argument("--balance-target", type=int, default=0,
                        help="Oversample weak classes up to this training count using augmentation; 0 disables")
    parser.add_argument("--output", type=str, default="gesture_model.joblib",
                        help="Output model path")
    parser.add_argument("--scaler", type=str, default="gesture_scaler.joblib",
                        help="Output scaler path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.mix:
        print("Mode: Mixed (real + synthetic)")
        print(f"  Real data: {args.mix}")
        print(f"  Synthetic per class: {args.synth_per_class}")
        print("  Note: --mix still uses the legacy stratified split. For reliable "
              "session/user holdout evaluation, prefer --real with imported data.")
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
        print("Mode: Real/imported data")
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
        print("Mode: Synthetic data")
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
