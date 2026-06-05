"""
ML手势分类器训练脚本

支持三种训练模式:
  python train_model.py                         # 合成数据 (默认)
  python train_model.py --real training_data/   # 真实采集数据
  python train_model.py --mix training_data/    # 真实 + 合成数据混合
"""

import argparse
import sys
from gesture_model import GestureModel


def main():
    parser = argparse.ArgumentParser(
        description="Train ML gesture classifier")
    parser.add_argument("--real", type=str, default=None,
                        help="Path to real training data directory")
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
    parser.add_argument("--output", type=str, default="gesture_model.joblib",
                        help="Output model path")
    parser.add_argument("--scaler", type=str, default="gesture_scaler.joblib",
                        help="Output scaler path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.mix:
        print(f"Mode: Mixed (real + synthetic)")
        print(f"  Real data: {args.mix}")
        print(f"  Synthetic per class: {args.synth_per_class}")
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
        print(f"Mode: Real data only")
        print(f"  Data dir: {args.real}")
        result = GestureModel.train_from_real(
            data_dir=args.real,
            test_size=args.test_size,
            model_path=args.output,
            scaler_path=args.scaler,
            seed=args.seed,
        )
    else:
        print(f"Mode: Synthetic data")
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
