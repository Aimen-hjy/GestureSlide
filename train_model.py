"""
ML手势分类器训练脚本

使用合成数据训练 MLP 模型，替换硬编码的静态手势规则。

用法:
  python train_model.py                      # 默认参数
  python train_model.py --samples 2000       # 每类2000样本
  python train_model.py --noise 0.02         # 自定义噪声
  python train_model.py --output my_model.joblib
"""

import argparse
import sys
from gesture_model import GestureModel, N_CLASSES


def main():
    parser = argparse.ArgumentParser(
        description="Train ML gesture classifier from synthetic data")
    parser.add_argument("--samples", type=int, default=1500,
                        help="Samples per gesture class (default: 1500)")
    parser.add_argument("--noise", type=float, default=0.015,
                        help="Gaussian noise std for data augmentation (default: 0.015)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test split ratio (default: 0.2)")
    parser.add_argument("--output", type=str, default="gesture_model.joblib",
                        help="Output model path (default: gesture_model.joblib)")
    parser.add_argument("--scaler", type=str, default="gesture_scaler.joblib",
                        help="Output scaler path (default: gesture_scaler.joblib)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    total = args.samples * N_CLASSES
    print(f"Configuration:")
    print(f"  Samples per class: {args.samples}")
    print(f"  Total samples:     {total}")
    print(f"  Noise std:         {args.noise}")
    print(f"  Test split:        {args.test_size}")
    print(f"  Output model:      {args.output}")
    print(f"  Output scaler:     {args.scaler}")
    print()

    result = GestureModel.train_from_synthetic(
        n_per_class=args.samples,
        noise_std=args.noise,
        test_size=args.test_size,
        model_path=args.output,
        scaler_path=args.scaler,
        seed=args.seed,
    )

    print(f"\nDone! {total} samples → {result['accuracy']:.2%} accuracy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
