#!/usr/bin/env bash
# Final local-data training pipeline for GestureSlide.
#
# The final demo model uses local camera data only. This script intentionally
# keeps the training path compact and reproducible:
#   1. syntax check
#   2. local training-data audit
#   3. feature augmentation
#   4. class balancing
#   5. lightweight model comparison and best-model export
#
# Usage:
#   bash tools/run_project_pipeline.sh
#
# Tunable examples:
#   BALANCE_TARGET=800 AUGMENT_FACTOR=2 bash tools/run_project_pipeline.sh
#   MODELS="hgb extra_trees random_forest" bash tools/run_project_pipeline.sh
#
# If training_data/imported contains old experiment files, this script stops by
# default. Remove/move those files for the final local-only model, or explicitly
# set INCLUDE_IMPORTED=1 for a non-final experiment.

set -euo pipefail

AUGMENT_FACTOR="${AUGMENT_FACTOR:-2}"
BALANCE_TARGET="${BALANCE_TARGET:-800}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-group}"
MODELS="${MODELS:-mlp svm random_forest extra_trees hgb}"
INCLUDE_IMPORTED="${INCLUDE_IMPORTED:-0}"

python -m py_compile \
  main.py \
  config.py \
  gesture_model.py \
  hand_detector.py \
  gesture_classifier.py \
  ppt_controller.py \
  action_controller.py \
  hud_window.py \
  training_pipeline.py \
  train_model.py \
  tools/audit_training_data.py \
  tools/compare_models.py \
  tools/evaluate_geometry_direction.py

mkdir -p training_data/imported reports

if [[ "${INCLUDE_IMPORTED}" != "1" ]]; then
  if find training_data/imported -maxdepth 1 -type f -name 'session_*.json' | grep -q .; then
    echo "[error] training_data/imported contains session_*.json files."
    echo "        The final pipeline expects local camera data only."
    echo "        Move or remove those imported files, then rerun:"
    echo "          rm -f training_data/imported/session_*.json"
    echo "        For a non-final experiment, rerun with INCLUDE_IMPORTED=1."
    exit 1
  fi
fi

python tools/audit_training_data.py training_data/

python tools/compare_models.py \
  --data training_data/ \
  --split-strategy "${SPLIT_STRATEGY}" \
  --augment \
  --augment-factor "${AUGMENT_FACTOR}" \
  --balance-target "${BALANCE_TARGET}" \
  --models ${MODELS} \
  --metric macro_f1 \
  --output gesture_model.joblib \
  --scaler gesture_scaler.joblib

printf '\nPipeline complete. Recommended demo command:\n  python main.py --headless --hud\n'
