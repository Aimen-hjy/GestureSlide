#!/usr/bin/env bash
# Mature classroom pipeline for GestureSlide.
#
# It keeps the original project architecture, but adds the best-value steps:
#   1. syntax check
#   2. data audit
#   3. optional lightweight RPS import (~219 MiB, not 40GB/class)
#   4. feature augmentation
#   5. model comparison and best-model export
#
# Usage:
#   bash tools/run_project_pipeline.sh
#
# Optional:
#   USE_RPS=1 bash tools/run_project_pipeline.sh
#   BALANCE_TARGET=800 AUGMENT_FACTOR=2 bash tools/run_project_pipeline.sh
#   CLEAN_RPS=0 USE_RPS=1 bash tools/run_project_pipeline.sh  # keep previous RPS imports
#
# Important:
#   RPS is supplemental and may hurt cross-session performance. The default
#   USE_RPS=0 trains only on local GestureSlide data.

set -euo pipefail

USE_RPS="${USE_RPS:-0}"
CLEAN_RPS="${CLEAN_RPS:-1}"
RPS_ROOT="${RPS_ROOT:-datasets/rps}"
AUGMENT_FACTOR="${AUGMENT_FACTOR:-2}"
BALANCE_TARGET="${BALANCE_TARGET:-800}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-group}"
MODELS="${MODELS:-mlp svm random_forest extra_trees hgb}"

python -m py_compile \
  config.py \
  gesture_model.py \
  hand_detector.py \
  gesture_classifier.py \
  ppt_controller.py \
  action_controller.py \
  training_pipeline.py \
  train_model.py \
  tools/audit_training_data.py \
  tools/import_image_folder.py \
  tools/compare_models.py

if [[ "${CLEAN_RPS}" == "1" ]]; then
  rm -f training_data/imported/session_rps_*.json 2>/dev/null || true
fi

python tools/audit_training_data.py training_data/

if [[ "${USE_RPS}" == "1" ]]; then
  python tools/download_rps_dataset.py --output-dir "${RPS_ROOT}"

  python tools/import_image_folder.py \
    --dataset-root "${RPS_ROOT}/rps" \
    --map rock=FIST paper=OPEN_PALM scissors=PEACE_UP \
    --source-name rps_train \
    --max-per-class 1000 \
    --output-dir training_data/imported

  python tools/import_image_folder.py \
    --dataset-root "${RPS_ROOT}/rps-test-set" \
    --map rock=FIST paper=OPEN_PALM scissors=PEACE_UP \
    --source-name rps_test \
    --max-per-class 500 \
    --output-dir training_data/imported

  python tools/audit_training_data.py training_data/
fi

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

printf '\nPipeline complete. Run the demo with:\n  python main.py\n'
