#!/usr/bin/env bash
# Mature classroom pipeline for GestureSlide.
#
# It keeps the original project architecture, but adds the best-value steps:
#   1. syntax check
#   2. data audit
#   3. optional web data import: HaGRID no_gesture for NONE, RPS for 3 easy classes
#   4. feature augmentation
#   5. model comparison and best-model export
#
# Usage:
#   bash tools/run_project_pipeline.sh
#
# Optional:
#   USE_HAGRID_NONE=1 bash tools/run_project_pipeline.sh
#   HAGRID_SKIP_DOWNLOAD=1 USE_HAGRID_NONE=1 bash tools/run_project_pipeline.sh
#   USE_RPS=1 bash tools/run_project_pipeline.sh
#   BALANCE_TARGET=800 AUGMENT_FACTOR=2 bash tools/run_project_pipeline.sh
#   CLEAN_RPS=0 USE_RPS=1 bash tools/run_project_pipeline.sh
#   CLEAN_HAGRID=0 USE_HAGRID_NONE=1 bash tools/run_project_pipeline.sh
#
# Important:
#   HaGRID no_gesture is the recommended web-data supplement because the local
#   dataset is weak in NONE. RPS is optional and may hurt full 11-class balance.

set -euo pipefail

USE_RPS="${USE_RPS:-0}"
USE_HAGRID_NONE="${USE_HAGRID_NONE:-0}"
HAGRID_SKIP_DOWNLOAD="${HAGRID_SKIP_DOWNLOAD:-0}"
CLEAN_RPS="${CLEAN_RPS:-1}"
CLEAN_HAGRID="${CLEAN_HAGRID:-1}"
RPS_ROOT="${RPS_ROOT:-datasets/rps}"
HAGRID_ROOT="${HAGRID_ROOT:-datasets/hagrid}"
HAGRID_REPO="${HAGRID_REPO:-external/hagrid}"
HAGRID_NONE_MAX="${HAGRID_NONE_MAX:-1500}"
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
  tools/import_hagrid.py \
  tools/compare_models.py

mkdir -p training_data/imported external datasets "${HAGRID_ROOT}"

if [[ "${CLEAN_RPS}" == "1" ]]; then
  rm -f training_data/imported/session_rps_*.json 2>/dev/null || true
fi
if [[ "${CLEAN_HAGRID}" == "1" ]]; then
  rm -f training_data/imported/session_hagrid_*.json 2>/dev/null || true
fi

python tools/audit_training_data.py training_data/

if [[ "${USE_HAGRID_NONE}" == "1" ]]; then
  if [[ "${HAGRID_SKIP_DOWNLOAD}" != "1" ]]; then
    if [[ ! -d "${HAGRID_REPO}/.git" ]]; then
      git clone https://github.com/hukenovs/hagrid.git "${HAGRID_REPO}"
    else
      git -C "${HAGRID_REPO}" pull --ff-only
    fi

    # Download only the lightweight no_gesture archive (~493.9 MB), not 40GB gesture classes.
    python "${HAGRID_REPO}/download.py" --save_path "${HAGRID_ROOT}" --dataset --targets no_gesture
  else
    echo "[info] HAGRID_SKIP_DOWNLOAD=1: using existing files under ${HAGRID_ROOT}"
  fi

  find "${HAGRID_ROOT}" -type f -name "*.zip" -print0 | while IFS= read -r -d '' archive; do
    unzip -n "${archive}" -d "${HAGRID_ROOT}"
  done

  # Some HaGRID no_gesture archives extract images flat into HAGRID_ROOT rather
  # than HAGRID_ROOT/no_gesture. Normalize that layout for the importer.
  mkdir -p "${HAGRID_ROOT}/no_gesture"
  find "${HAGRID_ROOT}" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) -print0 |
    while IFS= read -r -d '' image; do
      mv -n "${image}" "${HAGRID_ROOT}/no_gesture/"
    done

  python tools/import_hagrid.py \
    --dataset-root "${HAGRID_ROOT}" \
    --annotations-root "${HAGRID_ROOT}/hagrid_annotations" \
    --output-dir training_data/imported \
    --splits all \
    --targets no_gesture \
    --max-per-class "${HAGRID_NONE_MAX}"

  python tools/audit_training_data.py training_data/
fi

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
