#!/usr/bin/env bash
# Download a selected HaGRID/HaGRIDv2 subset, convert it to GestureSlide
# features, and train the small MLP model.
#
# Usage:
#   bash tools/run_hagrid_pipeline.sh
#
# Optional environment variables:
#   SAVE_ROOT=datasets/hagrid
#   HAGRID_REPO=external/hagrid
#   TARGETS="fist palm stop like peace peace_inverted three no_gesture"
#   SPLITS="train val"
#   MAX_PER_CLASS=2000
#   INCLUDE_DIRECTION=0   # set to 1 to include point/one direction auto-labeling
#   TRAINING_OUT=training_data/imported
#   DOWNLOAD_ANNOTATIONS=0 # set to 1 if the upstream annotation URL is accessible
#
# Note: MAX_PER_CLASS limits how many already-downloaded images are imported.
# The official downloader still downloads each requested class archive.

set -euo pipefail

SAVE_ROOT="${SAVE_ROOT:-datasets/hagrid}"
HAGRID_REPO="${HAGRID_REPO:-external/hagrid}"
TARGETS="${TARGETS:-fist palm stop like peace peace_inverted three no_gesture}"
SPLITS="${SPLITS:-train val}"
MAX_PER_CLASS="${MAX_PER_CLASS:-2000}"
INCLUDE_DIRECTION="${INCLUDE_DIRECTION:-0}"
TRAINING_OUT="${TRAINING_OUT:-training_data/imported}"
DOWNLOAD_ANNOTATIONS="${DOWNLOAD_ANNOTATIONS:-0}"

mkdir -p external datasets "${SAVE_ROOT}" "${TRAINING_OUT}"

if [[ ! -d "${HAGRID_REPO}/.git" ]]; then
  git clone https://github.com/hukenovs/hagrid.git "${HAGRID_REPO}"
else
  git -C "${HAGRID_REPO}" pull --ff-only
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if [[ -f "${HAGRID_REPO}/requirements.txt" ]]; then
  python -m pip install -r "${HAGRID_REPO}/requirements.txt"
fi

# The upstream annotations URL may return 403. Import can work without it by
# scanning class image folders directly, so annotations are opt-in.
if [[ "${DOWNLOAD_ANNOTATIONS}" == "1" ]]; then
  python "${HAGRID_REPO}/download.py" --save_path "${SAVE_ROOT}" --annotations || \
    echo "[warn] Annotation download failed; continuing with image-folder scan."
fi

# The official downloader supports --dataset and --targets.
# Dataset archives are large; make sure you have enough disk space before this step.
python "${HAGRID_REPO}/download.py" --save_path "${SAVE_ROOT}" --dataset --targets ${TARGETS}

# Unzip archives if any were downloaded. -n avoids overwriting already extracted files.
find "${SAVE_ROOT}" -type f -name "*.zip" -print0 | while IFS= read -r -d '' archive; do
  unzip -n "${archive}" -d "${SAVE_ROOT}"
done

IMPORT_ARGS=(
  --dataset-root "${SAVE_ROOT}"
  --annotations-root "${SAVE_ROOT}/hagrid_annotations"
  --output-dir "${TRAINING_OUT}"
  --splits ${SPLITS}
  --targets ${TARGETS}
  --max-per-class "${MAX_PER_CLASS}"
)

if [[ "${INCLUDE_DIRECTION}" == "1" ]]; then
  IMPORT_ARGS+=(--include-direction-classes)
fi

python tools/import_hagrid.py "${IMPORT_ARGS[@]}"
python tools/audit_training_data.py training_data/
python train_model.py --real training_data/ --split-strategy group
