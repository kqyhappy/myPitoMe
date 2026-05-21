#!/bin/bash
#SBATCH -J pitome_ic
#SBATCH -p i64m1tga800ue
#SBATCH -o image_classification_%j.out
#SBATCH -e image_classification_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-pitome}"
DEFAULT_MODE="${MODE:-eval}"
DEFAULT_MODEL="${MODEL:-DEIT-T-224}"
DEFAULT_ALGO="${ALGO:-pitome}"
FIRST_ARG="${1:-}"
SECOND_ARG="${2:-}"
THIRD_ARG="${3:-}"

GPU="${GPU:-0}"
DATA_SET="${DATA_SET:-IMNET}"
BATCH_SIZE="${BATCH_SIZE:-100}"
EPOCHS="${EPOCHS:-10}"
RATIOS="${RATIOS:-0.9125 0.8750 0.8375 0.8000}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/log/ic_temp}"
IC_CACHE_DIR="${IC_CACHE_DIR:-$ROOT_DIR/data/ic}"
NUM_WORKERS="${NUM_WORKERS:-10}"

usage() {
  cat <<USAGE
Usage:
  sbatch scripts/tasks/image_classification.sh [algo] [eval|train] [model]

Defaults:
  algo=${ALGO:-$DEFAULT_ALGO}
  mode=eval
  model=DEIT-T-224
  data_set=$DATA_SET
  ratios="$RATIOS"

Supported models:
  DEIT-T-224 DEIT-S-224 DEIT-B-224 DEIT-T-384 DEIT-S-384 DEIT-B-384
  MAE-B-224 MAE-L-224 MAE-H-224

Examples:
  sbatch scripts/tasks/image_classification.sh
  sbatch scripts/tasks/image_classification.sh tome
  sbatch scripts/tasks/image_classification.sh pitome eval DEIT-T-224
  sbatch --export=ALL,RATIOS="0.90 0.80 0.70",BATCH_SIZE=64 scripts/tasks/image_classification.sh

Backward-compatible form:
  sbatch scripts/tasks/image_classification.sh eval DEIT-T-224
USAGE
}

if [[ "$FIRST_ARG" == "-h" || "$FIRST_ARG" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$FIRST_ARG" == "eval" || "$FIRST_ARG" == "train" ]]; then
  ALGO="$DEFAULT_ALGO"
  MODE="$FIRST_ARG"
  MODEL="${SECOND_ARG:-$DEFAULT_MODEL}"
else
  ALGO="${FIRST_ARG:-$DEFAULT_ALGO}"
  MODE="${SECOND_ARG:-$DEFAULT_MODE}"
  MODEL="${THIRD_ARG:-$DEFAULT_MODEL}"
fi

if [[ "$MODE" != "eval" && "$MODE" != "train" ]]; then
  echo "ERROR: mode must be 'eval' or 'train', got '$MODE'." >&2
  usage
  exit 1
fi

case "$ALGO" in
  pitome|tome|none)
    ;;
  *)
    echo "ERROR: unsupported algo '$ALGO'. Supported algos: pitome tome none." >&2
    usage
    exit 1
    ;;
esac

case "$MODEL" in
  DEIT-T-224|DEIT-S-224|DEIT-B-224|DEIT-T-384|DEIT-S-384|DEIT-B-384|MAE-B-224|MAE-L-224|MAE-H-224)
    ;;
  *)
    echo "ERROR: unsupported model '$MODEL'." >&2
    usage
    exit 1
    ;;
esac

if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi

if ! python - <<'PY' >/dev/null 2>&1
import accelerate, datasets, skimage, timm, torch, torchvision
PY
then
  echo "ERROR: Python dependencies are missing. Submit sbatch scripts/env/init_environment.sh first, then resubmit this job." >&2
  exit 1
fi

prepare_imagenet_data() {
  if [[ "$DATA_SET" != "IMNET" ]]; then
    return
  fi

  if [[ ! -d "$IC_CACHE_DIR" ]]; then
    echo "ERROR: ImageNet-1k cache not found at $IC_CACHE_DIR." >&2
    echo "Run: sbatch scripts/env/download_ic_datasets.sh" >&2
    exit 1
  fi

  if [[ -z "$(find "$IC_CACHE_DIR" -mindepth 1 -maxdepth 2 -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: ImageNet-1k cache at $IC_CACHE_DIR looks empty." >&2
    echo "Run: sbatch scripts/env/download_ic_datasets.sh" >&2
    exit 1
  fi
}

run_one() {
  local algo="$1"
  local ratio="$2"

  prepare_imagenet_data

  echo
  echo "Running image classification:"
  echo "  mode=$MODE data_set=$DATA_SET model=$MODEL algo=$algo ratio=$ratio batch_size=$BATCH_SIZE"

  local common_args=(
    --model "$MODEL"
    --algo "$algo"
    --ratio "$ratio"
    --data-set "$DATA_SET"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --num_workers "$NUM_WORKERS"
    --output_dir "$OUTPUT_DIR"
  )

  if [[ "$MODE" == "eval" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python main_ic.py "${common_args[@]}" --eval
  else
    CUDA_VISIBLE_DEVICES="$GPU" python -m accelerate.commands.launch main_ic.py "${common_args[@]}"
  fi
}

mkdir -p "$ROOT_DIR/outputs/ic_outputs" "$OUTPUT_DIR"

echo "Batch image classification:"
echo "  mode=$MODE data_set=$DATA_SET model=$MODEL"
echo "  cache=$IC_CACHE_DIR"
echo "  baseline=none ratio=1.0"
echo "  algo=$ALGO ratios=$RATIOS"

run_one "none" "1.0"

if [[ "$ALGO" != "none" ]]; then
  for ratio in $RATIOS; do
    run_one "$ALGO" "$ratio"
  done
fi

echo
echo "All image classification runs finished."
