#!/bin/bash
#SBATCH -J pitome_tc
#SBATCH -p i64m1tga800ue
#SBATCH -o text_classification_%j.out
#SBATCH -e text_classification_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-pitome}"
DEFAULT_MODE="${MODE:-eval}"
DEFAULT_MODEL="${MODEL:-bert-base-uncased}"
DEFAULT_ALGO="${ALGO:-pitome}"
FIRST_ARG="${1:-}"
SECOND_ARG="${2:-}"
THIRD_ARG="${3:-}"
GPU="${GPU:-0}"
ALPHA="${ALPHA:-1.0}"
TASKS="${TASKS:-imdb sst2 rotten}"
RATIOS="${RATIOS:-0.80 0.70 0.60 0.50 0.40 0.30}"

DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/datasets/imdb}"
HF_DATASET_CACHE="${HF_DATASET_CACHE:-$ROOT_DIR/datasets/huggingface_cache}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-$ROOT_DIR/model/JiaqiLee_imdb-finetuned-bert-base-uncased}"
TC_DATA_DIR="$ROOT_DIR/data/tc"
TC_CACHE_DIR="$TC_DATA_DIR/.cache"

usage() {
  cat <<USAGE
Usage:
  sbatch text_classification.sh [algo] [eval|train] [model]

Defaults:
  algo=${ALGO:-$DEFAULT_ALGO}
  mode=eval
  model=bert-base-uncased
  tasks="$TASKS"
  ratios="$RATIOS"

Examples:
  sbatch text_classification.sh
  sbatch text_classification.sh tome
  sbatch text_classification.sh pitome eval bert-base-uncased
  sbatch text_classification.sh tome eval bert-base-uncased
  sbatch --export=ALL,TASKS="imdb",RATIOS="0.90 0.80 0.70 0.60 0.50" text_classification.sh
  sbatch --export=ALL,HF_DATASET_CACHE=/path/to/huggingface_cache text_classification.sh

Backward-compatible form:
  sbatch text_classification.sh eval bert-base-uncased
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

if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi

if ! python - <<'PY' >/dev/null 2>&1
import accelerate, datasets, ml_collections, torch, transformers
PY
then
  echo "ERROR: Python dependencies are missing. Submit sbatch init_environment.sh first, then resubmit this job." >&2
  exit 1
fi

prepare_task_data() {
  local task="$1"

  unset PITOME_TC_LOCAL_MODEL

  mkdir -p "$ROOT_DIR/data"
  if [[ -L "$TC_DATA_DIR" ]]; then
    :
  elif [[ -e "$TC_DATA_DIR" ]]; then
    if [[ "$task" == "imdb" && ! -d "$TC_DATA_DIR/aclImdb" ]]; then
      echo "ERROR: $TC_DATA_DIR exists but does not contain aclImdb." >&2
      echo "Move it aside or set up data/tc so main_tc.py can find the dataset." >&2
      exit 1
    fi
  else
    ln -s "$DATASET_DIR" "$TC_DATA_DIR"
  fi

  mkdir -p "$TC_CACHE_DIR"

  if [[ "$task" == "sst2" ]]; then
    link_hf_cache_dir "stanfordnlp___sst2"
    unset HF_HUB_OFFLINE
    unset TRANSFORMERS_OFFLINE
    return
  fi

  if [[ "$task" == "rotten" ]]; then
    link_hf_cache_dir "rotten_tomatoes"
    unset HF_HUB_OFFLINE
    unset TRANSFORMERS_OFFLINE
    return
  fi

  if [[ "$task" != "imdb" ]]; then
    echo "ERROR: unsupported task '$task'. Supported tasks: imdb sst2 rotten." >&2
    exit 1
  fi

  if [[ ! -d "$DATASET_DIR/aclImdb" ]]; then
    echo "ERROR: IMDB dataset not found at $DATASET_DIR/aclImdb." >&2
    echo "Set DATASET_DIR=/path/to/imdb_parent or place aclImdb under datasets/imdb." >&2
    exit 1
  fi

  if [[ "$MODEL" == "bert-base-uncased" && -d "$LOCAL_MODEL_DIR" ]]; then
    export PITOME_TC_LOCAL_MODEL="$LOCAL_MODEL_DIR"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  fi
}

link_hf_cache_dir() {
  local cache_name="$1"
  local source_dir="$HF_DATASET_CACHE/$cache_name"
  local target_dir="$TC_CACHE_DIR/$cache_name"

  if [[ ! -d "$source_dir" ]]; then
    echo "ERROR: Hugging Face dataset cache not found at $source_dir." >&2
    echo "Run: sbatch download_datasets.sh all" >&2
    exit 1
  fi

  if [[ -L "$target_dir" || -d "$target_dir" ]]; then
    return
  fi

  ln -s "$source_dir" "$target_dir"
  echo "Created $target_dir -> $source_dir"
}

run_one() {
  local task="$1"
  local algo="$2"
  local ratio="$3"

  prepare_task_data "$task"

  echo
  echo "Running text classification:"
  echo "  mode=$MODE task=$task model=$MODEL algo=$algo ratio=$ratio alpha=$ALPHA"

  if [[ "$MODE" == "eval" ]]; then
    python main_tc.py \
      --algo "$algo" \
      --ratio "$ratio" \
      --task "$task" \
      --model "$MODEL" \
      --alpha "$ALPHA" \
      --eval
  else
    CUDA_VISIBLE_DEVICES="$GPU" python -m accelerate.commands.launch main_tc.py \
      --model "$MODEL" \
      --algo "$algo" \
      --ratio "$ratio" \
      --task "$task" \
      --alpha "$ALPHA"
  fi
}

mkdir -p "$ROOT_DIR/outputs/tc_outputs"

echo "Batch text classification:"
echo "  mode=$MODE model=$MODEL alpha=$ALPHA"
echo "  tasks=$TASKS"
echo "  baseline=none ratio=1.0"
echo "  algo=$ALGO ratios=$RATIOS"

for task in $TASKS; do
  run_one "$task" "none" "1.0"

  if [[ "$ALGO" != "none" ]]; then
    for ratio in $RATIOS; do
      run_one "$task" "$ALGO" "$ratio"
    done
  fi
done

echo
echo "All text classification runs finished."
