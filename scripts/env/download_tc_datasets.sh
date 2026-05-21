#!/bin/bash
#SBATCH -J pitome_data
#SBATCH -p i64m1tga800ue
#SBATCH -o download_datasets_%j.out
#SBATCH -e download_datasets_%j.err
#SBATCH -n 4

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-pitome}"
DATASET="${1:-${DATASET:-imdb}}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/datasets}"
HF_DATASET_CACHE="${HF_DATASET_CACHE:-$DATA_ROOT/huggingface_cache}"
TC_DATA_DIR="$ROOT_DIR/data/tc"

usage() {
  cat <<USAGE
Usage:
  sbatch download_datasets.sh [imdb|sst2|rotten|all]

Defaults:
  dataset=imdb
  DATA_ROOT=$ROOT_DIR/datasets
  HF_DATASET_CACHE=$ROOT_DIR/datasets/huggingface_cache

Examples:
  sbatch download_datasets.sh
  sbatch download_datasets.sh imdb
  sbatch download_datasets.sh all
  sbatch --export=ALL,DATA_ROOT=/path/to/datasets download_datasets.sh imdb
  sbatch --export=ALL,HF_DATASET_CACHE=/path/to/huggingface_cache download_datasets.sh all
USAGE
}

if [[ "$DATASET" == "-h" || "$DATASET" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$DATASET" != "imdb" && "$DATASET" != "sst2" && "$DATASET" != "rotten" && "$DATASET" != "all" ]]; then
  echo "ERROR: dataset must be one of imdb, sst2, rotten, all. Got '$DATASET'." >&2
  usage
  exit 1
fi

if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi

mkdir -p "$DATA_ROOT"

download_imdb() {
  local imdb_dir="$DATA_ROOT/imdb"
  local tar_file="$imdb_dir/aclImdb_v1.tar.gz"
  local url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

  mkdir -p "$imdb_dir"

  if [[ -d "$imdb_dir/aclImdb/train" && -d "$imdb_dir/aclImdb/test" ]]; then
    echo "IMDB already exists at $imdb_dir/aclImdb"
  else
    echo "Downloading IMDB to $imdb_dir"
    if command -v wget >/dev/null 2>&1; then
      wget -c -O "$tar_file" "$url"
    elif command -v curl >/dev/null 2>&1; then
      curl -L -C - -o "$tar_file" "$url"
    else
      echo "ERROR: neither wget nor curl is available." >&2
      exit 1
    fi

    tar -xzf "$tar_file" -C "$imdb_dir"
  fi

  mkdir -p "$ROOT_DIR/data"
  if [[ -L "$TC_DATA_DIR" ]]; then
    echo "data/tc symlink already exists."
  elif [[ -e "$TC_DATA_DIR" ]]; then
    echo "data/tc already exists. Keeping it unchanged."
  else
    ln -s "$imdb_dir" "$TC_DATA_DIR"
    echo "Created data/tc -> $imdb_dir"
  fi
}

download_hf_text_dataset() {
  local hf_name="$1"
  local cache_dir="$HF_DATASET_CACHE"

  mkdir -p "$cache_dir"

  if ! python - <<'PY' >/dev/null 2>&1
import datasets
PY
  then
    echo "ERROR: Python package 'datasets' is missing. Submit sbatch init_environment.sh first." >&2
    exit 1
  fi

  echo "Caching Hugging Face dataset '$hf_name' under $cache_dir"
  HF_NAME="$hf_name" CACHE_DIR="$cache_dir" python - <<'PY'
import os
from datasets import load_dataset

name = os.environ["HF_NAME"]
cache_dir = os.environ["CACHE_DIR"]
load_dataset(name, cache_dir=cache_dir)
print(f"Cached {name} at {cache_dir}")
PY
}

case "$DATASET" in
  imdb)
    download_imdb
    ;;
  sst2)
    download_hf_text_dataset "stanfordnlp/sst2"
    ;;
  rotten)
    download_hf_text_dataset "rotten_tomatoes"
    ;;
  all)
    download_imdb
    download_hf_text_dataset "stanfordnlp/sst2"
    download_hf_text_dataset "rotten_tomatoes"
    ;;
esac

echo
echo "Dataset download finished."
