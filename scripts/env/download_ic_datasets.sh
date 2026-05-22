#!/bin/bash
#SBATCH -J pitome_ic_data
#SBATCH -p i64m512ue
#SBATCH -o download_ic_dataset_%j.out
#SBATCH -e download_ic_dataset_%j.err
#SBATCH -n 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-pitome}"
DATASET="${1:-${DATASET:-imagenet-1k}}"
IC_CACHE_DIR="${IC_CACHE_DIR:-$ROOT_DIR/data/ic}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"
IMAGENET_EXPORT_DIR="${IMAGENET_EXPORT_DIR:-}"
UPLOAD_TO="${UPLOAD_TO:-}"
UPLOAD_SOURCE="${UPLOAD_SOURCE:-}"
DELETE_REMOTE="${DELETE_REMOTE:-0}"

usage() {
  cat <<USAGE
Usage:
  bash scripts/env/download_ic_datasets.sh [imagenet-1k]
  sbatch scripts/env/download_ic_datasets.sh [imagenet-1k]

Defaults:
  dataset=imagenet-1k
  IC_CACHE_DIR=$ROOT_DIR/data/ic

Examples:
  sbatch scripts/env/download_ic_datasets.sh
  sbatch --export=ALL,IC_CACHE_DIR=/path/to/cache scripts/env/download_ic_datasets.sh
  sbatch --export=ALL,HF_TOKEN=hf_xxx scripts/env/download_ic_datasets.sh
  sbatch --export=ALL,HF_TOKEN_FILE=/path/to/hf_token scripts/env/download_ic_datasets.sh
  sbatch --export=ALL,UPLOAD_TO=user@server:/data/imagenet1k/ scripts/env/download_ic_datasets.sh
  sbatch --export=ALL,IMAGENET_EXPORT_DIR=/data/imagenet1k_imagefolder scripts/env/download_ic_datasets.sh

Notes:
  ImageNet-1k on Hugging Face is gated. Make sure your account has access and
  either run 'huggingface-cli login' beforehand or pass HF_TOKEN through sbatch.
  You can also store the token in a private file and pass HF_TOKEN_FILE.
  Passing HF_TOKEN only works after Hugging Face has approved your access.

  UPLOAD_TO uses rsync syntax, for example user@server:/data/imagenet1k/.
  Set DELETE_REMOTE=1 to mirror-delete files on the remote target.
USAGE
}

if [[ "$DATASET" == "-h" || "$DATASET" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$DATASET" != "imagenet-1k" && "$DATASET" != "imagenet" ]]; then
  echo "ERROR: dataset must be imagenet-1k. Got '$DATASET'." >&2
  usage
  exit 1
fi

if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi

if [[ -z "${HF_TOKEN:-}" && -n "$HF_TOKEN_FILE" ]]; then
  if [[ ! -f "$HF_TOKEN_FILE" ]]; then
    echo "ERROR: HF_TOKEN_FILE does not exist: $HF_TOKEN_FILE" >&2
    exit 1
  fi
  HF_TOKEN="$(head -n 1 "$HF_TOKEN_FILE" | tr -d '[:space:]')"
  export HF_TOKEN
fi

mkdir -p "$IC_CACHE_DIR"

if ! python - <<'PY' >/dev/null 2>&1
import datasets
PY
then
  echo "ERROR: Python package 'datasets' is missing. Submit sbatch scripts/env/init_environment.sh first." >&2
  exit 1
fi

cmd=(
  python scripts/env/download_imagenet1k.py
  --dataset-name "$DATASET"
  --cache-dir "$IC_CACHE_DIR"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  cmd+=(--hf-token "$HF_TOKEN")
fi

if [[ -n "$HF_TOKEN_FILE" ]]; then
  cmd+=(--hf-token-file "$HF_TOKEN_FILE")
fi

if [[ -n "$IMAGENET_EXPORT_DIR" ]]; then
  cmd+=(--export-imagefolder "$IMAGENET_EXPORT_DIR")
fi

if [[ -n "$UPLOAD_TO" ]]; then
  cmd+=(--upload-to "$UPLOAD_TO")
fi

if [[ -n "$UPLOAD_SOURCE" ]]; then
  cmd+=(--upload-source "$UPLOAD_SOURCE")
fi

if [[ "$DELETE_REMOTE" == "1" || "$DELETE_REMOTE" == "true" ]]; then
  cmd+=(--delete-remote)
fi

echo "Caching Hugging Face dataset '$DATASET' under $IC_CACHE_DIR"
"${cmd[@]}"

echo
echo "ImageNet-1k download finished."
