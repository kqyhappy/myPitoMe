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

Notes:
  ImageNet-1k on Hugging Face is gated. Make sure your account has access and
  either run 'huggingface-cli login' beforehand or pass HF_TOKEN through sbatch.
  You can also store the token in a private file and pass HF_TOKEN_FILE.
  Passing HF_TOKEN only works after Hugging Face has approved your access.
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

echo "Caching Hugging Face dataset 'imagenet-1k' under $IC_CACHE_DIR"
IC_CACHE_DIR="$IC_CACHE_DIR" python - <<'PY'
import os
import sys
from datasets import load_dataset

cache_dir = os.environ["IC_CACHE_DIR"]
token = os.environ.get("HF_TOKEN") or None

kwargs = {"cache_dir": cache_dir}
if token:
    kwargs["token"] = token

try:
    dataset = load_dataset("imagenet-1k", **kwargs)
except Exception as exc:
    message = str(exc)
    if "gated dataset" in message or "ask for access" in message:
        print(
            "ERROR: Hugging Face denied access to the gated dataset 'imagenet-1k'.\n"
            "Open https://huggingface.co/datasets/imagenet-1k and request access first.\n"
            "After access is approved, either run 'huggingface-cli login' on the cluster\n"
            "or submit with: sbatch --export=ALL,HF_TOKEN=hf_xxx scripts/env/download_ic_datasets.sh",
            file=sys.stderr,
        )
        sys.exit(2)
    raise

print(f"Cached imagenet-1k at {cache_dir}")
for split, split_dataset in dataset.items():
    print(f"{split}: {len(split_dataset)} examples")
PY

echo
echo "ImageNet-1k download finished."
