#!/bin/bash
#SBATCH -J pitome_env
#SBATCH -p i64m1tga800ue
#SBATCH -o init_environment_%j.out
#SBATCH -e init_environment_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-pitome}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_BACKEND="${PYTORCH_BACKEND:-auto}"
CUDA_VERSION="${CUDA_VERSION:-11.8}"

usage() {
  cat <<USAGE
Usage:
  sbatch init_environment.sh

Environment variables:
  ENV_NAME=pitome              Conda environment name
  PYTHON_VERSION=3.10          Python version
  PYTORCH_BACKEND=auto|cuda|cpu Install CUDA or CPU PyTorch
  CUDA_VERSION=11.8            CUDA version when PYTORCH_BACKEND=cuda

Examples:
  sbatch init_environment.sh
  sbatch --export=ALL,PYTORCH_BACKEND=cpu init_environment.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Removing existing conda environment: $ENV_NAME"
  conda env remove -y -n "$ENV_NAME"
fi

echo "Creating conda environment: $ENV_NAME"
conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"

conda activate "$ENV_NAME"

python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
conda remove -y pytorch torchvision torchaudio pytorch-cuda cpuonly >/dev/null 2>&1 || true

if [[ "$PYTORCH_BACKEND" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    PYTORCH_BACKEND="cuda"
  else
    PYTORCH_BACKEND="cpu"
  fi
fi

if [[ "$PYTORCH_BACKEND" == "cpu" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    conda install -y pytorch torchvision torchaudio -c pytorch
  else
    conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
  fi
elif [[ "$PYTORCH_BACKEND" == "cuda" ]]; then
  conda install -y pytorch torchvision torchaudio "pytorch-cuda=$CUDA_VERSION" -c pytorch -c nvidia
else
  echo "ERROR: PYTORCH_BACKEND must be 'auto', 'cuda', or 'cpu', got '$PYTORCH_BACKEND'." >&2
  exit 1
fi

conda install -y "mkl<2024.1" "intel-openmp<2024.1"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python - <<'PY'
import accelerate
import datasets
import ml_collections
import torch
import transformers

print("Environment check passed.")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
PY

echo
echo "Done. You can now submit:"
echo "  sbatch text_classification.sh"
