#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="flickr"
MODEL="blip"
ALGOS=("none" "tome" "pitome")
RATIOS=("0.925")
CFG="scripts/eval_scripts/blip_itr_flickr.yml"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

for algo in "${ALGOS[@]}"; do
  if [[ "$algo" == "none" ]]; then
    run_cmd python -m main_itr \
      --cfg-path "$CFG" \
      --dataset "$DATASET" \
      --model "$MODEL" \
      --algo "$algo" \
      --ratio 1.0 \
      --eval
    continue
  fi

  for ratio in "${RATIOS[@]}"; do
    run_cmd python -m main_itr \
      --cfg-path "$CFG" \
      --dataset "$DATASET" \
      --model "$MODEL" \
      --algo "$algo" \
      --ratio "$ratio" \
      --eval
  done
done
