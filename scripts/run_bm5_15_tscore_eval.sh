#!/usr/bin/env bash
# Evaluate Tscore on Shirali BM5_15complexes (DeepRank-GNN hold-out).
#
# E0: CHECKPOINT=pretrain  SCORE_TYPE=mdn
# E1: CHECKPOINT=finetune  SCORE_TYPE=energy
#
# Example (E1):
#   CHECKPOINT=Trained_models/finetune_pairwise_rank/Tscore_best.chk \
#   SCORE_TYPE=energy \
#   BM5_15_DIR=~/tscore_data/BM5_15complexes \
#   LABELS_CSV=~/tscore_data/BM5_scores\&labels.csv \
#   bash scripts/run_bm5_15_tscore_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TSCORE_DIR:-${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/tscore_path_lib.sh" ]]; then
  # shellcheck source=scripts/tscore_path_lib.sh
  source "$PROJECT_ROOT/scripts/tscore_path_lib.sh"
  tscore_source_env_files "$PROJECT_ROOT" || true
fi

BM5_15_DIR="${BM5_15_DIR:-$HOME/tscore_data/BM5_15complexes}"
LABELS_CSV="${LABELS_CSV:-$HOME/tscore_data/BM5_scores&labels.csv}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to Tscore_best.chk}"
OUT="${OUT:-$PROJECT_ROOT/results/bm5_15_tscore.csv}"
SCORE_TYPE="${SCORE_TYPE:-mdn}"
N_WORKERS="${N_WORKERS:-16}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LIMIT_TARGETS="${LIMIT_TARGETS:-0}"
LIMIT_PER_TARGET="${LIMIT_PER_TARGET:-0}"

if [[ ! -d "$BM5_15_DIR/PDBs" ]]; then
  echo "ERROR: missing $BM5_15_DIR/PDBs — extract BM5_15complexes.zip first" >&2
  exit 1
fi
if [[ ! -f "$LABELS_CSV" ]]; then
  echo "ERROR: missing labels CSV: $LABELS_CSV" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

echo "BM5_15_DIR=$BM5_15_DIR"
echo "LABELS_CSV=$LABELS_CSV"
echo "CHECKPOINT=$CHECKPOINT"
echo "SCORE_TYPE=$SCORE_TYPE"
echo "OUT=$OUT"
echo "GPU=$CUDA_VISIBLE_DEVICES workers=$N_WORKERS"

extra=()
if [[ "$LIMIT_TARGETS" != "0" ]]; then
  extra+=(--limit_targets "$LIMIT_TARGETS")
fi
if [[ "$LIMIT_PER_TARGET" != "0" ]]; then
  extra+=(--limit_per_target "$LIMIT_PER_TARGET")
fi

python -u example/eval_bm5_15_tscore.py \
  --data_dir "$BM5_15_DIR" \
  --labels_csv "$LABELS_CSV" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT" \
  --score_type "$SCORE_TYPE" \
  --n_workers "$N_WORKERS" \
  --device "$DEVICE" \
  "${extra[@]}"

echo "done -> $OUT"
echo "      -> ${OUT%.csv}.summary.csv"
echo "      -> ${OUT%.csv}.aggregate.csv"
