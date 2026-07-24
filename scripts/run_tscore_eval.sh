#!/usr/bin/env bash
# Tscore unified evaluation entry (BM5_15 / DB5 / CAPRI)
#
# EXP=E0  -> pretrained ckpt + score_type=mdn
# EXP=E1  -> finetuned ckpt + score_type=energy
#
# Usage:
#   EXP=E0 BENCH=bm5  bash scripts/run_tscore_eval.sh
#   EXP=E1 BENCH=db5  bash scripts/run_tscore_eval.sh
#   EXP=E1 BENCH=all  bash scripts/run_tscore_eval.sh
#
# BENCH: bm5 | db5 | capri | all  (default bm5)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TSCORE_DIR:-${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/tscore_path_lib.sh" ]]; then
  # shellcheck source=scripts/tscore_path_lib.sh
  source "$PROJECT_ROOT/scripts/tscore_path_lib.sh"
  tscore_source_env_files "$PROJECT_ROOT" || true
fi

EXP="${EXP:-E0}"
BENCH="${BENCH:-bm5}"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-Trained_models/pretrain_with_sasa/Tscore_best.chk}"
FT_CKPT="${FT_CKPT:-Trained_models/finetune_pairwise_rank/Tscore_best.chk}"

case "$EXP" in
  E0|e0)
    export CHECKPOINT="${CHECKPOINT:-$PRETRAIN_CKPT}"
    export SCORE_TYPE="${SCORE_TYPE:-mdn}"
    EXP_TAG="e0_mdn"
    ;;
  E1|e1)
    export CHECKPOINT="${CHECKPOINT:-$FT_CKPT}"
    export SCORE_TYPE="${SCORE_TYPE:-energy}"
    EXP_TAG="e1_energy"
    ;;
  *)
    echo "ERROR: EXP must be E0 or E1 (got: $EXP)" >&2
    exit 1
    ;;
esac

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p results
echo "=== Tscore eval: EXP=$EXP BENCH=$BENCH ==="
echo "CHECKPOINT=$CHECKPOINT"
echo "SCORE_TYPE=$SCORE_TYPE"
echo ""

run_bm5() {
  export OUT="${OUT_BM5:-$PROJECT_ROOT/results/bm5_15_tscore_${EXP_TAG}.csv}"
  echo "--- BM5_15 ---"
  bash scripts/run_bm5_15_tscore_eval.sh
}

run_db5() {
  echo "--- DB5 paper (holo + apo) ---"
  bash scripts/run_step8_eval_db5_paper.sh
}

run_capri() {
  echo "--- CAPRI 113 ---"
  bash scripts/run_step7_eval.sh
}

case "$BENCH" in
  bm5|BM5|bm5_15)
    run_bm5
    ;;
  db5|DB5)
    run_db5
    ;;
  capri|CAPRI)
    run_capri
    ;;
  all|ALL)
    run_bm5
    echo ""
    run_db5
    echo ""
    run_capri
    ;;
  *)
    echo "ERROR: BENCH must be bm5|db5|capri|all (got: $BENCH)" >&2
    exit 1
    ;;
esac

echo ""
echo "=== Tscore eval done (EXP=$EXP BENCH=$BENCH) ==="
