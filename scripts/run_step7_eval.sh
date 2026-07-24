#!/bin/bash
# Step 7: CAPRI Score_set 113 fast evaluation
#
# SCORE_TYPE: mdn (E0) | energy (E1)
# Unified entry: EXP=E0|E1 BENCH=capri bash scripts/run_tscore_eval.sh

set -e
PROJECT_ROOT="${TSCORE_DIR:-${TRADOCK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CAPRI_DIR="${CAPRI_DIR:-$PROJECT_ROOT/data/database}"
CHECKPOINT="${CHECKPOINT:-Trained_models/pretrain_with_sasa/Tscore_best.chk}"
SCORE_TYPE="${SCORE_TYPE:-mdn}"
OUT="${OUT_CAPRI:-results/capri_eval_113_fast.csv}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4

mkdir -p results

echo "=== CAPRI Score_set 113 fast eval ==="
echo "CHECKPOINT=$CHECKPOINT SCORE_TYPE=$SCORE_TYPE"

if [ -d "$CAPRI_DIR" ] && ls "$CAPRI_DIR"/S-T*.pdb 1>/dev/null 2>&1 && ls "$CAPRI_DIR"/S-T*.csv 1>/dev/null 2>&1 && [ -f "$CHECKPOINT" ]; then
    MAX_MODELS_ARG=()
    if [ -n "${MAX_MODELS:-}" ]; then
        MAX_MODELS_ARG=(--max_models "$MAX_MODELS")
    fi
    python -u example/eval_capri_fast.py \
        --data_dir "$CAPRI_DIR" \
        --checkpoint "$CHECKPOINT" \
        --out "$OUT" \
        --pos_metric classification --pos_threshold 0.3 \
        --success_denominator with_positives \
        --score_type "$SCORE_TYPE" \
        --n_workers "${N_WORKERS:-1}" \
        "${MAX_MODELS_ARG[@]}"
else
    echo "[skip] CAPRI eval: missing data ($CAPRI_DIR/S-T*.pdb + .csv) or $CHECKPOINT"
    echo "Set CAPRI_DIR=... CHECKPOINT=... SCORE_TYPE=energy"
fi

echo ""
echo "=== CAPRI eval done ==="
ls "${OUT}" 2>/dev/null && echo "  $OUT"
ls "${OUT%.csv}.summary.csv" 2>/dev/null && echo "  ${OUT%.csv}.summary.csv"
