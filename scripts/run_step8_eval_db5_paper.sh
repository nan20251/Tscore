#!/bin/bash
# Step 8: paper-aligned DB5 apo/holo evaluation for Tscore reranking.
#
# E0: SCORE_TYPE=mdn     CHECKPOINT=pretrain
# E1: SCORE_TYPE=energy  CHECKPOINT=finetune_pairwise_rank/...

set -e
PROJECT_ROOT="${TSCORE_DIR:-${TRADOCK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
PAPER_ROOT="${PAPER_ROOT:-/root/PPCBench}"
CHECKPOINT="${CHECKPOINT:-Trained_models/pretrain_with_sasa/Tscore_best.chk}"
DB5_HOLO_DATASET="${DB5_HOLO_DATASET:-DB5}"
DB5_APO_DATASET="${DB5_APO_DATASET:-DB5-u}"
HOLO_POSE_MODELS="${HOLO_POSE_MODELS:-hdock_1,hdock_2,hdock_3,hdock_4,hdock_5}"
APO_POSE_MODELS="${APO_POSE_MODELS:-hdock_1,hdock_2,hdock_3,hdock_4,hdock_5}"
MIN_TARGETS="${MIN_TARGETS:-218}"
SCORE_TYPE="${SCORE_TYPE:-mdn}"

cd "$PROJECT_ROOT"
mkdir -p results

echo "=== Step 8: Tscore DB5 paper-aligned apo/holo evaluation ==="
echo "PAPER_ROOT=$PAPER_ROOT"
echo "CHECKPOINT=$CHECKPOINT"
echo "SCORE_TYPE=$SCORE_TYPE"
echo ""

if [ ! -d "$PAPER_ROOT/dataset" ] || [ ! -d "$PAPER_ROOT/results" ]; then
    echo "[error] missing paper data dirs: $PAPER_ROOT/dataset and $PAPER_ROOT/results"
    echo "Use https://github.com/Yukki1777/PPCBench + Zenodo full datasets/results."
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "[error] missing Tscore checkpoint: $CHECKPOINT"
    exit 1
fi

echo "=== 8.1 DB5 holo (${DB5_HOLO_DATASET}) ==="
python -u example/eval_db5_paper_tscore.py \
    --paper_root "$PAPER_ROOT" \
    --dataset "$DB5_HOLO_DATASET" \
    --pose_models "$HOLO_POSE_MODELS" \
    --checkpoint "$CHECKPOINT" \
    --out "results/tscore_${DB5_HOLO_DATASET}_paper.csv" \
    --score_type "$SCORE_TYPE" \
    --min_targets "$MIN_TARGETS"

echo ""
echo "=== 8.2 DB5 apo (${DB5_APO_DATASET}) ==="
python -u example/eval_db5_paper_tscore.py \
    --paper_root "$PAPER_ROOT" \
    --dataset "$DB5_APO_DATASET" \
    --pose_models "$APO_POSE_MODELS" \
    --checkpoint "$CHECKPOINT" \
    --out "results/tscore_${DB5_APO_DATASET}_paper.csv" \
    --score_type "$SCORE_TYPE" \
    --min_targets "$MIN_TARGETS"

echo ""
echo "=== DB5 paper-aligned evaluation complete ==="
ls -1 results/tscore_${DB5_HOLO_DATASET}_paper*.csv 2>/dev/null || true
ls -1 results/tscore_${DB5_APO_DATASET}_paper*.csv 2>/dev/null || true
