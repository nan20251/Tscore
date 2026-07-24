#!/bin/bash
# Step 2: DIPS MDN pretrain
# max_nodes=1500 limits surface size to avoid OOM

set -e
PROJECT_ROOT="${TSCORE_DIR:-${TRADOCK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DIPS_SURFACES="${DIPS_SURFACES:-/root/autodl-tmp/dips_with_sasa_full}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "example/train.py" ]; then
    echo "[error] missing example/train.py; wrong project root: $PROJECT_ROOT" >&2
    exit 1
fi
if [ ! -f "$DIPS_SURFACES/pairs.csv" ]; then
    echo "[error] missing training data: $DIPS_SURFACES/pairs.csv" >&2
    echo "Set DIPS_SURFACES=/path/to/surfaces" >&2
    exit 1
fi

mkdir -p Trained_models results

PAIRS_CSV="${PAIRS_CSV:-results/dips_with_sasa_full.filtered_pairs.csv}"
python scripts/filter_bad_dips_pairs.py \
    --input "$DIPS_SURFACES/pairs.csv" \
    --output "$PAIRS_CSV" \
    --exclude 1u0c_A_B 1yk0_A_B \
    --exclude_file data/dips/exclude_capri.txt

python example/train.py \
    --data_dir "$DIPS_SURFACES" \
    --pairs_csv "$PAIRS_CSV" \
    --save_dir Trained_models/pretrain_with_sasa \
    --epochs 30 \
    --batch_size 2 \
    --lr 1e-4 \
    --contrast_weight 0.0

echo "=== Step 2 pretrain done ==="
echo "checkpoint: Trained_models/pretrain_with_sasa/Tscore_best.chk"
