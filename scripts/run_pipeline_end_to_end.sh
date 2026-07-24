#!/usr/bin/env bash
# End-to-end Tscore pipeline: DIPS prepare -> dedup vs CAPRI -> surfaces -> pretrain
# Optional: LightDock fine-tune + BM5/DB5/CAPRI eval.
#
# Required env:
#   DIPS_ROOT=/path/to/workdir     # will hold pdbs/, split_pdbs/, surfaces/
#   METADATA=data/dips/metadata.csv  (default in repo)
#
# Optional:
#   CAPRI_TAR=/path/to/Scoreset_*.tar.bz2   # rebuild capri_ref.fasta if set
#   SKIP_DEDUP=1                           # skip MMseqs2 (use existing exclude)
#   SKIP_PRETRAIN=1
#   PDB_DIR=/path/to/finetune_natives      # enables step6 fine-tune
#   RUN_EVAL=1 + BM5_15_DIR / PAPER_ROOT / CAPRI_DIR as needed
#
# Example (data prep + pretrain only):
#   DIPS_ROOT=/root/autodl-tmp/dips bash scripts/run_pipeline_end_to_end.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TSCORE_DIR:-${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
cd "$PROJECT_ROOT"

DIPS_ROOT="${DIPS_ROOT:?set DIPS_ROOT=/path/to/dips_workdir}"
METADATA="${METADATA:-$PROJECT_ROOT/data/dips/metadata.csv}"
CAPRI_FASTA="${CAPRI_FASTA:-$PROJECT_ROOT/data/capri_exclude/capri_ref.fasta}"
EXCLUDE_OUT="${EXCLUDE_OUT:-$PROJECT_ROOT/data/dips/exclude_capri.txt}"

PDB_CACHE="${PDB_CACHE:-$DIPS_ROOT/pdbs}"
SPLIT_DIR="${SPLIT_DIR:-$DIPS_ROOT/split_pdbs}"
SURFACES="${SURFACES:-$DIPS_ROOT/surfaces}"
DEDUP_WORK="${DEDUP_WORK:-$DIPS_ROOT/dedup}"
VOXEL_SIZE="${VOXEL_SIZE:-3.5}"
WORKERS="${WORKERS:-8}"
DL_WORKERS="${DL_WORKERS:-8}"
LIMIT_ARG=()
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

mkdir -p "$PDB_CACHE" "$SPLIT_DIR" "$SURFACES" "$DEDUP_WORK" results Trained_models

echo "=== Tscore end-to-end ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DIPS_ROOT=$DIPS_ROOT METADATA=$METADATA"
echo ""

if [ ! -f "$METADATA" ]; then
  echo "[error] missing metadata CSV: $METADATA" >&2
  exit 1
fi

# --- 0) CAPRI reference FASTA (optional rebuild) ---
if [ -n "${CAPRI_TAR:-}" ]; then
  echo "=== 0. extract CAPRI sequences ==="
  python example/extract_capri_sequences.py --tar "$CAPRI_TAR" --out "$CAPRI_FASTA"
elif [ ! -f "$CAPRI_FASTA" ]; then
  echo "[error] missing $CAPRI_FASTA (shipped in repo, or set CAPRI_TAR=...)" >&2
  exit 1
else
  echo "=== 0. using existing CAPRI fasta: $CAPRI_FASTA ==="
fi

# --- 1) Download + split only (for dedup sequences) ---
echo ""
echo "=== 1. DIPS download + split (no surfaces yet) ==="
python example/prep_dips.py \
  --metadata "$METADATA" \
  --pdb_dir "$PDB_CACHE" \
  --split_dir "$SPLIT_DIR" \
  --out_dir "$SURFACES" \
  --voxel_size "$VOXEL_SIZE" \
  --dl_workers "$DL_WORKERS" \
  --workers "$WORKERS" \
  --split_only \
  "${LIMIT_ARG[@]}"

# --- 2) Dedup vs CAPRI (MMseqs2) ---
if [ "${SKIP_DEDUP:-0}" != "1" ]; then
  echo ""
  echo "=== 2. DIPS vs CAPRI dedup (mmseqs2) ==="
  if ! command -v mmseqs >/dev/null 2>&1; then
    echo "[error] mmseqs not found. Install: conda install -c bioconda mmseqs2" >&2
    echo "  Or set SKIP_DEDUP=1 to use existing $EXCLUDE_OUT" >&2
    exit 1
  fi
  python example/dedup_dips_vs_capri.py \
    --capri_fasta "$CAPRI_FASTA" \
    --split_dir "$SPLIT_DIR" \
    --work_dir "$DEDUP_WORK" \
    --out_exclude "$EXCLUDE_OUT" \
    --id_thresh "${ID_THRESH:-0.30}" \
    --cov_thresh "${COV_THRESH:-0.50}" \
    --threads "${MMSEQS_THREADS:-8}"
else
  echo ""
  echo "=== 2. SKIP_DEDUP=1; using $EXCLUDE_OUT ==="
  if [ ! -f "$EXCLUDE_OUT" ]; then
    echo "[error] missing exclude file: $EXCLUDE_OUT" >&2
    exit 1
  fi
fi

# --- 3) Surfaces with exclude list ---
echo ""
echo "=== 3. DIPS surfaces (exclude CAPRI homologs) ==="
python example/prep_dips.py \
  --metadata "$METADATA" \
  --pdb_dir "$PDB_CACHE" \
  --split_dir "$SPLIT_DIR" \
  --out_dir "$SURFACES" \
  --voxel_size "$VOXEL_SIZE" \
  --dl_workers "$DL_WORKERS" \
  --workers "$WORKERS" \
  --skip_download \
  --exclude_file "$EXCLUDE_OUT" \
  "${LIMIT_ARG[@]}"

export DIPS_SURFACES="$SURFACES"

# --- 4) Pretrain ---
if [ "${SKIP_PRETRAIN:-0}" != "1" ]; then
  echo ""
  echo "=== 4. MDN pretrain ==="
  DIPS_SURFACES="$SURFACES" bash scripts/run_step2_pretrain.sh
else
  echo ""
  echo "=== 4. SKIP_PRETRAIN=1 ==="
fi

# --- 5) Optional fine-tune ---
if [ -n "${PDB_DIR:-}" ]; then
  echo ""
  echo "=== 5. LightDock fine-tune (PDB_DIR=$PDB_DIR) ==="
  INIT_FROM="${INIT_FROM:-Trained_models/pretrain_with_sasa/Tscore_best.chk}" \
    PDB_DIR="$PDB_DIR" \
    bash scripts/run_step6_lightdock_finetune.sh
else
  echo ""
  echo "=== 5. skip fine-tune (set PDB_DIR=... to enable) ==="
fi

# --- 6) Optional eval ---
if [ "${RUN_EVAL:-0}" = "1" ]; then
  echo ""
  echo "=== 6. eval EXP=${EXP:-E0} BENCH=${BENCH:-bm5} ==="
  EXP="${EXP:-E0}" BENCH="${BENCH:-bm5}" bash scripts/run_tscore_eval.sh
else
  echo ""
  echo "=== 6. skip eval (set RUN_EVAL=1 EXP=E0|E1 BENCH=bm5|db5|capri|all) ==="
fi

echo ""
echo "=== pipeline done ==="
echo "surfaces: $SURFACES"
echo "exclude:  $EXCLUDE_OUT"
echo "pretrain: Trained_models/pretrain_with_sasa/Tscore_best.chk"
