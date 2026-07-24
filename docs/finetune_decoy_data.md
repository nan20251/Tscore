# Fine-tune decoy data layout

## Directories

```text
<native_dir>/                 # optional with --decoy_only
  pairs.csv
  {name}_receptor.ply
  {name}_ligand.ply

<decoy_dir>/
  decoys.csv                  # required
  {name}_receptor.ply
  {name}_ligand.ply
```

`decoys.csv` columns (from `example/prep_lightdock_decoys.py`):

| column | meaning |
|--------|---------|
| `name` | surface stem, e.g. `1PPE_d00012` |
| `stem` | target id (same-target decoys share) |
| `decoy_id` | index |
| `dockq` | DockQ label for ranking |
| `fnat` / `irms` / `lrms` | optional |
| `classification` | optional CAPRI class |

## Protocol

1. **Train:** BM5 minus BM5_15; per target HDock/LightDock → DockQ → `decoys.csv` + PLY.
2. **Test:** BM5_15 only (`example/eval_bm5_15_tscore.py`); optional DB5-u / CAPRI.
3. After fine-tune: `--score_type energy`.

```bash
PDB_DIR=/path/to/bm5_minus_15_pdbs bash scripts/run_step6_lightdock_finetune.sh

SKIP_LIGHTDOCK=1 DECOY_ONLY=1 \
  DECOY_SURFACES=/path/to/decoy_surfaces \
  bash scripts/run_step6_lightdock_finetune.sh

python example/train_native_vs_decoy_v2.py \
  --decoy_only \
  --decoy_dir "$DECOY_SURFACES" \
  --decoy_csv "$DECOY_SURFACES/decoys.csv" \
  --init_from "$CHECKPOINT" \
  --save_dir Trained_models/finetune_pairwise_rank \
  --mdn_min_dockq 0.5
```

## Smoke

DIPS train subset → generate decoys; exclude BM5_15 / test PDBs. Validates pipeline only.
