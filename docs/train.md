# Tscore training

## Path defaults

```bash
export TSCORE_DIR=~/Desktop/Tscore   # or your clone path
export DIPS_ROOT=/path/to/dips_workdir
export DIPS_SURFACES=$DIPS_ROOT/surfaces
export CHECKPOINT=$TSCORE_DIR/Trained_models/pretrain_with_sasa/Tscore_best.chk
export FT_CHECKPOINT=$TSCORE_DIR/Trained_models/finetune_pairwise_rank/Tscore_best.chk
```

## Recommended: one-shot data + pretrain

```bash
cd "$TSCORE_DIR"
DIPS_ROOT="$DIPS_ROOT" bash scripts/run_pipeline_end_to_end.sh
```

See [data_pipeline.md](data_pipeline.md) for acquire + MMseqs dedup details.

## Pretrain only (surfaces ready)

```bash
cd "$TSCORE_DIR"
DIPS_SURFACES="$DIPS_SURFACES" bash scripts/run_step2_pretrain.sh
```

Equivalent:

```bash
python example/train.py \
  --data_dir "$DIPS_SURFACES" \
  --save_dir Trained_models/pretrain_with_sasa \
  --epochs 30 --batch_size 2 --lr 1e-4 --contrast_weight 0.0
```

## Fine-tune (pairwise + DockQ)

`L = λ_rank·pairwise + λ_mse·MSE(DockQ) + λ_mdn·MDN(DockQ≥mdn_min_dockq)`

```bash
python example/train_native_vs_decoy_v2.py \
  --decoy_only \
  --decoy_dir /path/to/decoy_surfaces \
  --decoy_csv /path/to/decoy_surfaces/decoys.csv \
  --init_from "$CHECKPOINT" \
  --save_dir Trained_models/finetune_pairwise_rank \
  --mdn_min_dockq 0.5
```

Or: `bash scripts/run_step6_lightdock_finetune.sh` (LightDock + surfaces + train).

**Protocol:** train on BM5 minus BM5_15; evaluate BM5_15 with `--score_type energy`.

After fine-tune, always evaluate with **`--score_type energy`**.
