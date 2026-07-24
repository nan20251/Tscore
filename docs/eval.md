# Tscore evaluation

Unified entry: `scripts/run_tscore_eval.sh`

## Matrix

| EXP | Checkpoint | SCORE_TYPE |
|-----|------------|------------|
| E0 | `Trained_models/pretrain_with_sasa/Tscore_best.chk` | `mdn` |
| E1 | `Trained_models/finetune_pairwise_rank/Tscore_best.chk` | `energy` |

| BENCH | Backend |
|-------|---------|
| bm5 | `run_bm5_15_tscore_eval.sh` → `example/eval_bm5_15_tscore.py` |
| db5 | `run_step8_eval_db5_paper.sh` → `example/eval_db5_paper_tscore.py` |
| capri | `run_step7_eval.sh` → `example/eval_capri_fast.py` |
| all | bm5 then db5 then capri |

```bash
EXP=E0 BENCH=bm5 bash scripts/run_tscore_eval.sh
EXP=E1 BENCH=bm5 bash scripts/run_tscore_eval.sh
EXP=E1 BENCH=db5 PAPER_ROOT=/path/to/PPCBench bash scripts/run_tscore_eval.sh
EXP=E1 BENCH=capri CAPRI_DIR=/path/to/capri/database bash scripts/run_tscore_eval.sh
```

Smoke:

```bash
LIMIT_TARGETS=1 LIMIT_PER_TARGET=20 EXP=E0 BENCH=bm5 bash scripts/run_tscore_eval.sh
```

Env overrides: `CHECKPOINT`, `SCORE_TYPE`, `BM5_15_DIR`, `LABELS_CSV`, `PAPER_ROOT`, `CAPRI_DIR`, `OUT` / `OUT_BM5` / `OUT_CAPRI`.

`TSCORE_DIR` preferred; `TRADOCK_DIR` still accepted as fallback.
