# Data pipeline (DIPS acquire + CAPRI dedup)

Full path from metadata to surfaces used by pretrain.

```text
metadata.csv
    -> prep_dips --split_only     # RCSB download + chain split
    -> dedup_dips_vs_capri        # MMseqs2 vs CAPRI ref FASTA
    -> prep_dips --exclude_file   # surfaces + pairs.csv
    -> run_step2_pretrain.sh
```

Orchestrator: `scripts/run_pipeline_end_to_end.sh`

## Inputs in repo

| Path | Role |
|------|------|
| `data/dips/metadata.csv` | DIPS complex list (`pdb_id,rec_chains,lig_chains`) |
| `data/dips/exclude_capri.txt` | Default / regenerated CAPRI-homolog PDB prefixes |
| `data/capri_exclude/capri_ref.fasta` | CAPRI query sequences for MMseqs |

## Scripts

| Script | Role |
|--------|------|
| `example/extract_capri_sequences.py` | Build `capri_ref.fasta` from Score_set `tar.bz2` |
| `example/prep_dips.py` | Download PDB, split chains, optional surfaces |
| `example/dedup_dips_vs_capri.py` | MMseqs2 identity filter → exclude list |
| `scripts/filter_bad_dips_pairs.py` | Filter `pairs.csv` at pretrain time |

## Manual steps (same as pipeline)

```bash
# 0) optional rebuild CAPRI fasta
python example/extract_capri_sequences.py \
  --tar /path/to/Scoreset_v2022_Scorers.tar.bz2 \
  --out data/capri_exclude/capri_ref.fasta

# 1) download + split only
python example/prep_dips.py \
  --metadata data/dips/metadata.csv \
  --pdb_dir $DIPS_ROOT/pdbs \
  --split_dir $DIPS_ROOT/split_pdbs \
  --out_dir $DIPS_ROOT/surfaces \
  --split_only --workers 8

# 2) dedup (needs mmseqs on PATH)
python example/dedup_dips_vs_capri.py \
  --capri_fasta data/capri_exclude/capri_ref.fasta \
  --split_dir $DIPS_ROOT/split_pdbs \
  --work_dir $DIPS_ROOT/dedup \
  --out_exclude data/dips/exclude_capri.txt

# 3) surfaces with exclude
python example/prep_dips.py \
  --metadata data/dips/metadata.csv \
  --pdb_dir $DIPS_ROOT/pdbs \
  --split_dir $DIPS_ROOT/split_pdbs \
  --out_dir $DIPS_ROOT/surfaces \
  --skip_download \
  --exclude_file data/dips/exclude_capri.txt \
  --workers 8
```

## BM5 / DB5 (eval data, not DIPS)

These are **separate** benchmarks; not produced by `prep_dips`:

- **BM5_15:** unzip `BM5_15complexes` + labels CSV → `run_bm5_15_tscore_eval.sh`
- **DB5:** PPCBench / Zenodo paper poses → `run_step8_eval_db5_paper.sh`
- **CAPRI Score_set:** decoy PDBs + CSVs → `run_step7_eval.sh`

Fine-tune natives (BM5\\15) go through LightDock via `run_step6_lightdock_finetune.sh`.
