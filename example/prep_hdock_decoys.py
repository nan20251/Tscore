"""
HDock decoys (BM5-clean layout) → Tscore fine-tune surfaces + decoys.csv

Expect:
  <native_root>/<ID>/<ID>_r_u.pdb
  <native_root>/<ID>/<ID>_l_u.pdb
  <native_root>/<ID>/<ID>_r_b-matched.pdb
  <native_root>/<ID>/<ID>_l_b-matched.pdb

  <hdock_root>/<ID>/<ID>_hdock_1.pdb ...

Bound (r_b + l_b) is used as DockQ native reference.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from example.prep_lightdock_decoys import (  # noqa: E402
    process_decoy,
    detect_chains_in_pdb,
)


def merge_bound_native(rb: str, lb: str, out_pdb: str) -> None:
    with open(out_pdb, 'w') as out:
        for path in (rb, lb):
            with open(path) as f:
                for line in f:
                    if line.startswith(('ATOM', 'HETATM', 'TER')):
                        out.write(line)
        out.write('END\n')


def collect_jobs(native_root, hdock_root, out_dir, voxel_size,
                 targets=None, max_per_target=None, limit=None):
    ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(native_root, '*'))
                 if os.path.isdir(p))
    if targets:
        wanted = set(targets)
        ids = [i for i in ids if i in wanted]
    if limit:
        ids = ids[:limit]

    cache = os.path.join(out_dir, '_native_cache')
    os.makedirs(cache, exist_ok=True)

    for stem in ids:
        nd = os.path.join(native_root, stem)
        md = os.path.join(hdock_root, stem)
        rb = os.path.join(nd, f'{stem}_r_b-matched.pdb')
        lb = os.path.join(nd, f'{stem}_l_b-matched.pdb')
        if not (os.path.isdir(md) and os.path.isfile(rb) and os.path.isfile(lb)):
            print(f'  [skip] {stem}: missing models or bound pdbs')
            continue

        native_pdb = os.path.join(cache, f'{stem}_native.pdb')
        if not os.path.isfile(native_pdb):
            merge_bound_native(rb, lb, native_pdb)

        # BM5-clean matched: receptor then ligand in file order
        native_chains = detect_chains_in_pdb(native_pdb)
        if len(native_chains) < 2:
            print(f'  [skip] {stem}: native chains < 2 ({native_chains})')
            continue
        # split half: first chain(s) as rec if 2 chains; else first half
        if len(native_chains) == 2:
            rec_chains, lig_chains = [native_chains[0]], [native_chains[1]]
        else:
            n_rec = max(1, len(native_chains) // 2)
            rec_chains = native_chains[:n_rec]
            lig_chains = native_chains[n_rec:]

        decoys = sorted(glob.glob(os.path.join(md, f'{stem}_hdock_*.pdb')))
        if not decoys:
            decoys = sorted(glob.glob(os.path.join(md, '*.pdb')))
        if max_per_target:
            decoys = decoys[:max_per_target]

        for gid, pdb in enumerate(decoys):
            yield (stem, gid, pdb, native_pdb,
                   rec_chains, lig_chains, None, None,
                   out_dir, voxel_size)


def main():
    p = argparse.ArgumentParser(description='HDock models → decoys.csv + PLY')
    p.add_argument('--native_root', required=True,
                   help='bm5_minus_15_natives root')
    p.add_argument('--hdock_root', required=True,
                   help='bm5_minus_15_hdock/models root')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--voxel_size', type=float, default=3.5)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max_per_target', type=int, default=None)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--targets', default='',
                   help='Comma list, e.g. 1A2K,1ACB')
    p.add_argument('--report_every', type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = [t.strip() for t in args.targets.split(',') if t.strip()] or None
    jobs = list(collect_jobs(
        args.native_root, args.hdock_root, args.out_dir, args.voxel_size,
        targets=targets, max_per_target=args.max_per_target, limit=args.limit,
    ))
    print(f'jobs={len(jobs)} out={args.out_dir} DockQ={os.environ.get("DOCKQ_BIN", "DockQ")}')
    if not jobs:
        sys.exit('no jobs')

    rows, fails = [], []
    n_ok = n_fail = done = 0
    t0 = time.time()

    def handle(r):
        nonlocal n_ok, n_fail, done
        (rows if r.get('ok') else fails).append(r)
        n_ok += int(bool(r.get('ok')))
        n_fail += int(not r.get('ok'))
        done += 1
        if done % args.report_every == 0 or done == len(jobs):
            print(f'[{done}/{len(jobs)}] ok={n_ok} fail={n_fail} '
                  f'{(time.time()-t0):.0f}s')

    if args.workers <= 1:
        for j in jobs:
            handle(process_decoy(j))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_decoy, j): j for j in jobs}
            while futs:
                done_f, _ = wait(futs, return_when=FIRST_COMPLETED)
                for fut in done_f:
                    futs.pop(fut, None)
                    try:
                        handle(fut.result())
                    except Exception as e:
                        handle({'ok': False, 'name': '?', 'msg': str(e)})

    rows.sort(key=lambda x: (x['stem'], x['decoy_id']))
    csv_path = os.path.join(args.out_dir, 'decoys.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'stem', 'decoy_id', 'dockq', 'fnat', 'irms', 'lrms',
                    'classification'])
        for r in rows:
            w.writerow([r['name'], r['stem'], r['decoy_id'],
                        f"{r['dockq']:.4f}", f"{r['fnat']:.4f}",
                        f"{r['irms']:.3f}", f"{r['lrms']:.3f}",
                        r['classification']])
    print(f'csv={csv_path} ok={n_ok} fail={n_fail}')
    if fails:
        log = os.path.join(args.out_dir, 'failures.log')
        with open(log, 'w') as f:
            for r in fails:
                f.write(f"{r.get('name','?')}\t{r.get('msg','')}\n")
        print(f'failures={log}')


if __name__ == '__main__':
    main()
