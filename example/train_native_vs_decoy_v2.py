"""
Tscore fine-tuning for Success@N (PPI decoy reranking).

Literature-aligned objective (PIsToN / DeepRank style), NOT BioScore affinity corr:

    L = λ_rank · pairwise_margin(same-target, by DockQ)
      + λ_mse  · MSE(pred_energy, DockQ)
      + λ_mdn  · MDN_NLL (only samples with DockQ >= mdn_min_dockq)

Supports decoy-only training (no native required) when each stem has ≥2 DockQ-labeled decoys.

After fine-tuning, evaluate with --score_type energy.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tscore.models import DeepDock_PPI, ppi_train_loss
from tscore.utils.data import match_feature_dim, read_ply


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class MixedDataset(Dataset):
    def __init__(self, native_dir, decoy_dir, decoy_csv,
                 stem_filter=None, max_per_stem_decoy=None,
                 in_channels=11, decoy_only=False):
        self.records = []
        self.in_channels = in_channels
        self.decoy_only = decoy_only

        native_csv = os.path.join(native_dir, 'pairs.csv') if native_dir else ''
        if (not decoy_only) and native_csv and os.path.exists(native_csv):
            with open(native_csv) as f:
                for row in csv.DictReader(f):
                    name = row.get('name') or row.get('pdb_id', '')
                    if not name:
                        continue
                    if stem_filter is not None and name not in stem_filter:
                        continue
                    rec = os.path.join(native_dir, f'{name}_receptor.ply')
                    lig = os.path.join(native_dir, f'{name}_ligand.ply')
                    if not (os.path.exists(rec) and os.path.exists(lig)):
                        continue
                    self.records.append({
                        'rec': rec, 'lig': lig,
                        'dockq': 1.0,
                        'is_native': True,
                        'stem': name,
                        'name': name,
                    })

        stem_counts = {}
        with open(decoy_csv) as f:
            for row in csv.DictReader(f):
                stem = row['stem']
                if stem_filter is not None and stem not in stem_filter:
                    continue
                if max_per_stem_decoy is not None:
                    if stem_counts.get(stem, 0) >= max_per_stem_decoy:
                        continue
                    stem_counts[stem] = stem_counts.get(stem, 0) + 1

                name = row['name']
                rec = os.path.join(decoy_dir, f'{name}_receptor.ply')
                lig = os.path.join(decoy_dir, f'{name}_ligand.ply')
                if not (os.path.exists(rec) and os.path.exists(lig)):
                    continue
                self.records.append({
                    'rec': rec, 'lig': lig,
                    'dockq': float(row['dockq']),
                    'is_native': False,
                    'stem': stem,
                    'name': name,
                })

        self.native_idx = [i for i, r in enumerate(self.records) if r['is_native']]
        self.decoy_idx = [i for i, r in enumerate(self.records) if not r['is_native']]

        self.decoy_by_stem = defaultdict(list)
        self.native_by_stem = {}
        for i, r in enumerate(self.records):
            if r['is_native']:
                self.native_by_stem[r['stem']] = i
            else:
                self.decoy_by_stem[r['stem']].append(i)

        # Within-target ranking: ≥2 decoys; native optional (decoy-only OK)
        if decoy_only or not self.native_idx:
            self.include_native = False
            self.rankable_stems = [
                s for s, di in self.decoy_by_stem.items() if len(di) >= 2
            ]
        else:
            self.include_native = True
            self.rankable_stems = [
                s for s, di in self.decoy_by_stem.items()
                if s in self.native_by_stem and len(di) >= 1
            ]
            # Prefer ≥2 decoys when available; fall back to ≥1 + native
            rich = [s for s in self.rankable_stems if len(self.decoy_by_stem[s]) >= 2]
            if rich:
                self.rankable_stems = rich

        self.decoy_near_native = [
            i for i in self.decoy_idx if 0.30 <= self.records[i]['dockq'] < 0.50
        ]
        self.decoy_hard = [
            i for i in self.decoy_idx if 0.10 <= self.records[i]['dockq'] < 0.30
        ]
        self.decoy_medium = [
            i for i in self.decoy_idx if 0.03 <= self.records[i]['dockq'] < 0.10
        ]
        self.decoy_easy = [
            i for i in self.decoy_idx if self.records[i]['dockq'] < 0.03
        ]

        mode = 'decoy-only' if not self.include_native else 'native+decoy'
        print(
            f'Mixed dataset ({mode}): native={len(self.native_idx)}, '
            f'decoy={len(self.decoy_idx)}, '
            f'rankable_stems={len(self.rankable_stems)} '
            f'(near={len(self.decoy_near_native)}, hard={len(self.decoy_hard)}, '
            f'med={len(self.decoy_medium)}, easy={len(self.decoy_easy)})'
        )
        if not self.decoy_idx:
            raise RuntimeError(f'Decoy 样本为 0。检查 {decoy_csv}')
        if self.include_native and not self.native_idx:
            raise RuntimeError(
                f'Native 样本为 0。检查 {native_csv}，或加 --decoy_only'
            )
        if not self.rankable_stems:
            raise RuntimeError(
                '没有可用于同靶 pairwise 的 stem（需要 ≥2 decoy，'
                '或 native+≥1 decoy）'
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        rec = match_feature_dim(read_ply(r['rec']), self.in_channels)
        lig = match_feature_dim(read_ply(r['lig']), self.in_channels)
        return (
            rec, lig,
            torch.tensor(r['dockq'], dtype=torch.float32),
            torch.tensor(r['is_native'], dtype=torch.bool),
            r['stem'], r['name'],
        )


class SameTargetSampler(Sampler):
    """
    Each batch = same-stem poses for within-target ranking.
    - include_native: 1 native + N decoys
    - decoy-only: N decoys (no native)
    Difficulty mix when enough decoys exist on that stem.
    """

    def __init__(
        self, dataset, batches_per_epoch=400, decoy_per_native=15,
        ratio_near_native=0.20, ratio_hard=0.40,
        ratio_medium=0.30, ratio_easy=0.10,
    ):
        self.ds = dataset
        self.batches_per_epoch = batches_per_epoch
        self.decoy_per_native = decoy_per_native
        self.include_native = getattr(dataset, 'include_native', True)
        n = decoy_per_native
        self.n_near = max(1, int(round(n * ratio_near_native)))
        self.n_hard = max(1, int(round(n * ratio_hard)))
        self.n_med = max(1, int(round(n * ratio_medium)))
        self.n_easy = max(0, n - self.n_near - self.n_hard - self.n_med)

    def _pick(self, pool, k, fallback):
        if k <= 0:
            return []
        src = pool if pool else fallback
        if not src:
            return []
        return [random.choice(src) for _ in range(k)]

    def __iter__(self):
        stems = self.ds.rankable_stems
        target_n = self.decoy_per_native
        for _ in range(self.batches_per_epoch):
            stem = random.choice(stems)
            decoy_ids = self.ds.decoy_by_stem[stem]
            near = [i for i in decoy_ids if 0.30 <= self.ds.records[i]['dockq'] < 0.50]
            hard = [i for i in decoy_ids if 0.10 <= self.ds.records[i]['dockq'] < 0.30]
            med = [i for i in decoy_ids if 0.03 <= self.ds.records[i]['dockq'] < 0.10]
            easy = [i for i in decoy_ids if self.ds.records[i]['dockq'] < 0.03]

            batch = []
            if self.include_native:
                batch.append(self.ds.native_by_stem[stem])
            batch += self._pick(near, self.n_near, decoy_ids)
            batch += self._pick(hard, self.n_hard, decoy_ids)
            batch += self._pick(med, self.n_med, decoy_ids)
            batch += self._pick(easy, self.n_easy, decoy_ids)

            need = target_n + (1 if self.include_native else 0)
            while len(batch) < need and decoy_ids:
                batch.append(random.choice(decoy_ids))
            yield batch[:need]

    def __len__(self):
        return self.batches_per_epoch


def collate(batch):
    recs, ligs, dockqs, is_natives, stems, names = zip(*batch)
    return (
        Batch.from_data_list(list(recs)),
        Batch.from_data_list(list(ligs)),
        torch.stack(dockqs),
        torch.stack(is_natives),
        list(stems),
        list(names),
    )


# ─────────────────────────────────────────────────────────────
# Losses: pairwise + MSE + MDN  (no BioScore corr)
# ─────────────────────────────────────────────────────────────

def pairwise_margin_loss(scores, labels, margin=0.1, label_gap=0.05):
    """
    Within-batch pairwise ranking (same target by construction).
    For every pair with label_i >= label_j + label_gap:
        hinge: max(0, margin - (score_i - score_j))
    """
    n = scores.numel()
    if n < 2:
        return scores.new_zeros(())

    # [n, n]
    s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    y_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
    mask = y_diff >= label_gap
    if not mask.any():
        return scores.new_zeros(())

    # Prefer higher label → higher score
    losses = F.relu(margin - s_diff)[mask]
    return losses.mean()


def ranking_finetune_loss(
    pred_energy,
    dockqs,
    pi, sigma, mu, dist,
    dist_threshold,
    lambda_rank=1.0,
    lambda_mse=0.5,
    lambda_mdn=0.3,
    pair_margin=0.1,
    label_gap=0.05,
    C_batch=None,
    mdn_min_dockq=0.5,
):
    score = pred_energy
    label = dockqs.clamp(0.0, 1.0)

    loss_rank = pairwise_margin_loss(
        score, label, margin=pair_margin, label_gap=label_gap,
    )
    loss_mse = F.mse_loss(score, label)

    loss_mdn = score.new_zeros(())
    if lambda_mdn != 0.0 and pi.numel() > 0 and dist.numel() > 0:
        if C_batch is not None and mdn_min_dockq > 0:
            # Only pairs from complexes with DockQ >= mdn_min_dockq
            keep_complex = label >= mdn_min_dockq
            pair_keep = keep_complex[C_batch]
            if pair_keep.any():
                loss_mdn = ppi_train_loss(
                    pi[pair_keep], sigma[pair_keep], mu[pair_keep],
                    dist[pair_keep], dist_threshold=dist_threshold,
                )
        else:
            loss_mdn = ppi_train_loss(
                pi, sigma, mu, dist, dist_threshold=dist_threshold,
            )
        if not torch.isfinite(loss_mdn):
            loss_mdn = score.new_zeros(())

    loss = (
        lambda_rank * loss_rank
        + lambda_mse * loss_mse
        + lambda_mdn * loss_mdn
    )
    return loss, {'rank': loss_rank, 'mse': loss_mse, 'mdn': loss_mdn}


# ─────────────────────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, dist_threshold, cfg, max_grad_norm=1.0):
    model.train()
    stats = {'total': 0.0, 'rank': 0.0, 'mse': 0.0, 'mdn': 0.0, 'n': 0}

    for rec_batch, lig_batch, dockqs, is_natives, stems, names in loader:
        rec_batch = rec_batch.to(device)
        lig_batch = lig_batch.to(device)
        dockqs = dockqs.to(device)

        optimizer.zero_grad()
        pi, sigma, mu, dist, C_batch, pred_energy = model(rec_batch, lig_batch)
        if (
            not torch.isfinite(pi).all()
            or not torch.isfinite(sigma).all()
            or not torch.isfinite(mu).all()
            or not torch.isfinite(dist).all()
            or not torch.isfinite(pred_energy).all()
        ):
            continue

        loss, parts = ranking_finetune_loss(
            pred_energy, dockqs, pi, sigma, mu, dist, dist_threshold,
            C_batch=C_batch, **cfg,
        )
        if not torch.isfinite(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        stats['total'] += float(loss.item())
        stats['rank'] += float(parts['rank'].item())
        stats['mse'] += float(parts['mse'].item())
        stats['mdn'] += float(parts['mdn'].item())
        stats['n'] += 1

    if stats['n'] == 0:
        return {k: float('inf') for k in ('total', 'rank', 'mse', 'mdn')}
    n = stats['n']
    return {k: v / n for k, v in stats.items() if k != 'n'}


@torch.no_grad()
def eval_epoch(model, loader, device, dist_threshold, cfg):
    model.eval()
    stats = {'total': 0.0, 'rank': 0.0, 'mse': 0.0, 'mdn': 0.0, 'n': 0}
    all_preds, all_labels = [], []
    all_native_preds, all_decoy_preds = [], []
    succ_top1, n_batches = 0, 0

    for rec_batch, lig_batch, dockqs, is_natives, stems, names in loader:
        rec_batch = rec_batch.to(device)
        lig_batch = lig_batch.to(device)
        dockqs = dockqs.to(device)
        is_natives = is_natives.to(device)

        pi, sigma, mu, dist, C_batch, pred_energy = model(rec_batch, lig_batch)
        if (
            not torch.isfinite(pi).all()
            or not torch.isfinite(sigma).all()
            or not torch.isfinite(mu).all()
            or not torch.isfinite(dist).all()
            or not torch.isfinite(pred_energy).all()
        ):
            continue

        loss, parts = ranking_finetune_loss(
            pred_energy, dockqs, pi, sigma, mu, dist, dist_threshold,
            C_batch=C_batch, **cfg,
        )
        if not torch.isfinite(loss):
            continue

        stats['total'] += float(loss.item())
        stats['rank'] += float(parts['rank'].item())
        stats['mse'] += float(parts['mse'].item())
        stats['mdn'] += float(parts['mdn'].item())
        stats['n'] += 1

        preds_np = pred_energy.detach().cpu().numpy()
        labels_np = dockqs.detach().cpu().numpy()
        is_nat_np = is_natives.cpu().numpy().astype(bool)
        all_preds.extend(preds_np.tolist())
        all_labels.extend(labels_np.tolist())
        all_native_preds.extend(preds_np[is_nat_np].tolist())
        all_decoy_preds.extend(preds_np[~is_nat_np].tolist())

        # Top-1 success: highest score is the highest-DockQ pose in the batch
        if preds_np.size >= 2:
            n_batches += 1
            if int(preds_np.argmax()) == int(labels_np.argmax()):
                succ_top1 += 1

    if stats['n'] == 0:
        return {
            'total': float('inf'), 'rank': float('inf'),
            'mse': float('inf'), 'mdn': float('inf'),
            'native_mean': 0.0, 'decoy_mean': 0.0, 'gap': 0.0,
            'succ_top1': 0.0, 'spearman': 0.0,
        }

    n = stats['n']
    avg = {k: v / n for k, v in stats.items() if k != 'n'}
    nat_arr = np.asarray(all_native_preds) if all_native_preds else np.array([0.0])
    dec_arr = np.asarray(all_decoy_preds) if all_decoy_preds else np.array([0.0])
    avg['native_mean'] = float(nat_arr.mean())
    avg['decoy_mean'] = float(dec_arr.mean())
    avg['gap'] = avg['native_mean'] - avg['decoy_mean']
    avg['succ_top1'] = succ_top1 / max(n_batches, 1)

    from scipy.stats import spearmanr
    if len(all_preds) > 2:
        r_s, _ = spearmanr(all_preds, all_labels)
        avg['spearman'] = float(r_s) if np.isfinite(r_s) else 0.0
    else:
        avg['spearman'] = 0.0
    return avg


def main():
    parser = argparse.ArgumentParser(
        description='Within-target pairwise + DockQ MSE + MDN fine-tuning',
    )
    parser.add_argument(
        '--native_dir', default='',
        help='Native surfaces dir (pairs.csv). Empty OK with --decoy_only',
    )
    parser.add_argument('--decoy_dir', required=True)
    parser.add_argument('--decoy_csv', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--init_from', type=str, default=None)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_gaussians', type=int, default=10)
    parser.add_argument('--n_tf_blocks', type=int, default=6)
    parser.add_argument('--tf_heads', type=int, default=4)
    parser.add_argument('--cross_heads', type=int, default=8)
    parser.add_argument('--n_cross_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--dist_threshold', type=float, default=10.0)
    parser.add_argument('--in_channels', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batches_per_epoch', type=int, default=400)
    parser.add_argument('--val_batches_per_epoch', type=int, default=80)
    parser.add_argument('--decoy_per_native', type=int, default=15,
                        help='Decoys per batch (same stem); with native = 1+N')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_per_stem_decoy', type=int, default=None)
    parser.add_argument('--val_targets', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda_rank', type=float, default=1.0)
    parser.add_argument('--lambda_mse', type=float, default=0.5)
    parser.add_argument('--lambda_mdn', type=float, default=0.3)
    parser.add_argument(
        '--mdn_min_dockq', type=float, default=0.5,
        help='MDN loss only on samples with DockQ >= this (0=all)',
    )
    parser.add_argument('--pair_margin', type=float, default=0.1)
    parser.add_argument('--label_gap', type=float, default=0.05,
                        help='Min DockQ gap to form a ranking pair')
    parser.add_argument(
        '--decoy_only', action='store_true',
        help='Ignore natives; sample only same-stem decoys with DockQ',
    )
    parser.add_argument('--ratio_near_native', type=float, default=0.20)
    parser.add_argument('--ratio_hard', type=float, default=0.40)
    parser.add_argument('--ratio_medium', type=float, default=0.30)
    parser.add_argument('--ratio_easy', type=float, default=0.10)
    parser.add_argument(
        '--best_metric',
        choices=['succ_top1', 'spearman', 'gap'],
        default='succ_top1',
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device}')
    print(
        'Fine-tune: within-target pairwise + DockQ MSE + MDN '
        f'(λ_rank={args.lambda_rank}, λ_mse={args.lambda_mse}, '
        f'λ_mdn={args.lambda_mdn}, mdn_min_dockq={args.mdn_min_dockq})'
    )
    if args.decoy_only:
        print('Mode: decoy-only (no native required)')
    print('Eval after training: use --score_type energy')

    stems = set()
    with open(args.decoy_csv) as f:
        for row in csv.DictReader(f):
            stems.add(row['stem'])
    stems = sorted(stems)
    if len(stems) < 2:
        raise RuntimeError('至少需要 2 个带 decoy 的 target 才能切 train/val')
    val_targets = min(args.val_targets, len(stems) - 1)
    rng = random.Random(args.seed)
    rng.shuffle(stems)
    val_stems = set(stems[:val_targets])
    train_stems = set(stems[val_targets:])
    print(f'Train stems: {len(train_stems)}, Val stems: {len(val_stems)}')

    train_set = MixedDataset(
        args.native_dir, args.decoy_dir, args.decoy_csv,
        stem_filter=train_stems,
        max_per_stem_decoy=args.max_per_stem_decoy,
        in_channels=args.in_channels,
        decoy_only=args.decoy_only,
    )
    val_set = MixedDataset(
        args.native_dir, args.decoy_dir, args.decoy_csv,
        stem_filter=val_stems,
        max_per_stem_decoy=args.max_per_stem_decoy,
        in_channels=args.in_channels,
        decoy_only=args.decoy_only,
    )

    sampler_kwargs = dict(
        decoy_per_native=args.decoy_per_native,
        ratio_near_native=args.ratio_near_native,
        ratio_hard=args.ratio_hard,
        ratio_medium=args.ratio_medium,
        ratio_easy=args.ratio_easy,
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=SameTargetSampler(
            train_set, args.batches_per_epoch, **sampler_kwargs),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=SameTargetSampler(
            val_set, args.val_batches_per_epoch, **sampler_kwargs),
        collate_fn=collate,
    )

    model = DeepDock_PPI(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        n_gaussians=args.n_gaussians,
        n_transformer_blocks=args.n_tf_blocks,
        transformer_heads=args.tf_heads,
        use_global_attn=True,
        global_attn_layers=2,
        cross_attn_heads=args.cross_heads,
        n_cross_attn_layers=args.n_cross_layers,
        dist_threshold=args.dist_threshold,
        dropout_rate=args.dropout,
    ).to(device)

    if args.init_from and os.path.exists(args.init_from):
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        cur_sd = model.state_dict()
        compatible = {
            k: v for k, v in sd.items()
            if k in cur_sd and tuple(v.shape) == tuple(cur_sd[k].shape)
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(
            f'Init from {args.init_from}: loaded={len(compatible)}, '
            f'missing={len(missing)}, unexpected={len(unexpected)}'
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'参数量: {n_params:,}')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    cfg = dict(
        lambda_rank=args.lambda_rank,
        lambda_mse=args.lambda_mse,
        lambda_mdn=args.lambda_mdn,
        pair_margin=args.pair_margin,
        label_gap=args.label_gap,
        mdn_min_dockq=args.mdn_min_dockq,
    )

    best_value = -float('inf')
    log_rows = []
    header = (
        f"{'Ep':>3} {'TrTot':>7} {'TrRank':>7} {'TrMSE':>7} {'TrMDN':>7} "
        f"{'VaTot':>7} {'Gap':>6} {'Sρ':>6} {'S@1':>6} {'LR':>8} {'T':>5}"
    )
    print(header)
    print('─' * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = datetime.now()
        tr = train_epoch(
            model, train_loader, optimizer, device, args.dist_threshold, cfg,
        )
        va = eval_epoch(
            model, val_loader, device, args.dist_threshold, cfg,
        )
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = (datetime.now() - t0).seconds

        print(
            f"{epoch:>3} {tr['total']:>7.4f} {tr['rank']:>7.4f} "
            f"{tr['mse']:>7.4f} {tr['mdn']:>7.4f} "
            f"{va['total']:>7.4f} {va['gap']:>6.3f} "
            f"{va['spearman']:>6.3f} {va['succ_top1']:>6.3f} "
            f"{lr_now:>8.1e} {elapsed:>4}s"
        )

        log_rows.append({
            'epoch': epoch,
            **{f'train_{k}': v for k, v in tr.items()},
            **{f'val_{k}': v for k, v in va.items()},
        })

        cur_value = va[args.best_metric]
        if cur_value > best_value:
            best_value = cur_value
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val': va,
                'args': vars(args),
                'score_type_recommend': 'energy',
                'finetune_style': 'pairwise_mse_mdn',
            }, os.path.join(args.save_dir, 'Tscore_best.chk'))

    pd.DataFrame(log_rows).to_csv(
        os.path.join(args.save_dir, 'training_log.csv'), index=False)
    print(f'\n训练完成. Best {args.best_metric}: {best_value:.4f}')
    print(f'Checkpoint: {args.save_dir}/Tscore_best.chk')
    print('Inference: use --score_type energy')


if __name__ == '__main__':
    main()
