"""Extract CAPRI Score_set reference sequences from a tar.bz2 of decoy PDBs.

Same target decoys share sequences; MODEL 1 is enough.
Writes FASTA for DIPS-vs-CAPRI dedup (query for MMseqs2).

Usage:
  python example/extract_capri_sequences.py \\
      --tar /path/to/Scoreset_v2022_Scorers.tar.bz2 \\
      --out data/capri_exclude/capri_ref.fasta
"""
from __future__ import annotations

import argparse
import bz2
import io
import os
import sys
import tarfile

THREE2ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'SEC': 'U', 'PYL': 'O', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
}


def model1_sequences(pdb_text):
    """Parse first MODEL in PDB text -> {chain: seq}."""
    seqs = {}
    seen = {}
    in_model = False
    started = False
    for line in pdb_text.splitlines():
        if line.startswith('MODEL'):
            if started:
                break
            in_model = True
            started = True
            continue
        if line.startswith('ENDMDL'):
            if in_model:
                break
        if line.startswith(('ATOM', 'HETATM')):
            resn = line[17:20].strip()
            one = THREE2ONE.get(resn)
            if one is None:
                continue
            chain = line[21].strip() or '_'
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (resseq, icode)
            s = seen.setdefault(chain, set())
            if key in s:
                continue
            s.add(key)
            seqs.setdefault(chain, []).append(one)
    return {c: ''.join(v) for c, v in seqs.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tar', required=True, help='CAPRI Score_set tar.bz2 (decoy PDBs)')
    p.add_argument(
        '--out', default='data/capri_exclude/capri_ref.fasta',
        help='Output FASTA path',
    )
    args = p.parse_args()

    if not os.path.isfile(args.tar):
        sys.exit(f'missing tar: {args.tar}')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)

    entries = []
    n_files = 0
    with tarfile.open(args.tar, 'r:bz2') as tf:
        members = [m for m in tf.getmembers() if m.name.endswith('.pdb')]
        members.sort(key=lambda m: m.name)
        for m in members:
            n_files += 1
            base = os.path.basename(m.name)[:-4]
            tag = base.replace('S-', '').replace('.', '_')
            f = tf.extractfile(m)
            text = f.read().decode('utf-8', errors='replace')
            seqs = model1_sequences(text)
            for chain, seq in seqs.items():
                if len(seq) >= 20:
                    entries.append((f'{tag}_{chain}', seq))

    with open(args.out, 'w') as out:
        for name, seq in entries:
            out.write(f'>{name}\n')
            for i in range(0, len(seq), 80):
                out.write(seq[i:i + 80] + '\n')

    print(f'parsed {n_files} decoy PDBs')
    print(f'wrote {len(entries)} chains -> {args.out}')


if __name__ == '__main__':
    main()
