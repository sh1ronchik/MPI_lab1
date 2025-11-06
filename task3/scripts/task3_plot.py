#!/usr/bin/env python3
# task3/scripts/task3_plot.py
import csv
import math
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

CSV_DIR = Path('./task3/data/csv')
PLOT_DIR = Path('./task3/data/plot')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    rows = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s: continue
            parts = [p.strip() for p in s.split(',')]
            if len(parts) < 5: continue
            try:
                procs = int(parts[0]); N = int(parts[1])
                overall = float(parts[2]); comp_max = float(parts[3]); comm_max = float(parts[4])
            except:
                continue
            rows.append({'procs': procs, 'N': N, 'overall': overall, 'comp': comp_max, 'comm': comm_max})
    return rows

def build_series(prefix):
    path = CSV_DIR / f"{prefix}_cannon.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    rows = read_csv(path)
    if not rows:
        raise SystemExit(f"No data in {path}")
    # baseline: entries with procs==1 (use latest for each N)
    baseline = {}
    for rec in rows:
        if rec['procs'] == 1:
            baseline[rec['N']] = rec['overall']
    # target procs = last row's procs
    target = rows[-1]['procs']
    # pick all rows with procs == target
    parallel = {rec['N']: rec['overall'] for rec in rows if rec['procs'] == target}
    Ns = sorted(parallel.keys())
    overall = [parallel[n] for n in Ns]
    speedup = []
    eff = []
    for n in Ns:
        T1 = baseline.get(n)
        Tp = parallel.get(n)
        if T1 is None or Tp == 0:
            speedup.append(float('nan')); eff.append(float('nan'))
        else:
            sp = T1 / Tp
            speedup.append(sp)
            eff.append(sp / float(target))
    return target, Ns, overall, speedup, eff

def plot_series(prefix, procs, Ns, overall, speedup, eff):
    labels = [str(n) for n in Ns]
    x = list(range(len(Ns)))
    plt.figure(figsize=(8,5))
    plt.plot(x, overall, marker='o', linestyle='-')
    plt.xticks(x, labels); plt.xlabel('N'); plt.ylabel('overall (s)')
    plt.title(f"{prefix} Cannon overall (procs={procs})"); plt.grid(True); plt.tight_layout()
    out1 = PLOT_DIR / f"{prefix}_cannon_overall_P{procs}.png"; plt.savefig(out1); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(x, speedup, marker='o', linestyle='-')
    plt.xticks(x, labels); plt.xlabel('N'); plt.ylabel('speedup')
    plt.title(f"{prefix} Cannon speedup (procs={procs})"); plt.grid(True); plt.tight_layout()
    out2 = PLOT_DIR / f"{prefix}_cannon_speedup_P{procs}.png"; plt.savefig(out2); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(x, eff, marker='o', linestyle='-')
    plt.xticks(x, labels); plt.xlabel('N'); plt.ylabel('efficiency')
    plt.title(f"{prefix} Cannon efficiency (procs={procs})"); plt.grid(True); plt.tight_layout()
    out3 = PLOT_DIR / f"{prefix}_cannon_eff_P{procs}.png"; plt.savefig(out3); plt.close()

    return out1, out2, out3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', '-p', default='task3')
    args = parser.parse_args()
    prefix = args.prefix
    procs, Ns, overall, speedup, eff = build_series(prefix)
    print("procs (target):", procs)
    print("Ns:", Ns)
    for n,o,s,e in zip(Ns, overall, speedup, eff):
        print(f"N={n} overall={o:.6e} speedup={s:.4f} eff={e:.4f}")
    out1,out2,out3 = plot_series(prefix, procs, Ns, overall, speedup, eff)
    print("Saved:", out1, out2, out3)
