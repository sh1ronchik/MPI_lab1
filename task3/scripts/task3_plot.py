#!/usr/bin/env python3
# task3/scripts/task3_plot.py
# -*- coding: utf-8 -*-
"""
Построение графиков для task3 (Cannon).
Ожидаемый CSV: ./task3/data/csv/<prefix>_cannon.csv
Формат строк: procs,N,overall

Построит:
 - overall vs N
 - speedup vs N  (использует baseline procs==1, если есть)
 - efficiency vs N
 - speedup vs procs (несколько кривых — для разных N, где есть baseline)
 - efficiency vs procs
"""
import argparse
from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

CSV_DIR = Path('./task3/data/csv')
PLOT_DIR = Path('./task3/data/plot')
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    rows = []
    with path.open() as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 3:
                print(f"Warning: пропускаю некорректную строку {path}:{lineno}: '{line}'")
                continue
            try:
                procs = int(parts[0])
                N = int(parts[1])
                overall = float(parts[2])
            except ValueError:
                print(f"Warning: не могу распарсить строку {path}:{lineno}: '{line}'")
                continue
            rows.append({'procs': procs, 'N': N, 'overall': overall, 'lineno': lineno})
    return rows


def build_lookup(rows):
    """Построить lookup структуры:
       - by_procs: {procs: {N: overall}}
       - baseline: {N: overall} для записей с procs==1 (последняя запись для N перезаписывает)
       - last_procs: procs из последней непустой строки
       - Ns_all: sorted set of all N
    """
    if not rows:
        return None

    by_procs = {}
    baseline = {}
    for rec in rows:
        by_procs.setdefault(rec['procs'], {})[rec['N']] = rec['overall']
        if rec['procs'] == 1:
            baseline[rec['N']] = rec['overall']

    last_procs = rows[-1]['procs']
    Ns_all = sorted({rec['N'] for rec in rows})
    return by_procs, baseline, last_procs, Ns_all


def plot_lines(x_labels, series_dict, xlabel, ylabel, title, outpath):
    """
    series_dict: {label: [y0,y1,...]} aligned to x_labels
    Пропускаются NaN при построении каждой кривой.
    """
    x = list(range(len(x_labels)))
    plt.figure(figsize=(10, 6))
    for label, ys in series_dict.items():
        xs_plot = [i for i, v in enumerate(ys) if not (isinstance(v, float) and math.isnan(v))]
        ys_plot = [v for v in ys if not (isinstance(v, float) and math.isnan(v))]
        if not xs_plot:
            continue
        plt.plot(xs_plot, ys_plot, marker='o', linestyle='-', label=label)
    plt.xticks(x, [str(xl) for xl in x_labels], rotation=45, ha='right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return outpath


def main():
    parser = argparse.ArgumentParser(description='Plot Cannon results')
    parser.add_argument('--prefix', '-p', default='task3', help='prefix for CSV file (default: task3)')
    args = parser.parse_args()
    prefix = args.prefix

    csv_path = CSV_DIR / f"{prefix}_cannon.csv"
    if not csv_path.exists():
        raise SystemExit(f"File not found: {csv_path}")

    rows = read_csv(csv_path)
    if not rows:
        raise SystemExit(f"No valid data in {csv_path}")

    by_procs, baseline, target_procs, Ns_all = build_lookup(rows)

    print(f"Target procs (from last row): {target_procs}")
    print(f"All measured N: {Ns_all}")
    print(f"Baseline (procs==1) has entries for N: {sorted(baseline.keys())}")

    # --- 1) overall / speedup / efficiency vs N for target_procs ---
    # collect Ns where we have measurement for target_procs
    parallel_map = by_procs.get(target_procs, {})
    Ns_target = sorted(parallel_map.keys())
    if not Ns_target:
        print(f"No entries found with procs={target_procs}. Cannot plot vs N.")
    else:
        overall_vs_N = [parallel_map.get(n, float('nan')) for n in Ns_target]
        speed_vs_N = []
        eff_vs_N = []
        for n in Ns_target:
            T_p = parallel_map.get(n)
            T1 = baseline.get(n)
            if T1 is None or T_p is None or T_p == 0.0:
                speed_vs_N.append(float('nan'))
                eff_vs_N.append(float('nan'))
            else:
                sp = T1 / T_p
                speed_vs_N.append(sp)
                eff_vs_N.append(sp / float(target_procs))

        out1 = plot_lines(Ns_target, {'overall': overall_vs_N},
                          'N', 'overall (s)', f'{prefix}: overall vs N (procs={target_procs})',
                          PLOT_DIR / f"{prefix}_cannon_overall_P{target_procs}.png")
        out2 = plot_lines(Ns_target, {'speedup': speed_vs_N},
                          'N', 'speedup', f'{prefix}: speedup vs N (procs={target_procs})',
                          PLOT_DIR / f"{prefix}_cannon_speedup_P{target_procs}.png")
        out3 = plot_lines(Ns_target, {'efficiency': eff_vs_N},
                          'N', 'efficiency', f'{prefix}: efficiency vs N (procs={target_procs})',
                          PLOT_DIR / f"{prefix}_cannon_eff_P{target_procs}.png")

        print("Saved size-based plots:")
        print(" ->", out1)
        print(" ->", out2)
        print(" ->", out3)

    # --- 2) speedup / efficiency vs procs (for each N that has baseline T1) ---
    # build list of procs available (sorted)
    procs_list = sorted(by_procs.keys())

    # choose Ns to plot: those that have baseline (procs==1) and at least one measurement with procs>1
    Ns_for_procs_plot = sorted([n for n in Ns_all if (n in baseline and any(n in by_procs[p] for p in procs_list))])

    if not Ns_for_procs_plot:
        print("No N has baseline (procs==1) AND other measurements — cannot plot speedup vs procs.")
    else:
        # build series: for each N, list of overall aligned to procs_list (NaN when missing)
        speed_series = {}
        eff_series = {}
        for n in Ns_for_procs_plot:
            T1 = baseline.get(n)
            ys_speed = []
            ys_eff = []
            for p in procs_list:
                Tp = by_procs.get(p, {}).get(n)
                if T1 is None or Tp is None or Tp == 0.0:
                    ys_speed.append(float('nan'))
                    ys_eff.append(float('nan'))
                else:
                    sp = T1 / Tp
                    ys_speed.append(sp)
                    ys_eff.append(sp / float(p))
            speed_series[f"N={n}"] = ys_speed
            eff_series[f"N={n}"] = ys_eff

        out4 = plot_lines(procs_list, speed_series,
                          'procs', 'speedup', f'{prefix}: speedup vs procs (various N)',
                          PLOT_DIR / f"{prefix}_cannon_speedup_vs_procs.png")
        out5 = plot_lines(procs_list, eff_series,
                          'procs', 'efficiency', f'{prefix}: efficiency vs procs (various N)',
                          PLOT_DIR / f"{prefix}_cannon_eff_vs_procs.png")
        print("Saved procs-based plots:")
        print(" ->", out4)
        print(" ->", out5)

    # --- печать таблицы для быстрой проверки ---
    print("\nSummary table:")
    header = "N | procs | overall(s) | speedup | efficiency"
    print(header)
    printed_rows = []
    for n in sorted(Ns_all):
        for p in sorted(by_procs.keys()):
            Tp = by_procs.get(p, {}).get(n)
            if Tp is None:
                continue
            T1 = baseline.get(n)
            if T1 is None or Tp == 0.0:
                sp = float('nan'); eff = float('nan')
            else:
                sp = T1 / Tp
                eff = sp / float(p)
            printed_rows.append((n, p, Tp, sp, eff))
            print(f"{n:4d} | {p:5d} | {Tp:12.6e} | {sp:7.3f} | {eff:7.3f}")

    print("\nDone. Plots saved to:", PLOT_DIR)


if __name__ == '__main__':
    main()
