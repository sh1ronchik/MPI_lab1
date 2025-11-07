#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task2_plot.py

Читает CSV-файлы (формат строк: procs,rows,cols,overall) для трёх алгоритмов:
  <prefix>_algo_row.csv
  <prefix>_algo_col.csv
  <prefix>_algo_block.csv

Генерирует 5 графиков:
  1) Time vs matrix size (используем p = chosen_target (majority из файлов))
  2) Speedup vs matrix size (тот же p, baseline - procs==1 для того же размера)
  3) Efficiency vs matrix size
  4) Speedup vs procs (для каждого алгоритма берём его N_max (наибольшая матрица) )
  5) Efficiency vs procs

Выход: ./task2/data/plot/<prefix>_*.png
"""
import csv
import math
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

CSV_DIR = Path('./task2/data/csv')
PLOT_DIR = Path('./task2/data/plot')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ALGOS = ['algo_row', 'algo_col', 'algo_block']
LABELS = {'algo_row': 'row', 'algo_col': 'col', 'algo_block': 'block'}
COLORS = {'algo_row': 'C0', 'algo_col': 'C1', 'algo_block': 'C2'}
MARKERS = {'algo_row': 'o', 'algo_col': 's', 'algo_block': '^'}


def find_file(prefix, algo):
    # возможные шаблоны имён (популярные варианты)
    candidates = [
        CSV_DIR / f"{prefix}_{algo}.csv",
        CSV_DIR / f"{prefix}_algo_{algo}.csv",
        CSV_DIR / f"{prefix}_algo{algo}.csv",
        CSV_DIR / f"{prefix}_{algo}.csv",
        CSV_DIR / f"{prefix}-{algo}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_csv_simple(path):
    """
    Считывает CSV в список записей:
    each = {'procs': int, 'rows': int, 'cols': int, 'overall': float}
    """
    rows = []
    with path.open('r') as f:
        rdr = csv.reader(f)
        for lineno, r in enumerate(rdr, start=1):
            if not r:
                continue
            # допускаем возможные пробелы
            parts = [c.strip() for c in r if c is not None]
            if len(parts) < 4:
                # игнорируем некорректные строки
                continue
            try:
                procs = int(parts[0])
                rows_i = int(parts[1])
                cols_i = int(parts[2])
                overall = float(parts[3])
            except Exception:
                continue
            rows.append({'procs': procs, 'rows': rows_i, 'cols': cols_i, 'overall': overall, 'lineno': lineno})
    return rows


def build_data_for_algo(prefix, algo):
    path = find_file(prefix, algo)
    if path is None:
        print(f"Warning: file for algorithm '{algo}' not found in {CSV_DIR}")
        return None
    rows = read_csv_simple(path)
    if not rows:
        print(f"Warning: file {path} empty or no valid rows")
        return {'all_rows': [], 'baseline': {}, 'target_procs': None, 'parallel': {}, 'path': path}

    # baseline: map (r,c) -> overall for procs == 1 (if multiple entries, last wins)
    baseline = {}
    for rec in rows:
        if rec['procs'] == 1:
            baseline[(rec['rows'], rec['cols'])] = rec['overall']

    # target_procs = procs value from last non-empty row in file
    target_procs = rows[-1]['procs']

    # parallel lookup for target_procs
    parallel = {}
    for rec in rows:
        if rec['procs'] == target_procs:
            parallel[(rec['rows'], rec['cols'])] = rec['overall']

    return {'all_rows': rows, 'baseline': baseline, 'target_procs': target_procs, 'parallel': parallel, 'path': path}


def build_combined_series(prefix):
    """
    Возвращает:
      chosen_target (p для графиков vs size),
      Ns (sorted list of (r,c) chosen),
      series: dict algo -> {'overall': [...], 'speedup': [...], 'eff': [...]}
    """
    algo_data = {}
    targets = []
    for algo in ALGOS:
        d = build_data_for_algo(prefix, algo)
        algo_data[algo] = d
        if d and d['target_procs'] is not None:
            targets.append(d['target_procs'])

    if not targets:
        raise SystemExit("No valid target_procs found in any CSV files.")

    # choose majority target_procs (в случае расхождений)
    counts = {}
    for t in targets:
        counts[t] = counts.get(t, 0) + 1
    chosen_target = max(counts.items(), key=lambda kv: kv[1])[0]
    if len(counts) > 1:
        print(f"Warning: different target_procs in files: {counts}. Chosen p = {chosen_target} (majority).")

    # collect union of (r,c) that have measurement with procs == chosen_target in any file
    Ns_set = set()
    for algo in ALGOS:
        d = algo_data.get(algo)
        if not d:
            continue
        for rec in d['all_rows']:
            if rec['procs'] == chosen_target:
                Ns_set.add((rec['rows'], rec['cols']))

    if not Ns_set:
        raise SystemExit(f"No entries with procs={chosen_target} found in CSV files.")

    # sort Ns: by area (rows*cols), then rows, then cols
    Ns = sorted(list(Ns_set), key=lambda rc: (rc[0] * rc[1], rc[0], rc[1]))

    # build series per algorithm aligned to Ns
    series = {}
    for algo in ALGOS:
        d = algo_data.get(algo)
        series[algo] = {'overall': [], 'speedup': [], 'eff': []}

        # lookup maps
        parallel_lookup = {}
        baseline_lookup = {}
        if d:
            for rec in d['all_rows']:
                if rec['procs'] == chosen_target:
                    parallel_lookup[(rec['rows'], rec['cols'])] = rec['overall']
                if rec['procs'] == 1:
                    baseline_lookup[(rec['rows'], rec['cols'])] = rec['overall']

        # for each (r,c) build overall, speedup, eff
        for rc in Ns:
            overall = parallel_lookup.get(rc, float('nan'))
            series[algo]['overall'].append(overall)

            # find baseline T1: try algorithm's own baseline, else search other algos' baselines
            T1 = baseline_lookup.get(rc)
            if T1 is None:
                for other in ALGOS:
                    if other == algo:
                        continue
                    od = algo_data.get(other)
                    if od and rc in od['baseline']:
                        T1 = od['baseline'][rc]
                        break

            if T1 is None or (isinstance(overall, float) and math.isnan(overall)):
                series[algo]['speedup'].append(float('nan'))
                series[algo]['eff'].append(float('nan'))
            else:
                if overall == 0.0:
                    series[algo]['speedup'].append(float('nan'))
                    series[algo]['eff'].append(float('nan'))
                else:
                    sp = T1 / overall
                    eff = sp / float(chosen_target)
                    series[algo]['speedup'].append(sp)
                    series[algo]['eff'].append(eff)

    return chosen_target, Ns, series


def plot_vs_size(prefix, p, Ns, series):
    """
    Рисует три графика по размерам: time, speedup, eff (все три алгоритма на каждом графике)
    X-ось — дискретные метки "RxC".
    """
    labels = [f"{r}x{c}" for (r, c) in Ns]
    x = list(range(len(labels)))

    def save_plot(y_values_dict, ylabel, title, outname):
        plt.figure(figsize=(10, 6))
        for algo in ALGOS:
            ys = y_values_dict[algo]
            # filter nan values but keep indices consistent
            xs_plot = [xi for xi, vv in zip(x, ys) if not (isinstance(vv, float) and math.isnan(vv))]
            ys_plot = [vv for vv in ys if not (isinstance(vv, float) and math.isnan(vv))]
            if not xs_plot:
                continue
            plt.plot(xs_plot, ys_plot, marker=MARKERS[algo], linestyle='-', color=COLORS[algo], label=LABELS.get(algo, algo))
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.xlabel('Matrix size R x C')
        plt.ylabel(ylabel)
        plt.title(f"{title} (p={p})")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        out_path = PLOT_DIR / outname
        plt.savefig(out_path)
        plt.close()
        return out_path

    out1 = save_plot({a: series[a]['overall'] for a in ALGOS}, 'time (s)', 'Overall time vs matrix size', f"{prefix}_time_vs_size_P{p}.png")
    out2 = save_plot({a: series[a]['speedup'] for a in ALGOS}, 'speedup', 'Speedup vs matrix size', f"{prefix}_speedup_vs_size_P{p}.png")
    out3 = save_plot({a: series[a]['eff'] for a in ALGOS}, 'efficiency', 'Efficiency vs matrix size', f"{prefix}_eff_vs_size_P{p}.png")
    return out1, out2, out3


def build_and_plot_vs_procs(prefix):
    """
    Для каждого алгоритма отдельно берём его максимальный размер матрицы (по area),
    затем формируем speedup/eff vs procs (и рисуем все алгоритмы на одном графике).
    """
    algo_points = {}  # algo -> dict with 'Nmax', 'procs_list', 'map' ( (p)->overall )
    procs_union = set()
    for algo in ALGOS:
        path = find_file(prefix, algo)
        if path is None:
            algo_points[algo] = None
            continue
        rows = read_csv_simple(path)
        if not rows:
            algo_points[algo] = None
            continue
        # find Nmax by area rows*cols (take the largest area present in the file)
        # if multiple entries for same area choose one with largest rows then cols
        best = None
        for rec in rows:
            area = rec['rows'] * rec['cols']
            if best is None:
                best = rec
            else:
                barea = best['rows'] * best['cols']
                if area > barea or (area == barea and (rec['rows'] > best['rows'] or rec['cols'] > best['cols'])):
                    best = rec
        if best is None:
            algo_points[algo] = None
            continue
        Nmax = (best['rows'], best['cols'])
        # build map p -> overall for this Nmax
        map_p = {}
        procs_list = set()
        for rec in rows:
            if (rec['rows'], rec['cols']) == Nmax:
                map_p[rec['procs']] = rec['overall']
                procs_list.add(rec['procs'])
                procs_union.add(rec['procs'])
        # also need baseline T1 for this Nmax: if not present in this algo, try find in other algos
        algo_points[algo] = {'Nmax': Nmax, 'map': map_p, 'procs': sorted(procs_list)}

    # union of procs values
    xs_procs = sorted(procs_union)
    if not xs_procs:
        raise SystemExit("No procs entries found for any algorithm to plot speedup vs procs.")

    # build series for plotting: each algo has xs (subset) and ys (speedup) and eff
    speed_series = {}
    eff_series = {}
    for algo in ALGOS:
        info = algo_points.get(algo)
        if not info:
            speed_series[algo] = ([], [])
            eff_series[algo] = ([], [])
            continue
        Nmax = info['Nmax']
        map_p = info['map']
        # find baseline T1: try own map at p==1, else search other algorithms' files for p==1 at this Nmax
        T1 = map_p.get(1)
        if T1 is None:
            # search other files
            for other in ALGOS:
                if other == algo:
                    continue
                other_path = find_file(prefix, other)
                if other_path:
                    other_rows = read_csv_simple(other_path)
                    for rec in other_rows:
                        if rec['procs'] == 1 and (rec['rows'], rec['cols']) == Nmax:
                            T1 = rec['overall']
                            break
                if T1 is not None:
                    break
        xs_algo = []
        ys_speed = []
        ys_eff = []
        for p in sorted(map_p.keys()):
            Tp = map_p.get(p)
            if Tp is None or T1 is None or Tp == 0.0:
                # skip or record nan
                xs_algo.append(p)
                ys_speed.append(float('nan'))
                ys_eff.append(float('nan'))
            else:
                S = T1 / Tp
                E = S / float(p)
                xs_algo.append(p)
                ys_speed.append(S)
                ys_eff.append(E)
        speed_series[algo] = (xs_algo, ys_speed)
        eff_series[algo] = (xs_algo, ys_eff)

    # plot speedup vs procs
    plt.figure(figsize=(10, 6))
    for algo in ALGOS:
        xs_algo, ys = speed_series[algo]
        if not xs_algo:
            continue
        # filter nan
        xs_plot = [x for x, y in zip(xs_algo, ys) if not (isinstance(y, float) and math.isnan(y))]
        ys_plot = [y for y in ys if not (isinstance(y, float) and math.isnan(y))]
        if not xs_plot:
            continue
        plt.plot(xs_plot, ys_plot, marker=MARKERS[algo], linestyle='-', color=COLORS[algo], label=f"{LABELS.get(algo,algo)} N={algo_points[algo]['Nmax'] if algo_points[algo] else 'N/A'}")
    plt.xlabel('procs')
    plt.ylabel('speedup')
    plt.title('Speedup vs procs (per-algo Nmax)')
    plt.xticks(sorted(xs_procs))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    out_speed = PLOT_DIR / f"{prefix}_speedup_vs_procs.png"
    plt.tight_layout(); plt.savefig(out_speed); plt.close()

    # plot efficiency vs procs
    plt.figure(figsize=(10, 6))
    for algo in ALGOS:
        xs_algo, ys = eff_series[algo]
        if not xs_algo:
            continue
        xs_plot = [x for x, y in zip(xs_algo, ys) if not (isinstance(y, float) and math.isnan(y))]
        ys_plot = [y for y in ys if not (isinstance(y, float) and math.isnan(y))]
        if not xs_plot:
            continue
        plt.plot(xs_plot, ys_plot, marker=MARKERS[algo], linestyle='-', color=COLORS[algo], label=f"{LABELS.get(algo,algo)} N={algo_points[algo]['Nmax'] if algo_points[algo] else 'N/A'}")
    plt.xlabel('procs')
    plt.ylabel('efficiency')
    plt.title('Efficiency vs procs (per-algo Nmax)')
    plt.xticks(sorted(xs_procs))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    out_eff = PLOT_DIR / f"{prefix}_eff_vs_procs.png"
    plt.tight_layout(); plt.savefig(out_eff); plt.close()

    return out_speed, out_eff


def print_table(Ns, series, p):
    header = ["size"] + [f"{a}_T" for a in ALGOS] + [f"{a}_S" for a in ALGOS] + [f"{a}_E" for a in ALGOS]
    print("\n" + "="*80)
    print(f"Results (p={p})")
    print(",".join(header))
    for i, rc in enumerate(Ns):
        row = [f"{rc[0]}x{rc[1]}"]
        for algo in ALGOS:
            val = series[algo]['overall'][i]
            row.append(f"{val:.6e}" if not (isinstance(val, float) and math.isnan(val)) else "nan")
        for algo in ALGOS:
            val = series[algo]['speedup'][i]
            row.append(f"{val:.4f}" if not (isinstance(val, float) and math.isnan(val)) else "nan")
        for algo in ALGOS:
            val = series[algo]['eff'][i]
            row.append(f"{val:.4f}" if not (isinstance(val, float) and math.isnan(val)) else "nan")
        print(",".join(row))
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot task2 results (matrix-vector) comparisons')
    parser.add_argument('--prefix', '-p', required=True, help='prefix for CSV files (e.g. myprefix)')
    args = parser.parse_args()
    prefix = args.prefix

    chosen_p, Ns, series = build_combined_series(prefix)

    print_table(Ns, series, chosen_p)

    out1, out2, out3 = plot_vs_size(prefix, chosen_p, Ns, series)
    print("Saved:", out1, out2, out3)

    out_speed, out_eff = build_and_plot_vs_procs(prefix)
    print("Saved:", out_speed, out_eff)


if __name__ == "__main__":
    main()
