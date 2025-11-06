#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task2_plot.py

Читает CSV, сгенерированные task2:
  <prefix>_algo_row.csv
  <prefix>_algo_col.csv
  <prefix>_algo_block.csv

Ожидаемый формат строки (без заголовка):
    procs,rows,cols,overall,comp_max,comm_max

Поведение:
- Строит три графика (для выбранного procs):
    - overall time (s)
    - speedup
    - efficiency
  сравнивая три алгоритма по набору размеров R x C.

Выход:
  ./task2/data/plot/<prefix>_overall_P<procs>.png
  ./task2/data/plot/<prefix>_speedup_P<procs>.png
  ./task2/data/plot/<prefix>_eff_P<procs>.png
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


def find_file(prefix, algo):
    """
    Ищем файл для алгоритма среди типичных шаблонов.
    """
    candidates = [
        CSV_DIR / f"{prefix}_{algo}.csv",
        CSV_DIR / f"{prefix}_algo_{algo}.csv",
        CSV_DIR / f"{prefix}_algo{algo}.csv",
        CSV_DIR / f"{prefix}_algo-{algo}.csv",
        CSV_DIR / f"{prefix}_algo.{algo}.csv",
        CSV_DIR / f"{prefix}_{algo}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_csv_file(path):
    """
    Считывает CSV-файл в список записей (в порядке файла).
    Каждая запись — dict: procs, rows, cols, overall, comp_max, comm_max
    """
    rows = []
    with path.open() as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 6:
                print(f"Warning: пропускаю некорректную строку {path}:{lineno}: '{line}'")
                continue
            try:
                procs = int(parts[0])
                r = int(parts[1])
                c = int(parts[2])
                overall = float(parts[3])
                comp_max = float(parts[4])
                comm_max = float(parts[5])
            except ValueError:
                print(f"Warning: не могу распарсить строку {path}:{lineno}: '{line}'")
                continue
            rows.append({'procs': procs, 'rows': r, 'cols': c,
                         'overall': overall, 'comp_max': comp_max, 'comm_max': comm_max,
                         'lineno': lineno})
    return rows


def build_data_for_algo(prefix, algo):
    """
    Возвращает словарь с:
      - all_rows: список всех записей
      - baseline: {(r,c): overall} — последние найденные строки с procs==1
      - target_procs: int (procs из последней непустой строки) или None
      - parallel: {(r,c): overall} — записи с procs==target_procs
      - path: путь к файлу
    """
    path = find_file(prefix, algo)
    if path is None:
        print(f"Warning: файл для алгоритма '{algo}' не найден в {CSV_DIR}")
        return None

    rows = read_csv_file(path)
    if not rows:
        print(f"Warning: файл {path} пуст или не содержит корректных записей.")
        return {'all_rows': [], 'baseline': {}, 'target_procs': None, 'parallel': {}, 'path': path}

    baseline = {}
    for rec in rows:
        if rec['procs'] == 1:
            baseline[(rec['rows'], rec['cols'])] = rec['overall']

    target_procs = rows[-1]['procs']

    parallel = {}
    for rec in rows:
        if rec['procs'] == target_procs:
            parallel[(rec['rows'], rec['cols'])] = rec['overall']

    return {'all_rows': rows, 'baseline': baseline, 'target_procs': target_procs, 'parallel': parallel, 'path': path}


def build_combined_series(prefix):
    """
    Считывает данные для всех алгоритмов, выбирает общий набор размеров (R,C)
    присутствующих при выбранном target_procs (выбираем majority target_procs среди файлов).
    Возвращает (chosen_target, Ns_sorted, series)
    где series[algo] = {'overall': [...], 'speedup': [...], 'eff': [...] } aligned to Ns_sorted
    """
    algo_data = {}
    targets = []
    for algo in ALGOS:
        d = build_data_for_algo(prefix, algo)
        algo_data[algo] = d
        if d and d['target_procs'] is not None:
            targets.append(d['target_procs'])

    if not targets:
        raise SystemExit("Ошибка: не найдено ни одной корректной записи в CSV для всех алгоритмов.")

    # majority choice for target procs
    target_counts = {}
    for t in targets:
        target_counts[t] = target_counts.get(t, 0) + 1
    chosen_target = max(target_counts.items(), key=lambda kv: kv[1])[0]
    if len(target_counts) > 1:
        print(f"Warning: разные значения target_procs в файлах: {target_counts}. Выбрано procs = {chosen_target} (majority).")

    # union of (r,c) where there's a measurement with procs==chosen_target (in any file)
    Ns_set = set()
    for algo in ALGOS:
        d = algo_data.get(algo)
        if d is None:
            continue
        # collect entries with procs == chosen_target
        for rec in d['all_rows']:
            if rec['procs'] == chosen_target:
                Ns_set.add((rec['rows'], rec['cols']))

    if not Ns_set:
        raise SystemExit(f"Не найдена ни одна запись с procs={chosen_target} в CSV файлах.")

    # sort by area then rows then cols
    Ns = sorted(list(Ns_set), key=lambda rc: (rc[0] * rc[1], rc[0], rc[1]))

    # build series
    series = {}
    for algo in ALGOS:
        d = algo_data.get(algo)
        series[algo] = {'overall': [], 'speedup': [], 'eff': []}

        # build lookup for parallel measurements with chosen_target
        parallel_lookup = {}
        if d:
            for rec in d['all_rows']:
                if rec['procs'] == chosen_target:
                    parallel_lookup[(rec['rows'], rec['cols'])] = rec['overall']

        # baseline lookup (procs==1) — try algorithm's own baseline first
        baseline_lookup = d['baseline'] if d else {}

        for rc in Ns:
            overall = parallel_lookup.get(rc, float('nan'))
            series[algo]['overall'].append(overall)

            # find T1: algorithm's baseline or fallback to other algorithms' baseline
            T1_val = baseline_lookup.get(rc)
            if T1_val is None:
                # search in other algos' baseline
                for other in ALGOS:
                    if other == algo:
                        continue
                    dd = algo_data.get(other)
                    if dd and rc in dd['baseline']:
                        T1_val = dd['baseline'][rc]
                        break

            if T1_val is None or (isinstance(overall, float) and math.isnan(overall)):
                series[algo]['speedup'].append(float('nan'))
                series[algo]['eff'].append(float('nan'))
            else:
                if overall == 0.0:
                    sp = float('nan')
                    eff = float('nan')
                else:
                    sp = T1_val / overall
                    eff = sp / float(chosen_target)
                series[algo]['speedup'].append(sp)
                series[algo]['eff'].append(eff)

    return chosen_target, Ns, series


def plot_comparison(prefix, procs, Ns, series):
    labels = [f"{r}x{c}" for (r, c) in Ns]
    x = list(range(len(labels)))

    def save_plot(y_values_dict, ylabel, title, outname):
        plt.figure(figsize=(10, 6))
        for algo in ALGOS:
            ys = y_values_dict[algo]
            xs_plot = [xx for xx, vv in zip(x, ys) if not (isinstance(vv, float) and math.isnan(vv))]
            ys_plot = [vv for vv in ys if not (isinstance(vv, float) and math.isnan(vv))]
            if not xs_plot:
                continue
            plt.plot(xs_plot, ys_plot, marker='o', linestyle='-', label=algo)
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.xlabel('Matrix size R x C')
        plt.ylabel(ylabel)
        plt.title(f"{title} (procs={procs})")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        out_path = PLOT_DIR / outname
        plt.savefig(out_path)
        plt.close()
        return out_path

    out_over = save_plot({a: series[a]['overall'] for a in ALGOS}, 'overall time (s)', 'Overall time comparison', f"{prefix}_overall_P{procs}.png")
    out_speed = save_plot({a: series[a]['speedup'] for a in ALGOS}, 'speedup', 'Speedup comparison', f"{prefix}_speedup_P{procs}.png")
    out_eff = save_plot({a: series[a]['eff'] for a in ALGOS}, 'efficiency', 'Efficiency comparison', f"{prefix}_eff_P{procs}.png")
    return out_over, out_speed, out_eff


def print_table(Ns, series, procs):
    """
    Печать таблицы результатов в консоль для быстрой проверки.
    """
    header = ["size"] + [f"{algo}_T" for algo in ALGOS] + [f"{algo}_S" for algo in ALGOS] + [f"{algo}_E" for algo in ALGOS]
    print("\n" + "="*80)
    print(f"Results (procs={procs})")
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
    parser = argparse.ArgumentParser(description='Plot overall / speedup / efficiency comparison for task2 outputs')
    parser.add_argument('--prefix', '-p', required=True, help='prefix used for CSV files (e.g. myprefix)')
    args = parser.parse_args()

    prefix = args.prefix

    chosen_procs, Ns, series = build_combined_series(prefix)

    # печатаем таблицу для проверки
    print_table(Ns, series, chosen_procs)

    out_over, out_speed, out_eff = plot_comparison(prefix, chosen_procs, Ns, series)

    print("Saved plots:")
    print(" ->", out_over)
    print(" ->", out_speed)
    print(" ->", out_eff)


if __name__ == '__main__':
    main()
