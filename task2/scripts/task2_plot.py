#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task2_plot_compare.py

Читает три CSV, сгенерированные task2:
  <prefix>_algo_block.csv
  <prefix>_algo_col.csv
  <prefix>_algo_row.csv

Ожидаемый формат строки (без заголовка):
    procs,N,eff,overall,speedup

Строит три сравнительных графика по N (matrix size):
  - efficiency
  - overall time
  - speedup

Сохраняет PNG в ./task2/data/plot:
  <prefix>_eff_P<procs>.png
  <prefix>_overall_P<procs>.png
  <prefix>_speedup_P<procs>.png

Usage:
  python3 task2/scripts/task2_plot_compare.py --prefix myprefix --procs 4

Если какие-то файлы или значения N отсутствуют — скрипт выдаст предупреждение.
"""
import argparse
from pathlib import Path
import csv
import matplotlib.pyplot as plt

CSV_DIR = Path('./task2/data/csv')
PLOT_DIR = Path('./task2/data/plot')
PLOT_DIR.mkdir(parents=True, exist_ok=True)
ALGOS = ['block', 'col', 'row']


def read_algo_file(prefix, algo):
    # Пробуем возможные шаблоны имён файлов
    candidates = [
        CSV_DIR / f"{prefix}_algo_{algo}.csv",
        CSV_DIR / f"{prefix}_algo{algo}.csv",
        CSV_DIR / f"{prefix}_algo-{algo}.csv",
        CSV_DIR / f"{prefix}_algo.{algo}.csv",
        CSV_DIR / f"{prefix}_{algo}.csv",
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break
    if path is None:
        print(f"Warning: file not found for algorithm '{algo}'; tried: {candidates}")
        return {}

    data = {}
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Пропускаем строковые / заголовочные записи
            if line.lower().startswith('algo') or line.lower().startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                continue
            try:
                procs = int(parts[0])
                N = int(parts[1])
                eff = float(parts[2])
                overall = float(parts[3])
                speedup = float(parts[4])
            except ValueError:
                # пропускаем непарсируемые строки
                continue
            data.setdefault(procs, {})[N] = {'eff': eff, 'overall': overall, 'speedup': speedup}
    return data


def build_series(prefix, procs):
    # Собираем данные по всем алгоритмам и единому множеству N
    algo_data = {algo: read_algo_file(prefix, algo) for algo in ALGOS}
    Ns = set()
    for algo in ALGOS:
        Ns.update(algo_data.get(algo, {}).get(procs, {}).keys())
    if not Ns:
        raise SystemExit(f"No data found for procs={procs} in CSVs under {CSV_DIR}")
    Ns = sorted(Ns)

    series = {}
    for algo in ALGOS:
        series[algo] = {'eff': [], 'overall': [], 'speedup': []}
        pdata = algo_data.get(algo, {}).get(procs, {})
        for N in Ns:
            vals = pdata.get(N)
            if vals is None:
                print(f"Warning: missing {algo} procs={procs} N={N}")
                series[algo]['eff'].append(float('nan'))
                series[algo]['overall'].append(float('nan'))
                series[algo]['speedup'].append(float('nan'))
            else:
                series[algo]['eff'].append(vals['eff'])
                series[algo]['overall'].append(vals['overall'])
                series[algo]['speedup'].append(vals['speedup'])
    return Ns, series


def plot_continuous(Ns, series, prefix, procs):
    # Маркеры/цвета для читаемости (имена алгоритмов оставлены на английском)
    markers = {'row': 'o', 'col': 's', 'block': '^'}
    colors = {'row': 'C0', 'col': 'C1', 'block': 'C2'}

    def plot_metric(key, ylabel, outname):
        # Рисуем непрерывные линии для каждого алгоритма, пропуская NaN
        plt.figure(figsize=(8,5))
        for algo in ALGOS:
            y = series[algo][key]
            xs = []
            ys = []
            for x, val in zip(Ns, y):
                if val == val:  # проверка на NaN
                    xs.append(x); ys.append(val)
            if not xs:
                continue
            plt.plot(xs, ys, marker=markers[algo], linestyle='-', label=algo, color=colors[algo])
        plt.xlabel('N (matrix size)')
        plt.xticks(Ns)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} (procs={procs}) — {prefix}")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        out = PLOT_DIR / outname
        plt.savefig(out)
        plt.close()
        return out

    out_eff = plot_metric('eff', 'efficiency', f"{prefix}_eff_P{procs}.png")
    out_over = plot_metric('overall', 'overall time (s)', f"{prefix}_overall_P{procs}.png")
    out_speed = plot_metric('speedup', 'speedup', f"{prefix}_speedup_P{procs}.png")
    return out_eff, out_over, out_speed


def main():
    parser = argparse.ArgumentParser(description='Plot efficiency/overall/speedup vs N for three algorithms')
    parser.add_argument('--prefix', '-p', required=True, help='prefix used for CSV files (e.g. myprefix)')
    parser.add_argument('--procs', type=int, required=True, help='number of processes to filter (e.g. 4)')
    args = parser.parse_args()

    Ns, series = build_series(args.prefix, args.procs)
    out = plot_continuous(Ns, series, args.prefix, args.procs)
    print('Saved plots:')
    for p in out:
        print(' ->', p)

if __name__ == '__main__':
    main()
