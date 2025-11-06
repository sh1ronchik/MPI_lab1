#!/usr/bin/env python3
"""
plot_task1.py

Строит столбчатые графики времени для трёх алгоритмов:
  A: Send/Recv, B: Reduce, C: Isend/Irecv
на основе CSV, сгенерированных C-программой:
  ./task1/data/csv/<prefix>_algo{A,B,C}.csv

Выход: PNG в task1/data/plots:
  <prefix>_overall.png, <prefix>_comp.png, <prefix>_comm.png,
  <prefix>_speedup.png, <prefix>_efficiency.png

Usage:
  python3 task1/data/scripts/plot_task1.py [prefix]
"""
import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR.parent / 'data' / 'csv'
PLOTS_DIR = SCRIPT_DIR.parent / 'data' / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_PREFIX = 'task1'
ALGOS = {'A': 'Send/Recv', 'B': 'Reduce', 'C': 'Isend/Irecv'}


def read_baseline_and_last(path):
    """
    Считывает CSV и возвращает пару (baseline_row, last_row) в виде списков строк.
    baseline_row — первая встреченная строка с procs == 1 (если есть).
    last_row — последняя непустая строка файла.
    Если соответствующая строка не найдена — возвращается None на её месте.
    """
    if not path.exists():
        return None, None
    baseline = None
    last = None
    with path.open('r', newline='') as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or not any(cell.strip() for cell in row):
                continue
            last = row
            # Попытка распарсить procs для определения baseline
            try:
                procs = int(row[0])
                if procs == 1 and baseline is None:
                    baseline = row
            except Exception:
                # Некорректная строка — пропускаем для baseline, но она всё равно может быть last
                pass
    return baseline, last


def parse_row(row):
    """Парсит строку в формат: procs,total_points,overall,comp_max,comm_max,pi"""
    if not row or len(row) < 6:
        return None
    try:
        procs = int(row[0])
        total_points = int(row[1])
        overall = float(row[2])
        comp_max = float(row[3])
        comm_max = float(row[4])
        pi_est = float(row[5])
        return {'procs': procs, 'total_points': total_points,
                'overall': overall, 'comp_max': comp_max, 'comm_max': comm_max, 'pi': pi_est}
    except ValueError:
        return None


def make_bar(values, labels, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(values))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PREFIX

    # Считываем baseline (procs==1 где угодно) и последнюю строку (parallel run) для каждого алгоритма
    data_baseline = {}
    data_last = {}
    for k in ['A', 'B', 'C']:
        fname = f"{prefix}_algo{k}.csv"
        path = CSV_DIR / fname
        baseline_row, last_row = read_baseline_and_last(path)
        data_baseline[k] = parse_row(baseline_row) if baseline_row else None
        data_last[k] = parse_row(last_row) if last_row else None
        if data_last[k] is None:
            print(f"Warning: file {path} missing or has invalid last-row content")
        if data_baseline[k] is None:
            print(f"Note: baseline (procs==1) not found in {path} — S/E won't be computed for this algorithm")

    # Подготовка bar-диаграмм для overall / comp_max / comm_max (используем последнюю строку)
    labels = []
    overall_values = []
    comp_values = []
    comm_values = []
    for k in ['A', 'B', 'C']:
        labels.append(f"{k}\n{ALGOS[k]}")
        dlast = data_last[k]
        if dlast is None:
            overall_values.append(0.0)
            comp_values.append(0.0)
            comm_values.append(0.0)
        else:
            overall_values.append(dlast['overall'])
            comp_values.append(dlast['comp_max'])
            comm_values.append(dlast['comm_max'])

    out_overall = PLOTS_DIR / f"{prefix}_overall.png"
    out_comp = PLOTS_DIR / f"{prefix}_comp.png"
    out_comm = PLOTS_DIR / f"{prefix}_comm.png"

    make_bar(overall_values, labels, 'seconds', 'Overall time by algorithm (parallel run)', out_overall)
    make_bar(comp_values, labels, 'seconds', 'Max compute time by algorithm (parallel run)', out_comp)
    make_bar(comm_values, labels, 'seconds', 'Max communication time by algorithm (parallel run)', out_comm)

    # Вычисление ускорения и эффективности на основе baseline (procs==1) и last (parallel)
    speedups = []
    efficiencies = []
    speedup_labels = []
    for k in ['A', 'B', 'C']:
        d_base = data_baseline[k]
        d_last = data_last[k]
        if d_base is None or d_last is None:
            speedups.append(0.0)
            efficiencies.append(0.0)
            speedup_labels.append(f"{k}\n(na)")
            continue
        # проверка: d_base['procs'] == 1 гарантирована при поиске, но проверим дополнительно
        if d_base['procs'] != 1:
            print(f"Warning: baseline row parsed for {k} does not have procs==1; skipping S/E")
            speedups.append(0.0)
            efficiencies.append(0.0)
            speedup_labels.append(f"{k}\n(na)")
            continue
        T1 = d_base['overall']
        Tp = d_last['overall']
        p = d_last['procs']
        if Tp <= 0 or T1 <= 0 or p <= 0:
            speedups.append(0.0)
            efficiencies.append(0.0)
            speedup_labels.append(f"{k}\n(na)")
            continue
        S = T1 / Tp
        E = S / p
        speedups.append(S)
        efficiencies.append(E)
        speedup_labels.append(f"{k}\n{p}p")

    out_speedup = PLOTS_DIR / f"{prefix}_speedup.png"
    out_eff = PLOTS_DIR / f"{prefix}_efficiency.png"

    make_bar(speedups, speedup_labels, 'speedup', 'Speedup (T1 / Tp) by algorithm', out_speedup)
    make_bar(efficiencies, speedup_labels, 'efficiency', 'Efficiency (S / p) by algorithm', out_eff)

    # Печать краткой таблички
    print("\nSaved plots:")
    print(out_overall, out_comp, out_comm, out_speedup, out_eff)
    print("\nSummary (baseline: first found row with procs==1; parallel: last row):")
    print("algo | baseline_procs | parallel_procs | T1(s)     | Tp(s)     | speedup  | efficiency | pi")
    for k in ['A', 'B', 'C']:
        bf = data_baseline[k]
        bl = data_last[k]
        if bf is None or bl is None:
            print(f"{k:4} |   -            |   -           |    -      |    -      |    -     |    -      |  -")
        else:
            T1 = bf['overall']
            Tp = bl['overall']
            p = bl['procs']
            S = T1 / Tp if Tp > 0 else float('nan')
            E = S / p if p > 0 else float('nan')
            print(f"{k:4} | {bf['procs']:13d} | {p:13d} | {T1:9.6f} | {Tp:9.6f} | {S:8.4f} | {E:10.4f} | {bl['pi']:8.6f}")

if __name__ == '__main__':
    main()
