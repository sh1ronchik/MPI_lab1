#!/usr/bin/env python3
"""
plot_task1.py

Строит столбчатые графики времени для трёх алгоритмов:
  A: Send/Recv, B: Reduce, C: Isend/Irecv
на основе CSV, сгенерированных C-программой:
  ./task1/data/csv/<prefix>_algo{A,B,C}.csv

Выход: PNG в task1/data/plots:
  <prefix>_overall.png, <prefix>_comp.png, <prefix>_comm.png

Usage:
  cd <project_root>
  python3 task1/data/scripts/plot_task1.py [prefix]

Поведение:
- читается последняя непустая строка из каждого CSV (без агрегации по запускам),
- при отсутствии или некорректном содержимом выдаются предупреждения.
"""
import os
import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# Константы (имена переменных/путей оставлены на English)
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR.parent / 'data' / 'csv'  # ../data/csv
PLOTS_DIR = SCRIPT_DIR.parent / 'data' / 'plots'  # ../data/plots
PLOTS_DIR.mkdir(exist_ok=True)
DEFAULT_PREFIX = 'task1'
ALGOS = {'A': 'Send/Recv', 'B': 'Reduce', 'C': 'Isend/Irecv'}


def read_last_row(path):
    # Возвращает последнюю непустую CSV-строку или None
    if not path.exists():
        return None
    last = None
    with path.open('r', newline='') as f:
        rdr = csv.reader(f)
        for row in rdr:
            if row and any(cell.strip() for cell in row):
                last = row
    return last


def parse_row(row):
    # Парсит строку в ожидаемый формат: procs,total_points,overall,comp_max,comm_max,pi
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
    # Рисует простой bar chart и сохраняет в out_path
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

    # Чтение последних строк из CSV для каждого алгоритма
    data = {}
    for k in ['A','B','C']:
        fname = f"{prefix}_algo{k}.csv"
        p = CSV_DIR / fname
        row = read_last_row(p)
        parsed = parse_row(row)
        if parsed is None:
            print(f"Warning: file {p} missing or has invalid content")
            data[k] = None
        else:
            data[k] = parsed

    # Подготовка значений для графиков (подставляем 0 для отсутствующих данных)
    labels = []
    overall_values = []
    comp_values = []
    comm_values = []

    for k in ['A','B','C']:
        if data[k] is None:
            labels.append(f"{k}\n(na)")
            overall_values.append(0.0)
            comp_values.append(0.0)
            comm_values.append(0.0)
        else:
            labels.append(f"{k}\n{ALGOS[k]}")
            overall_values.append(data[k]['overall'])
            comp_values.append(data[k]['comp_max'])
            comm_values.append(data[k]['comm_max'])

    # Имена выходных файлов
    out_overall = PLOTS_DIR / f"{prefix}_overall.png"
    out_comp = PLOTS_DIR / f"{prefix}_comp.png"
    out_comm = PLOTS_DIR / f"{prefix}_comm.png"

    # Сохранение трёх графиков
    make_bar(overall_values, labels, 'seconds', 'Overall time by algorithm', out_overall)
    make_bar(comp_values, labels, 'seconds', 'Max compute time by algorithm', out_comp)
    make_bar(comm_values, labels, 'seconds', 'Max communication time by algorithm', out_comm)

    print('Saved:', out_overall, out_comp, out_comm)

    # Краткая текстовая сводка
    print('\nSummary:')
    print('algo | procs | total_points | overall(s) | comp_max(s) | comm_max(s) | pi')
    for k in ['A','B','C']:
        d = data[k]
        if d is None:
            print(f"{k}   |  -    |     -       |    -       |     -      |    -      |  -")
        else:
            print(f"{k}   | {d['procs']:5d} | {d['total_points']:11d} | {d['overall']:10.6f} | {d['comp_max']:10.6f} | {d['comm_max']:10.6f} | {d['pi']:8.6f}")


if __name__ == '__main__':
    main()
