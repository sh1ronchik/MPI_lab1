#!/usr/bin/env python3
import sys
import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR.parent / "data" / "csv"
PLOTS_DIR = SCRIPT_DIR.parent / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PREFIX = "task1"

FILES = {
    "Send_Recv": "_Send_Recv.csv",
    "Reduce": "_Reduce.csv",
    "Isend_Irecv": "_Isend_Irecv.csv"
}

LABELS = {
    "Send_Recv": "Send/Recv",
    "Reduce": "Reduce",
    "Isend_Irecv": "Isend/Irecv"
}

COLORS = {"Send_Recv": "C0", "Reduce": "C1", "Isend_Irecv": "C2"}
MARKERS = {"Send_Recv": "o", "Reduce": "s", "Isend_Irecv": "^"}


def read_csv(path):
    rows = []
    with path.open("r") as f:
        rdr = csv.reader(f)
        for r in rdr:
            if len(r) < 3:
                continue
            rows.append({
                "procs": int(r[0]),
                "total": int(r[1]),
                "overall": float(r[2])
            })
    return rows


def plot(x, series, xlabel, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, xs, ys, color, marker in series:
        if xs:
            ax.plot(xs, ys, marker=marker, color=color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], rotation=30)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PREFIX

    # --- читаем данные ---
    data = {k: read_csv(CSV_DIR / f"{prefix}{sfx}") for k, sfx in FILES.items()}

    # === 1) Время / Ускорение / Эффективность vs total_points ===
    # найдём общее максимальное p
    common_procs = set.intersection(*(set(r["procs"] for r in rows) for rows in data.values()))
    p = max(common_procs)
    print(f"Используем p={p} для графиков vs total_points")

    # baseline (p=1)
    baseline = {(k, r["total"]): r["overall"]
                for k, rows in data.items()
                for r in rows if r["procs"] == 1}

    xs = sorted(set(
        r["total"] for rows in data.values() for r in rows if r["procs"] == p
    ))

    overall_series, speed_series, eff_series = [], [], []

    for k, rows in data.items():
        row_map = {(r["procs"], r["total"]): r["overall"] for r in rows}
        ys_over, ys_speed, ys_eff = [], [], []
        for n in xs:
            Tp = row_map.get((p, n), math.nan)
            T1 = baseline.get((k, n), None)
            ys_over.append(Tp)
            if T1:
                S = T1 / Tp
                ys_speed.append(S)
                ys_eff.append(S / p)
            else:
                ys_speed.append(math.nan)
                ys_eff.append(math.nan)

        lbl = f"{LABELS[k]} (p={p})"
        overall_series.append((lbl, xs, ys_over, COLORS[k], MARKERS[k]))
        speed_series.append((lbl, xs, ys_speed, COLORS[k], MARKERS[k]))
        eff_series.append((lbl, xs, ys_eff, COLORS[k], MARKERS[k]))

    plot(xs, overall_series, "total_points", "Time (s)", f"Time vs total_points (p={p})",
         PLOTS_DIR / f"{prefix}_time_vs_points.png")
    plot(xs, speed_series, "total_points", "Speedup", f"Speedup vs total_points (p={p})",
         PLOTS_DIR / f"{prefix}_speedup_vs_points.png")
    plot(xs, eff_series, "total_points", "Efficiency", f"Efficiency vs total_points (p={p})",
         PLOTS_DIR / f"{prefix}_eff_vs_points.png")

    # === 2) Ускорение / Эффективность vs procs ===
    speed_p_series, eff_p_series = [], []
    procs_set = set()

    for k, rows in data.items():
        Nmax = max(r["total"] for r in rows)
        row_map = {(r["procs"], r["total"]): r["overall"] for r in rows}
        T1 = row_map.get((1, Nmax), None)
        xs_p, ys_speed, ys_eff = [], [], []
        for p_ in sorted({r["procs"] for r in rows if r["total"] == Nmax}):
            Tp = row_map.get((p_, Nmax), None)
            if Tp and T1:
                xs_p.append(p_)
                ys_speed.append(T1 / Tp)
                ys_eff.append((T1 / Tp) / p_)
                procs_set.add(p_)
        lbl = f"{LABELS[k]} (N={Nmax})"
        speed_p_series.append((lbl, xs_p, ys_speed, COLORS[k], MARKERS[k]))
        eff_p_series.append((lbl, xs_p, ys_eff, COLORS[k], MARKERS[k]))

    xs_procs = sorted(procs_set)

    plot(xs_procs, speed_p_series, "procs", "Speedup", "Speedup vs procs",
         PLOTS_DIR / f"{prefix}_speedup_vs_procs.png")
    plot(xs_procs, eff_p_series, "procs", "Efficiency", "Efficiency vs procs",
         PLOTS_DIR / f"{prefix}_eff_vs_procs.png")

    print("Готово. Графики сохранены в:", PLOTS_DIR)


if __name__ == "__main__":
    main()
