#!/usr/bin/env python3
"""
Generate Figure 4: 2D subplots of total trial runtime vs station count, one per CPU count.

Each subplot: X = number of stations, Y = total trial runtime (s).
- Model colors distinguish models
- Marker shape: CPU=circle, 1 GPU=diamond, 2 GPUs=square (filled=Model Actor, open=Ripper)
- Plotted station counts: every 10 stations (10, 20, …, 220) plus 228
- Y axis: 0–60 s, ticks every 10 s (series above 60 s are clipped)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

BASE = Path(__file__).resolve().parents[2]
TRIALS = BASE / "results" / "trials"
OUT = BASE / "docs" / "figures"

MODEL_MAP = {
    "phasenet_original": "PhaseNet",
    "phasenetlight_stead": "PhaseNetLight",
    "eqtransformer_original": "EQTransformer",
    "eqtransformer_nonconservative": "EQT-NC",
    "eqcct": "EQCCT",
}
MODEL_COLORS = {
    "PhaseNet": "#2471A3",
    "PhaseNetLight": "#1E8449",
    "EQTransformer": "#D4AC0D",
    "EQT-NC": "#E67E22",
    "EQCCT": "#8E44AD",
}

MARKER_MAP = {
    ("Ripper", 0): ("o", "none"),
    ("Ripper", 1): ("D", "none"),
    ("Ripper", 2): ("s", "none"),
    ("ModelActor", 0): ("o", "full"),
    ("ModelActor", 1): ("D", "full"),
    ("ModelActor", 2): ("s", "full"),
}
GREY = "#666666"
_EXCLUDED_RAY_CPUS = frozenset({41, 46})


def _trial_ok(row) -> bool:
    v = (row.get("Trial Success") or "").strip().lower()
    return v in ("1", "true", "yes", "1.0")


def _ray_cpus_allowed(row) -> bool:
    try:
        c = int(float(row.get("Number of CPUs Allocated for Ray to Use", -1)))
    except (TypeError, ValueError):
        return False
    return c not in _EXCLUDED_RAY_CPUS


def parse_gpu_count(gpu_str):
    if not gpu_str or gpu_str == "[]" or str(gpu_str).strip() == "nan":
        return 0
    s = str(gpu_str).strip("[]")
    return len([x for x in s.split(",") if x.strip()]) if s else 0


def load_optimal_per_station(csv_path):
    rows = []
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if not _trial_ok(row) or not _ray_cpus_allowed(row):
                    continue
                try:
                    n = int(float(row.get("Number of Stations Used", 0)))
                    cpus = int(float(row.get("Number of CPUs Allocated for Ray to Use", 0)))
                    tt = float(row.get("Total Trial Time (s)", 0) or 0)
                    picking = float(row.get("Total Run time for Picker (s)", 0) or 0)
                    setup = float(row.get("Actor Creation Time (s)", 0) or row.get("Avg Model Load Time (s)", 0) or 0)
                    actors = int(float(row.get("N ModelActors", 0) or 0))
                    conc = int(float(row.get("Actual Ripper Concurrent Tasks", 0) or row.get("Number of Concurrent Station Tasks", 0) or 0))
                    gpus = parse_gpu_count(row.get("GPUs Used", "[]"))
                    rows.append((n, cpus, gpus, tt, setup, picking, actors if actors > 0 else conc))
                except (ValueError, TypeError):
                    continue
    except FileNotFoundError:
        pass
    return rows


def best_per_station(rows):
    by_n = {}
    for r in rows:
        n = r[0]
        if n not in by_n or r[3] < by_n[n][3]:
            by_n[n] = r
    return sorted(by_n.values(), key=lambda x: x[0])


def best_per_station_per_cpu(rows):
    """Best (min total trial time) per (station, cpu)."""
    by_n_cpu = {}
    for r in rows:
        n, cpus = r[0], r[1]
        key = (n, cpus)
        if key not in by_n_cpu or r[3] < by_n_cpu[key][3]:
            by_n_cpu[key] = r
    return by_n_cpu


def best_per_station_per_gpu(rows):
    """Best per (station, cpu, gpu)."""
    by_n_cpu_gpu = {}
    for r in rows:
        n, cpus, gpu = r[0], r[1], min(r[2], 2)
        key = (n, cpus, gpu)
        if key not in by_n_cpu_gpu or r[3] < by_n_cpu_gpu[key][3]:
            by_n_cpu_gpu[key] = r
    return by_n_cpu_gpu


def parse_trial_dir(name):
    if not name.startswith("eval_") or ("_modelactor" not in name and "_ripper" not in name):
        return None, None, None
    hw = "CPU" if name.startswith("eval_cpu_") else "GPU"
    method = "ModelActor" if "_modelactor" in name else "Ripper"
    frag = name.replace("eval_cpu_", "").replace("eval_gpu_", "").replace("_modelactor", "").replace("_ripper", "")
    model = MODEL_MAP.get(frag, None)
    return model, hw, method


def collect_parallel_data():
    """Use full test_results CSV for complete data; fall back to optimal_configurations."""
    data = {}
    for d in sorted(TRIALS.iterdir()):
        if not d.is_dir():
            continue
        model, hw, method = parse_trial_dir(d.name)
        if model is None:
            continue
        # Prefer full test_results for all data points
        test_path = d / f"{'cpu' if hw == 'CPU' else 'gpu'}_test_results.csv"
        opt_path = d / f"optimal_configurations_{'cpu' if hw == 'CPU' else 'gpu'}.csv"
        csv_path = test_path if test_path.exists() else opt_path
        if not csv_path.exists():
            continue
        rows = load_optimal_per_station(csv_path)
        if not rows:
            continue
        if hw == "CPU":
            by_n_cpu = best_per_station_per_cpu(rows)
            # Flatten to list of (n, cpus, tt, setup, picking, actors)
            flat = sorted(by_n_cpu.values(), key=lambda x: (x[1], x[0]))
            data[(model, method, 0)] = {
                "stations": [r[0] for r in flat],
                "cpus": [r[1] for r in flat],
                "tt": [r[3] for r in flat],
                "setup": [r[4] for r in flat],
                "picking": [r[5] for r in flat],
                "actors": [r[6] for r in flat],
            }
        else:
            by_ncg = best_per_station_per_gpu(rows)
            for gpu_count in [1, 2]:
                best = sorted([r for (n, c, g), r in by_ncg.items() if g == gpu_count],
                              key=lambda x: (x[1], x[0]))
                if not best:
                    continue
                data[(model, method, gpu_count)] = {
                    "stations": [r[0] for r in best],
                    "cpus": [r[1] for r in best],
                    "tt": [r[3] for r in best],
                    "setup": [r[4] for r in best],
                    "picking": [r[5] for r in best],
                    "actors": [r[6] for r in best],
                }
    return data


# Collect data
parallel_data = collect_parallel_data()

# Unique CPU counts (exclude 1; parallel trials only)
cpu_counts = set()
for (model, method, gpu_count), d in parallel_data.items():
    cpu_counts |= set(d["cpus"])
# Match paper protocol (5–20 Ray CPUs); ignore one-off blocks (e.g., 40 CPUs) for the 2×3 grid.
_PROTOCOL_CPUS = (5, 8, 11, 14, 17, 20)
cpu_counts = sorted(c for c in cpu_counts if c in _PROTOCOL_CPUS)

# Shared axes
X_MIN, X_MAX = 0, 228
X_TICKS = [0, 50, 100, 150, 200]
# Plotted points only: step 10 through 220, plus final 228 (trials use a denser sweep in CSV)
VALID_STATIONS = set(range(10, 221, 10)) | {228}
Y_MAX = 60
Y_TICKS = [0, 10, 20, 30, 40, 50, 60]

# Build subplots: 6 plots (5,8,11,14,17,20 CPUs) in 2 rows of 3
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)
axes = []
for i in range(6):
    ax = fig.add_subplot(gs[i // 3, i % 3], sharex=axes[0] if axes else None)
    axes.append(ax)

for idx, n_cpu in enumerate(cpu_counts):
    ax = axes[idx]
    ax.set_title(f"{n_cpu} CPU{'s' if n_cpu > 1 else ''}", fontsize=20, fontweight="bold")

    # Parallel data at this CPU count (points every 10 stations, plus 228)
    for (model, method, gpu_count), d in parallel_data.items():
        mask = np.array(d["cpus"]) == n_cpu
        if not np.any(mask):
            continue
        stations = np.array(d["stations"])[mask]
        tt = np.array(d["tt"])[mask]
        # Filter to valid station counts (10, 20, ..., 220, 228)
        valid_mask = np.array([s in VALID_STATIONS for s in stations])
        if not np.any(valid_mask):
            continue
        stations = stations[valid_mask]
        tt = tt[valid_mask]
        order = np.argsort(stations)
        stations, tt = stations[order], tt[order]
        color = MODEL_COLORS.get(model, "#888888")
        marker, fill = MARKER_MAP.get((method, min(gpu_count, 2)), ("o", "full"))
        fc = color if fill == "full" else "none"
        ax.plot(stations, tt, color=color, linewidth=2.5, alpha=0.8)
        ax.scatter(stations, tt, marker=marker, s=70, facecolors=fc, edgecolors=color, linewidths=1.5)

    ax.set_xlabel("Number of Stations", fontsize=18)
    ax.set_ylabel("Total Trial Runtime (s)", fontsize=18)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks(X_TICKS)
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks(Y_TICKS)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=16)

# Single legend: Grouped order (Models, then Hardware)
# PhaseNet, PhaseNetLight, EQTransformer, EQT-NC, EQCCT, CPU (0 GPUs), 1 GPU, 2 GPUs, Open: Ripper, Filled: Model Actor
model_handles = []
# 1. Models
for name, color in MODEL_COLORS.items():
    model_handles.append(mpatches.Patch(facecolor=color, edgecolor="white", label=name))

# 2. Hardware
hw_handles = []
hw_items = [
    ("o", "CPU (0 GPUs)", "none"),
    ("D", "1 GPU", "none"),
    ("s", "2 GPUs", "none"),
    ("o", "Open: Ripper", "none"),
    ("o", "Filled: Model Actor", GREY),
]
for marker, label, fc in hw_items:
    h = Line2D([0], [0], marker=marker, markersize=14, color="w", markerfacecolor=fc,
               markeredgecolor=GREY, markeredgewidth=1.5, linestyle="")
    h.set_label(label)
    hw_handles.append(h)

legend_handles = []
for m_handle, h_handle in zip(model_handles, hw_handles):
    legend_handles.append(m_handle)
    legend_handles.append(h_handle)

fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=20, framealpha=0.95,
          bbox_to_anchor=(0.05, -0.04, 0.9, 0.08), bbox_transform=fig.transFigure,
          columnspacing=3, handletextpad=2)

fig.suptitle("Total Trial Runtime vs Station Count by CPU Allocation", fontsize=28, fontweight="bold", y=1.02)
plt.subplots_adjust(bottom=0.12, top=0.94, left=0.06, right=0.98)
OUT.mkdir(parents=True, exist_ok=True)
out_path = OUT / "fig4_runtime_3d.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
