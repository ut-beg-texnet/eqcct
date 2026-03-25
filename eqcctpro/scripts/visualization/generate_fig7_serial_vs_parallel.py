#!/usr/bin/env python3
"""
Generate Figure 7 (Serial vs Ripper vs Amdahl) and Figure 8 (Serial vs Model Actor vs Amdahl).

Both figures use a 2x3 grid of subplots, one per CPU allocation present in the trial logs
(typically 5, 8, 11, 14, 17, 20). Curves are built from the station-count grid in the main
trial directory (5, 10, …, 225, 228). This is the scaling plot for the paper; 228-station
bar charts may use additional fixed-228 experiments not folded into these curves.

Fig 7 Ripper configs (fastest total time per (stations, cpus, gpu group)):
  CPU Ripper:   PhaseNet
  1 GPU Ripper: PhaseNetLight
  2 GPU Ripper: PhaseNetLight

Fig 8 Model Actor configs:
  CPU MA:   PhaseNet
  1 GPU MA: EQT-NC
  2 GPU MA: PhaseNet
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
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

SERIAL_TABLE = {
    ("PhaseNet", "CPU"): (1.264, 0.343, 33.70),
    ("PhaseNet", "GPU"): (1.309, 0.224, 27.11),
    ("PhaseNetLight", "CPU"): (1.184, 0.315, 1.43),
    ("PhaseNetLight", "GPU"): (1.180, 0.216, 1.43),
    ("EQTransformer", "CPU"): (1.197, 1.216, 12.20),
    ("EQTransformer", "GPU"): (1.215, 0.513, 12.22),
    ("EQT-NC", "CPU"): (1.190, 1.182, 8.19),
    ("EQT-NC", "GPU"): (1.171, 0.458, 8.18),
}

RIPPER_CONFIGS = [
    ("PhaseNet", "Ripper", 0),
    ("PhaseNetLight", "Ripper", 1),
    ("PhaseNetLight", "Ripper", 2),
]
MA_CONFIGS = [
    ("PhaseNet", "ModelActor", 0),
    ("EQT-NC", "ModelActor", 1),
    ("PhaseNet", "ModelActor", 2),
]
ALL_CONFIGS = RIPPER_CONFIGS + MA_CONFIGS

VALID_STATIONS = set(range(5, 226, 5)) | {228}
GREY = "#666666"
RED = "#CC0000"

BATCH_LOAD = 1.180
BATCH_ANNOTATE_228 = 0.216

MARKER_STYLE = {
    ("Ripper", 0): ("o", "none"),
    ("Ripper", 1): ("D", "none"),
    ("Ripper", 2): ("s", "none"),
    ("ModelActor", 0): ("o", "full"),
    ("ModelActor", 1): ("D", "full"),
    ("ModelActor", 2): ("s", "full"),
}

LINE_WIDTH = 3.0


def parse_gpu_count(gpu_str):
    if not gpu_str or gpu_str == "[]" or str(gpu_str).strip() == "nan":
        return 0
    s = str(gpu_str).strip("[]")
    return len([x for x in s.split(",") if x.strip()]) if s else 0


def parse_trial_dir(name):
    if not name.startswith("eval_") or ("_modelactor" not in name and "_ripper" not in name):
        return None, None, None
    hw = "CPU" if name.startswith("eval_cpu_") else "GPU"
    method = "ModelActor" if "_modelactor" in name else "Ripper"
    frag = name.replace("eval_cpu_", "").replace("eval_gpu_", "").replace("_modelactor", "").replace("_ripper", "")
    model = MODEL_MAP.get(frag, None)
    return model, hw, method


def load_all_rows(csv_path):
    rows = []
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    n = int(float(row.get("Number of Stations Used", 0)))
                    cpus = int(float(row.get("Number of CPUs Allocated for Ray to Use", 0)))
                    tt = float(row.get("Total Trial Time (s)", 0) or 0)
                    picking = float(row.get("Total Run time for Picker (s)", 0) or 0)
                    setup = float(
                        row.get("Actor Creation Time (s)", 0)
                        or row.get("Avg Model Load Time (s)", 0)
                        or 0
                    )
                    actors = int(float(row.get("N ModelActors", 0) or 0))
                    conc = int(
                        float(
                            row.get("Actual Ripper Concurrent Tasks", 0)
                            or row.get("Number of Concurrent Station Tasks", 0)
                            or 0
                        )
                    )
                    gpus = parse_gpu_count(row.get("GPUs Used", "[]"))
                    workers = actors if actors > 0 else conc
                    rows.append((n, cpus, gpus, tt, setup, picking, workers))
                except (ValueError, TypeError):
                    continue
    except FileNotFoundError:
        pass
    return rows


def collect_data_for_configs():
    best = {cfg: {} for cfg in ALL_CONFIGS}
    for d in sorted(TRIALS.iterdir()):
        if not d.is_dir():
            continue
        model, hw, method = parse_trial_dir(d.name)
        if model is None:
            continue
        test_path = d / f"{'cpu' if hw == 'CPU' else 'gpu'}_test_results.csv"
        opt_path = d / f"optimal_configurations_{'cpu' if hw == 'CPU' else 'gpu'}.csv"
        csv_path = test_path if test_path.exists() else opt_path
        if not csv_path.exists():
            continue
        rows = load_all_rows(csv_path)
        for n, cpus, gpus, tt, setup, picking, workers in rows:
            if n not in VALID_STATIONS or cpus <= 1:
                continue
            gpu_grp = 0 if hw == "CPU" else min(gpus, 2)
            cfg_key = (model, method, gpu_grp)
            if cfg_key not in best:
                continue
            sc_key = (n, cpus)
            if sc_key not in best[cfg_key] or tt < best[cfg_key][sc_key][0]:
                best[cfg_key][sc_key] = (tt, setup, picking, workers)
    return best


def amdahl_ideal_batch(load, annotate_228, workers, stations):
    if workers <= 0:
        return None
    t_per_station = annotate_228 / 228.0
    return [load + n * t_per_station / workers for n in stations]


def make_figure(configs, title, out_name, serial_whitelist: set[str]):
    config_data = collect_data_for_configs()
    cpu_counts = set()
    for cfg in configs:
        for (_n, cpus) in config_data[cfg]:
            if cpus > 1:
                cpu_counts.add(cpus)
    cpu_counts = sorted(cpu_counts)
    station_grid = np.array(sorted(VALID_STATIONS))

    fig_models = set(cfg[0] for cfg in configs)
    serial_models = fig_models & serial_whitelist
    serial_curves = {}
    for model in serial_models:
        key_cpu = (model, "CPU")
        if key_cpu in SERIAL_TABLE:
            _load, _ann228, cls228 = SERIAL_TABLE[key_cpu]
            serial_curves[(model, "streaming")] = np.array([cls228 * (n / 228.0) for n in station_grid])

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)
    axes = []
    for i in range(6):
        ax = fig.add_subplot(gs[i // 3, i % 3], sharex=axes[0] if axes else None)
        axes.append(ax)

    for idx, n_cpu in enumerate(cpu_counts):
        ax = axes[idx]
        ax.set_title(f"{n_cpu} CPUs", fontsize=20, fontweight="bold")

        for model in sorted(serial_models):
            color = MODEL_COLORS[model]
            str_key = (model, "streaming")
            if str_key in serial_curves:
                ax.plot(
                    station_grid,
                    serial_curves[str_key],
                    color=color,
                    linestyle="-.",
                    linewidth=LINE_WIDTH,
                    alpha=0.8,
                )

        for cfg in configs:
            model, method, gpu_grp = cfg
            color = MODEL_COLORS[model]
            marker, fill = MARKER_STYLE[(method, gpu_grp)]
            fc = color if fill == "full" else "none"

            pts = [(n, *vals) for (n, cpus), vals in config_data[cfg].items() if cpus == n_cpu and n in VALID_STATIONS]
            if not pts:
                continue
            pts.sort(key=lambda x: x[0])
            stations = np.array([p[0] for p in pts])
            tt = np.array([p[1] for p in pts])

            ax.plot(stations, tt, color=color, linewidth=2.5, alpha=0.8)
            ax.scatter(stations, tt, marker=marker, s=70, facecolors=fc, edgecolors=color, linewidths=1.5)

        ideal = amdahl_ideal_batch(BATCH_LOAD, BATCH_ANNOTATE_228, n_cpu, station_grid)
        if ideal is not None:
            ax.plot(station_grid, ideal, color=RED, linestyle=":", linewidth=LINE_WIDTH, alpha=0.9)

        ax.set_xlabel("Number of Stations", fontsize=18)
        ax.set_ylabel("Total Trial Runtime (s)", fontsize=18)
        ax.set_xlim(0, 228)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_ylim(0, 60)
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=16)

    model_order = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC", "EQCCT"]
    sorted_fig_models = [m for m in model_order if m in fig_models]
    legend_handles = []
    if len(sorted_fig_models) > 0:
        legend_handles.append(
            mpatches.Patch(facecolor=MODEL_COLORS[sorted_fig_models[0]], edgecolor="white", label=sorted_fig_models[0])
        )
    if len(sorted_fig_models) > 1:
        legend_handles.append(
            mpatches.Patch(facecolor=MODEL_COLORS[sorted_fig_models[1]], edgecolor="white", label=sorted_fig_models[1])
        )
    h_stream = Line2D([0], [0], color=GREY, linestyle="-.", linewidth=LINE_WIDTH)
    h_stream.set_label("Per-station streaming (serial)")
    legend_handles.append(h_stream)
    h_cpu = Line2D(
        [0],
        [0],
        marker="o",
        markersize=16,
        color="w",
        markerfacecolor="none",
        markeredgecolor=GREY,
        markeredgewidth=2.0,
        linestyle="",
    )
    h_cpu.set_label("CPU (0 GPUs)")
    legend_handles.append(h_cpu)
    h_1gpu = Line2D(
        [0],
        [0],
        marker="D",
        markersize=16,
        color="w",
        markerfacecolor="none",
        markeredgecolor=GREY,
        markeredgewidth=2.0,
        linestyle="",
    )
    h_1gpu.set_label("1 GPU")
    legend_handles.append(h_1gpu)
    h_2gpu = Line2D(
        [0],
        [0],
        marker="s",
        markersize=16,
        color="w",
        markerfacecolor="none",
        markeredgecolor=GREY,
        markeredgewidth=2.0,
        linestyle="",
    )
    h_2gpu.set_label("2 GPUs")
    legend_handles.append(h_2gpu)
    h_amdahl = Line2D([0], [0], color=RED, linestyle=":", linewidth=3.5)
    h_amdahl.set_label("Amdahl ideal (batch-based)")
    legend_handles.append(h_amdahl)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        fontsize=18,
        framealpha=0.95,
        bbox_to_anchor=(0.05, -0.06, 0.9, 0.08),
        bbox_transform=fig.transFigure,
        columnspacing=1.5,
        handletextpad=1.2,
    )
    fig.suptitle(title, fontsize=28, fontweight="bold", y=1.02)
    plt.subplots_adjust(bottom=0.12, top=0.94, left=0.06, right=0.98)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_figure(
        RIPPER_CONFIGS,
        "Serial vs Fastest Ripper vs Amdahl Ideal",
        "fig7_serial_vs_ripper.png",
        serial_whitelist={"PhaseNet", "PhaseNetLight"},
    )
    make_figure(
        MA_CONFIGS,
        "Serial vs Fastest Model Actor vs Amdahl Ideal",
        "fig8_serial_vs_modelactor.png",
        serial_whitelist={"PhaseNet", "PhaseNetLight", "EQT-NC"},
    )
