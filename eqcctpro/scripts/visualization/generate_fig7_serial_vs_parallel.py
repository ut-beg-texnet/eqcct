#!/usr/bin/env python3
"""
Generate Figure 7 (Serial vs Ripper vs Amdahl) and Figure 8 (Serial vs Model Actor vs Amdahl).

Both figures use a 2x3 grid of subplots, one per CPU allocation present in the trial logs
(typically 5, 8, 11, 14, 17, 20). Curves are built from the station-count grid in the main
trial directory (5, 10, …, 225, 228). This is the scaling plot for the paper; 228-station
bar charts may use additional fixed-228 experiments not folded into these curves.

Ripper and Model Actor curves are chosen from trial CSVs: for each method and
hardware class (CPU, 1 GPU, 2 GPUs), we take the model with the lowest mean
successful total trial time over the station grid (5, 10, …, 225, 228) and
Ray CPU allocations (5, 8, 11, 14, 17, 20). If no data exist for a slot, the
script falls back to the previous static choice for that slot.

Serial (per-station streaming) curves read ``docs/tables/serial_classify_spotcheck_cpu.json``
when present (from ``benchmark_serial_classify_spotcheck.py``) and interpolate CPU
``classify()`` minima; otherwise they scale linearly from Table~1 ``Classify-Per-Stn`` at 228.
"""
import csv
import json
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

ALL_MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC", "EQCCT"]

FALLBACK_RIPPER = [
    ("PhaseNet", "Ripper", 0),
    ("PhaseNetLight", "Ripper", 1),
    ("PhaseNetLight", "Ripper", 2),
]
FALLBACK_MA = [
    ("PhaseNet", "ModelActor", 0),
    ("EQT-NC", "ModelActor", 1),
    ("PhaseNet", "ModelActor", 2),
]

_PROTOCOL_CPUS = (5, 8, 11, 14, 17, 20)
ALL_CANDIDATE_CONFIGS = [
    (m, method, g)
    for m in ALL_MODELS
    for method in ("Ripper", "ModelActor")
    for g in (0, 1, 2)
]

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

SERIAL_SPOTCHECK_JSON = BASE / "docs" / "tables" / "serial_classify_spotcheck_cpu.json"


def _load_serial_spotcheck() -> dict | None:
    if not SERIAL_SPOTCHECK_JSON.is_file():
        return None
    try:
        return json.loads(SERIAL_SPOTCHECK_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _serial_streaming_times(model: str, station_grid: np.ndarray, empirical: dict | None) -> np.ndarray:
    """
    CPU-only ``classify()`` wall time vs. station count (no model load in the plotted time).

    With empirical JSON: use measured minima at 10,20,30,40,…,228 and linearly interpolate
    on ``station_grid``; below the smallest / above the largest anchor, fall back to
    ``Classify-Per-Stn`` at 228 scaled by *n*/228.
    """
    key_cpu = (model, "CPU")
    if key_cpu not in SERIAL_TABLE:
        return np.zeros(len(station_grid))
    _load, _ann228, cls228 = SERIAL_TABLE[key_cpu]
    analytic = np.array([cls228 * (float(n) / 228.0) for n in station_grid], dtype=float)
    if not empirical:
        return analytic
    raw = empirical.get("models", {}).get(model)
    if not raw:
        return analytic
    d = {int(k): float(v) for k, v in raw.items()}
    if not d:
        return analytic
    xp = sorted(d.keys())
    fp = [d[x] for x in xp]
    out = []
    for n in station_grid:
        nn = int(n)
        if nn in d:
            out.append(d[nn])
        elif nn < xp[0] or nn > xp[-1]:
            out.append(cls228 * (nn / 228.0))
        else:
            out.append(float(np.interp(nn, xp, fp)))
    return np.array(out)


def _trial_ok(row) -> bool:
    v = (row.get("Trial Success") or "").strip().lower()
    return v in ("1", "true", "yes", "1.0")


def _ray_cpus_allowed(row) -> bool:
    try:
        c = int(float(row.get("Number of CPUs Allocated for Ray to Use", -1)))
    except (TypeError, ValueError):
        return False
    return c in _PROTOCOL_CPUS


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
                if not _trial_ok(row) or not _ray_cpus_allowed(row):
                    continue
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
    best = {cfg: {} for cfg in ALL_CANDIDATE_CONFIGS}
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


def pick_fastest_config(config_data, method: str, gpu_grp: int):
    best = None
    best_key = None
    for m in ALL_MODELS:
        cfg = (m, method, gpu_grp)
        tts = [
            vals[0]
            for (n, cpus), vals in config_data.get(cfg, {}).items()
            if n in VALID_STATIONS and cpus in _PROTOCOL_CPUS
        ]
        if not tts:
            continue
        mean_tt = sum(tts) / len(tts)
        key = (mean_tt, -len(tts), m)
        if best_key is None or key < best_key:
            best_key = key
            best = cfg
    return best


def build_method_configs(config_data, method: str, fallbacks):
    out = []
    for gpu_grp, fb in enumerate(fallbacks):
        chosen = pick_fastest_config(config_data, method, gpu_grp)
        out.append(chosen if chosen is not None else fb)
    return out


def amdahl_ideal_batch(load, annotate_228, workers, stations):
    if workers <= 0:
        return None
    t_per_station = annotate_228 / 228.0
    return [load + n * t_per_station / workers for n in stations]


def _ripper_legend_model_label(model: str) -> str:
    if model == "EQTransformer":
        return "EQT"
    return model


def _ripper_fig7_legend_handles(configs):
    """
    Legend order: PhaseNet, CPU, EQT, 1 GPU, EQT-NC, 2 GPUs, Amdahl ideal,
    per-station streaming (interleaved model patch + hardware marker per Ripper slot).
    """
    ms = 14
    mew = 1.5
    hw_short = {0: "CPU", 1: "1 GPU", 2: "2 GPUs"}
    handles = []
    for model, mth, gpu_grp in sorted(configs, key=lambda c: c[2]):
        handles.append(
            mpatches.Patch(
                facecolor=MODEL_COLORS[model],
                edgecolor="white",
                label=_ripper_legend_model_label(model),
            )
        )
        marker, _fill = MARKER_STYLE[(mth, gpu_grp)]
        h = Line2D(
            [0],
            [0],
            marker=marker,
            markersize=ms,
            color="w",
            markerfacecolor="none",
            markeredgecolor=GREY,
            markeredgewidth=mew,
            linestyle="",
        )
        h.set_label(hw_short[gpu_grp])
        handles.append(h)
    h_amdahl = Line2D([0], [0], color=RED, linestyle=":", linewidth=LINE_WIDTH)
    h_amdahl.set_label("Amdahl ideal")
    handles.append(h_amdahl)
    h_stream = Line2D([0], [0], color=GREY, linestyle="-.", linewidth=LINE_WIDTH)
    h_stream.set_label("Per-station streaming")
    handles.append(h_stream)
    return handles




def make_figure(
    configs,
    title,
    out_name,
    serial_whitelist: set[str],
    config_data=None,
    ripper_interleaved_legend: bool = False,
):
    if config_data is None:
        config_data = collect_data_for_configs()
    cpu_counts = set()
    for cfg in configs:
        for (_n, cpus) in config_data[cfg]:
            if cpus > 1 and cpus in _PROTOCOL_CPUS:
                cpu_counts.add(cpus)
    cpu_counts = sorted(cpu_counts)
    station_grid = np.array(sorted(VALID_STATIONS))

    fig_models = set(cfg[0] for cfg in configs)
    serial_models = fig_models & serial_whitelist
    serial_empirical = _load_serial_spotcheck()
    serial_curves = {}
    for model in serial_models:
        key_cpu = (model, "CPU")
        if key_cpu in SERIAL_TABLE:
            serial_curves[(model, "streaming")] = _serial_streaming_times(
                model, station_grid, serial_empirical
            )

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
    if ripper_interleaved_legend:
        legend_handles = _ripper_fig7_legend_handles(configs)
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=4,
            fontsize=20,
            framealpha=0.95,
            bbox_to_anchor=(0.05, -0.04, 0.9, 0.08),
            bbox_transform=fig.transFigure,
            columnspacing=3,
            handletextpad=2,
        )
    else:
        for m in sorted_fig_models:
            legend_handles.append(
                mpatches.Patch(facecolor=MODEL_COLORS[m], edgecolor="white", label=m)
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
            ncol=min(len(legend_handles), 11),
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


def _serial_whitelist_for_configs(configs):
    return {m for (m, _, _) in configs if (m, "CPU") in SERIAL_TABLE}


if __name__ == "__main__":
    full_data = collect_data_for_configs()
    ripper_cfgs = build_method_configs(full_data, "Ripper", FALLBACK_RIPPER)
    ma_cfgs = build_method_configs(full_data, "ModelActor", FALLBACK_MA)
    print("Fig 7 Ripper (method, gpu_grp):", ripper_cfgs)
    print("Fig 8 ModelActor:", ma_cfgs)

    make_figure(
        ripper_cfgs,
        "Serial vs Fastest Ripper vs Amdahl Ideal",
        "fig7_serial_vs_ripper.png",
        serial_whitelist=_serial_whitelist_for_configs(ripper_cfgs),
        config_data=full_data,
        ripper_interleaved_legend=True,
    )
    make_figure(
        ma_cfgs,
        "Serial vs Fastest Model Actor vs Amdahl Ideal",
        "fig8_serial_vs_modelactor.png",
        serial_whitelist=_serial_whitelist_for_configs(ma_cfgs),
        config_data=full_data,
    )
