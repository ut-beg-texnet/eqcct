#!/usr/bin/env python3
"""
Generate fig5: Ripper vs. Model-Actor minimum total runtime at 228 stations.
Two subplots — CPU (left) and GPU (right).

Data sources:
  Model-Actor: minimum Total Trial Time (s) at 228 stations from results/trials.
  CPU Ripper: results/ripper_228_conc_sweep/.../cpu_test_results.csv (successful rows).
  GPU Ripper: minimum successful 228-station time over results/trials eval_gpu_*_ripper
    and results/ripper_228_conc_sweep eval_sweep_ripper_gpu_* (see ripper_combined_minima.py).

  Ripper bars:      forward-slash hatching  ///
  Model-Actor bars: dot hatching            ...
  Each model has its own colour.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

import sys

_VIZ = Path(__file__).resolve().parent
if str(_VIZ) not in sys.path:
    sys.path.insert(0, str(_VIZ))
from ripper_combined_minima import best_gpu_ripper_bar_value

BASE_ROOT = Path(__file__).resolve().parents[2] / "results"
BASE = BASE_ROOT / "trials"
RIPPER_CPU_228 = BASE_ROOT / "ripper_228_conc_sweep"

RIPPER_SWEEP_SUBDIRS = {
    "PhaseNet": "eval_sweep_ripper_cpu_phasenet_original",
    "PhaseNetLight": "eval_sweep_ripper_cpu_phasenetlight_stead",
    "EQTransformer": "eval_sweep_ripper_cpu_eqtransformer_original",
    "EQT-NC": "eval_sweep_ripper_cpu_eqtransformer_nc",
    "EQCCT": "eval_sweep_ripper_cpu_eqcct",
}

MODEL_MAP = {
    "phasenet_original":             "PhaseNet",
    "phasenetlight_stead":           "PhaseNetLight",
    "eqtransformer_original":        "EQTransformer",
    "eqtransformer_nonconservative": "EQT-NC",
    "eqcct":                         "EQCCT",
}

def _trial_ok(row: dict) -> bool:
    v = (row.get("Trial Success") or "").strip().lower()
    return v in ("1", "true", "yes", "1.0")


_EXCLUDED_RAY_CPUS = frozenset({41, 46})


def _ray_cpus_allowed(row: dict) -> bool:
    try:
        c = int(float(row["Number of CPUs Allocated for Ray to Use"]))
    except (KeyError, ValueError, TypeError):
        return False
    return c not in _EXCLUDED_RAY_CPUS


def min_total_228(csv_path, *, require_success: bool = False):
    """Return (min_total_trial_time, workers) at 228 stations.

    When ``require_success`` is True, only rows with ``Trial Success`` are considered
    (used for ``results/ripper_228_conc_sweep`` CPU Ripper). Model-Actor and the initial
    GPU Ripper scan use ``require_success=False`` (trial logs); GPU Ripper values are then
    replaced by the combined successful minimum over trials + GPU sweep (see module docstring).
    """
    best = None
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    n = int(float(row["Number of Stations Used"]))
                except (KeyError, ValueError):
                    continue
                if n != 228:
                    continue
                if not _trial_ok(row):
                    continue
                if not _ray_cpus_allowed(row):
                    continue
                try:
                    tt = float(row["Total Trial Time (s)"])
                    actors = int(float(row.get("N ModelActors", 0) or 0))
                    conc = int(
                        float(
                            row.get("Number of Concurrent Station Tasks", 0)
                            or row.get("Actual Ripper Concurrent Tasks", 0)
                            or 0
                        )
                    )
                    workers = actors if actors > 0 else conc
                except (KeyError, ValueError):
                    continue
                if best is None or tt < best[0]:
                    best = (tt, workers)
    except FileNotFoundError:
        pass
    return best

raw = {}
for d in BASE.iterdir():
    if not d.is_dir():
        continue
    name = d.name
    if name.startswith("eval_cpu_"):
        hw, frag = "CPU", name[len("eval_cpu_"):]
    elif name.startswith("eval_gpu_"):
        hw, frag = "GPU", name[len("eval_gpu_"):]
    else:
        continue
    if frag.endswith("_modelactor"):
        orch, mfrag = "MA", frag[:-len("_modelactor")]
    elif frag.endswith("_ripper"):
        orch, mfrag = "Ripper", frag[:-len("_ripper")]
    else:
        continue
    model = MODEL_MAP.get(mfrag)
    if model is None:
        continue
    csv_file = d / f"{'cpu' if hw == 'CPU' else 'gpu'}_test_results.csv"
    val = min_total_228(csv_file)
    if val is not None:
        raw[(model, hw, orch)] = val

for model, sub in RIPPER_SWEEP_SUBDIRS.items():
    sweep_csv = RIPPER_CPU_228 / sub / "cpu_test_results.csv"
    v = min_total_228(sweep_csv, require_success=True)
    if v is not None:
        raw[(model, "CPU", "Ripper")] = v

gpu_combined = best_gpu_ripper_bar_value(BASE, RIPPER_CPU_228)
# Table 3 / ripper_combined_minima use "EQTransformer-NC"; this figure's MODEL_MAP uses "EQT-NC".
def _fig5_gpu_model(name: str) -> str:
    return "EQT-NC" if name == "EQTransformer-NC" else name

for m, (tt, w) in gpu_combined.items():
    key = _fig5_gpu_model(m)
    cur = raw.get((key, "GPU", "Ripper"))
    if cur is None or tt < cur[0]:
        raw[(key, "GPU", "Ripper")] = (tt, w)

MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC", "EQCCT"]
MODEL_COLORS = {
    "PhaseNet":      "#2471A3",
    "PhaseNetLight": "#1E8449",
    "EQTransformer": "#D4AC0D",
    "EQT-NC":        "#E67E22",
    "EQCCT":         "#8E44AD",
}

def vals(hw, orch):
    return [raw.get((m, hw, orch), (float("nan"), 0))[0] for m in MODELS]

cpu_ripper = vals("CPU", "Ripper")
cpu_ma     = vals("CPU", "MA")
gpu_ripper = vals("GPU", "Ripper")
gpu_ma     = vals("GPU", "MA")

print("Verified data (min Total Trial Time at 228 stations):")
print(f"{'Model':<18} {'CPU Ripper':>12} {'CPU MA':>8} {'GPU Ripper':>12} {'GPU MA':>8}")
for i, m in enumerate(MODELS):
    print(f"{m:<18} {cpu_ripper[i]:>12.2f} {cpu_ma[i]:>8.2f} {gpu_ripper[i]:>12.2f} {gpu_ma[i]:>8.2f}")

DEADLINE = 30.0
BAR_W    = 0.32
gap      = 0.06
offsets  = [-(BAR_W / 2 + gap / 2), (BAR_W / 2 + gap / 2)]

YTICKS = list(range(0, 41, 10)) + list(range(60, 111, 20))
x = np.arange(len(MODELS))

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
fig.subplots_adjust(wspace=0.08)

def draw_panel(ax, ripper_vals, ma_vals, hw, title, ymax):
    for i, m in enumerate(MODELS):
        c = MODEL_COLORS[m]
        rh = 0.0 if (ripper_vals[i] != ripper_vals[i]) else float(ripper_vals[i])
        mh = 0.0 if (ma_vals[i] != ma_vals[i]) else float(ma_vals[i])
        ax.bar(x[i] + offsets[0], rh, BAR_W,
               color=c, edgecolor="white", linewidth=0.7,
               hatch="///", zorder=3)
        ax.bar(x[i] + offsets[1], mh, BAR_W,
               color=c, edgecolor="white", linewidth=0.7,
               hatch="...", zorder=3)

    ax.axhline(DEADLINE, color="#E74C3C", linewidth=1.8,
               linestyle="--", zorder=5)

    label_pad  = ymax * 0.015
    label_clip = ymax * 0.97
    for i, m in enumerate(MODELS):
        c = MODEL_COLORS[m]
        for j, v in enumerate([ripper_vals[i], ma_vals[i]]):
            if v != v or v <= 0:
                continue
            y_text = min(v + label_pad, label_clip)
            va = "bottom" if v + label_pad <= label_clip else "top"
            ax.text(x[i] + offsets[j], y_text,
                    f"{v:.1f}s", ha="center", va=va,
                    fontsize=8, color=c, fontweight="bold", clip_on=True)

    # Build X-axis labels with worker counts
    xlabels = []
    for m in MODELS:
        r_w = raw.get((m, hw, "Ripper"), (float("nan"), 0))[1]
        a_w = raw.get((m, hw, "MA"), (float("nan"), 0))[1]
        xlabels.append(f"{m}\n(R:{r_w} / MA:{a_w})")

    visible_ticks = [t for t in YTICKS if t <= ymax]
    ax.set_yticks(visible_ticks)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Minimum Total Runtime (s)", fontsize=11)
    ax.set_xlabel("Model", fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

draw_panel(axes[0], cpu_ripper, cpu_ma, "CPU", "CPU Trials",  ymax=105)
draw_panel(axes[1], gpu_ripper, gpu_ma, "GPU", "GPU Trials",  ymax=105)

legend_handles = [
    mpatches.Patch(facecolor="#888888", edgecolor="white",
                   hatch="///", label="Ripper"),
    mpatches.Patch(facecolor="#888888", edgecolor="white",
                   hatch="...", label="Model-Actor"),
    plt.Line2D([0], [0], color="#E74C3C", linewidth=1.8,
               linestyle="--", label="30 s real-time target"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center", ncol=3,
    fontsize=11, framealpha=0.9, edgecolor="#cccccc",
    bbox_to_anchor=(0.5, -0.15),
)

fig.suptitle(
    f"Ripper vs. Model-Actor:\nMinimum Total Runtime at 228 Stations",
    fontsize=14, fontweight="bold", y=1.01,
)

out = Path(__file__).resolve().parents[2] / "docs" / "figures" / "fig5.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"\nSaved {out}")
