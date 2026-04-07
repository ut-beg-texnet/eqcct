#!/usr/bin/env python3
"""
Generate fig5: Ripper vs. Model-Actor minimum total runtime at 228 stations.
Two subplots — CPU (left) and GPU (right).

Selection rules (paper):
  - Model-Actor: global minimum successful Total Trial Time at 228 with Ray CPUs
    in {5, 8, 11, 14, 17, 20}.
  - Ripper: same Ray CPU grid; GPU Ripper bars use the faster of
    the best 1-GPU and best 2-GPU Ripper totals (same CPU constraint).

  CPU Ripper: eval_cpu_*_ripper + ripper_228_conc_sweep cpu_test_results.
  GPU Ripper: ripper_combined_minima.best_gpu_ripper_bar_value.

  Ripper bars:      forward-slash hatching  ///
  Model-Actor bars: dot hatching            ...
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import sys

_VIZ = Path(__file__).resolve().parent
if str(_VIZ) not in sys.path:
    sys.path.insert(0, str(_VIZ))

from ripper_combined_minima import paper_runtime_raw_dict  # noqa: E402

BASE_ROOT = Path(__file__).resolve().parents[2] / "results"
BASE = BASE_ROOT / "trials"
RIPPER_CPU_228 = BASE_ROOT / "ripper_228_conc_sweep"

MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC", "EQCCT"]
MODEL_COLORS = {
    "PhaseNet":      "#2471A3",
    "PhaseNetLight": "#1E8449",
    "EQTransformer": "#D4AC0D",
    "EQT-NC":        "#E67E22",
    "EQCCT":         "#8E44AD",
}

raw = paper_runtime_raw_dict(BASE, RIPPER_CPU_228)


def vals(hw, orch):
    return [raw.get((m, hw, orch), (float("nan"), 0))[0] for m in MODELS]

cpu_ripper = vals("CPU", "Ripper")
cpu_ma     = vals("CPU", "MA")
gpu_ripper = vals("GPU", "Ripper")
gpu_ma     = vals("GPU", "MA")

print("Paper data (228 stn; Ray CPUs ∈ {5,8,11,14,17,20}; MA & Ripper = global min in grid):")
print(f"{'Model':<18} {'CPU Ripper':>12} {'CPU MA':>8} {'GPU Ripper':>12} {'GPU MA':>8}")
for i, m in enumerate(MODELS):
    print(f"{m:<18} {cpu_ripper[i]:>12.2f} {cpu_ma[i]:>8.2f} {gpu_ripper[i]:>12.2f} {gpu_ma[i]:>8.2f}")

DEADLINE = 30.0
BAR_W    = 0.32
gap      = 0.06
offsets  = [-(BAR_W / 2 + gap / 2), (BAR_W / 2 + gap / 2)]

YTICKS = list(range(0, 41, 10)) + list(range(60, 111, 20)) + [120]
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
    ax.set_ylabel("Minimum total runtime (s)", fontsize=11)
    ax.set_xlabel("Model", fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


ymax = 125
draw_panel(axes[0], cpu_ripper, cpu_ma, "CPU", "CPU trials", ymax=ymax)
draw_panel(axes[1], gpu_ripper, gpu_ma, "GPU", "GPU trials", ymax=ymax)

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
    "Ripper vs. Model-Actor: minimum total runtime at 228 stations\n"
    "(Ray CPUs 5–20; Ripper GPU bar = faster of best 1-GPU vs 2-GPU Ripper; MA: min total in grid)",
    fontsize=13, fontweight="bold", y=1.02,
)

out = Path(__file__).resolve().parents[2] / "docs" / "figures" / "fig5.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"\nSaved {out}")
