#!/usr/bin/env python3
"""
Generate fig4: Runtime Reduction barplot comparing Ripper vs Model-Actor at 228 stations.
Runtime Reduction = (1 - MA Total / Ripper Total) × 100%

Uses the same row-selection rules as generate_fig5_runtime_comparison.py (via paper_runtime_raw_dict).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np
from pathlib import Path
import sys

_VIZ = Path(__file__).resolve().parent
if str(_VIZ) not in sys.path:
    sys.path.insert(0, str(_VIZ))

from ripper_combined_minima import paper_runtime_raw_dict  # noqa: E402

BASE_ROOT = Path(__file__).resolve().parents[2] / "results"
TRIALS = BASE_ROOT / "trials"
RIPPER_CPU_228 = BASE_ROOT / "ripper_228_conc_sweep"

models = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC", "EQCCT"]
raw = paper_runtime_raw_dict(TRIALS, RIPPER_CPU_228)

cpu_vals = []
gpu_vals = []
for m in models:
    r_cpu = raw.get((m, "CPU", "Ripper"), (None, 0))[0]
    m_cpu = raw.get((m, "CPU", "MA"), (None, 0))[0]
    r_gpu = raw.get((m, "GPU", "Ripper"), (None, 0))[0]
    m_gpu = raw.get((m, "GPU", "MA"), (None, 0))[0]
    cpu_vals.append(
        round((1 - m_cpu / r_cpu) * 100) if r_cpu and m_cpu and r_cpu > 0 else 0
    )
    gpu_vals.append(
        round((1 - m_gpu / r_gpu) * 100) if r_gpu and m_gpu and r_gpu > 0 else 0
    )

x        = np.arange(len(models))
bar_w    = 0.35
fig, ax  = plt.subplots(figsize=(10, 5.5))

CPU_COLOR = "#4878CF"
GPU_COLOR = "#F28522"

bars_cpu = ax.bar(x - bar_w / 2, cpu_vals, bar_w,
                  color=CPU_COLOR, edgecolor="white", linewidth=0.8,
                  label="CPU")
bars_gpu = ax.bar(x + bar_w / 2, gpu_vals, bar_w,
                  color=GPU_COLOR, edgecolor="white", linewidth=0.8,
                  label="GPU")

for bar in bars_cpu:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
            f"{int(h)}%", ha="center", va="bottom",
            fontsize=10, color=CPU_COLOR)

for bar in bars_gpu:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
            f"{int(h)}%", ha="center", va="bottom",
            fontsize=10, color=GPU_COLOR)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel("Runtime Reduction over Ripper (%)", fontsize=12)
ax.set_xlabel("Model", fontsize=12, labelpad=20)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100))
ax.set_yticks(range(0, 101, 10))

ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

ax.set_title(
    "Model-Actor Runtime Reduction over Ripper\nat 228 Stations (CPU vs. GPU)",
    fontsize=13, pad=12,
)

cpu_patch = mpatches.Patch(color=CPU_COLOR, label="CPU")
gpu_patch = mpatches.Patch(color=GPU_COLOR, label="GPU")
fig.legend(
    handles=[cpu_patch, gpu_patch],
    ncol=2,
    fontsize=10,
    framealpha=0.9,
    edgecolor="#cccccc",
    loc="upper center",
    bbox_to_anchor=(0.505, 0.15),
    bbox_transform=fig.transFigure,
)

fig.subplots_adjust(bottom=0.30, top=0.92)
fig.savefig(
    Path(__file__).resolve().parents[2] / "docs" / "figures" / "fig4.png",
    dpi=180,
    bbox_inches="tight",
    pad_inches=0.15,
)
print(f"Saved {Path(__file__).resolve().parents[2] / 'docs/figures/fig4.png'}")
print(f"CPU reductions: {dict(zip(models, cpu_vals))}")
print(f"GPU reductions: {dict(zip(models, gpu_vals))}")
