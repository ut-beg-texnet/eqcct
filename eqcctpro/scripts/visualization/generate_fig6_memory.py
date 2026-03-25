#!/usr/bin/env python3
"""
Generate fig6: Ripper vs Model-Actor memory at 228 stations.

Left:  CPU — Process-Tree RAM (GB)
Right: GPU — Process-Tree VRAM (GB)

Data from scripts/benchmark_peak_memory.py: N model instances loaded
simultaneously via Ray actors; RAM/VRAM via psutil + pynvml.

/// = Ripper, ... = Model-Actor.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
MEASURED_FILE = PROJECT / "results" / "benchmark_results" / "peak_memory_measured.json"

with open(MEASURED_FILE) as f:
    measured = json.load(f)

for key in measured:
    d = measured[key]
    d["tree_ram_gb"] = d["tree_ram_mb"] / 1024
    d["tree_vram_gb"] = d["tree_vram_mb"] / 1024

MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC", "EQCCT"]
MODEL_COLORS = {
    "PhaseNet":      "#2471A3",
    "PhaseNetLight": "#1E8449",
    "EQTransformer": "#D4AC0D",
    "EQT-NC":        "#E67E22",
    "EQCCT":         "#8E44AD",
}

def get(model, hw, orch, field):
    key = f"{model}_{hw}_{orch}"
    return measured[key][field]

BAR_W   = 0.32
gap     = 0.06
offsets = [-(BAR_W / 2 + gap / 2), (BAR_W / 2 + gap / 2)]
x       = np.arange(len(MODELS))
YMAX    = 120
YTICKS  = list(range(0, YMAX + 1, 20))

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
fig.subplots_adjust(wspace=0.08)


def draw_panel(ax, hw, mem_field, title, ylabel):
    for i, m in enumerate(MODELS):
        c = MODEL_COLORS[m]
        r_val = get(m, hw, "Ripper", mem_field)
        a_val = get(m, hw, "MA", mem_field)

        bar_r = min(r_val, YMAX)
        bar_a = min(a_val, YMAX)

        ax.bar(x[i] + offsets[0], bar_r, BAR_W,
               color=c, edgecolor="white", linewidth=0.7,
               hatch="///", zorder=3)
        ax.bar(x[i] + offsets[1], bar_a, BAR_W,
               color=c, edgecolor="white", linewidth=0.7,
               hatch="...", zorder=3)

        pad = YMAX * 0.008
        for j, v in enumerate([r_val, a_val]):
            if v > 0.3:
                display_v = min(v, YMAX)
                y_text = min(display_v + pad, YMAX * 0.96)
                va = "bottom" if display_v + pad <= YMAX * 0.96 else "top"
                ax.text(x[i] + offsets[j], y_text,
                        f"{v:.1f}", ha="center", va=va,
                        fontsize=8, fontweight="bold", color=c, clip_on=True)

    xlabels = []
    for i, m in enumerate(MODELS):
        r_n = get(m, hw, "Ripper", "n_instances")
        a_n = get(m, hw, "MA", "n_instances")
        xlabels.append(f"{m}\n(R:{r_n} / MA:{a_n})")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9, rotation=25, ha="right")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Model", fontsize=11, labelpad=10)
    ax.set_ylim(0, YMAX)
    ax.set_yticks(YTICKS)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


draw_panel(axes[0], "CPU", "tree_ram_gb",
           "CPU — Peak RAM at 228 Stations", "RAM (GB)")
draw_panel(axes[1], "GPU", "tree_vram_gb",
           "GPU — Peak VRAM at 228 Stations", "VRAM (GB)")

legend_handles = [
    mpatches.Patch(facecolor="#888888", edgecolor="white",
                   hatch="///", label="Ripper"),
    mpatches.Patch(facecolor="#888888", edgecolor="white",
                   hatch="...", label="Model-Actor"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center", ncol=2,
    fontsize=11, framealpha=0.9, edgecolor="#cccccc",
    bbox_to_anchor=(0.5, -0.12),
)

fig.suptitle(
    "Ripper vs. Model-Actor:\nPeak Memory at 228 Stations\n",
    fontsize=14, fontweight="bold", y=1.01,
)

out = PROJECT / "docs" / "figures" / "fig6.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved {out}")

print("\nData verification (GB at 228 stations):")
for hw in ["CPU", "GPU"]:
    mem_f = "tree_ram_gb" if hw == "CPU" else "tree_vram_gb"
    print(f"  {hw}:")
    for m in MODELS:
        r_v = get(m, hw, "Ripper", mem_f)
        a_v = get(m, hw, "MA", mem_f)
        r_n = get(m, hw, "Ripper", "n_instances")
        a_n = get(m, hw, "MA", "n_instances")
        print(f"    {m}: R={r_v:.1f} GB (n={r_n}) | MA={a_v:.1f} GB (n={a_n})")
