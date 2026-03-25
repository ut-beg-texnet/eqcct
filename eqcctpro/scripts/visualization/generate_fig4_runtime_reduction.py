#!/usr/bin/env python3
"""
Generate fig4: Runtime Reduction barplot comparing Ripper vs Model-Actor at 228 stations.
Runtime Reduction = (1 - MA Total / Ripper Total) × 100%
Data: minimum Total Trial Time at 228 stations from trial CSVs (same source as fig5).
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "results" / "trials"
MODEL_MAP = {
    "phasenet_original":             "PhaseNet",
    "phasenetlight_stead":           "PhaseNetLight",
    "eqtransformer_original":        "EQTransformer",
    "eqtransformer_nonconservative": "EQT-NC",
    "eqcct":                         "EQCCT",
}

def min_tt_228(csv_path):
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
                tt = float(row.get("Total Trial Time (s)", 0) or 0)
                if best is None or tt < best:
                    best = tt
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
    val = min_tt_228(csv_file)
    if val is not None:
        raw[(model, hw, orch)] = val

models = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC", "EQCCT"]
cpu_vals = []
gpu_vals = []
for m in models:
    r_cpu = raw.get((m, "CPU", "Ripper"))
    m_cpu = raw.get((m, "CPU", "MA"))
    r_gpu = raw.get((m, "GPU", "Ripper"))
    m_gpu = raw.get((m, "GPU", "MA"))
    cpu_vals.append(round((1 - m_cpu / r_cpu) * 100) if r_cpu and m_cpu and r_cpu > 0 else 0)
    gpu_vals.append(round((1 - m_gpu / r_gpu) * 100) if r_gpu and m_gpu and r_gpu > 0 else 0)

# ── Layout ─────────────────────────────────────────────────────────────────────
x        = np.arange(len(models))
bar_w    = 0.35
fig, ax  = plt.subplots(figsize=(10, 5.5))

# Colours consistent with the rest of the paper figures (muted blue / orange)
CPU_COLOR = "#4878CF"
GPU_COLOR = "#F28522"

bars_cpu = ax.bar(x - bar_w / 2, cpu_vals, bar_w,
                  color=CPU_COLOR, edgecolor="white", linewidth=0.8,
                  label="CPU")
bars_gpu = ax.bar(x + bar_w / 2, gpu_vals, bar_w,
                  color=GPU_COLOR, edgecolor="white", linewidth=0.8,
                  label="GPU")

# ── Value labels on top of each bar ───────────────────────────────────────────
for bar in bars_cpu:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
            f"{int(h)}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=CPU_COLOR)

for bar in bars_gpu:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
            f"{int(h)}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=GPU_COLOR)

# ── Axes formatting ────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel("Runtime Reduction over Ripper (%)", fontsize=12)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100))
ax.set_yticks(range(0, 101, 10))

# Light horizontal grid
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Model-Actor Runtime Reduction over Ripper\nat 228 Stations (CPU vs. GPU)",
    fontsize=13, fontweight="bold", pad=12
)

# ── Legend ────────────────────────────────────────────────────────────────────
cpu_patch = mpatches.Patch(color=CPU_COLOR, label="CPU")
gpu_patch = mpatches.Patch(color=GPU_COLOR, label="GPU")
ax.legend(handles=[cpu_patch, gpu_patch],
          title="Device", title_fontsize=10,
          fontsize=10, framealpha=0.9,
          loc="lower right", edgecolor="#cccccc")

fig.tight_layout()

out = Path(__file__).resolve().parents[2] / "docs" / "figures" / "fig4.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved {out}")
