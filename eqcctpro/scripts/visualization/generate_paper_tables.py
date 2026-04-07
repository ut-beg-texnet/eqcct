#!/usr/bin/env python3
"""
Generate paper tables for EQCCTPro methodology section.

Creates:
- Table 1: Per-model memory requirements (from parallelization.py)
- Table 2: Optimal configuration picking times and actor creation (from trial results)
- Table 3: Best Ripper at 228 (Ray CPUs 5–20): one CPU row per model + best 1-GPU and 2-GPU rows per model, sorted by runtime
- Table 4–5: Model-Actor minima at 228 (paper Ray CPU grid); Table 4 includes separate 1- and 2-GPU rows when present, sorted by total time

Usage:
    python generate_paper_tables.py [--output_dir PATH]
"""

import ast
import csv
import argparse
from pathlib import Path

# Import memory constants from parallelization
import sys

_VIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(_VIZ))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ripper_combined_minima import (  # noqa: E402
    paper_ma_ray_cpus_ok,
    table3_ripper_rows_sorted,
    table4_ma_entries_with_raw,
    table4_ma_rows_sorted,
    trial_success,
)
from eqcctpro.parallelization import (
    SEISBENCH_MODEL_VRAM_MB,
    SEISBENCH_MODEL_RAM_MB,
    SEISBENCH_MODEL_CPU_RAM_MB,
    VRAM_BUFFER_MB,
    RAM_BUFFER_MB,
    EQCCT_GPU_VRAM_MB,
    EQCCT_GPU_RAM_MB,
    EQCCT_CPU_RAM_MB,
    get_seisbench_model_vram_mb_ripper,
)


def _safe_float(val, default=0.0):
    if val is None or val == "" or str(val).strip() == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def build_table1_memory(output_path: Path):
    """Table 1: Per-model memory requirements (MB)."""
    # Models used in the study
    models = [
        ("PhaseNet", "original", "PyTorch"),
        ("PhaseNetLight", "stead", "PyTorch"),
        ("EQTransformer", "original", "PyTorch"),
        ("EQTransformer-NC", "original_nonconservative", "PyTorch"),
        ("EQCCT", None, "TensorFlow"),
    ]
    rows = []
    for m in models:
        parent, child, fw = m
        if parent == "EQCCT":
            base_vram = EQCCT_GPU_VRAM_MB
            base_ram_gpu = EQCCT_GPU_RAM_MB
            base_ram_cpu = EQCCT_CPU_RAM_MB
            ma_vram = base_vram + VRAM_BUFFER_MB
            ma_ram_gpu = base_ram_gpu + RAM_BUFFER_MB
            ma_ram_cpu = base_ram_cpu + RAM_BUFFER_MB
            ripper_vram = EQCCT_GPU_VRAM_MB * 2.0
        else:
            p_name = "EQTransformer" if parent == "EQTransformer-NC" else parent
            key = (p_name, child)
            base_vram = SEISBENCH_MODEL_VRAM_MB.get(key, 500)
            base_ram_gpu = SEISBENCH_MODEL_RAM_MB.get(key, 870)
            base_ram_cpu = SEISBENCH_MODEL_CPU_RAM_MB.get(key, 502)
            ma_vram = base_vram + VRAM_BUFFER_MB
            ma_ram_gpu = base_ram_gpu + RAM_BUFFER_MB
            ma_ram_cpu = base_ram_cpu + RAM_BUFFER_MB
            ripper_vram = get_seisbench_model_vram_mb_ripper(p_name, child)

        name = parent
        rows.append({
            "Model": name,
            "Framework": fw,
            "RAM CPU (Base)": int(base_ram_cpu),
            "RAM CPU (Actor)": int(ma_ram_cpu),
            "RAM GPU (Base)": int(base_ram_gpu),
            "RAM GPU (Actor)": int(ma_ram_gpu),
            "VRAM GPU (Base)": int(base_vram),
            "VRAM GPU (Ripper)": int(ripper_vram),
            "VRAM GPU (Actor)": int(ma_vram),
        })

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {output_path}")
    return rows


def _gpu_count_from_field(raw: str | None) -> int:
    raw = (raw or "").strip()
    if not raw or raw == "[]":
        return 0
    try:
        v = ast.literal_eval(raw)
        if isinstance(v, (list, tuple)):
            return len(v)
    except (ValueError, SyntaxError, TypeError):
        pass
    return 0


def build_table3_ripper(
    ripper_cpu_root: Path,
    trials_root: Path,
    output_path: Path,
):
    """Table 3: global-best Ripper at 228 per model (Ray CPUs 5–20); GPU split by 1 vs 2 GPUs; sorted by time."""
    rows_out = table3_ripper_rows_sorted(trials_root, ripper_cpu_root)

    if not rows_out:
        print("No data for Table 3")
        return []
    fieldnames = list(rows_out[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {output_path}")
    return rows_out


def build_table2_timing(results_root: Path, output_path: Path):
    """Table 2: Optimal picking times and actor creation (228 stations, ModelActor only).
    Uses the row with MINIMUM picking time for 228 stations (not the first row)."""
    trials_dir = results_root / "trials"
    if not trials_dir.exists():
        trials_dir = results_root
    rows = []
    seen = {}
    for d in sorted(trials_dir.iterdir()):
        if not d.is_dir() or "ripper" in d.name: continue  # ModelActor only
        for hw in ["cpu", "gpu"]:
            f = d / f"optimal_configurations_{hw}.csv"
            if not f.exists(): continue
            candidates = []
            with open(f) as fp:
                r = csv.DictReader(fp)
                for row in r:
                    n = int(_safe_float(row.get("Number of Stations Used", 0)))
                    if n != 228:
                        continue
                    if not paper_ma_ray_cpus_ok(row):
                        continue
                    pt = _safe_float(row.get("Total Run time for Picker (s)", 0))
                    tt = _safe_float(row.get("Total Trial Time (s)", 0))
                    act = _safe_float(row.get("Actor Creation Time (s)", 0))
                    candidates.append((pt, tt, act))
            if not candidates:
                continue
            # Select row with minimum picking time
            best = min(candidates, key=lambda x: x[0])
            pt, tt, act = best
            if "eqcct" in d.name: model = "EQCCT"
            elif "phasenet_original" in d.name: model = "PhaseNet"
            elif "phasenetlight" in d.name: model = "PhaseNetLight"
            elif "eqtransformer_nonconservative" in d.name: model = "EQTransformer-NC"
            elif "eqtransformer_original" in d.name: model = "EQTransformer"
            else: model = "?"
            key = (model, hw.upper())
            if key in seen and seen[key][0] <= pt: continue
            pct = (act / tt * 100) if tt > 0 else 0
            seen[key] = (pt, tt, act, pct)

    # Sort by hardware (CPU then GPU) and picking time
    for (model, hw), (pt, tt, act, pct) in sorted(seen.items(), key=lambda x: (x[1], x[0])):
        rows.append({
            "Model": model,
            "Device": hw,
            "Min Picking (s)": round(pt, 2),
            "Min Total Trial (s)": round(tt, 2),
            "Setup Time (s)": round(act, 2),
            "Setup Overhead (%)": round(pct, 1),
        })

    if not rows:
        print("No trial data found for Table 2")
        return []
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {output_path}")
    return rows


def build_table4_modelactor(trials_root: Path, output_path: Path):
    """Table 4: Model-Actor minima at 228 (paper CPU grid); separate 1- and 2-GPU rows when present."""
    trials_dir = trials_root / "trials" if (trials_root / "trials").is_dir() else trials_root
    rows_out = table4_ma_rows_sorted(trials_dir)
    if not rows_out:
        print("No data for Table 4")
        return []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {output_path}")
    return rows_out


def build_table5_memory(trials_root: Path, output_path: Path):
    """Process-tree / requested RAM & VRAM at each Table 4 Model-Actor configuration."""
    trials_dir = trials_root / "trials" if (trials_root / "trials").is_dir() else trials_root
    entries = table4_ma_entries_with_raw(trials_dir)
    rows_out: list[dict] = []
    for disp, row in entries:
        rows_out.append({
            "Model": disp["Model"],
            "Device": disp["Device"],
            "CPUs": disp["CPUs"],
            "GPUs": disp["GPUs"],
            "Act.": disp["Actors"],
            "Req. RAM": int(round(_safe_float(row.get("Total Requested RAM (MB)"), 0))),
            "Tree RAM": int(round(_safe_float(row.get("Process Tree RAM (MB)"), 0))),
            "Req. VRAM": int(round(_safe_float(row.get("Total Requested VRAM (MB)"), 0))),
            "Tree VRAM": int(round(_safe_float(row.get("Process Tree VRAM (MB)"), 0))),
            "Peak RAM": int(round(_safe_float(row.get("Peak RAM (MB)"), 0))),
            "_sort_tot": disp["Total (s)"],
        })
    rows_out.sort(key=lambda r: r["_sort_tot"])
    for r in rows_out:
        del r["_sort_tot"]
    if not rows_out:
        print("No data for Table 5")
        return []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {output_path}")
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="docs/tables", help="Output directory for CSV tables")
    ap.add_argument("--results_root", default="results", help="Root of results/ (contains trials/)")
    ap.add_argument(
        "--ripper-cpu-root",
        default=None,
        help="Ripper CPU sweep directory (default: <results_root>/ripper_228_conc_sweep)",
    )
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(args.results_root)
    ripper_cpu = Path(args.ripper_cpu_root) if args.ripper_cpu_root else results / "ripper_228_conc_sweep"
    trials = results / "trials" if (results / "trials").is_dir() else results
    build_table1_memory(out / "table1_memory_requirements.csv")
    build_table2_timing(results, out / "table2_optimal_picking_times.csv")
    build_table3_ripper(ripper_cpu, trials, out / "table3_ripper_228_stations.csv")
    build_table4_modelactor(trials, out / "table4_modelactor_228.csv")
    build_table5_memory(trials, out / "table5_modelactor_memory.csv")


if __name__ == "__main__":
    main()
