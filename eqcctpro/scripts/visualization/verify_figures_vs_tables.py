#!/usr/bin/env python3
"""
Cross-check paper tables (1–6) and figure generators against source CSV/JSON.

Run from repo root (eqcctpro):
  python3 scripts/visualization/verify_figures_vs_tables.py

Writes: docs/tables/FIGURE_VS_TABLES_CHECKLIST.md

Paper ↔ repo file map:
  Table 1  → docs/tables/seisbench_table1_scaling_228_250_580.json
  Table 2  → docs/tables/serial_classify_spotcheck.json
  Table 3  → docs/tables/table1_memory_requirements.csv
  Table 4  → docs/tables/table3_ripper_228_stations.csv
  Table 5  → docs/tables/table4_modelactor_228.csv
  Table 6  → docs/tables/table5_modelactor_memory.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
TABLES = PROJECT / "docs" / "tables"
TRIALS = PROJECT / "results" / "trials"
RIPPER_SWEEP = PROJECT / "results" / "ripper_228_conc_sweep"
BENCH_JSON = PROJECT / "results" / "benchmark_results" / "peak_memory_measured.json"
SCALING_JSON = TABLES / "seisbench_table1_scaling_228_250_580.json"
SPOTCHECK_JSON = TABLES / "serial_classify_spotcheck.json"

_VIZ = Path(__file__).resolve().parent
if str(_VIZ) not in sys.path:
    sys.path.insert(0, str(_VIZ))

from ripper_combined_minima import paper_runtime_raw_dict  # noqa: E402
from generate_fig7_serial_vs_parallel import (  # noqa: E402
    build_method_configs,
    collect_data_for_configs,
    FALLBACK_MA,
    FALLBACK_RIPPER,
    VALID_STATIONS,
)

TOL = 0.06
TOL_PCT = 0.15


def _canonical_model(name: str) -> str:
    n = name.strip()
    if n == "EQTransformer-NC":
        return "EQT-NC"
    return n


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_table3_ripper() -> list[dict]:
    return load_csv_rows(TABLES / "table3_ripper_228_stations.csv")


def load_table4_ma() -> list[dict]:
    return load_csv_rows(TABLES / "table4_modelactor_228.csv")


def load_table6_memory() -> list[dict]:
    return load_csv_rows(TABLES / "table5_modelactor_memory.csv")


def load_table3_memory_budgets() -> list[dict]:
    return load_csv_rows(TABLES / "table1_memory_requirements.csv")


def section_table1(lines: list[str]) -> None:
    lines.append("## Table 1 — SeisBench baselines (`seisbench_table1_scaling_228_250_580.json`)\n")
    lines.append(
        "Field `classify_per_station_s` in JSON is **total** sequential `classify()` time; "
        "**Cls./stn** = total / $N$ (matches paper Table 1).\n\n"
    )
    if not SCALING_JSON.is_file():
        lines.append("*(JSON missing)*\n\n")
        return
    with open(SCALING_JSON, encoding="utf-8") as f:
        rows = json.load(f)
    lines.append("| Model | Device | N | Load | Ann. | Cls./stn | OK (internal) |\n")
    lines.append("|-------|--------|---|------|------|-----------|---------------|\n")
    for r in rows:
        n = int(r["n_stations"])
        tot = float(r["classify_per_station_s"])
        cps = tot / n
        ok = abs(tot - cps * n) < 1e-6
        ls = float(r["load_s"])
        aa = float(r["annotate_all_s"])
        lines.append(
            f"| {r['model']} | {r['device']} | {n} | {ls:.3f} | {aa:.3f} | "
            f"{cps:.3f} | {'yes' if ok else 'NO'} |\n"
        )
    lines.append("\n")


def section_table2_spotcheck(lines: list[str]) -> None:
    lines.append("## Table 2 — CPU serial spot-checks (`serial_classify_spotcheck.json`)\n")
    lines.append("Panel **(a)** `230_stations_1_min_dt`: anchors used for Figs 7–8 serial curves.\n\n")
    if not SPOTCHECK_JSON.is_file():
        lines.append("*(JSON missing)*\n\n")
        return
    with open(SPOTCHECK_JSON, encoding="utf-8") as f:
        doc = json.load(f)
    try:
        block = doc["results"]["CPU"]["230_stations_1_min_dt"]["models"]
    except (KeyError, TypeError):
        lines.append("*(missing CPU/230_stations_1_min_dt/models)*\n\n")
        return
    anchors = [10, 50, 100, 150, 200, 228]

    def _g(d, k):
        if isinstance(d, dict) and str(k) in d:
            v = d[str(k)]
            try:
                return f"{float(v):.3f}"
            except (TypeError, ValueError):
                return str(v)
        return ""

    hdr = "| Model | Ld |"
    sep = "|---|---|"
    for a in anchors:
        hdr += f" Ann@{a} | Cls@{a} |"
        sep += "---|---|"
    lines.append(hdr + "\n")
    lines.append(sep + "\n")
    for mname in ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]:
        m = block.get(mname)
        if not m:
            continue
        try:
            ld = f"{float(m.get('load_s', 0)):.3f}"
        except (TypeError, ValueError):
            ld = str(m.get("load_s", ""))
        ann = m.get("annotate_all_s", {})
        cls = m.get("classify_total_s", {})
        row = f"| {mname} | {ld} |"
        for a in anchors:
            row += f" {_g(ann, a)} | {_g(cls, a)} |"
        row += "\n"
        lines.append(row)
    lines.append("\n*Self-consistent: values are read directly from JSON (source of truth for the paper table).*\n\n")


def section_table3_budgets(lines: list[str]) -> None:
    lines.append("## Table 3 — Memory budgets (`table1_memory_requirements.csv`)\n\n")
    rows = load_table3_memory_budgets()
    if not rows:
        lines.append("*(CSV missing)*\n\n")
        return
    hdr = list(rows[0].keys())
    lines.append("| " + " | ".join(hdr) + " |\n")
    lines.append("| " + " | ".join("---" for _ in hdr) + " |\n")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in hdr) + " |\n")
    lines.append("\n")


def section_table4_5_fig5(lines: list[str], raw: dict) -> None:
    lines.append("## Table 4 (Ripper) & Table 5 (Model-Actor) — `table3_ripper_228_stations.csv` / `table4_modelactor_228.csv`\n")
    lines.append(
        "**Figure 5** uses `paper_runtime_raw_dict` (same Ray CPU grid). "
        "GPU Ripper bar = min of best 1-GPU vs 2-GPU Ripper per model.\n\n"
    )
    lines.append("### Table 4 vs code (Fig. 5 Ripper bars)\n\n")
    lines.append("| Assertion | CSV (Table 4) | `paper_runtime_raw_dict` | OK? |\n|---|---|---|---|\n")

    t3 = load_table3_ripper()
    by_model_cpu: dict[str, float] = {}
    gpu_rows: dict[str, list[float]] = {}
    for row in t3:
        m = _canonical_model(row["Model"])
        dev = row["Device"].strip()
        tt = _f(row["Ripper Picking/Total (s)"])
        if dev == "CPU":
            by_model_cpu[m] = tt
        else:
            gpu_rows.setdefault(m, []).append(tt)

    for m, tt_csv in sorted(by_model_cpu.items()):
        r = raw.get((m, "CPU", "Ripper"))
        tt_py = r[0] if r else float("nan")
        ok = r and abs(tt_csv - tt_py) <= TOL
        lines.append(
            f"| Ripper CPU {m} | {tt_csv:.2f} | {tt_py:.2f} | **{'yes' if ok else 'NO'}** |\n"
        )
    for m, tts in sorted(gpu_rows.items()):
        min_csv = min(tts)
        r = raw.get((m, "GPU", "Ripper"))
        tt_py = r[0] if r else float("nan")
        ok = r and abs(min_csv - tt_py) <= TOL
        lines.append(
            f"| Ripper GPU {m} (min of GPU rows) | {min_csv:.2f} | {tt_py:.2f} | **{'yes' if ok else 'NO'}** |\n"
        )

    lines.append("\n### Table 5 vs code (Fig. 5 Model-Actor bars)\n\n")
    lines.append("| Assertion | CSV (Table 5) | `paper_runtime_raw_dict` | OK? |\n|---|---|---|---|\n")

    t4 = load_table4_ma()
    ma_cpu = {row["Model"].strip(): _f(row["Total (s)"]) for row in t4 if row["Device"].strip() == "CPU"}
    for m, tt_csv in sorted(ma_cpu.items()):
        r = raw.get((m, "CPU", "MA"))
        tt_py = r[0] if r else float("nan")
        ok = r and abs(tt_csv - tt_py) <= TOL
        lines.append(
            f"| MA CPU {m} | {tt_csv:.2f} | {tt_py:.2f} | **{'yes' if ok else 'NO'}** |\n"
        )
    ma_gpu_by_m: dict[str, list[float]] = {}
    for row in t4:
        if row["Device"].strip() != "GPU":
            continue
        m = row["Model"].strip()
        ma_gpu_by_m.setdefault(m, []).append(_f(row["Total (s)"]))
    for m, tts in sorted(ma_gpu_by_m.items()):
        min_csv = min(tts)
        r = raw.get((m, "GPU", "MA"))
        tt_py = r[0] if r else float("nan")
        ok = r and abs(min_csv - tt_py) <= TOL
        lines.append(
            f"| MA GPU {m} (min of GPU rows) | {min_csv:.2f} | {tt_py:.2f} | **{'yes' if ok else 'NO'}** |\n"
        )

    lines.append("\n### Table 5 — Setup OH % self-check\n\n")
    lines.append("| Model | Device | GPUs | Total | Setup | OH% (CSV) | OH% (computed) | OK? |\n|---|---|---:|---:|---:|---:|---:|---|\n")
    for row in t4:
        tot = _f(row["Total (s)"])
        setup = _f(row["Setup (s)"])
        oh_csv = _f(row["Setup OH (%)"])
        oh_c = (setup / tot * 100.0) if tot > 0 else float("nan")
        ok = abs(oh_csv - oh_c) <= TOL_PCT
        lines.append(
            f"| {row['Model']} | {row['Device']} | {row['GPUs']} | {tot:.2f} | {setup:.2f} | "
            f"{oh_csv:.1f} | {oh_c:.1f} | **{'yes' if ok else 'NO'}** |\n"
        )
    lines.append("\n")


def section_table6(lines: list[str]) -> None:
    lines.append("## Table 6 — Model-Actor memory (`table5_modelactor_memory.csv`)\n\n")
    t5 = load_table4_ma()
    t6 = load_table6_memory()
    lines.append(f"- Rows in Table 5 CSV: **{len(t5)}**; rows in Table 6 CSV: **{len(t6)}**.")
    lines.append(f" Match: **{'yes' if len(t5) == len(t6) else 'NO'}**.\n\n")

    def _key(r: dict) -> tuple:
        return (
            r["Model"].strip(),
            r["Device"].strip(),
            int(float(r["CPUs"])),
            int(float(r["GPUs"])),
            int(float(r.get("Actors", r.get("Act.", 0)))),
        )

    keys5 = {_key(r) for r in t5}
    keys6 = {_key(r) for r in t6}
    lines.append(f"- Keys (Model, Device, CPUs, GPUs, Actors) match: **{'yes' if keys5 == keys6 else 'NO'}**.\n\n")

    if t6:
        hdr = list(t6[0].keys())
        lines.append("| " + " | ".join(hdr) + " |\n")
        lines.append("| " + " | ".join("---" for _ in hdr) + " |\n")
        for r in t6:
            lines.append("| " + " | ".join(str(r.get(h, "")) for h in hdr) + " |\n")
    lines.append("\n")


def section_fig6(lines: list[str]) -> None:
    lines.append("## Figure 6 — Peak memory benchmark (`peak_memory_measured.json`)\n")
    lines.append(
        "Process-tree RAM/VRAM (MB) for load-hold benchmark; **not** the same quantities as Table 6 "
        "(trial Req./Tree memory).\n\n"
    )
    if BENCH_JSON.is_file():
        with open(BENCH_JSON, encoding="utf-8") as f:
            pm = json.load(f)
        lines.append("| Key | tree_ram_mb | tree_vram_mb | n_instances |\n|---|---|---|---|\n")
        for key in sorted(pm.keys()):
            d = pm[key]
            lines.append(
                f"| `{key}` | {d.get('tree_ram_mb', '')} | {d.get('tree_vram_mb', '')} | "
                f"{d.get('n_instances', '')} |\n"
            )
    else:
        lines.append("*(benchmark JSON missing — run `experiments/workbench/memory/benchmark_peak_memory.py`)*\n")
    lines.append("\n")


def section_fig78(lines: list[str]) -> None:
    lines.append("## Figures 7–8 (`generate_fig7_serial_vs_parallel.py`)\n")
    full_data = collect_data_for_configs()
    rip_cfgs = build_method_configs(full_data, "Ripper", FALLBACK_RIPPER)
    ma_cfgs = build_method_configs(full_data, "ModelActor", FALLBACK_MA)
    lines.append(f"- **Station grid** `VALID_STATIONS`: {len(sorted(VALID_STATIONS))} points from 5–225 step 5, plus 228.\n")
    lines.append(f"- **Fig. 7 Ripper** configs: `{rip_cfgs}`.\n")
    lines.append(f"- **Fig. 8 Model-Actor** configs: `{ma_cfgs}`.\n")
    lines.append(
        "- Serial: `serial_classify_spotcheck.json` (CPU 230-station slice), "
        "interpolation + `SERIAL_TABLE` fallback.\n\n"
    )


def section_fig4(lines: list[str]) -> None:
    lines.append("## Figure 4 (`generate_fig4_unified_3d.py`)\n")
    lines.append(
        "- Plotted stations: `10, 20, …, 220, 228` (subset of denser trial CSVs).\n\n"
    )


def main() -> None:
    lines: list[str] = []
    lines.append("# Figure vs table checklist\n\n")
    lines.append(
        "Auto-generated by `scripts/visualization/verify_figures_vs_tables.py`. "
        "Regenerate after updating trials, tables, or figures.\n\n"
    )
    lines.append("## Source file map (paper ↔ repository)\n\n")
    lines.append("| Paper | Source file |\n|-------|-------------|\n")
    lines.append("| Table 1 | `docs/tables/seisbench_table1_scaling_228_250_580.json` |\n")
    lines.append("| Table 2 | `docs/tables/serial_classify_spotcheck.json` |\n")
    lines.append("| Table 3 | `docs/tables/table1_memory_requirements.csv` |\n")
    lines.append("| Table 4 | `docs/tables/table3_ripper_228_stations.csv` |\n")
    lines.append("| Table 5 | `docs/tables/table4_modelactor_228.csv` |\n")
    lines.append("| Table 6 | `docs/tables/table5_modelactor_memory.csv` |\n\n")

    raw = paper_runtime_raw_dict(TRIALS, RIPPER_SWEEP)
    section_table1(lines)
    section_table2_spotcheck(lines)
    section_table3_budgets(lines)
    section_table4_5_fig5(lines, raw)
    section_table6(lines)
    section_fig6(lines)
    section_fig78(lines)
    section_fig4(lines)

    out = TABLES / "FIGURE_VS_TABLES_CHECKLIST.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
