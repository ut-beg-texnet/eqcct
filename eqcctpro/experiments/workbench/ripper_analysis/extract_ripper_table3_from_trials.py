#!/usr/bin/env python3
"""
Recompute optimal Ripper rows for Table 3 from raw trial CSVs only.

Uses:
  results/trials/eval_*_ripper/cpu_test_results.csv
  results/trials/eval_*_ripper/gpu_test_results.csv

Does NOT read best_overall_usecase*.csv.

For each model×device, selects successful trials (Trial Success) with exactly
228 stations and minimum Total Trial Time (s). Prints markdown/LaTeX rows and a
Python TABLE3 snippet for benchmark_ripper_table3_rerun.py.

Run from repo root:
  python scripts/extract_ripper_table3_from_trials.py
"""
from __future__ import annotations

import ast
import csv
import glob
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TRIALS = PROJECT / "results" / "trials"

# eval dir name -> (paper model name, mtype, parent, child)
RIPPER_EVAL_MAP: dict[str, tuple[str, str, str | None, str | None]] = {
    "eval_cpu_phasenet_original_ripper": ("PhaseNet", "seisbench", "PhaseNet", "original"),
    "eval_gpu_phasenet_original_ripper": ("PhaseNet", "seisbench", "PhaseNet", "original"),
    "eval_cpu_phasenetlight_stead_ripper": ("PhaseNetLight", "seisbench", "PhaseNetLight", "stead"),
    "eval_gpu_phasenetlight_stead_ripper": ("PhaseNetLight", "seisbench", "PhaseNetLight", "stead"),
    "eval_cpu_eqtransformer_original_ripper": ("EQTransformer", "seisbench", "EQTransformer", "original"),
    "eval_gpu_eqtransformer_original_ripper": ("EQTransformer", "seisbench", "EQTransformer", "original"),
    "eval_cpu_eqtransformer_nonconservative_ripper": (
        "EQTransformer-NC",
        "seisbench",
        "EQTransformer",
        "original_nonconservative",
    ),
    "eval_gpu_eqtransformer_nonconservative_ripper": (
        "EQTransformer-NC",
        "seisbench",
        "EQTransformer",
        "original_nonconservative",
    ),
    "eval_cpu_eqcct_ripper": ("EQCCT", "eqcct", None, None),
    "eval_gpu_eqcct_ripper": ("EQCCT", "eqcct", None, None),
}

# Table 3 row order (match paper)
ROW_ORDER = [
    ("PhaseNet", "CPU"),
    ("PhaseNetLight", "CPU"),
    ("EQTransformer", "CPU"),
    ("EQTransformer-NC", "CPU"),
    ("PhaseNetLight", "GPU"),
    ("EQTransformer", "GPU"),
    ("EQTransformer-NC", "GPU"),
    ("PhaseNet", "GPU"),
    ("EQCCT", "CPU"),
    ("EQCCT", "GPU"),
]


def _trial_ok(row: dict[str, str]) -> bool:
    s = str(row.get("Trial Success", "")).strip()
    return s in ("1", "1.0", "True")


def _float(x: str | None) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _parse_gpu_list(raw: str) -> list[int]:
    raw = (raw or "").strip()
    if not raw or raw == "[]":
        return []
    try:
        v = ast.literal_eval(raw)
        if isinstance(v, list):
            return [int(x) for x in v]
    except (ValueError, SyntaxError, TypeError):
        pass
    m = re.findall(r"\d+", raw)
    return [int(x) for x in m]


def best_row_for_csv(path: Path) -> dict[str, str] | None:
    best_t: float | None = None
    best: dict[str, str] | None = None
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                nsta = int(float(row["Number of Stations Used"]))
            except (KeyError, ValueError):
                continue
            if nsta != 228 or not _trial_ok(row):
                continue
            t = _float(row.get("Total Trial Time (s)"))
            if t is None:
                continue
            if best_t is None or t < best_t:
                best_t = t
                best = row
    return best


def row_to_table3_tuple(
    paper_name: str, dev: str, br: dict[str, str], mtype: str, parent: str | None, child: str | None
) -> tuple:
    cpus = int(float(br["Number of CPUs Allocated for Ray to Use"]))
    gpus_used = _parse_gpu_list(br.get("GPUs Used", ""))
    n_gpus = len(gpus_used)
    actual_conc = int(float(br["Actual Ripper Concurrent Tasks"]))
    return (paper_name, dev, cpus, n_gpus, actual_conc, mtype, parent, child)


def main() -> int:
    by_key: dict[tuple[str, str], tuple[dict[str, str], tuple[str, str, str | None, str | None], Path]] = {}
    for csv_path in sorted(TRIALS.glob("eval_*_ripper/*_test_results.csv")):
        eval_dir = csv_path.parent.name
        meta = RIPPER_EVAL_MAP.get(eval_dir)
        if not meta:
            print(f"skip unmapped: {csv_path}", file=sys.stderr)
            continue
        paper_name, mtype, parent, child = meta
        dev = "GPU" if "gpu_" in eval_dir else "CPU"
        key = (paper_name, dev)
        br = best_row_for_csv(csv_path)
        if br is None:
            print(f"no 228-station success row: {csv_path}", file=sys.stderr)
            continue
        by_key[key] = (br, (paper_name, mtype, parent, child), csv_path)

    missing = [k for k in ROW_ORDER if k not in by_key]
    if missing:
        print(f"Missing keys: {missing}", file=sys.stderr)
        return 1

    print("# Table 3 from trial CSVs (min Total Trial Time @ 228 st, Trial Success)\n")
    for key in ROW_ORDER:
        br, meta, src = by_key[key]
        paper_name, mtype, parent, child = meta
        t = _float(br["Total Trial Time (s)"])
        pick = _float(br.get("Total Run time for Picker (s)"))
        cpus = int(float(br["Number of CPUs Allocated for Ray to Use"]))
        gpus_used = _parse_gpu_list(br.get("GPUs Used", ""))
        req_c = br.get("Number of Concurrent Station Tasks")
        act_c = br.get("Actual Ripper Concurrent Tasks")
        print(f"{paper_name} {key[1]}: trial {br.get('Trial Number')}  total={t:.4f}s pick={pick:.4f}s")
        print(f"  CPUs={cpus} GPUs={gpus_used} conc_req={req_c} actual_conc={act_c}")
        print(f"  <- {src.relative_to(PROJECT)}\n")

    print("\n# Python TABLE3 (conc_tasks = actual concurrent Ripper tasks)\n")
    for key in ROW_ORDER:
        br, (_, mtype, parent, child), _ = by_key[key]
        tup = row_to_table3_tuple(key[0], key[1], br, mtype, parent, child)
        print(f"    {tup!r},")

    print("\n# Markdown table (2 dp)\n")
    for key in ROW_ORDER:
        br, _, _ = by_key[key]
        t = _float(br["Total Trial Time (s)"])
        cpus = int(float(br["Number of CPUs Allocated for Ray to Use"]))
        gpus_used = _parse_gpu_list(br.get("GPUs Used", ""))
        n_gpu = len(gpus_used)
        act_c = int(float(br["Actual Ripper Concurrent Tasks"]))
        assert t is not None
        print(f"| {key[0]:16} | {key[1]:5} | {cpus:4} | {n_gpu:4} | {act_c:11} | {t:24.2f} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
