"""
Best successful Ripper total time at 228 stations over the combined log corpus:
  - GPU: results/trials eval_gpu_*_ripper + results/ripper_228_conc_sweep eval_sweep_ripper_gpu_*
  - CPU: results/ripper_228_conc_sweep eval_sweep_ripper_cpu_* (trial CPU Ripper is superseded when sweep exists)
"""
from __future__ import annotations

import csv
from pathlib import Path


def trial_success(row: dict) -> bool:
    v = (row.get("Trial Success") or "").strip().lower()
    return v in ("1", "true", "yes", "1.0")


# Exploratory high-CPU runs are omitted from paper tables / figures.
_EXCLUDED_RAY_CPUS = frozenset({41, 46})


def ray_cpus_allowed(row: dict) -> bool:
    try:
        c = int(float(row.get("Number of CPUs Allocated for Ray to Use") or -1))
    except (TypeError, ValueError):
        return False
    return c not in _EXCLUDED_RAY_CPUS


def _gpu_count(raw: str | None) -> int:
    raw = (raw or "").strip()
    if not raw or raw == "[]":
        return 0
    import ast

    try:
        v = ast.literal_eval(raw)
        if isinstance(v, (list, tuple)):
            return len(v)
    except (ValueError, SyntaxError, TypeError):
        pass
    return 0


def _scan_gpu_csv(path: Path) -> tuple[float, dict] | None:
    if not path.is_file():
        return None
    best_tt = None
    best_row: dict | None = None
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                n = int(float(row.get("Number of Stations Used") or 0))
            except (ValueError, TypeError):
                continue
            if n != 228 or not trial_success(row) or not ray_cpus_allowed(row):
                continue
            try:
                tt = float(row.get("Total Trial Time (s)") or 1e9)
            except (ValueError, TypeError):
                continue
            if best_tt is None or tt < best_tt:
                best_tt = tt
                best_row = row
    if best_row is None:
        return None
    return (best_tt, best_row)


# trials dirname -> canonical model label (matches paper Table 3)
TRIALS_GPU_RIPPER = {
    "eval_gpu_phasenet_original_ripper": "PhaseNet",
    "eval_gpu_phasenetlight_stead_ripper": "PhaseNetLight",
    "eval_gpu_eqtransformer_original_ripper": "EQTransformer",
    "eval_gpu_eqtransformer_nonconservative_ripper": "EQTransformer-NC",
    "eval_gpu_eqcct_ripper": "EQCCT",
}

# (model, subdir under ripper_228_conc_sweep)
RIPPER_SWEEP_GPU = [
    ("PhaseNet", "eval_sweep_ripper_gpu_phasenet_original_1gpu"),
    ("PhaseNet", "eval_sweep_ripper_gpu_phasenet_original"),
    ("PhaseNetLight", "eval_sweep_ripper_gpu_phasenetlight_stead"),
    ("PhaseNetLight", "eval_sweep_ripper_gpu_phasenetlight_stead_2gpu"),
    ("EQTransformer", "eval_sweep_ripper_gpu_eqtransformer_original"),
    ("EQTransformer", "eval_sweep_ripper_gpu_eqtransformer_original_2gpu"),
    ("EQTransformer-NC", "eval_sweep_ripper_gpu_eqtransformer_nc"),
    ("EQTransformer-NC", "eval_sweep_ripper_gpu_eqtransformer_nc_2gpu"),
    ("EQCCT", "eval_sweep_ripper_gpu_eqcct_1gpu"),
    ("EQCCT", "eval_sweep_ripper_gpu_eqcct"),
]


def best_gpu_ripper_row_at_228(
    trials_root: Path,
    ripper_sweep_root: Path,
) -> dict[str, dict]:
    """model -> row dict of CSV row that achieved minimum Total Trial Time at 228 among successful trials."""
    winners: dict[str, tuple[float, dict]] = {}

    for subdir, model in TRIALS_GPU_RIPPER.items():
        p = trials_root / subdir / "gpu_test_results.csv"
        got = _scan_gpu_csv(p)
        if got is None:
            continue
        tt, row = got
        if model not in winners or tt < winners[model][0]:
            winners[model] = (tt, row)

    for model, subdir in RIPPER_SWEEP_GPU:
        p = ripper_sweep_root / subdir / "gpu_test_results.csv"
        got = _scan_gpu_csv(p)
        if got is None:
            continue
        tt, row = got
        if model not in winners or tt < winners[model][0]:
            winners[model] = (tt, row)

    return {m: row for m, (_tt, row) in winners.items()}


def row_to_table3_gpu_entry(model: str, row: dict) -> dict:
    cpus = int(float(row.get("Number of CPUs Allocated for Ray to Use") or 0))
    ng = _gpu_count(row.get("GPUs Used"))
    conc = int(
        float(
            row.get("Actual Ripper Concurrent Tasks")
            or row.get("Number of Concurrent Station Tasks")
            or 0
        )
    )
    tt = round(float(row.get("Total Trial Time (s)") or 0), 2)
    return {
        "Model": model,
        "Device": "GPU",
        "CPUs": cpus,
        "GPUs": ng,
        "Conc. Tasks": conc,
        "Ripper Picking/Total (s)": tt,
    }


def best_gpu_ripper_bar_value(
    trials_root: Path,
    ripper_sweep_root: Path,
) -> dict[str, tuple[float, int]]:
    """model -> (min_tt, workers) for Figure 5 GPU Ripper bars."""
    rows = best_gpu_ripper_row_at_228(trials_root, ripper_sweep_root)
    out: dict[str, tuple[float, int]] = {}
    for model, row in rows.items():
        tt = float(row.get("Total Trial Time (s)") or 0)
        actors = int(float(row.get("N ModelActors", 0) or 0))
        conc = int(
            float(
                row.get("Number of Concurrent Station Tasks", 0)
                or row.get("Actual Ripper Concurrent Tasks", 0)
                or 0
            )
        )
        workers = actors if actors > 0 else conc
        out[model] = (tt, workers)
    return out
