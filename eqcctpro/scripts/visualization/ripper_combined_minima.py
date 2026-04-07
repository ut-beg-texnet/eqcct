"""
Best successful Ripper total time at 228 stations over the combined log corpus:
  - GPU: results/trials eval_gpu_*_ripper + results/ripper_228_conc_sweep eval_sweep_ripper_gpu_*
  - CPU: results/ripper_228_conc_sweep eval_sweep_ripper_cpu_* (trial CPU Ripper is superseded when sweep exists)

Paper reporting:
  - Ripper: Ray CPUs in {5, 8, 11, 14, 17, 20} only; global minimum Total Trial Time at 228.
  - GPU Table 3: separate rows for the best 1-GPU and best 2-GPU run per model (when present).
  - Model-Actor (Table 4): same Ray CPU grid; one CPU row per model; GPU rows as separate
    1-GPU and 2-GPU minima when both exist; sorted by total time.
"""
from __future__ import annotations

import csv
from pathlib import Path

PAPER_RAY_CPU_GRID = frozenset(range(5, 21, 3))

RIPPER_CPU_SWEEP_SUBDIRS = {
    "PhaseNet": "eval_sweep_ripper_cpu_phasenet_original",
    "PhaseNetLight": "eval_sweep_ripper_cpu_phasenetlight_stead",
    "EQTransformer": "eval_sweep_ripper_cpu_eqtransformer_original",
    "EQT-NC": "eval_sweep_ripper_cpu_eqtransformer_nc",
    "EQCCT": "eval_sweep_ripper_cpu_eqcct",
}

MODEL_DIR_TO_LABEL = {
    "phasenet_original": "PhaseNet",
    "phasenetlight_stead": "PhaseNetLight",
    "eqtransformer_original": "EQTransformer",
    "eqtransformer_nonconservative": "EQT-NC",
    "eqcct": "EQCCT",
}


def trial_success(row: dict) -> bool:
    v = (row.get("Trial Success") or "").strip().lower()
    return v in ("1", "true", "yes", "1.0")


def paper_ripper_ray_cpus_ok(row: dict) -> bool:
    try:
        c = int(float(row.get("Number of CPUs Allocated for Ray to Use") or -1))
    except (TypeError, ValueError):
        return False
    return c in PAPER_RAY_CPU_GRID


def paper_ma_ray_cpus_ok(row: dict) -> bool:
    """Model-Actor paper rows: same Ray CPU grid as Ripper (5–20), global min at 228 within that grid."""
    return paper_ripper_ray_cpus_ok(row)


def workers_from_row(row: dict) -> int:
    actors = int(float(row.get("N ModelActors", 0) or 0))
    conc = int(
        float(
            row.get("Number of Concurrent Station Tasks", 0)
            or row.get("Actual Ripper Concurrent Tasks", 0)
            or 0
        )
    )
    return actors if actors > 0 else conc


def min_total_time_at_228_from_csv(path: Path, *, ripper: bool) -> tuple[float, int] | None:
    """Minimum Total Trial Time at 228 stations."""
    if not path.is_file():
        return None
    best: tuple[float, int] | None = None
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                n = int(float(row.get("Number of Stations Used") or 0))
            except (ValueError, TypeError):
                continue
            if n != 228 or not trial_success(row):
                continue
            if ripper:
                if not paper_ripper_ray_cpus_ok(row):
                    continue
            elif not paper_ma_ray_cpus_ok(row):
                continue
            try:
                tt = float(row.get("Total Trial Time (s)") or 1e9)
            except (ValueError, TypeError):
                continue
            w = workers_from_row(row)
            if best is None or tt < best[0]:
                best = (tt, w)
    return best


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


def _ripper_row_ok_at_228(row: dict) -> bool:
    try:
        n = int(float(row.get("Number of Stations Used") or 0))
    except (ValueError, TypeError):
        return False
    return n == 228 and trial_success(row) and paper_ripper_ray_cpus_ok(row)


def _best_ripper_row_from_csv(path: Path) -> tuple[float, dict] | None:
    """Best Ripper row at 228 (any GPU count in row)."""
    if not path.is_file():
        return None
    best: tuple[float, dict] | None = None
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if not _ripper_row_ok_at_228(row):
                continue
            try:
                tt = float(row.get("Total Trial Time (s)") or 1e9)
            except (ValueError, TypeError):
                continue
            if best is None or tt < best[0]:
                best = (tt, row)
    return best


def _consider_gpu_row(
    winners: dict[str, dict[int, tuple[float, dict]]],
    model: str,
    row: dict,
) -> None:
    if not _ripper_row_ok_at_228(row):
        return
    ng = _gpu_count(row.get("GPUs Used"))
    if ng not in (1, 2):
        return
    try:
        tt = float(row.get("Total Trial Time (s)") or 1e9)
    except (ValueError, TypeError):
        return
    mwin = winners.setdefault(model, {})
    if ng not in mwin or tt < mwin[ng][0]:
        mwin[ng] = (tt, row)


def best_gpu_ripper_by_ngpus_at_228(
    trials_root: Path,
    ripper_sweep_root: Path,
) -> dict[str, dict[int, dict]]:
    """model -> {1: row, 2: row} best Ripper CSV row per GPU count (228 stn, paper CPUs)."""
    winners: dict[str, dict[int, tuple[float, dict]]] = {}

    for subdir, model in TRIALS_GPU_RIPPER.items():
        p = trials_root / subdir / "gpu_test_results.csv"
        if not p.is_file():
            continue
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                _consider_gpu_row(winners, model, row)

    for model, subdir in RIPPER_SWEEP_GPU:
        p = ripper_sweep_root / subdir / "gpu_test_results.csv"
        if not p.is_file():
            continue
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                _consider_gpu_row(winners, model, row)

    return {m: {k: v[1] for k, v in d.items()} for m, d in winners.items()}


TRIALS_GPU_RIPPER = {
    "eval_gpu_phasenet_original_ripper": "PhaseNet",
    "eval_gpu_phasenetlight_stead_ripper": "PhaseNetLight",
    "eval_gpu_eqtransformer_original_ripper": "EQTransformer",
    "eval_gpu_eqtransformer_nonconservative_ripper": "EQTransformer-NC",
    "eval_gpu_eqcct_ripper": "EQCCT",
}

TRIALS_CPU_RIPPER = {
    "eval_cpu_phasenet_original_ripper": "PhaseNet",
    "eval_cpu_phasenetlight_stead_ripper": "PhaseNetLight",
    "eval_cpu_eqtransformer_original_ripper": "EQTransformer",
    "eval_cpu_eqtransformer_nonconservative_ripper": "EQTransformer-NC",
    "eval_cpu_eqcct_ripper": "EQCCT",
}


def best_cpu_ripper_row_at_228(
    trials_root: Path,
    ripper_cpu_root: Path,
) -> dict[str, dict]:
    """Best CPU Ripper row per model (228 stn, Ray CPUs in paper grid)."""
    winners: dict[str, tuple[float, dict]] = {}

    for subdir, model in TRIALS_CPU_RIPPER.items():
        p = trials_root / subdir / "cpu_test_results.csv"
        got = _best_ripper_row_from_csv(p)
        if got is None:
            continue
        tt, row = got
        if model not in winners or tt < winners[model][0]:
            winners[model] = (tt, row)

    for model, subdir in RIPPER_CPU_SWEEP_SUBDIRS.items():
        p = ripper_cpu_root / subdir / "cpu_test_results.csv"
        got = _best_ripper_row_from_csv(p)
        if got is None:
            continue
        tt, row = got
        if model not in winners or tt < winners[model][0]:
            winners[model] = (tt, row)

    return {m: row for m, (_tt, row) in winners.items()}


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
    """Per model, the faster of the best 1-GPU and best 2-GPU Ripper rows (228 stn, paper CPUs)."""
    byn = best_gpu_ripper_by_ngpus_at_228(trials_root, ripper_sweep_root)
    out: dict[str, dict] = {}
    for model, d in byn.items():
        best_tt: float | None = None
        best_row: dict | None = None
        for _ng, row in sorted(d.items()):
            tt = float(row.get("Total Trial Time (s)") or 1e9)
            if best_tt is None or tt < best_tt:
                best_tt = tt
                best_row = row
        if best_row is not None:
            out[model] = best_row
    return out


def row_to_table3_cpu_entry(model: str, row: dict) -> dict:
    cpus = int(float(row.get("Number of CPUs Allocated for Ray to Use") or 0))
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
        "Device": "CPU",
        "CPUs": cpus,
        "GPUs": 0,
        "Conc. Tasks": conc,
        "Ripper Picking/Total (s)": tt,
    }


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
    """model -> (min_tt, workers) for Figure 5 GPU Ripper bars (faster of 1- vs 2-GPU best)."""
    rows = best_gpu_ripper_row_at_228(trials_root, ripper_sweep_root)
    out: dict[str, tuple[float, int]] = {}
    for model, row in rows.items():
        tt = float(row.get("Total Trial Time (s)") or 0)
        out[model] = (tt, workers_from_row(row))
    return out


def paper_runtime_raw_dict(
    trials_root: Path,
    ripper_cpu_root: Path,
) -> dict[tuple[str, str, str], tuple[float, int]]:
    """(model, CPU|GPU, Ripper|MA) -> (total_trial_time_s, workers)."""
    raw: dict[tuple[str, str, str], tuple[float, int]] = {}

    for d in trials_root.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith("eval_cpu_"):
            hw, frag = "CPU", name[len("eval_cpu_") :]
        elif name.startswith("eval_gpu_"):
            hw, frag = "GPU", name[len("eval_gpu_") :]
        else:
            continue
        if frag.endswith("_modelactor"):
            orch, mfrag = "MA", frag[: -len("_modelactor")]
        elif frag.endswith("_ripper"):
            orch, mfrag = "Ripper", frag[: -len("_ripper")]
        else:
            continue
        if orch == "Ripper" and hw == "CPU":
            continue
        model = MODEL_DIR_TO_LABEL.get(mfrag)
        if model is None:
            continue
        suffix = "cpu" if hw == "CPU" else "gpu"
        csv_path = d / f"{suffix}_test_results.csv"
        val = min_total_time_at_228_from_csv(csv_path, ripper=(orch == "Ripper"))
        if val is not None:
            raw[(model, hw, orch)] = val

    for model, row in best_cpu_ripper_row_at_228(trials_root, ripper_cpu_root).items():
        tt = float(row["Total Trial Time (s)"])
        mkey = "EQT-NC" if model == "EQTransformer-NC" else model
        raw[(mkey, "CPU", "Ripper")] = (tt, workers_from_row(row))

    gpu_combined = best_gpu_ripper_bar_value(trials_root, ripper_cpu_root)

    def _fig5_gpu_model(n: str) -> str:
        return "EQT-NC" if n == "EQTransformer-NC" else n

    for m, (tt, w) in gpu_combined.items():
        key = _fig5_gpu_model(m)
        cur = raw.get((key, "GPU", "Ripper"))
        if cur is None or tt < cur[0]:
            raw[(key, "GPU", "Ripper")] = (tt, w)

    return raw


def table3_ripper_rows_sorted(
    trials_root: Path,
    ripper_cpu_root: Path,
) -> list[dict]:
    """Table 3: CPU row per model + GPU rows for best 1-GPU and 2-GPU (each model), sorted by time."""
    rows: list[dict] = []

    cpu_models = [
        "PhaseNet",
        "PhaseNetLight",
        "EQTransformer",
        "EQTransformer-NC",
        "EQCCT",
    ]
    cpu_winners = best_cpu_ripper_row_at_228(trials_root, ripper_cpu_root)
    for model in cpu_models:
        row = cpu_winners.get(model)
        if row is None:
            continue
        rows.append(row_to_table3_cpu_entry(model, row))

    gpu_by_n = best_gpu_ripper_by_ngpus_at_228(trials_root, ripper_cpu_root)
    for model in cpu_models:
        d = gpu_by_n.get(model, {})
        for ng in (1, 2):
            r = d.get(ng)
            if r is None:
                continue
            rows.append(row_to_table3_gpu_entry(model, r))

    rows.sort(key=lambda r: r["Ripper Picking/Total (s)"])
    return rows


def _ma_228_paper_rows_from_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                n = int(float(row.get("Number of Stations Used") or 0))
            except (ValueError, TypeError):
                continue
            if n != 228 or not trial_success(row) or not paper_ma_ray_cpus_ok(row):
                continue
            out.append(row)
    return out


def _ma_dir_candidates(d: Path, suffix: str) -> list[dict]:
    return _ma_228_paper_rows_from_csv(
        d / f"optimal_configurations_{suffix}.csv"
    ) + _ma_228_paper_rows_from_csv(d / f"{suffix}_test_results.csv")


def _best_ma_cpu_row_at_228_dir(trial_dir: Path) -> dict | None:
    candidates = _ma_dir_candidates(trial_dir, "cpu")
    best: tuple[float, dict] | None = None
    for row in candidates:
        if _gpu_count(row.get("GPUs Used")) != 0:
            continue
        try:
            tt = float(row.get("Total Trial Time (s)") or 1e9)
        except (ValueError, TypeError):
            continue
        if best is None or tt < best[0]:
            best = (tt, row)
    return best[1] if best else None


def _best_ma_gpu_by_ngpus_at_228_dir(trial_dir: Path) -> dict[int, dict]:
    candidates = _ma_dir_candidates(trial_dir, "gpu")
    winners: dict[int, tuple[float, dict]] = {}
    for row in candidates:
        ng = _gpu_count(row.get("GPUs Used"))
        if ng not in (1, 2):
            continue
        try:
            tt = float(row.get("Total Trial Time (s)") or 1e9)
        except (ValueError, TypeError):
            continue
        if ng not in winners or tt < winners[ng][0]:
            winners[ng] = (tt, row)
    return {k: v[1] for k, v in winners.items()}


def _row_to_table4_display(model: str, row: dict) -> dict:
    cpus = int(float(row.get("Number of CPUs Allocated for Ray to Use") or 0))
    ng = _gpu_count(row.get("GPUs Used"))
    hw = "GPU" if ng else "CPU"
    actors = int(float(row.get("N ModelActors", 0) or 0))
    setup = float(row.get("Actor Creation Time (s)") or 0)
    pick = float(row.get("Total Run time for Picker (s)") or 0)
    tot = float(row.get("Total Trial Time (s)") or 0)
    pct = (setup / tot * 100) if tot > 0 else 0.0
    return {
        "Model": model,
        "Device": hw,
        "CPUs": cpus,
        "GPUs": ng,
        "Actors": actors,
        "Setup (s)": round(setup, 2),
        "Pick (s)": round(pick, 2),
        "Total (s)": round(tot, 2),
        "Setup OH (%)": round(pct, 1),
    }


def table4_ma_rows_sorted(trials_root: Path) -> list[dict]:
    """Table 4: CPU row per model + best 1-GPU and 2-GPU MA rows per model, sorted by total time."""
    acc: list[tuple[dict, float]] = []
    for d in sorted(trials_root.iterdir()):
        if not d.is_dir() or not d.name.endswith("_modelactor"):
            continue
        if d.name.startswith("eval_cpu_"):
            mfrag = d.name[len("eval_cpu_") : -len("_modelactor")]
            model = MODEL_DIR_TO_LABEL.get(mfrag)
            if model is None:
                continue
            raw = _best_ma_cpu_row_at_228_dir(d)
            if raw is None:
                continue
            disp = _row_to_table4_display(model, raw)
            acc.append((disp, disp["Total (s)"]))
        elif d.name.startswith("eval_gpu_"):
            mfrag = d.name[len("eval_gpu_") : -len("_modelactor")]
            model = MODEL_DIR_TO_LABEL.get(mfrag)
            if model is None:
                continue
            for _ng in (1, 2):
                raw = _best_ma_gpu_by_ngpus_at_228_dir(d).get(_ng)
                if raw is None:
                    continue
                disp = _row_to_table4_display(model, raw)
                acc.append((disp, disp["Total (s)"]))
        else:
            continue
    acc.sort(key=lambda x: x[1])
    return [d for d, _ in acc]


def table4_ma_entries_with_raw(trials_root: Path) -> list[tuple[dict, dict]]:
    """(display row, source CSV row) in same order as table4_ma_rows_sorted (for Table 5 memory)."""
    triples: list[tuple[dict, dict, float]] = []
    for d in sorted(trials_root.iterdir()):
        if not d.is_dir() or not d.name.endswith("_modelactor"):
            continue
        if d.name.startswith("eval_cpu_"):
            mfrag = d.name[len("eval_cpu_") : -len("_modelactor")]
            model = MODEL_DIR_TO_LABEL.get(mfrag)
            if model is None:
                continue
            raw = _best_ma_cpu_row_at_228_dir(d)
            if raw is None:
                continue
            disp = _row_to_table4_display(model, raw)
            triples.append((disp, raw, disp["Total (s)"]))
        elif d.name.startswith("eval_gpu_"):
            mfrag = d.name[len("eval_gpu_") : -len("_modelactor")]
            model = MODEL_DIR_TO_LABEL.get(mfrag)
            if model is None:
                continue
            for _ng in (1, 2):
                raw = _best_ma_gpu_by_ngpus_at_228_dir(d).get(_ng)
                if raw is None:
                    continue
                disp = _row_to_table4_display(model, raw)
                triples.append((disp, raw, disp["Total (s)"]))
        else:
            continue
    triples.sort(key=lambda x: x[2])
    return [(a, b) for a, b, _ in triples]
