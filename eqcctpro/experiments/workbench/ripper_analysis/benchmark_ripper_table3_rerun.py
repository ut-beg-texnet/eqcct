#!/usr/bin/env python3
"""
Re-run Ripper trials using Table 3 hardware + concurrency (paper), 228 stations,
5 repeats per configuration. Records mean Total Trial Time and mean Picker runtime.

Table 3 rows match the fastest successful 228-station trials in
results/trials/eval_*_ripper/cpu_test_results.csv and gpu_test_results.csv
(minimum Total Trial Time (s)); regenerate with:
  python scripts/extract_ripper_table3_from_trials.py

Outputs:
  results/benchmark_results/ripper_table3_rerun.json
  results/benchmark_results/ripper_table3_rerun_runs.csv
"""
from __future__ import annotations

import csv
import json
import logging
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import psutil
import ray
from ray.util.queue import Queue

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from eqcctpro.parallelization import (  # noqa: E402
    get_eqcct_ram_mb,
    get_eqcct_vram_mb,
    get_seisbench_model_ram_mb,
    get_seisbench_model_vram_mb,
    mseed_predictor,
)
from eqcctpro.tools import build_station_list_from_dir  # noqa: E402

# --- Paths (match experiments/main/run.py) ---
DATASET = PROJECT / "data" / "230_stations_1_min_dt"
TIMECHUNK = "20241215T120000Z_20241215T120100Z"
MODELS_DIR = PROJECT / "models" / "EQCCT"
P_MODEL = str(MODELS_DIR / "test_trainer_024.h5")
S_MODEL = str(MODELS_DIR / "test_trainer_021.h5")
OUT_DIR = PROJECT / "results" / "benchmark_results"
N_STATIONS = 228
N_RUNS = 5
VRAM_PER_GPU_MB = 46550.0
INTRA = 1
INTER = 1
START_TIME = "2024-12-15 12:00:00"
END_TIME = "2024-12-15 12:01:00"

# Table 3: (paper_name, device, n_cpus, n_gpus, conc_tasks, model_type, parent, child)
# conc_tasks = Actual Ripper Concurrent Tasks in the winning trial (see extract script).
TABLE3 = [
    ("PhaseNet", "CPU", 20, 0, 150, "seisbench", "PhaseNet", "original"),
    ("PhaseNetLight", "CPU", 20, 0, 45, "seisbench", "PhaseNetLight", "stead"),
    ("EQTransformer", "CPU", 20, 0, 146, "seisbench", "EQTransformer", "original"),
    ("EQTransformer-NC", "CPU", 20, 0, 90, "seisbench", "EQTransformer", "original_nonconservative"),
    ("PhaseNetLight", "GPU", 20, 1, 22, "seisbench", "PhaseNetLight", "stead"),
    ("EQTransformer", "GPU", 20, 1, 22, "seisbench", "EQTransformer", "original"),
    ("EQTransformer-NC", "GPU", 20, 1, 22, "seisbench", "EQTransformer", "original_nonconservative"),
    ("PhaseNet", "GPU", 20, 2, 44, "seisbench", "PhaseNet", "original"),
    ("EQCCT", "CPU", 20, 0, 45, "eqcct", None, None),
    ("EQCCT", "GPU", 20, 2, 24, "eqcct", None, None),
]


def _fixed_stations() -> list[str]:
    tc = DATASET / TIMECHUNK
    if not tc.is_dir():
        raise FileNotFoundError(f"Missing timechunk directory: {tc}")
    all_sta = sorted(build_station_list_from_dir(str(tc)))
    if len(all_sta) < N_STATIONS:
        raise RuntimeError(f"Need at least {N_STATIONS} stations, found {len(all_sta)}")
    return all_sta[:N_STATIONS]


def _model_vram_mb(model_type: str, parent: str | None, child: str | None) -> float:
    if model_type == "eqcct":
        return float(get_eqcct_vram_mb())
    return float(
        get_seisbench_model_vram_mb(parent, child, default_mb=600.0, logger=None)
    )


def _model_ram_mb_gpu(model_type: str, parent: str | None, child: str | None) -> float:
    if model_type == "eqcct":
        return float(get_eqcct_ram_mb(use_gpu=True))
    return float(
        get_seisbench_model_ram_mb(
            parent, child, use_gpu=True, default_mb=1000.0, logger=None
        )
    )


def run_one_ripper(
    *,
    use_gpu: bool,
    cpu_ids: list[int],
    gpu_ids: list[int],
    conc: int,
    model_type: str,
    parent: str | None,
    child: str | None,
    timechunk_path: str,
    fixed_station_list: list[str],
    trial_csv: str,
    output_dir: str,
    tmp_dir: str,
    strict_affinity: bool = False,
) -> tuple[float, float, int]:
    """Returns (total_trial_s, picker_s, actual_ripper_conc)."""
    total_analysis_time = datetime.strptime(END_TIME, "%Y-%m-%d %H:%M:%S") - datetime.strptime(
        START_TIME, "%Y-%m-%d %H:%M:%S"
    )
    n_gpus = len(gpu_ids)
    vram_mb_total = VRAM_PER_GPU_MB * max(1, n_gpus)
    gpu_vram_per_actor = _model_vram_mb(model_type, parent, child)

    workspace_root = str(PROJECT)
    runtime_env = {
        "env_vars": {"PYTHONPATH": f"{workspace_root}:{os.environ.get('PYTHONPATH', '')}"}
    }

    if use_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        ray.init(
            ignore_reinit_error=True,
            num_gpus=n_gpus,
            num_cpus=len(cpu_ids),
            logging_level=logging.FATAL,
            log_to_driver=False,
            _temp_dir=tmp_dir,
            runtime_env=runtime_env,
        )
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        ray.init(
            ignore_reinit_error=True,
            num_cpus=len(cpu_ids),
            num_gpus=0,
            logging_level=logging.FATAL,
            log_to_driver=False,
            _temp_dir=tmp_dir,
            runtime_env=runtime_env,
        )

    log_queue = Queue()
    proc = psutil.Process()
    try:
        proc.cpu_affinity(cpu_ids)
    except Exception as e:
        if strict_affinity:
            raise RuntimeError(
                f"Driver process cpu_affinity({cpu_ids}) failed: {e}. "
                "Use --cpu-ids/--cpu-start that your OS allows for this user."
            ) from e

    ref = mseed_predictor.options(num_gpus=0, num_cpus=1).remote(
        input_dir=timechunk_path,
        output_dir=output_dir,
        log_queue=log_queue,
        P_threshold=0.001 if model_type == "eqcct" else 0.3,
        S_threshold=0.02 if model_type == "eqcct" else 0.3,
        p_model=P_MODEL,
        s_model=S_MODEL,
        number_of_concurrent_station_predictions=conc,
        ray_cpus=cpu_ids,
        use_gpu=use_gpu,
        gpu_id=gpu_ids if use_gpu else None,
        gpu_memory_limit_mb=gpu_vram_per_actor if use_gpu else None,
        total_vram_pool_mb=vram_mb_total if use_gpu else None,
        stations2use=None,
        fixed_station_list=fixed_station_list,
        timechunk_id=TIMECHUNK,
        waveform_overlap=0,
        total_timechunks=1,
        number_of_concurrent_timechunk_predictions=1,
        total_analysis_time=total_analysis_time,
        testing_gpu=True,
        test_csv_filepath=trial_csv,
        intra_threads=INTRA,
        inter_threads=INTER,
        timechunk_dt=1,
        model_type=model_type,
        seisbench_parent_model=parent,
        seisbench_child_model=child,
        Detection_threshold=0.3,
        ram_safety_cap=0.95,
        cudnn_headroom=0.25,
        ripper=True,
    )
    ray.get(ref)
    ray.shutdown()
    time.sleep(1.0)

    with open(trial_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No CSV rows written: {trial_csv}")
    last = rows[-1]
    total = float(last["Total Trial Time (s)"])
    pick = float(last["Total Run time for Picker (s)"])
    actual = int(float(last.get("Actual Ripper Concurrent Tasks") or last.get("Number of Concurrent Station Tasks") or 0))
    return total, pick, actual


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Re-run Ripper Table 3 configs (5x each).")
    ap.add_argument("--only", type=int, default=None, help="Run only TABLE3 row index (0-based).")
    ap.add_argument("--runs", type=int, default=N_RUNS, help="Repeats per configuration.")
    args = ap.parse_args()
    n_runs = max(1, args.runs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixed = _fixed_stations()
    timechunk_path = str(DATASET / TIMECHUNK)
    summary = []
    run_rows = []

    rows_to_run = TABLE3
    if args.only is not None:
        rows_to_run = [TABLE3[args.only]]

    for name, dev, n_cpus, n_gpus_table, conc, mtype, parent, child in rows_to_run:
        use_gpu = dev.upper() == "GPU"
        cpu_ids = list(range(n_cpus))
        gpu_ids = list(range(n_gpus_table)) if use_gpu else []
        key = f"{name}_{dev}"

        totals, picks, actuals = [], [], []
        print(f"\n{'='*60}\n{key}: CPUs={n_cpus} GPUs={n_gpus_table} conc={conc}\n{'='*60}")

        for run_i in range(n_runs):
            tmp = tempfile.mkdtemp(prefix="ray_ripper_", dir="/tmp")
            out = tempfile.mkdtemp(prefix="ripper_out_", dir="/tmp")
            trial_csv = str(OUT_DIR / f"ripper_table3_{key}_run{run_i+1}.csv")
            if os.path.exists(trial_csv):
                os.remove(trial_csv)

            t0 = time.perf_counter()
            try:
                total_s, pick_s, actual_conc = run_one_ripper(
                    use_gpu=use_gpu,
                    cpu_ids=cpu_ids,
                    gpu_ids=gpu_ids,
                    conc=conc,
                    model_type=mtype,
                    parent=parent,
                    child=child,
                    timechunk_path=timechunk_path,
                    fixed_station_list=fixed,
                    trial_csv=trial_csv,
                    output_dir=out,
                    tmp_dir=tmp,
                )
            except Exception as e:
                print(f"  run {run_i+1} FAILED: {e}")
                raise
            elapsed = time.perf_counter() - t0
            totals.append(total_s)
            picks.append(pick_s)
            actuals.append(actual_conc)
            print(
                f"  run {run_i+1}/{n_runs}: total={total_s:.2f}s pick={pick_s:.2f}s "
                f"actual_conc={actual_conc} (wall {elapsed:.1f}s)"
            )
            run_rows.append(
                {
                    "model": name,
                    "device": dev,
                    "run": run_i + 1,
                    "total_trial_s": round(total_s, 3),
                    "picker_s": round(pick_s, 3),
                    "requested_conc": conc,
                    "actual_conc": actual_conc,
                }
            )

        summary.append(
            {
                "model": name,
                "device": dev,
                "cpus": n_cpus,
                "gpus": n_gpus_table,
                "conc_tasks": conc,
                "mean_total_s": round(statistics.mean(totals), 2),
                "mean_picker_s": round(statistics.mean(picks), 2),
                "stdev_total_s": round(statistics.pstdev(totals), 3) if len(totals) > 1 else 0.0,
                "stdev_picker_s": round(statistics.pstdev(picks), 3) if len(picks) > 1 else 0.0,
                "actual_conc_sample": actuals[-1],
            }
        )

    json_path = OUT_DIR / "ripper_table3_rerun.json"
    with open(json_path, "w") as f:
        json.dump({"n_runs": n_runs, "n_stations": N_STATIONS, "results": summary}, f, indent=2)

    csv_path = OUT_DIR / "ripper_table3_rerun_runs.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        w.writeheader()
        w.writerows(run_rows)

    print(f"\nWrote {json_path}\nWrote {csv_path}")


if __name__ == "__main__":
    main()
