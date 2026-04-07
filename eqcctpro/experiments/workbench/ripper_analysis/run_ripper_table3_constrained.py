#!/usr/bin/env python3
"""
Run Ripper benchmarks for Table 3 (paper) with explicit CPU/GPU caps.
Table 3 rows are defined in benchmark_ripper_table3_rerun.TABLE3 (from trial CSVs;
see scripts/extract_ripper_table3_from_trials.py).

This script drives EQCCTPro the same way as benchmark_ripper_table3_rerun.py but
lets you choose *which* logical CPUs and CUDA devices to use so you do not claim
more cores or GPUs than the paper row specifies.

Resource model (matches EQCCTPro parallelization.py):
  - The Ray driver calls psutil.Process().cpu_affinity(cpu_ids), which uses the
    Linux sched_setaffinity syscall on the driver process.
  - The remote mseed_predictor task receives ``ray_cpus=cpu_ids`` and sets
    affinity again inside the worker (see parallelization.py), so the trial
    stays on that CPU set.
  - ray.init(num_cpus=len(cpu_ids), num_gpus=len(gpu_ids)) tells Ray the pool
    size; CUDA_VISIBLE_DEVICES is set to only the listed GPU indices for GPU rows.

Examples (from eqcctpro repo root, eqcctpro conda env):

  # CPU PhaseNet row (17 cores): pin to cores 0..16 (default)
  python scripts/run_ripper_table3_constrained.py --only 0 --runs 5 --strict-affinity

  # Same row but isolate to cores 20..36 (17 cores) on a shared machine
  python scripts/run_ripper_table3_constrained.py --only 0 --runs 5 \\
      --cpu-start 20 --strict-affinity

  # PhaseNet GPU row: 20 CPUs + 2 GPUs as PCI devices 0 and 1
  python scripts/run_ripper_table3_constrained.py --only 7 --runs 5 \\
      --cpu-start 0 --gpu-ids 0,1 --strict-affinity

  # Dry run: print resolved affinity and Ray/CUDA settings only
  python scripts/run_ripper_table3_constrained.py --only 0 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import psutil

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SCRIPTS))

import benchmark_ripper_table3_rerun as bt3  # noqa: E402


def _parse_int_list(spec: str, *, name: str) -> list[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"{name}: invalid integer {p!r}") from e
    return out


def resolve_cpu_ids(
    n_cpus: int,
    *,
    cpu_start: int | None,
    cpu_ids_arg: str | None,
) -> list[int]:
    if cpu_ids_arg is not None:
        ids = _parse_int_list(cpu_ids_arg, name="--cpu-ids")
        if len(ids) != n_cpus:
            raise ValueError(
                f"--cpu-ids must list exactly {n_cpus} cores for this Table 3 row; got {len(ids)}"
            )
        return ids
    start = 0 if cpu_start is None else cpu_start
    return list(range(start, start + n_cpus))


def resolve_gpu_ids(n_gpus: int, gpu_ids_arg: str | None) -> list[int]:
    if gpu_ids_arg is None:
        return list(range(n_gpus))
    ids = _parse_int_list(gpu_ids_arg, name="--gpu-ids")
    if len(ids) != n_gpus:
        raise ValueError(
            f"--gpu-ids must list exactly {n_gpus} devices for this Table 3 row; got {len(ids)}"
        )
    return ids


def validate_cpus_allowed(cpu_ids: list[int]) -> None:
    proc = psutil.Process()
    try:
        allowed = set(proc.cpu_affinity())
    except Exception as e:
        raise RuntimeError(f"Cannot read process CPU affinity (need Linux + permissions): {e}") from e
    bad = [c for c in cpu_ids if c not in allowed]
    if bad:
        raise ValueError(
            f"CPU ids {bad} are not in this process's schedulable set {sorted(allowed)}. "
            "Run inside a cpuset/cgroup that exposes those cores, or pick different ids."
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Table 3 Ripper runs with explicit CPU/GPU pinning (EQCCTPro)."
    )
    ap.add_argument(
        "--only",
        type=int,
        default=None,
        metavar="N",
        help="Run only TABLE3 row index N (0..9). Default: all rows.",
    )
    ap.add_argument("--runs", type=int, default=bt3.N_RUNS, help="Repeats per row.")
    ap.add_argument(
        "--cpu-start",
        type=int,
        default=None,
        metavar="K",
        help="Use logical CPUs K, K+1, ... (length = row's CPU count). Ignored if --cpu-ids set.",
    )
    ap.add_argument(
        "--cpu-ids",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated logical CPU ids; length must match row's CPU count.",
    )
    ap.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated CUDA device indices (as seen before hiding); "
        "length must match row's GPU count. CPU rows ignore this.",
    )
    ap.add_argument(
        "--strict-affinity",
        action="store_true",
        help="Fail if driver cpu_affinity cannot be applied (recommended).",
    )
    ap.add_argument(
        "--skip-allowed-check",
        action="store_true",
        help="Do not verify cpu_ids are in the current process affinity mask.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved cpu_ids, gpu_ids, conc, and exit without running Ray.",
    )
    args = ap.parse_args()
    n_runs = max(1, args.runs)

    rows = bt3.TABLE3 if args.only is None else [bt3.TABLE3[args.only]]
    if args.only is not None and not (0 <= args.only < len(bt3.TABLE3)):
        ap.error(f"--only must be 0..{len(bt3.TABLE3) - 1}")

    bt3.OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixed = bt3._fixed_stations()
    timechunk_path = str(bt3.DATASET / bt3.TIMECHUNK)

    for name, dev, n_cpus, n_gpus_table, conc, mtype, parent, child in rows:
        use_gpu = dev.upper() == "GPU"
        try:
            cpu_ids = resolve_cpu_ids(n_cpus, cpu_start=args.cpu_start, cpu_ids_arg=args.cpu_ids)
            gpu_ids = resolve_gpu_ids(n_gpus_table, args.gpu_ids) if use_gpu else []
        except ValueError as e:
            ap.error(str(e))

        if not args.skip_allowed_check:
            validate_cpus_allowed(cpu_ids)

        key = f"{name}_{dev}"
        print(
            f"\n{'=' * 60}\n{key}\n"
            f"  Table 3: CPUs={n_cpus} GPUs={n_gpus_table} conc_tasks={conc}\n"
            f"  Resolved cpu_ids={cpu_ids}\n"
            f"  Resolved gpu_ids={gpu_ids if use_gpu else []}\n"
            f"  ray.init num_cpus={len(cpu_ids)} num_gpus={len(gpu_ids)}\n"
            f"  EQCCTPro: mseed_predictor(..., ray_cpus=cpu_ids, ripper=True, ...)\n"
            f"{'=' * 60}"
        )

        if args.dry_run:
            continue

        totals, picks, actuals = [], [], []
        run_rows: list[dict] = []

        for run_i in range(n_runs):
            tmp = tempfile.mkdtemp(prefix="ray_ripper_", dir="/tmp")
            out = tempfile.mkdtemp(prefix="ripper_out_", dir="/tmp")
            trial_csv = str(bt3.OUT_DIR / f"ripper_table3_{key}_run{run_i + 1}.csv")
            if os.path.exists(trial_csv):
                os.remove(trial_csv)

            t0 = time.perf_counter()
            total_s, pick_s, actual_conc = bt3.run_one_ripper(
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
                strict_affinity=args.strict_affinity,
            )
            elapsed = time.perf_counter() - t0
            totals.append(total_s)
            picks.append(pick_s)
            actuals.append(actual_conc)
            print(
                f"  run {run_i + 1}/{n_runs}: total={total_s:.2f}s pick={pick_s:.2f}s "
                f"actual_conc={actual_conc} (wall {elapsed:.1f}s)"
            )
            run_rows.append(
                {
                    "model": name,
                    "device": dev,
                    "run": run_i + 1,
                    "cpu_ids": ",".join(map(str, cpu_ids)),
                    "gpu_ids": ",".join(map(str, gpu_ids)),
                    "total_trial_s": round(total_s, 3),
                    "picker_s": round(pick_s, 3),
                    "requested_conc": conc,
                    "actual_conc": actual_conc,
                }
            )

        summary_path = bt3.OUT_DIR / f"ripper_table3_constrained_{key}.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "n_runs": n_runs,
                    "n_stations": bt3.N_STATIONS,
                    "cpu_ids": cpu_ids,
                    "gpu_ids": gpu_ids,
                    "conc_tasks": conc,
                    "mean_total_s": round(statistics.mean(totals), 2),
                    "mean_picker_s": round(statistics.mean(picks), 2),
                    "stdev_total_s": round(statistics.pstdev(totals), 3) if len(totals) > 1 else 0.0,
                    "stdev_picker_s": round(statistics.pstdev(picks), 3) if len(picks) > 1 else 0.0,
                    "actual_conc_sample": actuals[-1],
                },
                f,
                indent=2,
            )

        csv_path = bt3.OUT_DIR / f"ripper_table3_constrained_{key}_runs.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
            w.writeheader()
            w.writerows(run_rows)

        print(f"Wrote {summary_path}\nWrote {csv_path}")

    if args.dry_run:
        print("\nDry run complete (no trials executed).")


if __name__ == "__main__":
    main()
