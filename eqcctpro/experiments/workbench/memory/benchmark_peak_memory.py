#!/usr/bin/env python3
"""
Benchmark peak memory when N model instances are loaded simultaneously.

For each (model, hardware, strategy) combination matching the paper trial
selection at 228 stations (Ripper and Model-Actor: Ray CPUs 5–20 only; global
minimum total time within that grid), this script:
  1. Starts Ray with the same CPU/GPU constraints as the trial
  2. Creates N Ray actors (Ripper = concurrent workers, Model-Actor = actors),
     each loading the model — no picking workload; load-only
  3. Waits until every instance reports ready (all models resident)
  4. Keeps all actors alive and samples process-tree RAM + GPU VRAM on an
     interval; records the maximum observed (peak)
  5. Shuts down Ray and writes results

Re-run with conda + python3:
  conda run -n eqcctpro python3 experiments/workbench/memory/benchmark_peak_memory.py

Output: results/benchmark_results/peak_memory_measured.json
"""
import argparse
import os
import sys
import json
import time
import csv
import gc
import psutil
from pathlib import Path

DEFAULT_HOLD_SECONDS = 25.0
DEFAULT_SAMPLE_INTERVAL = 1.25

# .../eqcctpro/experiments/workbench/memory/benchmark_peak_memory.py -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "visualization"))
from ripper_combined_minima import paper_ma_ray_cpus_ok, paper_ripper_ray_cpus_ok  # noqa: E402

EQCCT_P = "/home/skevofilaxc/model/ModelPS/test_trainer_024.h5"
EQCCT_S = "/home/skevofilaxc/model/ModelPS/test_trainer_021.h5"

TRIAL_BASE = REPO_ROOT / "results" / "trials"
MODEL_MAP = {
    "phasenet_original":             ("PhaseNet",      "PhaseNet",      "original"),
    "phasenetlight_stead":           ("PhaseNetLight", "PhaseNetLight", "stead"),
    "eqtransformer_original":        ("EQTransformer", "EQTransformer", "original"),
    "eqtransformer_nonconservative": ("EQT-NC",        "EQTransformer", "original_nonconservative"),
    "eqcct":                         ("EQCCT",         None,            None),
}

def get_optimal_configs():
    configs = {}
    for hw in ["cpu", "gpu"]:
        HW = hw.upper()
        for mfrag, (display, sb_parent, sb_child) in MODEL_MAP.items():
            for orch_suffix, orch_key in [("ripper", "Ripper"), ("modelactor", "MA")]:
                csv_file = TRIAL_BASE / f"eval_{hw}_{mfrag}_{orch_suffix}" / f"{hw}_test_results.csv"
                if not csv_file.exists():
                    continue
                best_tt, best = None, None
                with open(csv_file) as f:
                    for row in csv.DictReader(f):
                        n = int(float(row["Number of Stations Used"]))
                        if n != 228:
                            continue
                        v = (row.get("Trial Success") or "").strip().lower()
                        if v not in ("1", "true", "yes", "1.0"):
                            continue
                        if orch_key == "Ripper":
                            if not paper_ripper_ray_cpus_ok(row):
                                continue
                        else:
                            if not paper_ma_ray_cpus_ok(row):
                                continue
                        tt = float(row.get("Total Trial Time (s)", 0) or 0)
                        if best_tt is None or tt < best_tt:
                            best_tt = tt
                            conc = int(float(row.get("Actual Ripper Concurrent Tasks", 0) or
                                             row.get("Number of Concurrent Station Tasks", 0) or 0))
                            actors = int(float(row.get("N ModelActors", 0) or 0))
                            cpus = int(float(row.get("Number of CPUs Allocated for Ray to Use", 0) or 0))
                            gpu_str = str(row.get("GPUs Used", "")).strip("[] ")
                            gpus = len(gpu_str.split(",")) if gpu_str else 0
                            ram_per = float(row.get("Requested RAM per Actor (MB)", 0) or 0)
                            vram_per = float(row.get("Requested VRAM per Actor (MB)", 0) or 0)
                            best = {
                                "count": actors if orch_key == "MA" else conc,
                                "cpus": cpus,
                                "gpus": gpus,
                                "ram_per": ram_per,
                                "vram_per": vram_per,
                            }
                if best:
                    configs[(display, HW, orch_key)] = best
    return configs


def get_process_tree_ram_mb():
    proc = psutil.Process(os.getpid())
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / 1e6


def get_vram_mb(gpu_indices):
    if not gpu_indices:
        return 0.0
    try:
        import pynvml
        pynvml.nvmlInit()
        pid = os.getpid()
        all_pids = {pid}
        proc = psutil.Process(pid)
        for child in proc.children(recursive=True):
            all_pids.add(child.pid)

        total_vram = 0.0
        for gi in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gi)
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if p.pid in all_pids:
                    total_vram += p.usedGpuMemory
        pynvml.nvmlShutdown()
        return total_vram / 1e6
    except Exception as e:
        print(f"  VRAM measurement error: {e}")
        return 0.0


def measure_loaded_memory(
    display_name,
    sb_parent,
    sb_child,
    hw,
    n_instances,
    num_cpus,
    gpu_indices,
    ram_per_mb,
    vram_per_mb,
    hold_seconds: float,
    sample_interval: float,
):
    import ray

    use_gpu = hw == "GPU"
    is_eqcct = sb_parent is None

    print(f"\n{'='*60}")
    print(f"  {display_name} {hw} — loading {n_instances} instances")
    print(f"  CPUs={num_cpus}, GPUs={gpu_indices}")
    print(f"{'='*60}")

    ram_before = get_process_tree_ram_mb()

    ray_gpu_count = len(gpu_indices) if use_gpu else 0
    ray.init(
        num_cpus=num_cpus,
        num_gpus=ray_gpu_count,
        include_dashboard=False,
        log_to_driver=False,
    )

    if is_eqcct:
        @ray.remote
        class MemTestActor:
            def __init__(self, gpu_id, use_gpu):
                if not use_gpu:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                import tensorflow as tf
                if use_gpu:
                    gpus = tf.config.list_physical_devices("GPU")
                    if gpus:
                        tf.config.set_visible_devices(gpus[0], "GPU")
                        tf.config.experimental.set_memory_growth(gpus[0], True)
                from eqcctpro.eqcct_tf_models import load_eqcct_model
                self.model = load_eqcct_model(EQCCT_P, EQCCT_S)

            def ready(self):
                return True
    else:
        @ray.remote
        class MemTestActor:
            def __init__(self, parent, child, gpu_id, use_gpu):
                if not use_gpu:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                import seisbench.models as sbm
                import torch
                model_class = getattr(sbm, parent)
                self.model = model_class.from_pretrained(child)
                if use_gpu:
                    self.model.to(torch.device("cuda:0"))

            def ready(self):
                return True

    gpu_frac = (ray_gpu_count / n_instances) if (use_gpu and n_instances > 0) else 0

    actors = []
    for i in range(n_instances):
        if is_eqcct:
            actor_cls = MemTestActor.options(
                num_cpus=0.01,
                num_gpus=gpu_frac if use_gpu else 0,
            )
            actors.append(actor_cls.remote(gpu_indices[0] if gpu_indices else -1, use_gpu))
        else:
            actor_cls = MemTestActor.options(
                num_cpus=0.01,
                num_gpus=gpu_frac if use_gpu else 0,
            )
            actors.append(actor_cls.remote(sb_parent, sb_child,
                                           gpu_indices[0] if gpu_indices else -1, use_gpu))

    print(f"  Waiting for {n_instances} actors to load...")
    ready_refs = [a.ready.remote() for a in actors]
    ray.get(ready_refs)
    print(
        f"  All {n_instances} workers loaded; holding {hold_seconds:.0f}s "
        f"(sample every {sample_interval:.1f}s) for peak RAM/VRAM..."
    )

    deadline = time.monotonic() + hold_seconds
    peak_ram = 0.0
    peak_vram = 0.0
    n_samples = 0
    while time.monotonic() < deadline:
        ram_s = get_process_tree_ram_mb()
        vram_s = get_vram_mb(gpu_indices) if use_gpu else 0.0
        peak_ram = max(peak_ram, ram_s)
        peak_vram = max(peak_vram, vram_s)
        n_samples += 1
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(sample_interval, max(0.05, remaining)))

    budget_ram = n_instances * ram_per_mb / 1024
    budget_vram = n_instances * vram_per_mb / 1024

    print(f"  Peak process-tree RAM (max over {n_samples} samples):  {peak_ram:.0f} MB  ({peak_ram/1024:.1f} GB)")
    print(f"  Peak process-tree VRAM: {peak_vram:.0f} MB  ({peak_vram/1024:.1f} GB)")
    print(f"  Budget cap RAM:    {budget_ram:.1f} GB")
    print(f"  Budget cap VRAM:   {budget_vram:.1f} GB")

    del actors
    ray.shutdown()
    gc.collect()
    time.sleep(2)

    return {
        "tree_ram_mb": round(peak_ram, 1),
        "tree_vram_mb": round(peak_vram, 1),
        "budget_ram_gb": round(budget_ram, 1),
        "budget_vram_gb": round(budget_vram, 1),
        "n_instances": n_instances,
        "hold_seconds": hold_seconds,
        "sample_interval_s": sample_interval,
        "n_peak_samples": n_samples,
    }


def main():
    ap = argparse.ArgumentParser(description="Peak RAM/VRAM with all model workers loaded and held.")
    ap.add_argument(
        "--hold-seconds",
        type=float,
        default=DEFAULT_HOLD_SECONDS,
        help=f"Keep all actors loaded this long while sampling peak memory (default {DEFAULT_HOLD_SECONDS})",
    )
    ap.add_argument(
        "--sample-interval",
        type=float,
        default=DEFAULT_SAMPLE_INTERVAL,
        help=f"Seconds between RAM/VRAM samples while holding (default {DEFAULT_SAMPLE_INTERVAL})",
    )
    args = ap.parse_args()

    configs = get_optimal_configs()
    results = {}

    test_order = []
    for (display, hw, orch), cfg in sorted(configs.items()):
        mfrag = [k for k, v in MODEL_MAP.items() if v[0] == display][0]
        sb_parent = MODEL_MAP[mfrag][1]
        sb_child = MODEL_MAP[mfrag][2]
        test_order.append((display, sb_parent, sb_child, hw, orch, cfg))

    for display, sb_parent, sb_child, hw, orch, cfg in test_order:
        n = cfg["count"]
        if n == 0:
            continue
        gpu_indices = list(range(cfg["gpus"])) if hw == "GPU" else []

        result = measure_loaded_memory(
            display, sb_parent, sb_child, hw, n,
            cfg["cpus"], gpu_indices,
            cfg["ram_per"], cfg["vram_per"],
            hold_seconds=args.hold_seconds,
            sample_interval=args.sample_interval,
        )
        key = f"{display}_{hw}_{orch}"
        results[key] = result

    out_dir = REPO_ROOT / "results" / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "peak_memory_measured.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, val in sorted(results.items()):
        print(f"  {key:40s}: RAM={val['tree_ram_mb']/1024:.1f} GB, "
              f"VRAM={val['tree_vram_mb']/1024:.1f} GB, N={val['n_instances']}")


if __name__ == "__main__":
    main()
