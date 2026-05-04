import os
import sys
import gc
import time
import logging
import statistics
import numpy as np
import pandas as pd
import psutil
import torch
import obspy
from eqcctpro import EvaluateSystem
from eqcctpro.parallelization import _stream_select_for_station_task
from eqcctpro.seisbench_models import (
    SeisBenchModels,
    mseed2stream_3c,
    process_raw_station_stream_3c,
)
from eqcctpro.eqcct_tf_models import load_eqcct_model
from eqcctpro.tools import build_station_list_from_dir
from eqcctpro.timing_util import cuda_synchronize_best_effort, monotonic_s

# --- Configuration ---
base_dir = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro'
input_base_dir = os.path.join(base_dir, 'data/230_stations_1_min_dt')
timechunk_dir = '20241215T120000Z_20241215T120100Z'
input_mseed_dir = os.path.join(input_base_dir, timechunk_dir)
models_dir = os.path.join(base_dir, 'models/EQCCT')
output_base = os.path.join(base_dir, 'results/benchmark_results/requested_benchmarks')
tmp_dir = '/lambda1a/skevofilaxc/tmp'
VRAM_PER_GPU = 46550  # MB

# Station counts for raw serial, reload, serial Ray eval, and parallel Ray eval (slice from discovered list).
BENCHMARK_STATION_COUNTS = (1, 50, 100, 200, 228)

os.makedirs(output_base, exist_ok=True)

# CSV for raw serial baseline results (station counts in BENCHMARK_STATION_COUNTS; CPU and GPU).
# BaselineMode: warm_serial = one load, sequential classify; reload_per_waveform = load → classify → unload per station (SeisBench only).
#
# All durations use timing_util.monotonic_s (perf_counter), matching parallelization.run_eqcctpro_native_process.
# GPU SeisBench: cuda_synchronize_best_effort() after load/.to and after classify/predict (same policy as Ripper/tasks).
#
# Timing matches parallelization.append_trial_row / tools.CANONICAL_CSV_HEADER:
#   Total Trial Time (s) — wall (trial_start after CUDA env → picker_end); cf. Ray end − trial_start_time.
#   Total Run time for Picker (s) — wall (picker_start before merge → picker_end); cf. Ray end − start_time.
#   Actor Creation Time (s) — one-time model load (warm only); Ripper/reload leave empty.
#   Avg Model Load Time (s) — mean of per-station load_model intervals (reload only); cf. Ripper worker tuple.
#   Waveform Processing Time (s) — mean of per-station prep intervals (subset + process_raw OR mseed2stream); excludes driver merge.
#   LoadTime_s / ProcessTime_s / TotalWallTime_s — Actor load or sum(loads); mean inference; same as Total Trial.
serial_csv_path = os.path.join(output_base, 'serial_baseline.csv')
serial_results = []

# List of models to benchmark
models = [
    {"name": "EQCCT", "type": "eqcct", "parent": None, "child": None},
    {"name": "PhaseNet", "type": "seisbench", "parent": "PhaseNet", "child": "original"},
    {"name": "EQTransformer", "type": "seisbench", "parent": "EQTransformer", "child": "original"},
    {"name": "EQT_NonConservative", "type": "seisbench", "parent": "EQTransformer", "child": "original_nonconservative"},
    {"name": "PhaseNetLight", "type": "seisbench", "parent": "PhaseNetLight", "child": "stead"},
]


def _station_files_flat_dir(input_mseed_dir, station):
    """Match flat timechunk layout used by this bench (filenames starting with station id)."""
    return [
        os.path.join(input_mseed_dir, f)
        for f in os.listdir(input_mseed_dir)
        if f.startswith(station)
    ]


def build_premerged_stream_benchmark(input_mseed_dir, station_list):
    """
    One driver-side merge of all station waveforms (Ripper/Model-Actor-style preload).
    Falls back to empty Stream if nothing was loaded (caller may read per station like Ray tasks).
    """
    full = obspy.Stream()
    for station in station_list:
        files = _station_files_flat_dir(input_mseed_dir, station)
        if not files:
            continue
        try:
            stream, _, _ = mseed2stream_3c({}, files, station)
            full += stream
        except Exception:
            continue
    return full


def _r6(x):
    return round(float(x), 6)


def _mean_or_empty(values: list) -> str | float:
    if not values:
        return ""
    return _r6(statistics.mean(values))


def _append_serial_baseline_row(
    *,
    model_name,
    baseline_mode,
    device,
    n_stations,
    n_processed,
    total_trial_s,
    actor_creation_s,
    model_load_times,
    waveform_load_times,
    inference_times,
    picker_wall_s,
    waveform_merge_s,
):
    """
    Canonical columns match Ray trial CSV (see module docstring). Driver merge is inside Total Run time for Picker only,
    not in Waveform Processing Time (per-station mean, like Ripper workers).
    """
    total_run_picker = _r6(picker_wall_s)
    total_trial = _r6(total_trial_s)

    if baseline_mode == "warm_serial":
        actor_creation = _r6(actor_creation_s) if actor_creation_s is not None else ""
        avg_model_load = ""
        load_s = _r6(actor_creation_s) if actor_creation_s is not None else ""
        comment = (
            "Serial warm baseline. Waveform Processing = mean per-station prep (subset + process_raw_station_stream_3c "
            "or mseed2stream_3c), matching worker wedge in parallel_predict_seisbench. "
            "Total Trial = wall from trial_start (after CUDA env) through picker end. "
            "Ray trials add CPU affinity / station list / object-store put not timed here."
        )
    else:
        actor_creation = ""
        avg_model_load = _mean_or_empty(model_load_times)
        load_s = _r6(sum(model_load_times)) if model_load_times else ""
        comment = (
            "Serial reload-per-waveform; per-station order matches ripper_parallel_predict_seisbench (load model, "
            "waveform prep, classify). Waveform / Avg Model Load are means of per-station intervals. "
            "Total Trial = wall from trial_start through picker end (Ripper CSV also counts earlier trial_start_time "
            "on worker before this phase)."
        )

    avg_waveform_cell = _mean_or_empty(waveform_load_times)
    process_s = _mean_or_empty(inference_times)

    serial_results.append(
        {
            "Model Used": model_name,
            "BaselineMode": baseline_mode,
            "Device": device,
            "Number of Stations Used": int(n_stations),
            "Stations Processed": int(n_processed),
            "Total Trial Time (s)": total_trial,
            "Actor Creation Time (s)": actor_creation,
            "Avg Model Load Time (s)": avg_model_load,
            "Waveform Processing Time (s)": avg_waveform_cell,
            "Total Run time for Picker (s)": total_run_picker,
            "LoadTime_s": load_s,
            "WaveformMergeTime_s": _r6(waveform_merge_s),
            "ProcessTime_s": process_s,
            "TotalWallTime_s": total_trial,
            "Comments": comment,
        }
    )


def _seisbench_classify_kwargs():
    """Defaults aligned with ripper_parallel_predict_seisbench."""
    return dict(
        P_threshold=0.3,
        S_threshold=0.3,
        Detection_threshold=0.3,
        strict=False,
        flexible_horizontal_components=True,
    )


def run_raw_serial(model_info, station_list, use_gpu=False):
    """Run serial (1-at-a-time, no Ray) processing. Covers 1 and 50 stations on CPU and GPU."""
    device = "GPU" if use_gpu else "CPU"
    n_stations = len(station_list)
    print(f"\n>>> [RAW SERIAL BASELINE] {model_info['name']} on {device} (Stations: {n_stations})")
    
    # Set GPU growth if needed
    if use_gpu:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU for serial baseline
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    trial_start = monotonic_s()
    start_load = monotonic_s()
    try:
        if model_info["type"] == "eqcct":
            model = load_eqcct_model(
                os.path.join(models_dir, "test_trainer_024.h5"),
                os.path.join(models_dir, "test_trainer_021.h5"),
            )
        else:
            sb_manager = SeisBenchModels(model_info["parent"], model_info["child"])
            model = sb_manager.load_model()
            if use_gpu and torch.cuda.is_available():
                model.to(torch.device("cuda"))
                cuda_synchronize_best_effort()
        load_time = monotonic_s() - start_load

        picker_start = monotonic_s()
        t_merge0 = monotonic_s()
        merged = build_premerged_stream_benchmark(input_mseed_dir, station_list)
        waveform_merge_s = monotonic_s() - t_merge0
        use_merged = len(merged) > 0

        waveform_load_times: list[float] = []
        inference_times: list[float] = []
        processed = 0
        for i, station in enumerate(station_list):
            try:
                wf0 = monotonic_s()
                if use_merged:
                    st_sel = _stream_select_for_station_task(merged, station)
                    if len(st_sel) == 0:
                        continue
                    if model_info["type"] == "eqcct":
                        pass  # wf ends after numpy pack below (same window)
                    else:
                        stream3c, _, _ = process_raw_station_stream_3c({}, st_sel, station)
                else:
                    files = _station_files_flat_dir(input_mseed_dir, station)
                    if not files:
                        continue
                    if model_info["type"] == "eqcct":
                        stream3c, _, _ = mseed2stream_3c({}, files, station)
                        st_sel = stream3c
                    else:
                        stream3c, _, _ = mseed2stream_3c({}, files, station)

                if model_info["type"] == "eqcct":
                    stream = st_sel if use_merged else stream3c
                    data = np.array([tr.data[:6000] for tr in stream]).T.reshape(1, 6000, 3)
                    waveform_load_times.append(monotonic_s() - wf0)
                    inf0 = monotonic_s()
                    model.predict(data, verbose=0)
                    if use_gpu:
                        cuda_synchronize_best_effort()
                    inference_times.append(monotonic_s() - inf0)
                else:
                    waveform_load_times.append(monotonic_s() - wf0)
                    inf0 = monotonic_s()
                    model.classify(stream3c, **_seisbench_classify_kwargs())
                    if use_gpu:
                        cuda_synchronize_best_effort()
                    inference_times.append(monotonic_s() - inf0)
                processed += 1
            except Exception:
                pass
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{len(station_list)}...")

        picker_end = monotonic_s()
        total_run_picker_s = picker_end - picker_start
        total_trial_s = picker_end - trial_start
        print(
            f"    Done. Model load: {load_time:.5f}s | Merge: {waveform_merge_s:.5f}s | "
            f"Total picker: {total_run_picker_s:.5f}s | Total trial: {total_trial_s:.5f}s | processed: {processed}"
        )
        _append_serial_baseline_row(
            model_name=model_info["name"],
            baseline_mode="warm_serial",
            device=device,
            n_stations=n_stations,
            n_processed=processed,
            total_trial_s=total_trial_s,
            actor_creation_s=load_time,
            model_load_times=[],
            waveform_load_times=waveform_load_times,
            inference_times=inference_times,
            picker_wall_s=total_run_picker_s,
            waveform_merge_s=waveform_merge_s,
        )
        return total_run_picker_s
    except Exception as e:
        print(f"    FAILED Raw Serial: {e}")
        return None


def run_raw_serial_reload_per_waveform(model_info, station_list, use_gpu=False):
    """SeisBench only: load_model → classify → tear down after every station (pathological single-slot baseline)."""
    if model_info["type"] != "seisbench":
        return None
    device_str = "GPU" if use_gpu else "CPU"
    n_stations = len(station_list)
    print(f"\n>>> [RELOAD PER WAVEFORM] {model_info['name']} on {device_str} (Stations: {n_stations})")

    if use_gpu:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    torch_device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    trial_start = monotonic_s()
    try:
        # Picker window: driver merge then per-station work (same wall-clock definition as Ray start_time..end_time).
        picker_start = monotonic_s()
        t_merge0 = monotonic_s()
        merged = build_premerged_stream_benchmark(input_mseed_dir, station_list)
        waveform_merge_s = monotonic_s() - t_merge0
        use_merged = len(merged) > 0

        model_load_times: list[float] = []
        waveform_load_times: list[float] = []
        inference_times: list[float] = []
        processed = 0

        for i, station in enumerate(station_list):
            sb_manager = None
            try:
                sb_manager = SeisBenchModels(
                    model_info["parent"], model_info["child"], validate_pretrained=False
                )
                ml0 = monotonic_s()
                sb_manager.load_model()
                if use_gpu and torch.cuda.is_available():
                    sb_manager.model.to(torch_device)
                if use_gpu:
                    cuda_synchronize_best_effort()
                ml = monotonic_s() - ml0

                wf0 = monotonic_s()
                if use_merged:
                    st_sel = _stream_select_for_station_task(merged, station)
                    if len(st_sel) == 0:
                        continue
                    stream3c, _, _ = process_raw_station_stream_3c({}, st_sel, station)
                else:
                    files = _station_files_flat_dir(input_mseed_dir, station)
                    if not files:
                        continue
                    stream3c, _, _ = mseed2stream_3c({}, files, station)
                wf = monotonic_s() - wf0

                inf0 = monotonic_s()
                sb_manager.classify(stream3c, **_seisbench_classify_kwargs())
                if use_gpu:
                    cuda_synchronize_best_effort()
                inf = monotonic_s() - inf0

                model_load_times.append(ml)
                waveform_load_times.append(wf)
                inference_times.append(inf)
                processed += 1
            except Exception:
                pass
            finally:
                if sb_manager is not None:
                    del sb_manager
                gc.collect()
                if use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{len(station_list)}...")

        picker_end = monotonic_s()
        total_run_picker_s = picker_end - picker_start
        total_trial_s = picker_end - trial_start
        sum_load = sum(model_load_times)
        sum_classify = sum(inference_times)

        print(
            f"    Done (reload each station). Merge: {waveform_merge_s:.5f}s | Sum load: {sum_load:.5f}s | "
            f"Sum classify: {sum_classify:.5f}s | Total picker: {total_run_picker_s:.5f}s | "
            f"Total trial: {total_trial_s:.5f}s | processed: {processed}"
        )
        _append_serial_baseline_row(
            model_name=model_info["name"],
            baseline_mode="reload_per_waveform",
            device=device_str,
            n_stations=n_stations,
            n_processed=processed,
            total_trial_s=total_trial_s,
            actor_creation_s=None,
            model_load_times=model_load_times,
            waveform_load_times=waveform_load_times,
            inference_times=inference_times,
            picker_wall_s=total_run_picker_s,
            waveform_merge_s=waveform_merge_s,
        )
        return total_run_picker_s
    except Exception as e:
        print(f"    FAILED Reload-Per-Waveform: {e}")
        return None

def run_serial_eval_bench(model, mode, station_count, cpu_ids, gpus=None, ripper=False):
    """Run EvaluateSystem with serial processing (one concurrent station task at a time)."""
    method = "Ripper" if ripper else "ModelActor"
    output_dir = os.path.join(output_base, model['name'], method, mode, f"serial_{station_count}st")
    print(f"    [SERIAL EVAL] {model['name']} {mode.upper()} {station_count} station(s)")
    
    eval_sys = EvaluateSystem(
        eval_mode=mode, model_type=model['type'],
        p_model_filepath=os.path.join(models_dir, 'test_trainer_024.h5'),
        s_model_filepath=os.path.join(models_dir, 'test_trainer_021.h5'),
        seisbench_parent_model=model['parent'], seisbench_child_model=model['child'],
        input_dir=input_base_dir, output_dir=output_dir,
        log_filepath=os.path.join(output_dir, 'eqcctpro.log'),
        csv_dir=os.path.join(output_dir, 'csv'),
        cpu_id_list=list(cpu_ids),
        min_cpu_amount=1,
        cpu_test_step_size=1,
        stations2use=station_count,
        starting_amount_of_stations=station_count,
        station_list_step_size=1,
        min_conc_stations=1,
        conc_station_tasks_step_size=1,
        selected_gpus=gpus,
        max_vram_mb=len(gpus) * VRAM_PER_GPU if gpus else None,
        tmp_dir=tmp_dir,
        start_time='2024-12-15 12:00:00',
        end_time='2024-12-15 12:01:00',
        timechunk_dt=1,
        ripper=ripper
    )
    if mode == 'cpu':
        eval_sys.evaluate_cpu()
    else:
        eval_sys.evaluate_gpu()


def run_eval_bench(model, mode, cpu_ids, cpu_range, gpus=None, ripper=False, station_count=50):
    """Run EvaluateSystem with parallel processing requesting max concurrency for ``station_count`` stations."""
    method = "Ripper" if ripper else "ModelActor"
    output_dir = os.path.join(
        output_base, model["name"], method, mode, f"parallel_{station_count}st"
    )

    # EvaluateSystem caps actors/tasks to available RAM/VRAM when concurrency is high.
    eval_sys = EvaluateSystem(
        eval_mode=mode, model_type=model['type'],
        p_model_filepath=os.path.join(models_dir, 'test_trainer_024.h5'),
        s_model_filepath=os.path.join(models_dir, 'test_trainer_021.h5'),
        seisbench_parent_model=model['parent'], seisbench_child_model=model['child'],
        input_dir=input_base_dir, output_dir=output_dir,
        log_filepath=os.path.join(output_dir, 'eqcctpro.log'),
        csv_dir=os.path.join(output_dir, 'csv'),
        cpu_id_list=list(cpu_ids), 
        min_cpu_amount=cpu_range[0], 
        cpu_test_step_size=cpu_range[2],
        stations2use=station_count,
        starting_amount_of_stations=station_count,
        station_list_step_size=1,
        min_conc_stations=station_count,
        conc_station_tasks_step_size=1,
        selected_gpus=gpus, 
        max_vram_mb=len(gpus) * VRAM_PER_GPU if gpus else None,
        tmp_dir=tmp_dir, 
        start_time='2024-12-15 12:00:00', 
        end_time='2024-12-15 12:01:00', 
        timechunk_dt=1,
        ripper=ripper
    )
    if mode == 'cpu': eval_sys.evaluate_cpu()
    else: eval_sys.evaluate_gpu()

if __name__ == "__main__":
    all_stations = build_station_list_from_dir(input_mseed_dir)
    n_avail = len(all_stations)
    station_slices: dict[int, list] = {}
    for n in BENCHMARK_STATION_COUNTS:
        if n > n_avail:
            print(
                f"Skipping {n} stations: only {n_avail} available under {input_mseed_dir}"
            )
            continue
        station_slices[n] = all_stations[:n]

    for m in models:
        print(f"\n{'='*30} MODEL: {m['name']} {'='*30}")

        for n_stations, st_list in station_slices.items():
            print(f"\n--- Station count: {n_stations} ---")

            # 1. RAW SERIAL BASELINES (no Ray), CPU + GPU
            run_raw_serial(m, st_list, use_gpu=False)
            run_raw_serial(m, st_list, use_gpu=True)

            if m["type"] == "seisbench":
                run_raw_serial_reload_per_waveform(m, st_list, use_gpu=False)
                run_raw_serial_reload_per_waveform(m, st_list, use_gpu=True)

            # 2. SERIAL Ray EVAL (one concurrent station task)
            for ripper_mode in [False, True]:
                run_serial_eval_bench(
                    m, "cpu", n_stations, range(0, 1), gpus=None, ripper=ripper_mode
                )
                run_serial_eval_bench(
                    m, "gpu", n_stations, range(0, 1), gpus=[0], ripper=ripper_mode
                )

            # 3. PARALLEL Ray EVAL (max concurrency for this station count)
            for ripper_mode in [False, True]:
                meth = "RIPPER" if ripper_mode else "MODELACTOR"
                print(
                    f"\n>>> Parallel Benchmarks ({meth}) - {n_stations} Stations, Max Concurrency"
                )
                run_eval_bench(
                    m, "cpu", range(0, 1), (1, 1, 1), ripper=ripper_mode, station_count=n_stations
                )
                run_eval_bench(
                    m,
                    "cpu",
                    range(0, 20),
                    (5, 20, 3),
                    ripper=ripper_mode,
                    station_count=n_stations,
                )
                run_eval_bench(
                    m,
                    "gpu",
                    range(20, 40),
                    (5, 5, 1),
                    gpus=[0, 1],
                    ripper=ripper_mode,
                    station_count=n_stations,
                )

    # Save raw serial baseline results to CSV (column order matches trial CSV timing blocks)
    if serial_results:
        _serial_cols_pref = [
            "Model Used",
            "BaselineMode",
            "Device",
            "Number of Stations Used",
            "Stations Processed",
            "Total Trial Time (s)",
            "Actor Creation Time (s)",
            "Avg Model Load Time (s)",
            "Waveform Processing Time (s)",
            "Total Run time for Picker (s)",
            "LoadTime_s",
            "WaveformMergeTime_s",
            "ProcessTime_s",
            "TotalWallTime_s",
            "Comments",
        ]
        df = pd.DataFrame(serial_results)
        _tail = [c for c in df.columns if c not in _serial_cols_pref]
        _head = [c for c in _serial_cols_pref if c in df.columns]
        df = df[_head + _tail]
        df.to_csv(serial_csv_path, index=False)
        print(f"\nSerial baseline results saved to {serial_csv_path}")

    print("\nBenchmark completed. All results saved to results/benchmark_results/requested_benchmarks/")
