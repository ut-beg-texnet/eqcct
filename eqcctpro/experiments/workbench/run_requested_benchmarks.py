import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import psutil
from eqcctpro import EvaluateSystem
from eqcctpro.seisbench_models import SeisBenchModels, mseed2stream_3c
from eqcctpro.eqcct_tf_models import load_eqcct_model
from eqcctpro.tools import build_station_list_from_dir

# --- Configuration ---
base_dir = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro'
input_base_dir = os.path.join(base_dir, 'data/230_stations_1_min_dt')
timechunk_dir = '20241215T120000Z_20241215T120100Z'
input_mseed_dir = os.path.join(input_base_dir, timechunk_dir)
models_dir = os.path.join(base_dir, 'models/EQCCT')
output_base = os.path.join(base_dir, 'results/benchmark_results/requested_benchmarks')
tmp_dir = '/lambda1a/skevofilaxc/tmp'
VRAM_PER_GPU = 46550  # MB

os.makedirs(output_base, exist_ok=True)

# List of models to benchmark
models = [
    {"name": "EQCCT", "type": "eqcct", "parent": None, "child": None},
    {"name": "PhaseNet", "type": "seisbench", "parent": "PhaseNet", "child": "original"},
    {"name": "EQTransformer", "type": "seisbench", "parent": "EQTransformer", "child": "original"},
    {"name": "EQT_NonConservative", "type": "seisbench", "parent": "EQTransformer", "child": "original_nonconservative"},
    {"name": "PhaseNetLight", "type": "seisbench", "parent": "PhaseNetLight", "child": "stead"},
]

def run_raw_serial(model_info, station_list, use_gpu=False):
    device = "GPU" if use_gpu else "CPU"
    print(f"\n>>> [RAW SERIAL BASELINE] {model_info['name']} on {device} (Stations: {len(station_list)})")
    
    # Set GPU growth if needed
    if use_gpu: 
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Use first GPU for serial baseline
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # 1. Load Model
    start_load = time.time()
    try:
        if model_info["type"] == "eqcct":
            model = load_eqcct_model(os.path.join(models_dir, 'test_trainer_024.h5'), 
                                     os.path.join(models_dir, 'test_trainer_021.h5'))
        else:
            sb_manager = SeisBenchModels(model_info["parent"], model_info["child"])
            model = sb_manager.load_model()
        load_time = time.time() - start_load

        # 2. Process
        total_start = time.time()
        for i, station in enumerate(station_list):
            files = [os.path.join(input_mseed_dir, f) for f in os.listdir(input_mseed_dir) if f.startswith(station)]
            if not files: continue
            try:
                stream, _, _ = mseed2stream_3c({}, files, station)
                if model_info["type"] == "eqcct":
                    data = np.array([tr.data[:6000] for tr in stream]).T.reshape(1, 6000, 3)
                    model.predict(data, verbose=0)
                else:
                    model.classify(stream)
            except Exception: pass
            if (i+1) % 10 == 0: print(f"    Progress: {i+1}/{len(station_list)}...")
        
        duration = time.time() - total_start
        print(f"    Done. Load: {load_time:.2f}s | Process: {duration:.2f}s")
        return duration
    except Exception as e:
        print(f"    FAILED Raw Serial: {e}")
        return None

def run_eval_bench(model, mode, cpu_ids, cpu_range, gpus=None, ripper=False):
    method = "Ripper" if ripper else "ModelActor"
    output_dir = os.path.join(output_base, model['name'], method, mode)
    
    # For parallel modes, we only test 50 stations in 1 trial with max concurrency
    # EvaluateSystem will automatically cap the number of actors to what fits in RAM/VRAM
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
        stations2use=50, 
        starting_amount_of_stations=50, # Enforce 50 stations only
        station_list_step_size=1,
        min_conc_stations=50,           # Request max concurrency (50). System will cap to available memory.
        conc_station_tasks_step_size=1,
        selected_gpus=gpus, 
        max_vram_mb=len(gpus) * VRAM_PER_GPU if gpus else None,
        tmp_dir=tmp_dir, 
        start_time='2024-12-15 12:00:00', 
        end_time='2024-12-15 12:01:00', 
        ripper=ripper
    )
    if mode == 'cpu': eval_sys.evaluate_cpu()
    else: eval_sys.evaluate_gpu()

if __name__ == "__main__":
    all_stations = build_station_list_from_dir(input_mseed_dir)
    st_1, st_50 = all_stations[:1], all_stations[:50]

    for m in models:
        print(f"\n{'='*30} MODEL: {m['name']} {'='*30}")
        
        # 1. RAW SERIAL BASELINES (Sequential processing, No Ray)
        run_raw_serial(m, st_1, use_gpu=False)
        run_raw_serial(m, st_50, use_gpu=False)
        run_raw_serial(m, st_1, use_gpu=True)
        run_raw_serial(m, st_50, use_gpu=True)

        # 2. PARALLEL TRIALS (Ray-based EvaluateSystem)
        # Only testing 50 stations with max concurrency
        for ripper_mode in [False, True]:
            meth = "RIPPER" if ripper_mode else "MODELACTOR"
            print(f"\n>>> Parallel Benchmarks ({meth}) - 50 Stations, Max Concurrency")
            # a) 1 CPU
            run_eval_bench(m, "cpu", range(0, 1), (1, 1, 1), ripper=ripper_mode)
            # b) 5, 8, 11, 14, 17, 20 CPUs
            run_eval_bench(m, "cpu", range(0, 20), (5, 20, 3), ripper=ripper_mode)
            # c) 1 & 2 GPUs
            run_eval_bench(m, "gpu", range(20, 40), (5, 5, 1), gpus=[0, 1], ripper=ripper_mode)

    print("\nBenchmark completed. All results saved to results/benchmark_results/requested_benchmarks/")
