"""
Resource Usage Profiling Script for SeisBench and EQCCT Models
============================================================

This script measures both VRAM (GPU memory) and RAM (System memory) footprints 
of various deep learning models used in the EQCCT Pro pipeline.

Key Feature: Process Isolation Per Model
---------------------------------------
Each model is tested in its own subprocess to capture the TRUE "first-load" 
memory footprint. This includes:
- Library initialization costs (PyTorch/TensorFlow)
- Architecture definition loading
- Model weight loading
- Inference buffer allocation

This gives accurate values for capacity planning when each Ray ModelActor 
loads its own copy of the model.

Usage:
------
1. Run the script directly:
   $ python3 test_resource_usage.py
2. Results are printed in JSON format and can be used to update 
   SEISBENCH_MODEL_VRAM_MB in parallelization.py

Note: Uses psutil for RAM and Torch/NVML for VRAM tracking.
"""

import os
import time
import gc
import json
import logging
import numpy as np
import pynvml
import psutil
import multiprocessing
from obspy import Stream, Trace

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_process_ram():
    """
    Retrieves the current Resident Set Size (RSS) of the Python process.
    
    Returns:
        float: RAM usage in Megabytes (MB).
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def get_current_process_gpu_mem(gpu_index=0):
    """
    Retrieves the VRAM currently allocated to the current Python process.
    
    Returns:
        float: VRAM usage of the current process in Megabytes (MB).
    """
    try:
        pid = os.getpid()
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        mem = 0.0
        for p in processes:
            if p.pid == pid:
                mem = p.usedGpuMemory / 1024**2
                break
        pynvml.nvmlShutdown()
        return mem
    except Exception:
        return 0.0

def create_dummy_stream():
    """
    Creates a synthetic Obspy Stream object for inference testing.
    
    Returns:
        obspy.core.stream.Stream: A dummy stream object.
    """
    st = Stream()
    for ch in ['HH1', 'HH2', 'HHZ']:
        tr = Trace(data=np.random.randn(6000).astype(np.float32))
        tr.stats.sampling_rate = 100
        tr.stats.station = "TEST"
        tr.stats.channel = ch
        st.append(tr)
    return st

def _test_seisbench_isolated(parent, child, eval_mode, gpu_index, return_dict):
    """
    Worker function that runs in a subprocess to test a single SeisBench model.
    Captures the TOTAL process footprint (Overhead + Model).
    """
    if eval_mode == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    import torch
    import seisbench.models as sbm
    
    try:
        # 1. Initialize framework (Trigger overhead)
        if eval_mode == 'gpu':
            torch.cuda.init()
            device = torch.device(f'cuda:{gpu_index}')
            # Trigger full CUDA context creation
            _ = torch.zeros(1).to(device)
        else:
            device = torch.device('cpu')
        
        # 2. Load and run model
        model_class = getattr(sbm, parent)
        model = model_class.from_pretrained(child)
        model.to(device)
        
        # 3. Dummy inference to trigger workspace allocations
        st = create_dummy_stream()
        model.classify(st, P_threshold=0.3, S_threshold=0.3)
        
        # 4. Measure TOTAL process footprint
        time.sleep(1) # Allow NVML/OS to stabilize
        ram_total = get_process_ram()
        vram_total = get_current_process_gpu_mem(gpu_index) if eval_mode == 'gpu' else 0.0
        
        logger.info(f"SeisBench {parent}/{child} -> Total RAM: {ram_total:.2f} MB, Total VRAM: {vram_total:.2f} MB")
        
        return_dict[f"SeisBench_{parent}_{child}"] = {
            "ram_mb": round(ram_total, 2),
            "vram_mb": round(vram_total, 2)
        }
        
    except Exception as e:
        logger.error(f"Failed SeisBench test {parent}/{child}: {e}")
        return_dict[f"SeisBench_{parent}_{child}"] = None

def _test_eqcct_isolated(p_path, s_path, eval_mode, gpu_index, return_dict):
    """
    Worker function that runs in a subprocess to test EQCCT model.
    Captures the TOTAL process footprint.
    """
    if eval_mode == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    import tensorflow as tf
    
    if eval_mode == 'gpu':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
    
    try:
        from eqcctpro.eqcct_tf_models import load_eqcct_model, PreLoadGeneratorTest
        model = load_eqcct_model(p_path, s_path)
        
        # Dummy inference
        test_id = '2024-12-15 12:00:00'
        st_data = {test_id: np.random.randn(6000, 3).astype(np.float32)}
        gen = PreLoadGeneratorTest([test_id], st_data, batch_size=1)
        model.predict(gen, verbose=0)
        
        time.sleep(1)
        ram_total = get_process_ram()
        vram_total = get_current_process_gpu_mem(gpu_index) if eval_mode == 'gpu' else 0.0
        
        logger.info(f"EQCCT -> Total RAM: {ram_total:.2f} MB, Total VRAM: {vram_total:.2f} MB")
        
        return_dict["EQCCT"] = {
            "ram_mb": round(ram_total, 2),
            "vram_mb": round(vram_total, 2)
        }
    except Exception as e:
        logger.error(f"Failed EQCCT test: {e}")
        return_dict["EQCCT"] = None

def run_isolated_model_test(test_func, *args):
    """
    Runs a model test function in a completely isolated subprocess.
    Returns the result via a shared dictionary.
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    p = multiprocessing.Process(target=test_func, args=(*args, return_dict))
    p.start()
    p.join()
    
    return dict(return_dict)

def run_full_suite(seisbench_list, eqcct_p, eqcct_s, eval_mode, gpu_index=0):
    """
    Runs the complete test suite with each model in its own isolated process.
    """
    logger.info(f"{'='*50}")
    logger.info(f"Starting {eval_mode.upper()} Suite - ISOLATED PROCESS PER MODEL")
    logger.info(f"{'='*50}")
    
    results = {"models": {}}
    
    # Test each SeisBench model in isolation
    for parent, child in seisbench_list:
        model_result = run_isolated_model_test(
            _test_seisbench_isolated, parent, child, eval_mode, gpu_index
        )
        results["models"].update(model_result)
        time.sleep(0.5)  # Brief pause between processes
    
    # Test EQCCT in isolation
    if os.path.exists(eqcct_p) and os.path.exists(eqcct_s):
        eqcct_result = run_isolated_model_test(
            _test_eqcct_isolated, eqcct_p, eqcct_s, eval_mode, gpu_index
        )
        results["models"].update(eqcct_result)
    
    return results

if __name__ == "__main__":
    final_results = {
        "gpu_mode": {},
        "cpu_mode": {}
    }

    seisbench_list = [
        ('PhaseNet', 'original'), ('PhaseNet', 'stead'), ('PhaseNet', 'ethz'),
        ('PhaseNet', 'scedc'), ('PhaseNet', 'pisdl'), ('PhaseNet', 'instance'),
        ('PhaseNetLight', 'stead'), ('PhaseNetLight', 'ethz'),
        ('PhaseNetLight', 'scedc'), ('PhaseNetLight', 'instance'),
        ('EQTransformer', 'original'), ('EQTransformer', 'original_nonconservative'),
        ('EQTransformer', 'stead'), ('EQTransformer', 'ethz'),
        ('EQTransformer', 'scedc'), ('EQTransformer', 'instance'),
        ('GPD', 'original')
    ]
    
    eqcct_p = '/home/skevofilaxc/model/ModelPS/test_trainer_024.h5'
    eqcct_s = '/home/skevofilaxc/model/ModelPS/test_trainer_021.h5'

    # 1. Run GPU Suite (each model in isolated process)
    final_results["gpu_mode"] = run_full_suite(seisbench_list, eqcct_p, eqcct_s, 'gpu', 0)

    # 2. Run CPU Suite (each model in isolated process)
    final_results["cpu_mode"] = run_full_suite(seisbench_list, eqcct_p, eqcct_s, 'cpu', 0)

    # Final Summary
    print("\n" + "="*60)
    print("FINAL RESOURCE USAGE SUMMARY (MB) - ISOLATED PROCESS PER MODEL")
    print("="*60)
    print(json.dumps(final_results, indent=4))
    print("="*60)
    
    # Generate code snippet for parallelization.py
    print("\n" + "="*60)
    print("SUGGESTED CONSTANTS FOR parallelization.py:")
    print("="*60)
    
    print("\n# GPU Mode - VRAM requirements (MB) for each model actor")
    print("SEISBENCH_MODEL_VRAM_MB = {")
    for key, val in final_results["gpu_mode"]["models"].items():
        if val and "SeisBench_" in key:
            parts = key.replace("SeisBench_", "").split("_", 1)
            if len(parts) == 2:
                parent, child = parts
                print(f"    ('{parent}', '{child}'): {val['vram_mb']},")
    print("}")
    
    print("\n# GPU Mode - RAM requirements (MB) for each model actor") 
    print("SEISBENCH_MODEL_RAM_MB = {")
    for key, val in final_results["gpu_mode"]["models"].items():
        if val and "SeisBench_" in key:
            parts = key.replace("SeisBench_", "").split("_", 1)
            if len(parts) == 2:
                parent, child = parts
                print(f"    ('{parent}', '{child}'): {val['ram_mb']},")
    print("}")
    
    if final_results["gpu_mode"]["models"].get("EQCCT"):
        eqcct = final_results["gpu_mode"]["models"]["EQCCT"]
        print(f"\nEQCCT_GPU_VRAM_MB = {eqcct['vram_mb']}")
        print(f"EQCCT_GPU_RAM_MB = {eqcct['ram_mb']}")
    
    print("\n# CPU Mode - RAM requirements (MB)")
    print("SEISBENCH_MODEL_CPU_RAM_MB = {")
    for key, val in final_results["cpu_mode"]["models"].items():
        if val and "SeisBench_" in key:
            parts = key.replace("SeisBench_", "").split("_", 1)
            if len(parts) == 2:
                parent, child = parts
                print(f"    ('{parent}', '{child}'): {val['ram_mb']},")
    print("}")
    
    if final_results["cpu_mode"]["models"].get("EQCCT"):
        eqcct_cpu = final_results["cpu_mode"]["models"]["EQCCT"]
        print(f"\nEQCCT_CPU_RAM_MB = {eqcct_cpu['ram_mb']}")
