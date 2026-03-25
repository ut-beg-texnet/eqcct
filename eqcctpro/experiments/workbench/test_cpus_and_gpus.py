import os
import sys
import argparse
# Note: Removed sys.path.insert - using installed package for Ray worker compatibility
# To use local development version, run: pip install -e /path/to/eqcctpro --no-deps
from eqcctpro import RunEQCCTPro, EvaluateSystem, OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder

# --- Common Directory Paths ---
base_dir = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro'
input_mseed_directory_path = os.path.join(base_dir, 'data/6_hours_tx2026burunl')
output_pick_directory_path = os.path.join(base_dir, 'results/trials/logs')
csv_filepath = os.path.join(base_dir, 'results/trials')
models_dir = os.path.join(base_dir, 'models/EQCCT')
tmp_dir = '/lambda1a/skevofilaxc/tmp'

# ==============================================================================
# ARGUMENT PARSING - For Parallel tmux sessions
# ==============================================================================
parser = argparse.ArgumentParser(description='Run EQCCTPro Benchmarks in Parallel')
parser.add_argument('--mode', type=str, choices=['cpu', 'gpu'], default='cpu', help='Hardware mode')
parser.add_argument('--arch', type=str, choices=['modelactor', 'ripper'], default='modelactor', 
                    help='Architecture mode: modelactor (persistent) or ripper (per-task)')
parser.add_argument('--model', type=str, default='eqcct', 
                    choices=['eqcct', 'phasenet', 'eqtransformer', 'phasenetlight', 'eqtransformer_non_conservative', 'all'],
                    help='Model to evaluate')
parser.add_argument('--cpu_start', type=int, default=0, help='Start of CPU core range')
parser.add_argument('--cpu_end', type=int, default=40, help='End of CPU core range')
parser.add_argument('--gpus', type=str, default='0,1', help='Comma-separated GPU indices (e.g. 0,1)')
args = parser.parse_args()

# Map arguments to settings
MODE = args.mode
ARCH = args.arch
RIPPER = (ARCH == 'ripper')
SELECTED_GPUS = [int(g) for g in args.gpus.split(',')]
VRAM_PER_GPU = 46550   # MB (RTX 6000 Ada)
CPU_RANGE = range(args.cpu_start, args.cpu_end)
# ==============================================================================

# Helper to build hardware params dynamically
hw_params = {
    'eval_mode': MODE,
    'cpu_id_list': CPU_RANGE,
    'selected_gpus': SELECTED_GPUS if MODE == 'gpu' else None,
    'max_vram_mb': (len(SELECTED_GPUS) * VRAM_PER_GPU) if MODE == 'gpu' else None,
}

# Common Evaluation Settings
eval_defaults = {
    'input_dir': input_mseed_directory_path,
    'stations2use': 240,
    'starting_amount_of_stations': 1,
    'station_list_step_size': 1,
    'min_cpu_amount': 5,
    'cpu_test_step_size': 3,
    'min_conc_stations': 1,
    'conc_station_tasks_step_size': 1,
    'ram_safety_cap': 0.95,
    'tmp_dir': tmp_dir,
    'start_time': '2026-01-26 10:22:07',
    'end_time': '2026-01-26 10:23:07',
    'ripper': RIPPER,
    **hw_params
}

# --- A: EQCCT Model ---
def run_eqcct():
    tag = f"eval_{MODE}_eqcct_{ARCH}"
    print(f"\n>>> Running EQCCT Evaluation ({MODE.upper()}, {ARCH.upper()}) on CPU Cores {list(CPU_RANGE)} <<<")
    eval_obj = EvaluateSystem(
        model_type='eqcct',
        p_model_filepath=os.path.join(models_dir, 'test_trainer_024.h5'),
        s_model_filepath=os.path.join(models_dir, 'test_trainer_021.h5'),
        output_dir=os.path.join(output_pick_directory_path, tag),
        log_filepath=os.path.join(output_pick_directory_path, tag, 'eqcctpro.log'),
        csv_dir=os.path.join(csv_filepath, tag),
        **eval_defaults
    )
    eval_obj.evaluate()

# --- B: PhaseNet ---
def run_phasenet():
    tag = f"eval_{MODE}_phasenet_original_{ARCH}"
    print(f"\n>>> Running PhaseNet Evaluation ({MODE.upper()}, {ARCH.upper()}) on CPU Cores {list(CPU_RANGE)} <<<")
    eval_obj = EvaluateSystem(
        model_type='seisbench',
        seisbench_parent_model='PhaseNet',
        seisbench_child_model='original',
        output_dir=os.path.join(output_pick_directory_path, tag),
        log_filepath=os.path.join(output_pick_directory_path, tag, 'eqcctpro.log'),
        csv_dir=os.path.join(csv_filepath, tag),
        **eval_defaults
    )
    eval_obj.evaluate()

# --- C: EQTransformer ---
def run_eqtransformer():
    tag = f"eval_{MODE}_eqtransformer_original_{ARCH}"
    print(f"\n>>> Running EQTransformer Evaluation ({MODE.upper()}, {ARCH.upper()}) on CPU Cores {list(CPU_RANGE)} <<<")
    eval_obj = EvaluateSystem(
        model_type='seisbench',
        seisbench_parent_model='EQTransformer',
        seisbench_child_model='original',
        output_dir=os.path.join(output_pick_directory_path, tag),
        log_filepath=os.path.join(output_pick_directory_path, tag, 'eqcctpro.log'),
        csv_dir=os.path.join(csv_filepath, tag),
        **eval_defaults
    )
    eval_obj.evaluate()

# --- D: PhaseNetLight ---
def run_phasenetlight():
    tag = f"eval_{MODE}_phasenetlight_stead_{ARCH}"
    print(f"\n>>> Running PhaseNetLight Evaluation ({MODE.upper()}, {ARCH.upper()}) on CPU Cores {list(CPU_RANGE)} <<<")
    eval_obj = EvaluateSystem(
        model_type='seisbench',
        seisbench_parent_model='PhaseNetLight',
        seisbench_child_model='stead',
        output_dir=os.path.join(output_pick_directory_path, tag),
        log_filepath=os.path.join(output_pick_directory_path, tag, 'eqcctpro.log'),
        csv_dir=os.path.join(csv_filepath, tag),
        **eval_defaults
    )
    eval_obj.evaluate()

# --- E: EQTransformer Non-Conservative ---
def run_eqtransformer_non_conservative():
    tag = f"eval_{MODE}_eqtransformer_nonconservative_{ARCH}"
    print(f"\n>>> Running EQTransformer Non-Conservative Evaluation ({MODE.upper()}, {ARCH.upper()}) on CPU Cores {list(CPU_RANGE)} <<<")
    eval_obj = EvaluateSystem(
        model_type='seisbench',
        seisbench_parent_model='EQTransformer',
        seisbench_child_model='original_nonconservative',
        output_dir=os.path.join(output_pick_directory_path, tag),
        log_filepath=os.path.join(output_pick_directory_path, tag, 'eqcctpro.log'),
        csv_dir=os.path.join(csv_filepath, tag),
        **eval_defaults
    )
    eval_obj.evaluate()

if __name__ == "__main__":
    model_map = {
        'eqcct': run_eqcct,
        'phasenet': run_phasenet,
        'eqtransformer': run_eqtransformer,
        'phasenetlight': run_phasenetlight,
        'eqtransformer_non_conservative': run_eqtransformer_non_conservative
    }

    if args.model == 'all':
        for name, func in model_map.items():
            func()
    else:
        model_map[args.model]()
