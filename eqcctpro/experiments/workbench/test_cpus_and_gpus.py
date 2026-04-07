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
# Example: 228 stations, 41 CPUs, paper-style concurrency (20% steps), both architectures,
#          appends to existing results/trials/eval_cpu_*_{modelactor,ripper}/cpu_test_results.csv
#
#   python3 experiments/workbench/test_cpus_and_gpus.py --mode cpu --arch both \
#     --model phasenet --cpu_start 0 --cpu_end 41 --single_cpu_block --stations_cap 228
#
# Max concurrency only (N concurrent tasks for N stations, no fractional grid): --conc_max_only
# Dense sweep (every level): --conc_step 1
# Skip CPU RAM heuristics for ModelActor + Ripper: --ignore_cpu_ram_cap (OOM risk)
# ==============================================================================
parser = argparse.ArgumentParser(description='Run EQCCTPro Benchmarks in Parallel')
parser.add_argument('--mode', type=str, choices=['cpu', 'gpu'], default='cpu', help='Hardware mode')
parser.add_argument('--arch', type=str, choices=['modelactor', 'ripper', 'both'], default='modelactor',
                    help='Architecture mode: modelactor, ripper, or both (runs each, appends to each CSV)')
parser.add_argument('--model', type=str, default='eqcct',
                    choices=['eqcct', 'phasenet', 'eqtransformer', 'phasenetlight', 'eqtransformer_non_conservative', 'all'],
                    help='Model to evaluate')
parser.add_argument('--cpu_start', type=int, default=0, help='Start of CPU core range (inclusive)')
parser.add_argument('--cpu_end', type=int, default=40, help='End of CPU core range (exclusive, like range())')
parser.add_argument('--gpus', type=str, default='0,1', help='Comma-separated GPU indices (e.g. 0,1)')
parser.add_argument(
    '--conc_step',
    type=int,
    default=0,
    metavar='N',
    help=('Station-task concurrency step in EvaluateSystem: 0 = automatic 20%% of station count '
          '(paper default); use N>0 for fixed step (e.g. 1 tests every concurrency level). '
          'Ignored if --conc_max_only is set.'),
)
parser.add_argument(
    '--conc_max_only',
    action='store_true',
    help=('Run only the maximum concurrency trial (N concurrent station tasks for N stations). '
          'Skips intermediate 20%%/step grid without using --conc_step 1.'),
)
parser.add_argument(
    '--stations_cap',
    type=int,
    default=None,
    metavar='N',
    help=('If set, only benchmark station count N (e.g. 228). Same csv_dir tags as usual; new rows append '
          'to existing cpu_test_results.csv / gpu_test_results.csv.'),
)
parser.add_argument(
    '--single_cpu_block',
    action='store_true',
    help=('Use the full cpu_start:cpu_end range as one allocation only (no sweep from min_cpu_amount upward). '
          'Example: --cpu_start 0 --cpu_end 41 --single_cpu_block runs only with 41 CPUs.'),
)
parser.add_argument(
    '--ripper_ignore_cpu_ram_cap',
    action='store_true',
    help=('CPU Ripper only: skip RAM-based max_pending_tasks clamp. '
          'For both ModelActor and Ripper, prefer --ignore_cpu_ram_cap.'),
)
parser.add_argument(
    '--ignore_cpu_ram_cap',
    action='store_true',
    help=('CPU: skip RAM-based caps for both ModelActor (actor count + eval grid) and Ripper '
          '(max_pending_tasks). Same as --ripper_ignore_cpu_ram_cap for Ripper only. OOM risk.'),
)
args = parser.parse_args()

# Map arguments to settings
MODE = args.mode
SELECTED_GPUS = [int(g) for g in args.gpus.split(',')]
VRAM_PER_GPU = 46550   # MB (RTX 6000 Ada)
CPU_RANGE = range(args.cpu_start, args.cpu_end)
if len(CPU_RANGE) <= 0:
    print('Error: cpu_end must be greater than cpu_start.', file=sys.stderr)
    sys.exit(1)

ARCH_SEQUENCE = ['modelactor', 'ripper'] if args.arch == 'both' else [args.arch]

# ==============================================================================

def build_eval_defaults(ripper: bool) -> dict:
    """Per-run kwargs for EvaluateSystem. Existing CSVs under csv_dir are resumed/appended."""
    hw = {
        'eval_mode': MODE,
        'cpu_id_list': CPU_RANGE,
        'selected_gpus': SELECTED_GPUS if MODE == 'gpu' else None,
        'max_vram_mb': (len(SELECTED_GPUS) * VRAM_PER_GPU) if MODE == 'gpu' else None,
    }
    if args.stations_cap is not None:
        stations2use = args.stations_cap
        starting_amount = args.stations_cap
        station_step = 1
    else:
        stations2use = 240
        starting_amount = 1
        station_step = 1
    if args.single_cpu_block:
        min_cpu = len(CPU_RANGE)
        cpu_step = 1
    else:
        min_cpu = 5
        cpu_step = 3
    return {
        'input_dir': input_mseed_directory_path,
        'stations2use': stations2use,
        'starting_amount_of_stations': starting_amount,
        'station_list_step_size': station_step,
        'min_cpu_amount': min_cpu,
        'cpu_test_step_size': cpu_step,
        'min_conc_stations': 1,
        'conc_station_tasks_step_size': args.conc_step,
        'conc_station_tasks_max_only': args.conc_max_only,
        'ram_safety_cap': 0.95,
        'tmp_dir': tmp_dir,
        'start_time': '2026-01-26 10:22:07',
        'end_time': '2026-01-26 10:23:07',
        'ripper': ripper,
        'ripper_ignore_cpu_ram_cap': bool(ripper and args.ripper_ignore_cpu_ram_cap),
        'ignore_cpu_ram_cap': bool(args.ignore_cpu_ram_cap),
        **hw,
    }

# --- A: EQCCT Model ---
def run_eqcct(arch: str):
    eval_defaults = build_eval_defaults(ripper=(arch == 'ripper'))
    tag = f"eval_{MODE}_eqcct_{arch}"
    print(f"\n>>> Running EQCCT Evaluation ({MODE.upper()}, {arch.upper()}) on CPU Cores {list(CPU_RANGE)} <<<")
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
def run_phasenet(arch: str):
    eval_defaults = build_eval_defaults(ripper=(arch == 'ripper'))
    tag = f"eval_{MODE}_phasenet_original_{arch}"
    print(f"\n>>> Running PhaseNet Evaluation ({MODE.upper()}, {arch.upper()}) on CPU Cores {list(CPU_RANGE)} <<<")
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
def run_eqtransformer(arch: str):
    eval_defaults = build_eval_defaults(ripper=(arch == 'ripper'))
    tag = f"eval_{MODE}_eqtransformer_original_{arch}"
    print(f"\n>>> Running EQTransformer Evaluation ({MODE.upper()}, {arch.upper()}) on CPU Cores {list(CPU_RANGE)} <<<")
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
def run_phasenetlight(arch: str):
    eval_defaults = build_eval_defaults(ripper=(arch == 'ripper'))
    tag = f"eval_{MODE}_phasenetlight_stead_{arch}"
    print(f"\n>>> Running PhaseNetLight Evaluation ({MODE.upper()}, {arch.upper()}) on CPU Cores {list(CPU_RANGE)} <<<")
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
def run_eqtransformer_non_conservative(arch: str):
    eval_defaults = build_eval_defaults(ripper=(arch == 'ripper'))
    tag = f"eval_{MODE}_eqtransformer_nonconservative_{arch}"
    print(f"\n>>> Running EQTransformer Non-Conservative Evaluation ({MODE.upper()}, {arch.upper()}) on CPU Cores {list(CPU_RANGE)} <<<")
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

    for arch in ARCH_SEQUENCE:
        if args.model == 'all':
            for _name, func in model_map.items():
                func(arch)
        else:
            model_map[args.model](arch)
