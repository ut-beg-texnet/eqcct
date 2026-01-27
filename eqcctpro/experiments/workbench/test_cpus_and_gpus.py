import os
import sys
# Note: Removed sys.path.insert - using installed package for Ray worker compatibility
# To use local development version, run: pip install -e /path/to/eqcctpro --no-deps
from eqcctpro import RunEQCCTPro, EvaluateSystem, OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder

# --- Common Directory Paths (Modify for your local system) ---
base_dir = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro'
input_mseed_directory_path = os.path.join(base_dir, 'data/230_stations_1_min_dt')
output_pick_directory_path = os.path.join(base_dir, 'results/csv/logs')
csv_filepath = os.path.join(base_dir, 'results/csv')
models_dir = os.path.join(base_dir, 'models/EQCCT')
tmp_dir = '/lambda1a/skevofilaxc/tmp'

# --- NOTE: RIPPER MODE - Use old task-based approach instead of ModelActors ---
# RIPPER mode allows more flexible GPU memory sharing by loading the model inside each task
# instead of using persistent ModelActors. This bypasses the MIN_FRACTIONAL_GPU constraints
# and allows for dynamic GPU memory allocation, similar to the old methodology.
#
# Pros:
#   - More flexible GPU memory sharing (no MIN_FRACTIONAL_GPU constraint)
#   - Can run more concurrent predictions when VRAM allows
#   - Dynamic memory allocation per task
#   - Memory released after each task completes
#
# Cons:
#   - Model loading overhead per task (slightly slower)
#   - Less memory efficient for repeated predictions
#
# OOM Prevention in Ripper Mode:
#   - VRAM-aware concurrency limiting: If requested concurrency exceeds VRAM capacity,
#     it is automatically capped to prevent OOM (see logs for "RIPPER VRAM LIMIT" warnings)
#   - Automatic Ray restart: Between trials, if memory would be exceeded, Ray is restarted
#     (see "[RAY RESTART]" notes in Error Message column)
#   - Task-level cleanup: Each task explicitly releases model memory after completion

# --- Example A: Evaluate System capability using EQCCT Model (CPU) ---
# eval_eqcct_cpu = EvaluateSystem(
#     eval_mode='cpu',
#     model_type='eqcct',
#     p_model_filepath=os.path.join(models_dir, 'test_trainer_024.h5'), # P Model name for EQCCT model (test_trainer_024.h5)
#     s_model_filepath=os.path.join(models_dir, 'test_trainer_021.h5'), # S Model name for EQCCT model (test_trainer_021.h5)
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_gpu_eqcct_modelactor'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_gpu_eqcct_modelactor', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_gpu_eqcct_modelactor'),
#     cpu_id_list=range(0, 20),
#     min_cpu_amount=5,
#     cpu_test_step_size=3,
#     stations2use=100,
#     starting_amount_of_stations=1,
#     station_list_step_size=1,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     ram_safety_cap=0.95,             # Limit system RAM usage to 95% of total
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00',
#     ripper=False
# )
# eval_eqcct_cpu.evaluate()

# --- Example B: Evaluate System capability using EQCCT Model (GPU) ---
# eval_eqcct_gpu = EvaluateSystem(
#     eval_mode='gpu',
#     model_type='eqcct',
#     p_model_filepath=os.path.join(models_dir, 'test_trainer_024.h5'),
#     s_model_filepath=os.path.join(models_dir, 'test_trainer_021.h5'),
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_gpu_eqcct_modelactor'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_gpu_eqcct_modelactor', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_gpu_eqcct_modelactor'), # eval_gpu_eqcct_modelactor | eval_gpu_eqcct_modelactor
#     selected_gpus=[0, 1],
#     max_vram_mb=93100,
#     cpu_id_list=range(20, 40),
#     min_cpu_amount=5,
#     cpu_test_step_size=3,
#     stations2use=100,                                               
#     starting_amount_of_stations=1,
#     station_list_step_size=1,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     ram_safety_cap=0.95,             # Limit system RAM usage to 95% of total
#     tmp_dir=tmp_dir, 
#     start_time='2024-12-15 12:00:00',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
#     end_time='2024-12-15 12:01:00',
#     ripper=False
# )
# eval_eqcct_gpu.evaluate()

# --- Example C: Evaluate System capability using PhaseNet Model (CPU/GPU) ---
# eval_phasenet_original = EvaluateSystem(
#     # eval_mode='cpu',
#     eval_mode='gpu',
#     selected_gpus=[0, 1],
#     max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNet',
#     seisbench_child_model='original',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_gpu_phasenet_original_modelactor'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_gpu_phasenet_original_modelactor', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_gpu_phasenet_original_modelactor'),
#     cpu_id_list=range(40, 60),
#     min_cpu_amount=5,
#     cpu_test_step_size=3,
#     stations2use=100,
#     starting_amount_of_stations=1,
#     station_list_step_size=1,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     ram_safety_cap=0.95,             # Limit system RAM usage to 95% of total
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00',
#     ripper=False
# )
# eval_phasenet_original.evaluate()

# --- Example D: Evaluate System capability using PhaseNetLight Model (CPU/GPU) ---
# eval_phasenetlight_stead = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNetLight',
#     seisbench_child_model='stead',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_gpu_phasenetlight_stead_modelactor'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_gpu_phasenetlight_stead_modelactor', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_gpu_phasenetlight_stead_modelactor'),
#     cpu_id_list=range(60, 80),
#     min_cpu_amount=5,
#     cpu_test_step_size=3,
#     stations2use=100,
#     starting_amount_of_stations=1,
#     station_list_step_size=1,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     ram_safety_cap=0.95,             # Limit system RAM usage to 95% of total
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00',
#     ripper=False
# )
# eval_phasenetlight_stead.evaluate()


# --- Example E: Evaluate System capability using EQTransformer Model (CPU/GPU) ---
# eval_eqtransformer_original = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='EQTransformer',
#     seisbench_child_model='original',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_gpu_eqtransformer_original_modelactor'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_gpu_eqtransformer_original_modelactor', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_gpu_eqtransformer_original_modelactor'),
#     cpu_id_list=range(80, 100),
#     min_cpu_amount=5,
#     cpu_test_step_size=3,
#     stations2use=100,
#     starting_amount_of_stations=1,
#     station_list_step_size=1,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     ram_safety_cap=0.95,             # Limit system RAM usage to 95% of total
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00',
#     ripper=False
# )
# eval_eqtransformer_original.evaluate()

# --- Example F: Evaluate System capability using EQTransformer Model (CPU/GPU) ---
# eval_eqtransformer_original_nonconservative = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='EQTransformer',
#     seisbench_child_model='original_nonconservative',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_gpu_eqtransformer_nonconservative_modelactor'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_gpu_eqtransformer_nonconservative_modelactor', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_gpu_eqtransformer_nonconservative_modelactor'),
#     cpu_id_list=range(100, 120),
#     min_cpu_amount=5,
#     cpu_test_step_size=3,
#     stations2use=100,
#     starting_amount_of_stations=1,
#     station_list_step_size=1,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     ram_safety_cap=0.95,             # Limit system RAM usage to 95% of total
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00',
#     ripper=False
# )
# eval_eqtransformer_original_nonconservative.evaluate()
