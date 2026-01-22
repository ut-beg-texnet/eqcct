import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Add the parent directory to sys.path to import the local eqcctpro module
from eqcctpro import RunEQCCTPro, EvaluateSystem, OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder

# --- Common Directory Paths (Modify for your local system) ---
base_dir = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro'
input_mseed_directory_path = os.path.join(base_dir, 'data/230_stations_1_min_dt')
output_pick_directory_path = os.path.join(base_dir, 'results/csv/logs')
csv_filepath = os.path.join(base_dir, 'results/csv')
models_dir = os.path.join(base_dir, 'models/EQCCT')
tmp_dir = '/lambda1a/skevofilaxc/tmp'

# --- Example A: Evaluate System capability using EQCCT Model (CPU) ---
# eval_eqcct_cpu = EvaluateSystem(
#     eval_mode='cpu',
#     model_type='eqcct',
#     p_model_filepath=os.path.join(models_dir, 'test_trainer_024.h5'), # P Model name for EQCCT model (test_trainer_024.h5)
#     s_model_filepath=os.path.join(models_dir, 'test_trainer_021.h5'), # S Model name for EQCCT model (test_trainer_021.h5)
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_eqcct'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_eqcct', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_eqcct'),
#     cpu_id_list=range(0, 20),
#     min_cpu_amount=5,
#     cpu_test_step_size=1,
#     stations2use=100,
#     starting_amount_of_stations=1,
#     station_list_step_size=1,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     ram_safety_cap=0.95,             # Limit system RAM usage to 95% of total
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00'
# )
# eval_eqcct_cpu.evaluate()

# --- Example B: Evaluate System capability using EQCCT Model (GPU) ---
# eval_eqcct_gpu = EvaluateSystem(
#     eval_mode='gpu',
#     model_type='eqcct',
#     p_model_filepath=os.path.join(models_dir, 'test_trainer_024.h5'),
#     s_model_filepath=os.path.join(models_dir, 'test_trainer_021.h5'),
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_gpu_eqcct'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_gpu_eqcct', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_gpu_eqcct'),
#     selected_gpus=[0, 1],
#     max_vram_mb=93100,
#     cpu_id_list=range(20, 40),
#     min_cpu_amount=5,
#     cpu_test_step_size=1,
#     stations2use=100,                                               
#     starting_amount_of_stations=1,
#     station_list_step_size=1,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     ram_safety_cap=0.95,             # Limit system RAM usage to 95% of total
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
#     end_time='2024-12-15 12:01:00'
# )
# eval_eqcct_gpu.evaluate()

# --- Example C: Evaluate System capability using PhaseNet Model (CPU/GPU) ---
# eval_phasenet_original = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNet',
#     seisbench_child_model='original',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_phasenet_original'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_phasenet_original', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_phasenet_original'),
#     cpu_id_list=range(40, 60),
#     min_cpu_amount=5,
#     cpu_test_step_size=1,
#     stations2use=100,
#     starting_amount_of_stations=1,
#     station_list_step_size=1,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     ram_safety_cap=0.95,             # Limit system RAM usage to 95% of total
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00'
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
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_phasenetlight_stead'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_phasenetlight_stead', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_phasenetlight_stead'),
#     cpu_id_list=range(60, 80),
#     min_cpu_amount=5,
#     cpu_test_step_size=1,
#     stations2use=100,
#     starting_amount_of_stations=1,
#     station_list_step_size=1,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     ram_safety_cap=0.95,             # Limit system RAM usage to 95% of total
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00'
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
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_eqtransformer_original'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_eqtransformer_original', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_eqtransformer_original'),
#     cpu_id_list=range(80, 100),
#     min_cpu_amount=5,
#     cpu_test_step_size=1,
#     stations2use=100,
#     starting_amount_of_stations=1,
#     station_list_step_size=1,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     ram_safety_cap=0.95,             # Limit system RAM usage to 95% of total
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00'
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
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_eqtransformer_nonconservative'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_eqtransformer_nonconservative', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_eqtransformer_nonconservative'),
#     cpu_id_list=range(100, 120),
#     min_cpu_amount=5,
#     cpu_test_step_size=1,
#     stations2use=100,
#     starting_amount_of_stations=1,
#     station_list_step_size=1,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     ram_safety_cap=0.95,             # Limit system RAM usage to 95% of total
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00'
# )
# eval_eqtransformer_original_nonconservative.evaluate()                                  