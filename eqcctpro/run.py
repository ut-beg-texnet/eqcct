import os 
from eqcctpro import RunEQCCTPro, EvaluateSystem, OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder

# --- Common Directory Paths (Modify for your local system) ---
input_mseed_directory_path = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/230_stations_1_min_dt'
output_pick_directory_path = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/outputs'
log_file_path = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/outputs/eqcctpro.log'
csv_filepath = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/csv'
tmp_dir = '/lambda1a/skevofilaxc/tmp'

# =============================================================================
# 1. RunEQCCTPro Examples (Single Instance Runs)
# =============================================================================

# --- Example A: EQCCT Model on CPU ---
# runner_eqcct_cpu = RunEQCCTPro(
#     use_gpu=False,
#     model_type='eqcct',
#     p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5',
#     s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5',
#     input_dir=input_mseed_directory_path,
#     output_dir=output_pick_directory_path,
#     log_filepath=log_file_path,
#     cpu_id_list=range(0, 10),
#     number_of_concurrent_station_predictions=5,
#     number_of_concurrent_timechunk_predictions=1,
#     P_threshold=0.001,
#     S_threshold=0.02,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00',
#     timechunk_dt=1,
#     waveform_overlap=0,
#     tmp_dir=tmp_dir
# )
# runner_eqcct_cpu.run_eqcctpro()

# --- Example B: EQCCT Model on GPU ---
# runner_eqcct_gpu = RunEQCCTPro(
#     use_gpu=True,
#     model_type='eqcct',
#     p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5',
#     s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5',
#     input_dir=input_mseed_directory_path,
#     output_dir=output_pick_directory_path,
#     log_filepath=log_file_path,
#     selected_gpus=[0],
#     vram_mb=4000, # Per station task per GPU
#     cpu_id_list=range(0, 5),
#     number_of_concurrent_station_predictions=5,
#     number_of_concurrent_timechunk_predictions=1,
#     P_threshold=0.001,
#     S_threshold=0.02,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00',
#     timechunk_dt=1,
#     waveform_overlap=0,
#     tmp_dir=tmp_dir
# )
# runner_eqcct_gpu.run_eqcctpro()

# --- Example C: SeisBench PhaseNet (original) on CPU ---
# runner_phasenet_cpu = RunEQCCTPro(
#     use_gpu=False,
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNet',
#     seisbench_child_model='original',
#     input_dir=input_mseed_directory_path,
#     output_dir=output_pick_directory_path,
#     log_filepath=log_file_path,
#     cpu_id_list=range(0, 10),
#     number_of_concurrent_station_predictions=5,
#     P_threshold=0.3,
#     S_threshold=0.3,
#     Detection_threshold=0.3,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00',
#     tmp_dir=tmp_dir
# )
# runner_phasenet_cpu.run_eqcctpro()

# --- Example D: SeisBench EQTransformer (original_nonconservative) on GPU ---
# runner_eqt_gpu = RunEQCCTPro(
#     use_gpu=True,
#     model_type='seisbench',
#     seisbench_parent_model='EQTransformer',
#     seisbench_child_model='original_nonconservative',
#     input_dir=input_mseed_directory_path,
#     output_dir=output_pick_directory_path,
#     log_filepath=log_file_path,
#     selected_gpus=[0, 1],
#     vram_mb=3500,
#     number_of_concurrent_station_predictions=8,
#     P_threshold=0.3,
#     S_threshold=0.3,
#     Detection_threshold=0.3,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00',
#     tmp_dir=tmp_dir
# )
# runner_eqt_gpu.run_eqcctpro()


# =============================================================================
# 2. EvaluateSystem Examples (Benchmarking / Resource Tuning)
# =============================================================================

# --- Example E: Evaluate System capability using EQCCT Model (CPU) ---
# eval_eqcct_cpu = EvaluateSystem(
#     eval_mode='cpu',
#     model_type='eqcct',
#     p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5',
#     s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5',
#     input_dir=input_mseed_directory_path,
#     output_dir=output_pick_directory_path,
#     log_filepath=log_file_path,
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_eqcct'),
#     cpu_id_list=range(0, 20),
#     min_cpu_amount=5,
#     cpu_test_step_size=5,
#     stations2use=50,
#     starting_amount_of_stations=10,
#     station_list_step_size=10,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=5,
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00'
# )
# eval_eqcct_cpu.evaluate()

# --- Example F: Evaluate System capability using SeisBench Model (CPU) ---
# eval_seisbench_cpu = EvaluateSystem(
#     eval_mode='cpu',
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNet',
#     seisbench_child_model='original',
#     input_dir=input_mseed_directory_path,
#     output_dir=output_pick_directory_path,
#     log_filepath=log_file_path,
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_seisbench'),
#     cpu_id_list=range(0, 20),
#     min_cpu_amount=5,
#     cpu_test_step_size=5,
#     stations2use=50,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=5,
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00'
# )
# eval_seisbench_cpu.evaluate()

# --- Example G: Evaluate System capability using EQCCT Model (GPU) ---
# eval_eqcct_gpu = EvaluateSystem(
#     eval_mode='gpu',
#     model_type='eqcct',
#     p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5',
#     s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5',
#     input_dir=input_mseed_directory_path,
#     output_dir=output_pick_directory_path,
#     log_filepath=log_file_path,
#     csv_dir=os.path.join(csv_filepath, 'eval_gpu_eqcct'),
#     selected_gpus=[0, 1],
#     max_vram_mb=40000, # Total VRAM pool to test across GPUs
#     gpu_vram_safety_cap=0.95,
#     cpu_id_list=range(113,128),
#     min_cpu_amount=5,
#     cpu_test_step_size=5,
#     stations2use=100,
#     starting_amount_of_stations=10,
#     min_conc_stations=1,
#     conc_station_tasks_step_size=1,
#     timechunk_dt=1,
#     waveform_overlap=0,
#     tmp_dir=tmp_dir,
#     start_time='2024-12-15 12:00:00',
#     end_time='2024-12-15 12:01:00'
# )
# eval_eqcct_gpu.evaluate()

# --- Example H: Evaluate System capability using SeisBench Model (GPU) ---
eval_seisbench_gpu = EvaluateSystem(
    eval_mode='gpu',
    model_type='seisbench',
    seisbench_parent_model='EQTransformer',
    seisbench_child_model='original_nonconservative',
    input_dir=input_mseed_directory_path,
    output_dir=output_pick_directory_path,
    log_filepath=log_file_path,
    csv_dir=os.path.join(csv_filepath, 'eval_gpu_seisbench'),
    selected_gpus=[0, 1],
    max_vram_mb=40000,
    cpu_id_list=range(113,128),
    min_cpu_amount=5,
    cpu_test_step_size=5,
    stations2use=100,
    min_conc_stations=1,
    conc_station_tasks_step_size=5,
    tmp_dir=tmp_dir,
    start_time='2024-12-15 12:00:00',
    end_time='2024-12-15 12:01:00'
)
eval_seisbench_gpu.evaluate()


# =============================================================================
# 3. Optimal Configuration Search (Using Evaluation Data)
# =============================================================================

# --- CPU Configuration Search ---
# cpu_finder = OptimalCPUConfigurationFinder(
#     eval_sys_results_dir=os.path.join(csv_filepath, 'eval_cpu_eqcct'), 
#     log_file_path=log_file_path
# )
# cpu_finder.find_best_overall_usecase()
# cpu_finder.find_optimal_for(cpu=10, station_count=50)

# --- GPU Configuration Search ---
# gpu_finder = OptimalGPUConfigurationFinder(
#     eval_sys_results_dir=os.path.join(csv_filepath, 'eval_gpu_seisbench'), 
#     log_file_path=log_file_path
# )
# gpu_finder.find_best_overall_usecase()
# gpu_finder.find_optimal_for(num_cpus=10, gpu_list=[0, 1], station_count=100)
