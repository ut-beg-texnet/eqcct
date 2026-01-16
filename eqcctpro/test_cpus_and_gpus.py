import os 
from eqcctpro import RunEQCCTPro, EvaluateSystem, OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder

# --- Common Directory Paths (Modify for your local system) ---
input_mseed_directory_path = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/230_stations_1_min_dt'
output_pick_directory_path = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/outputs'
log_file_path = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/outputs/eqcctpro.log'
csv_filepath = '/home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/csv'
tmp_dir = '/lambda1a/skevofilaxc/tmp'

# --- Example E: Evaluate System capability using EQCCT Model (CPU) ---
# eval_eqcct_cpu = EvaluateSystem(
#     eval_mode='cpu',
#     model_type='eqcct',
#     p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5',
#     s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_eqcct'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_eqcct', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_eqcct'),
#     cpu_id_list=range(0, 15),
#     min_cpu_amount=1,
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

# --- Example F: Evaluate System capability using EQCCT Model (GPU) ---
# eval_eqcct_gpu = EvaluateSystem(
#     eval_mode='gpu',
#     model_type='eqcct',
#     p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5',
#     s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_gpu_eqcct'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_gpu_eqcct', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_gpu_eqcct'),
#     selected_gpus=[0, 1],
#     max_vram_mb=93100,
#     cpu_id_list=range(15, 30),
#     min_cpu_amount=1,
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

# --- Example G: Evaluate System capability using PhaseNet Model (CPU/GPU) ---
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
#     cpu_id_list=range(30, 45),
#     min_cpu_amount=1,
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

# --- Example H: Evaluate System capability using PhaseNet Model (CPU/GPU) ---
# eval_phasenet_stead = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNet',
#     seisbench_child_model='stead',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_phasenet_stead'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_phasenet_stead', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_phasenet_stead'),
#     cpu_id_list=range(45, 60),
#     min_cpu_amount=1,
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
# eval_phasenet_stead.evaluate()

# --- Example I: Evaluate System capability using PhaseNet Model (CPU/GPU) ---
# eval_phasenet_ethz = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNet',
#     seisbench_child_model='ethz',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_phasenet_ethz'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_phasenet_ethz', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_phasenet_ethz'),
#     cpu_id_list=range(60, 75),
#     min_cpu_amount=1,
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
# eval_phasenet_ethz.evaluate()

# --- Example J: Evaluate System capability using PhaseNet Model (CPU/GPU) ---
# eval_phasenet_scedc = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNet',
#     seisbench_child_model='scedc',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_phasenet_scedc'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_phasenet_scedc', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_phasenet_scedc'),
#     cpu_id_list=range(75, 90),
#     min_cpu_amount=1,
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
# eval_phasenet_scedc.evaluate()

# --- Example K: Evaluate System capability using PhaseNet Model (CPU/GPU) ---
# eval_phasenet_pisdl = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNet',
#     seisbench_child_model='pisdl',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_phasenet_pisdl'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_phasenet_pisdl', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_phasenet_pisdl'),
#     cpu_id_list=range(90, 105),
#     min_cpu_amount=1,
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
# eval_phasenet_pisdl.evaluate()

# --- Example L: Evaluate System capability using PhaseNet Model (CPU/GPU) ---
# eval_phasenet_instance = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNet',
#     seisbench_child_model='instance',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_phasenet_instance'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_phasenet_instance', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_phasenet_instance'),
#     cpu_id_list=range(105, 120),
#     min_cpu_amount=1,
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
# eval_phasenet_instance.evaluate()

# --- Example M: Evaluate System capability using PhaseNetLight Model (CPU/GPU) ---
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
#     cpu_id_list=range(120, 135),
#     min_cpu_amount=1,
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

# --- Example N: Evaluate System capability using PhaseNetLight Model (CPU/GPU) ---
# eval_phasenetlight_ethz = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNetLight',
#     seisbench_child_model='ethz',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_phasenetlight_ethz'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_phasenetlight_ethz', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_phasenetlight_ethz'),
#     cpu_id_list=range(135, 150),
#     min_cpu_amount=1,
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
# eval_phasenetlight_ethz.evaluate()

# --- Example O: Evaluate System capability using PhaseNetLight Model (CPU/GPU) ---
# eval_phasenetlight_scedc = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNetLight',
#     seisbench_child_model='scedc',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_phasenetlight_scedc'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_phasenetlight_scedc', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_phasenetlight_scedc'),
#     cpu_id_list=range(150, 165),
#     min_cpu_amount=1,
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
# eval_phasenetlight_scedc.evaluate()

# --- Example P: Evaluate System capability using PhaseNetLight Model (CPU/GPU) ---
# eval_phasenetlight_instance = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='PhaseNetLight',
#     seisbench_child_model='instance',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_phasenetlight_instance'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_phasenetlight_instance', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_phasenetlight_instance'),
#     cpu_id_list=range(165, 180),
#     min_cpu_amount=1,
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
# eval_phasenetlight_instance.evaluate()

# --- Example Q: Evaluate System capability using EQTransformer Model (CPU/GPU) ---
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
#     cpu_id_list=range(180, 195),
#     min_cpu_amount=1,
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

# --- Example R: Evaluate System capability using EQTransformer Model (CPU/GPU) ---
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
#     cpu_id_list=range(195, 210),
#     min_cpu_amount=1,
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

# --- Example S: Evaluate System capability using EQTransformer Model (CPU/GPU) ---
# eval_eqtransformer_stead = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='EQTransformer',
#     seisbench_child_model='stead',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_eqtransformer_stead'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_eqtransformer_stead', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_eqtransformer_stead'),
#     cpu_id_list=range(210, 225),
#     min_cpu_amount=1,
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
# eval_eqtransformer_stead.evaluate()

# --- Example T: Evaluate System capability using EQTransformer Model (CPU/GPU) ---
# eval_eqtransformer_ethz = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='EQTransformer',
#     seisbench_child_model='ethz',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_eqtransformer_ethz'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_eqtransformer_ethz', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_eqtransformer_ethz'),
#     cpu_id_list=range(225, 240),
#     min_cpu_amount=1,
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
# eval_eqtransformer_ethz.evaluate()

# --- Example U: Evaluate System capability using EQTransformer Model (CPU/GPU) ---
# eval_eqtransformer_scedc = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='EQTransformer',
#     seisbench_child_model='scedc',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_eqtransformer_scedc'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_eqtransformer_scedc', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_eqtransformer_scedc'),
#     cpu_id_list=range(240, 255),
#     min_cpu_amount=1,
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
# eval_eqtransformer_scedc.evaluate()

# --- Example V: Evaluate System capability using EQTransformer Model (CPU/GPU) ---
# eval_eqtransformer_instance = EvaluateSystem(
#     eval_mode='cpu',
#     # eval_mode='gpu',
#     # selected_gpus=[0, 1],
#     # max_vram_mb=93100,
#     model_type='seisbench',
#     seisbench_parent_model='EQTransformer',
#     seisbench_child_model='instance',
#     input_dir=input_mseed_directory_path,
#     output_dir=os.path.join(output_pick_directory_path, 'eval_cpu_eqtransformer_instance'),
#     log_filepath=os.path.join(output_pick_directory_path, 'eval_cpu_eqtransformer_instance', 'eqcctpro.log'),
#     csv_dir=os.path.join(csv_filepath, 'eval_cpu_eqtransformer_instance'),
#     cpu_id_list=range(255, 270),
#     min_cpu_amount=1,
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
# eval_eqtransformer_instance.evaluate()


