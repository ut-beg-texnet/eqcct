import os 
from eqcctpro import RunEQCCTPro, EvaluateSystem, OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder
input_mseed_directory_path = '/home/skevofilaxc/workspace/eqcct/eqcctpro/230_stations_1_min_dt' # Change to local path 
output_pick_directory_path = '/home/skevofilaxc/workspace/eqcct/eqcctpro/outputs' # Change
log_file_path = '/home/skevofilaxc/workspace/eqcct/eqcctpro/outputs/eqcctpro.log' # Change
csv_filepath = '/home/skevofilaxc/workspace/eqcct/eqcctpro/csv/test_cpu' # Change
tmp_dir = '/home/skevofilaxc/tmp' # Change

# EQCCTPro can run EQCCT on a given input dir on either your GPU or CPU     
eqcctpro_cpu_runner = RunEQCCTPro(use_gpu=False, # Defines if you use the GPU to run EQCCTMSeedRunner (bool)
                                intra_threads=1, # Defines the number of intra-parallelism threads (int)
                                inter_threads=1, # Defines the number of inter-parallelism threads (int)
                                cpu_id_list=range(0,50), # Defines which specific CPU cores that sched_setaffinity will allocate for executing the current EQCCTPro process (list)
                                input_dir=input_mseed_directory_path, # Directory path to the the mSEED directory (str)
                                output_dir=output_pick_directory_path, # Directory path to where the output picks and logs will be sent (str)
                                log_filepath=log_file_path, # Filepath to where the EQCCTPro log will be written to and stored (str)
                                P_threshold=0.001, # Threshold in which the P probabilities above it will be considered as P arrival (float)
                                S_threshold=0.02, # Threshold in which the S probabilities above it will be considered as S arrival (float)
                                p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', # Filepath to where the P EQCCT detection model is stored (str)
                                s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5', # Filepath to where the S EQCCT detection model is stored (str)
                                number_of_concurrent_station_predictions=20, # Number of stations that are being analyzed by EQCCT in parallel (int)
                                number_of_concurrent_timechunk_predictions=1, # Number of timechunks that are being analyzed by EQCCT in parallel (int)
                                best_usecase_config=False, # If True, will override inputted cpu_id_list, number_of_concurrent_predictions, intra_threads, inter_threads values for the best overall use-case configurations (bool)
                                csv_dir=csv_filepath, # Directory path containing the CSV's outputted by EvaluateSystem that contain the trial data that will be used to find the best_usecase_config (str)
                                selected_gpus=None, # List of GPU IDs on your computer you want to use if use_gpu = True (list). Can be None if not using gpus 
                                vram_mb=None, # Maximum amount of VRAM each Raylet can use (float). vram_mb = (GPU VRAM * .95 (to be safe)) / number_of_concurrent_station_predictions * number_of_concurrent_timechunk_predictions. Code will check for you if the requested allocation is valid for your hardware. Can set to None if not using gpus.
                                start_time='2024-12-15 12:00:00', # The start time of the area of time that is being analyzed | Must follow the following convention YYYY-MO-DA HR:MI:SC (str)
                                end_time='2024-12-15 12:01:00', # The end time of the area of time that is being analyzed | Must follow the following convention YYYY-MO-DA HR:MI:SC (str)
                                timechunk_dt=1, # The length each time chunk is (in minutes) (int)
                                waveform_overlap=0, # The duration (in minutes) for which each waveform overlaps with the others (int)
                                specific_stations='AT01, BP01, DG05') # String that contains the "list" of stations you want to only analyze (str)


eqcctpro_cpu_runner.run_eqcctpro() # Runs EQCCTPro on one instance of waveform data using CPU(s) as the main computing driver

# # EQCCTPro can run EQCCT on a given input dir on either your GPU or CPU     
# eqcctpro_gpu_runner = RunEQCCTPro(use_gpu=True, # Defines if you use the GPU to run EQCCTMSeedRunner (bool)
#                                 intra_threads=1, # Defines the number of intra-parallelism threads (int)
#                                 inter_threads=1, # Defines the number of inter-parallelism threads (int)
#                                 cpu_id_list=range(0,50), # Defines which specific CPU cores that sched_setaffinity will allocate for executing the current EQCCTPro process (list)
#                                 input_dir=input_mseed_directory_path, # Directory path to the the mSEED directory (str)
#                                 output_dir=output_pick_directory_path, # Directory path to where the output picks and logs will be sent (str)
#                                 log_filepath=log_file_path, # Filepath to where the EQCCTPro log will be written to and stored (str)
#                                 P_threshold=0.001, # Threshold in which the P probabilities above it will be considered as P arrival (float)
#                                 S_threshold=0.02, # Threshold in which the S probabilities above it will be considered as S arrival (float)
#                                 p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', # Filepath to where the P EQCCT detection model is stored (str)
#                                 s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5', # Filepath to where the S EQCCT detection model is stored (str)
#                                 number_of_concurrent_station_predictions=20, # Number of stations that are being analyzed by EQCCT in parallel (int)
#                                 number_of_concurrent_timechunk_predictions=1, # Number of timechunks that are being analyzed by EQCCT in parallel (int)
#                                 best_usecase_config=False, # If True, will override inputted cpu_id_list, number_of_concurrent_predictions, intra_threads, inter_threads values for the best overall use-case configurations (bool)
#                                 csv_dir=csv_filepath, # Directory path containing the CSV's outputted by EvaluateSystem that contain the trial data that will be used to find the best_usecase_config (str)
#                                 selected_gpus=[0], # List of GPU IDs on your computer you want to use if use_gpu = True (list)
#                                 vram_mb=500, # Maximum amount of VRAM each Raylet can use (float). vram_mb = (GPU VRAM * .95 (to be safe)) / number_of_concurrent_station_predictions * number_of_concurrent_timechunk_predictions
#                                 start_time='2024-12-15 12:00:00', # The start time of the area of time that is being analyzed | Must follow the following convention YYYY-MO-DA HR:MI:SC (str)
#                                 end_time='2024-12-15 12:01:00', # The end time of the area of time that is being analyzed | Must follow the following convention YYYY-MO-DA HR:MI:SC (str)
#                                 timechunk_dt=1, # The length each time chunk is (in minutes) (int)
#                                 waveform_overlap=0, # The duration (in minutes) for which each waveform overlaps with the others (int)
#                                 specific_stations='AT01, BP01, DG05') # String that contains the "list" of stations you want to only analyze (str)

# eqcctpro_gpu_runner.run_eqcctpro() # Runs EQCCTPro on one instance of waveform data using GPU(s) as the main computing driver

# # We can also evaluate your systems hardware capabilities for running several waveforms in parallel to see what configuration of resources enables real-time processing 
# eval_cpu = EvaluateSystem(
#                 eval_mode='cpu', # Tells EvaluateSystem which computing approach the trials should it iterate with, either 'cpu' or 'gpu' (str)
#                 intra_threads=1, # Defines the number of intra-parallelism threads (int)
#                 inter_threads=1, # Defines the number of inter-parallelism threads (int)
#                 input_dir=input_mseed_directory_path, # Directory path to the the mSEED directory (str)
#                 output_dir=output_pick_directory_path, # Directory path to where the output picks and logs will be sent (str)
#                 log_filepath=log_file_path, # Filepath to where the EQCCTPro log will be written to and stored (str)
#                 csv_dir='/home/skevofilaxc/workspace/eqcct/eqcctpro/csv/test_cpu', # Directory path where the CSV's outputted by EvaluateSystem will be saved, doesn't need to exist, will be created if doesn't exist (str)
#                 P_threshold=0.001,  # Threshold in which the P probabilities above it will be considered as P arrival (float)
#                 S_threshold=0.02,   # Threshold in which the S probabilities above it will be considered as S arrival (float)
#                 p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', # Filepath to where the P EQCCT detection model is stored (str)
#                 s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5', # Filepath to where the S EQCCT detection model is stored (str)
#                 cpu_id_list=range(0,10), # Defines which specific CPU cores that sched_setaffinity will allocate for executing the current EQCCTPro process and is the maximum amount of cores EvaluteSystem can use in its trial iterations (list)
#                 min_cpu_amount=5, # Is the minimum amount of CPUs you want to start your trials with (int)
#                 cpu_test_step_size=5, # Is the desired step size for the trials will march from min_cpu_amount to len(cpu_id_list) (int)
#                 stations2use=50, # Controls the maximum amount of stations EvaluateSystem can use in its trial iterations (int)
#                 starting_amount_of_stations=45, # For evaluating your system, you have the option to set a starting amount of stations you want to use in the test (default=1, int)
#                 station_list_step_size=5, # Set a step size for the station list that is generated (default stepsize of 1 for stations 1-10, then stepsize of 5 up to stations2use) (int)
#                 min_conc_stations=45, # Is the minimum amount of concurrent stations predictions you want each trial iteration to start with (int) | By default, if min_conc_predictions and conc_predictions_step_size are set to 1, a custom step size iteration will be applied (README)
#                 conc_station_tasks_step_size=5, # Is the concurrent station predictions step size you want each trial iteration to iterate with (int)
#                 start_time='2024-12-15 12:00:00', # The start time of the area of time that is being analyzed | Must follow the following convention YYYY-MO-DA HR:MI:SC (str)
#                 end_time='2024-12-15 12:01:00', # The end time of the area of time that is being analyzed | Must follow the following convention YYYY-MO-DA HR:MI:SC (str)
#                 conc_timechunk_tasks_step_size=1, # Is the concurrent timechunk predictions step size you want each trial iteration to iterate with
#                 timechunk_dt=1, # The length each time chunk is (in minutes) (int)
#                 waveform_overlap=0, # The duration (in minutes) for which each waveform overlaps with the others (int)
#                 tmp_dir=tmp_dir) # A temporary directory to store all temp files produced by EQCCTPro (str) | Used to help ease system cleanup and to not write to system's default temporary directory

# eval_cpu.evaluate()  # This triggers evaluate_cpu() with cpu mode

# cpu_finder = OptimalCPUConfigurationFinder(eval_sys_results_dir=csv_filepath, log_file_path=log_file_path)
# cpu_finder.find_best_overall_usecase()
# cpu_finder.find_optimal_for(cpu=5, station_count=45)

# eval_gpu = EvaluateSystem(
#                 eval_mode='gpu', # Tells EvaluateSystem which computing approach the trials should it iterate with, either 'cpu' or 'gpu' (str)
#                 intra_threads=1, # Defines the number of intra-parallelism threads (int)
#                 inter_threads=1, # Defines the number of inter-parallelism threads (int)
#                 input_dir=input_mseed_directory_path, # Directory path to the the mSEED directory (str)
#                 output_dir=output_pick_directory_path, # Directory path to where the output picks and logs will be sent (str)
#                 log_filepath=log_file_path, # Filepath to where the EQCCTPro log will be written to and stored (str)
#                 csv_dir='/home/skevofilaxc/workspace/eqcct/eqcctpro/csv/test_gpu', # Directory path where the CSV's outputted by EvaluateSystem will be saved, doesn't need to exist, will be created if doesn't exist (str)
#                 P_threshold=0.001,  # Threshold in which the P probabilities above it will be considered as P arrival (float)
#                 S_threshold=0.02,   # Threshold in which the S probabilities above it will be considered as S arrival (float)
#                 p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', # Filepath to where the P EQCCT detection model is stored (str)
#                 s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5', # Filepath to where the S EQCCT detection model is stored (str)
#                 cpu_id_list=range(108,128), # Defines which specific CPU cores that sched_setaffinity will allocate for executing the current EQCCTPro process and is the maximum amount of cores EvaluteSystem can use in its trial iterations (list)
#                 min_cpu_amount=20, # Is the minimum amount of CPUs you want to start your trials with (int)
#                 cpu_test_step_size=5, # Is the desired step size for the trials will march from min_cpu_amount to len(cpu_id_list) (int)
#                 stations2use=1, # Controls the maximum amount of stations EvaluateSystem can use in its trial iterations (int)
#                 starting_amount_of_stations=1, # For evaluating your system, you have the option to set a starting amount of stations you want to use in the test (default=1, int)
#                 station_list_step_size=1, # Set a step size for the station list that is generated (default stepsize of 1 for stations 1-10, then stepsize of 5 up to stations2use) (int)
#                 min_conc_stations=1, # Is the minimum amount of concurrent stations predictions you want each trial iteration to start with (int) | By default, if min_conc_predictions and conc_predictions_step_size are set to 1, a custom step size iteration will be applied (README)
#                 conc_station_tasks_step_size=5, # Is the concurrent station predictions step size you want each trial iteration to iterate with (int)
#                 start_time='2024-12-15 12:00:00', # The start time of the area of time that is being analyzed | Must follow the following convention YYYY-MO-DA HR:MI:SC (str)
#                 end_time='2024-12-15 12:01:00', # The end time of the area of time that is being analyzed | Must follow the following convention YYYY-MO-DA HR:MI:SC (str)
#                 conc_timechunk_tasks_step_size=1, # Is the concurrent timechunk predictions step size you want each trial iteration to iterate with
#                 timechunk_dt=1, # The length each time chunk is (in minutes) (int)
#                 waveform_overlap=0, # The duration (in minutes) for which each waveform overlaps with the others (int)
#                 tmp_dir=tmp_dir, # A temporary directory to store all temp files produced by EQCCTPro (str) | Used to help ease system cleanup and to not write to system's default temporary directory
#                 vram_mb=10000, # The maximum amount of VRAM each Raylet can use (float); It reflect the upper limit of the iterative amount of VRAM we can test the Raylets with | Must be a real value that is based on your hardware's physical memory space, if it exceeds the space the code will break due to OutOfMemoryError
#                 gpu_vram_safety_cap=0.95, # Safety cap for VRAM usage to help avoid OutOfMemory errors (float between 0 and 1)
#                 selected_gpus=[0]) # List of GPU IDs on your computer you want to use (list) | Non-existing GPU IDs will cause the code to exit

# eval_gpu.evaluate()  # This triggers evaluate_gpu() with gpu mode

# gpu_finder = OptimalGPUConfigurationFinder(eval_sys_results_dir=csv_filepath, log_file_path=log_file_path)
# gpu_finder.find_best_overall_usecase()
# gpu_finder.find_optimal_for(num_cpus=20, gpu_list=[0], station_count=1)