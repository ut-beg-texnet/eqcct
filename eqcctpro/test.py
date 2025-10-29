import os 
from functionality import RunEQCCTPro # , EvaluateSystem, OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder
input_mseed_directory_path = '/home/skevofilaxc/workspace/eqcct/eqcctpro/230_stations_1_min_dt' # Change to local path 
output_pick_directory_path = '/home/skevofilaxc/workspace/eqcct/eqcctpro/outputs' # Change
log_file_path = '/home/skevofilaxc/workspace/eqcct/eqcctpro/outputs/eqcctpro.log' # Change
csv_filepath = '/home/skevofilaxc/workspace/eqcct/eqcctpro/csv/test_cpu' # Change
tmp_dir = '/home/skevofilaxc/tmp' # Change

# EQCCTPro can run EQCCT on a given input dir on either your GPU or CPU     
eqcct_runner = RunEQCCTPro(use_gpu=True, # Defines if you use the GPU to run EQCCTMSeedRunner (bool)
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
                selected_gpus=[0], # List of GPU IDs on your computer you want to use if use_gpu = True (list)
                vram_mb=500, # Maximum amount of VRAM each Raylet can use (float). vram_mb = (GPU VRAM * .95 (to be safe)) / number_of_concurrent_station_predictions * number_of_concurrent_timechunk_predictions
                start_time='2024-12-15 12:00:00', # The start time of the area of time that is being analyzed | Must follow the following convention YYYY-MO-DA HR:MI:SC (str)
                end_time='2024-12-15 12:01:00', # The end time of the area of time that is being analyzed | Must follow the following convention YYYY-MO-DA HR:MI:SC (str)
                timechunk_dt=1, # The length each time chunk is (in minutes) (int)
                waveform_overlap=0, # The duration (in minutes) for which each waveform overlaps with the others (int)
                specific_stations='AT01, BP01, DG05') # String that contains the "list" of stations you want to only analyze (str)


eqcct_runner.run_eqcctpro() # Runs EQCCTPro on one instance of waveform data
