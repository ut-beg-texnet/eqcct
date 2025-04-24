
import os 
from eqcctpro import EQCCTMSeedRunner, EvaluateSystem, OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder
input_mseed_directory_path = '/home/skevofilaxc/workspace/eqcct/eqcctpro/230_stations_2hr_5_min_dt'   
output_pick_directory_path = '/home/skevofilaxc/workspace/eqcct/eqcctpro/outputs'
log_file_path = '/home/skevofilaxc/workspace/eqcct/eqcctpro/outputs/eqcctpro.log'
csv_filepath = '/home/skevofilaxc/workspace/eqcct/eqcctpro/csv'
tmp_dir = '/home/skevofilaxc/tmp'

# Can run EQCCT on a given input dir on GPU or CPU 
# Can also specify the number of stations you want to use as well  

# eqcct_runner = EQCCTMSeedRunner(use_gpu=False, 
#                 intra_threads=1, 
#                 inter_threads=1, 
#                 cpu_id_list=[127, 126, 125, 124],
#                 input_dir=input_mseed_directory_path, 
#                 output_dir=output_pick_directory_path, 
#                 log_filepath=log_file_path, 
#                 P_threshold=0.001, 
#                 S_threshold=0.02, 
#                 p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', 
#                 s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5', 
#                 number_of_concurrent_predictions=2,
#                 best_usecase_config=False,
#                 csv_dir=csv_filepath,
#                 selected_gpus=[0],
#                 set_vram_mb=24750,
#                 specific_stations='AT01, BP01, DG05',
#                 start_time='2024-12-15 12:00:00',
#                 end_time='2024-12-15 12:10:00',
#                 timechunk_dt=5, 
#                 waveform_overlap=2,
#                 number_of_concurrent_timechunk_predictions=2)

# eqcct_runner.run_eqcctpro()
# 'AT01, BP01, DG05'


eval_cpu = EvaluateSystem('cpu',
                intra_threads=1,
                inter_threads=1,
                input_dir=input_mseed_directory_path, 
                output_dir=output_pick_directory_path, 
                log_filepath=log_file_path,
                csv_dir=csv_filepath,
                P_threshold=0.001, 
                S_threshold=0.02, 
                p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', 
                s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5',
                stations2use=225,
                cpu_id_list=range(77,128),
                used_cpu_test_step_size=5, 
                starting_amount_of_stations=100, 
                station_list_step_size=5,
                min_cpu_amount=20,
                min_conc_stations=75,
                conc_station_tasks_step_size=5,
                start_time='2024-12-15 12:00:00',
                end_time='2024-12-15 14:00:00',
                conc_timechunk_tasks_step_size=4,
                timechunk_dt=5, 
                waveform_overlap=2,
                tmp_dir=tmp_dir)

eval_cpu.evaluate()  # This triggers evaluate_cpu() if mode is 'cpu'

# cpu_finder = OptimalCPUConfigurationFinder(csv_filepath)
# best_cpu_config = cpu_finder.find_best_overall_usecase()
# print(best_cpu_config)


# optimal_cpu_config = cpu_finder.find_optimal_for(cpu=5, station_count=1)
# print(optimal_cpu_config)


# gpu_finder = OptimalGPUConfigurationFinder(csv_filepath)
# best_gpu_config = gpu_finder.find_best_overall_usecase()
# print(best_gpu_config)

# optimal_gpu_config = gpu_finder.find_optimal_for(num_cpus=1, gpu_list=[0], station_count=1)
# print(optimal_gpu_config)
