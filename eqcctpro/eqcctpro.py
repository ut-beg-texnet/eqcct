
import os 
from eqcctpro import EQCCTMSeedRunner, EvaluateSystem, OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder
input_mseed_directory_path = '/home/skevofilaxc/workspace/eqcct/eqcctpro/20250304T115800Z_20250304T120100Z'   
output_pick_directory_path = '/home/skevofilaxc/workspace/eqcct/eqcctpro//outputs'
log_file_path = '/home/skevofilaxc/workspace/eqcct/eqcctpro/outputs/eqcctpro.log'
csv_filepath = '/home/skevofilaxc/workspace/eqcct/eqcctpro/csv'

# Can run EQCCT on a given input dir on GPU or CPU 
# Can also specify the number of stations you want to use as well  

# eqcct_runner = EQCCTMSeedRunner(use_gpu=True, 
#                 intra_threads=1, 
#                 inter_threads=1, 
#                 cpu_id_list=[0,1,2,3,4],
#                 input_dir=input_mseed_directory_path, 
#                 output_dir=output_pick_directory_path, 
#                 log_filepath=log_file_path, 
#                 P_threshold=0.001, 
#                 S_threshold=0.02, 
#                 p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', 
#                 s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5', 
#                 number_of_concurrent_predictions=5,
#                 best_usecase_config=True,
#                 csv_dir=csv_filepath,
#                 selected_gpus=[0],
#                 set_vram_mb=24750,
#                 specific_stations='AT01, BP01, DG05')

# eqcct_runner.run_eqcctpro()


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
                stations2use=230,
                cpu_id_list=range(87,127), 
                starting_amount_of_stations=50, 
                station_list_step_size=10,
                min_cpu_amount=15)
eval_cpu.evaluate()  # This triggers evaluate_cpu() if mode is 'cpu'

# cpu_finder = OptimalCPUConfigurationFinder(csv_filepath)
# best_cpu_config = cpu_finder.find_best_overall_usecase()
# print(best_cpu_config)


# optimal_cpu_config = cpu_finder.find_optimal_for(cpu=3, station_count=2)
# print(optimal_cpu_config)


# gpu_finder = OptimalGPUConfigurationFinder(csv_filepath)
# best_gpu_config = gpu_finder.find_best_overall_usecase()
# print(best_gpu_config)

# optimal_gpu_config = gpu_finder.find_optimal_for(num_cpus=1, gpu_list=[0], station_count=1)
# print(optimal_gpu_config)
