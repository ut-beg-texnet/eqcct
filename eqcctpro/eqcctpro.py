
import os 
from eqcctpro import EQCCTMSeedRunner, EvaluateSystem, OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder
input_mseed_directory_path = '/home/skevofilaxc/eqcctpro/mseed/20241215T115800Z_20241215T120100Z'   
output_pick_directory_path = '/home/skevofilaxc/eqcctpro/outputs'
log_file_path = '/home/skevofilaxc/eqcctpro/outputs/eqcctpro.log'
csv_filepath = '/home/skevofilaxc/eqcctpro/csv'

# Can run EQCCT on a given input dir on GPU or CPU 
# Can also specify the number of stations you want to use as well  

eqcct_runner = EQCCTMSeedRunner(use_gpu=True, 
                intra_threads=1, 
                inter_threads=1, 
                cpu_id_list=[0,1,2,3,4],
                input_dir=input_mseed_directory_path, 
                output_dir=output_pick_directory_path, 
                log_filepath=log_file_path, 
                P_threshold=0.001, 
                S_threshold=0.02, 
                p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', 
                s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5', 
                number_of_concurrent_predictions=5,
                best_usecase_config=True,
                csv_dir=csv_filepath,
                selected_gpus=[0],
                set_vram_mb=24750,
                specific_stations='AT01, BP01, DG05')

eqcct_runner.run_eqcctpro()


# eval_gpu = EvaluateSystem('gpu',
#                 intra_threads=1,
#                 inter_threads=1,
#                 input_dir=input_mseed_directory_path, 
#                 output_dir=output_pick_directory_path, 
#                 log_filepath=log_file_path,
#                 csv_dir=csv_filepath,
#                 P_threshold=0.001, 
#                 S_threshold=0.02, 
#                 p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', 
#                 s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5',
#                 stations2use=2,
#                 cpu_id_list=range(0,1,2),
#                 set_vram_mb=24750, 
#                 selected_gpus=[0])
# eval_gpu.evaluate()  # This triggers evaluate_gpu() if mode is 'gpu'

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
