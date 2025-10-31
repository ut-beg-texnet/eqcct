"""
functionality.py controls all the functionality of EQCCTPro, specifically how we access mseed_predictor() and parallel_predict. 
It is a level of abstraction so we can make the code more concise and cleaner
"""
import os
import gc
import ray
import sys
import ast
import math
import queue 
import psutil
import random
import numbers
import logging
import resource
import threading
from .tools import *
from pathlib import Path
from .parallelization import *
from obspy import UTCDateTime
from ray.util.queue import Queue
from datetime import datetime, timedelta
from .tools import _parse_gpus_field
from logging.handlers import QueueHandler, QueueListener


class RunEQCCTPro():  
    """RunEQCCTPro class for running the RunEQCCTPro functions for multiple instances of the class"""
    def __init__(self, # self is 'this instance' of the class 
                use_gpu: bool, 
                input_dir: str, 
                output_dir: str, 
                log_filepath: str, 
                p_model_filepath: str, 
                s_model_filepath: str, 
                number_of_concurrent_station_predictions: int,
                number_of_concurrent_timechunk_predictions: int, 
                intra_threads: int = 1, 
                inter_threads: int = 1, 
                P_threshold: float = 0.001, 
                S_threshold: float = 0.02,
                specific_stations: str = None,
                csv_dir: str = None,
                best_usecase_config: bool = None,
                vram_mb: float = None,
                selected_gpus: list = None,
                cpu_id_list: list = [1],
                start_time:str = None, 
                end_time:str = None, 
                timechunk_dt:int = None,
                waveform_overlap:int = None): 
         
        self.use_gpu = use_gpu  # 'this instance' of the classes object, use_gpu = use_gpu 
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.log_filepath = log_filepath
        self.p_model_filepath = p_model_filepath
        self.s_model_filepath = s_model_filepath
        self.number_of_concurrent_station_predictions = number_of_concurrent_station_predictions
        self.number_of_concurrent_timechunk_predictions = number_of_concurrent_timechunk_predictions
        self.intra_threads = intra_threads
        self.inter_threads = inter_threads
        self.P_threshold = P_threshold
        self.S_threshold = S_threshold
        self.specific_stations = specific_stations
        self.csv_dir = csv_dir
        self.best_usecase_config = best_usecase_config
        self.vram_mb = vram_mb
        self.selected_gpus = selected_gpus # a list of the GPU IDs 
        self.cpu_id_list = cpu_id_list 
        self.cpu_count = len(cpu_id_list)
        self.start_time = start_time
        self.end_time = end_time
        self.timechunk_dt = timechunk_dt
        self.waveform_overlap = waveform_overlap  

        # Ensures that the output_dir exists. If it doesn't, we create it 
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up main logger and logger queue to retrive queued logs from Raylets to be passed to the main logger
        self.logger = logging.getLogger("eqcctpro") # We named the logger eqcctpro (can be any name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False # if true, events logged to this logger will be passed to the handlers of higher level (ancestor) loggers, in addition to any handlers attached to this logger
        if not self.logger.handlers: # avoid duplicating inits 
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_h = logging.FileHandler(self.log_filepath) # Writes logs to file 
            stream_h = logging.StreamHandler() # Sends logs to console
            file_h.setFormatter(fmt)
            stream_h.setFormatter(fmt)
            self.logger.addHandler(file_h)
            self.logger.addHandler(stream_h)

        self.logger.info("")
        self.logger.info(f"------- Welcome to EQCCTPro -------")
        self.logger.info("")

        # If the user passed a GPU but no valid VRAM, need to exit 
        if self.use_gpu and not (isinstance(self.vram_mb, numbers.Real) and math.isfinite(self.vram_mb) and self.vram_mb > 0): 
            self.logger.error(f"No numerical VRAM passed. Please provide vram_mb (MB per Raylet per GPU) as a positive real number. Exiting...")
            sys.exit(1)

        # We need to ensure that the vram specified does not exceed the capabilities of the system, if not, we need to exit safely before it happens
        if self.use_gpu: 
            check_vram_per_gpu_style(
                selected_gpus=self.selected_gpus,
                get_gpu_vram_fn=lambda gid: get_gpu_vram(gpu_index=gid),
                intended_workers=self.number_of_concurrent_station_predictions * self.number_of_concurrent_timechunk_predictions,
                vram_mb=self.vram_mb,
                model_vram_mb=1500.0,   # your safety reserve for EQCCT
                safety_cap=0.95,
                eqcct_overhead_gb=0.0,
                logger=self.logger)
    
    # To-Do: merge dt_task_generator and chunk_time into one function and concatenate the objects so we dont have so much stuff running around
    # Generates the dt tasks list 
    def dt_task_generator(self): 
        # Modifies the times_list values (see chunk_time()) so it can be in a format the mseed_predictor can use 
        tasks = [[f"({i+1}/{len(self.times_list)})", f"{self.times_list[i][0].strftime(format='%Y%m%dT%H%M%SZ')}_{self.times_list[i][1].strftime(format='%Y%m%dT%H%M%SZ')}"] for i in range((len(self.times_list)))]
        self.tasks_picker = tasks
    
    def chunk_time(self):
        # Creates the timechunks, EI. from X specific time to Y specific time to generate the dt tasks (timechunk tasks that are run in parallel first at the top level)
        # EX. [[UTCDateTime(2024, 12, 15, 11, 58), UTCDateTime(2024, 12, 15, 13, 0)], [UTCDateTime(2024, 12, 15, 12, 58), UTCDateTime(2024, 12, 15, 14, 0)]]
        starttime = UTCDateTime(self.start_time) - (self.waveform_overlap * 60)
        endtime = UTCDateTime(self.end_time)

        times_list = []
        start = starttime
        end = start + (self.waveform_overlap * 60) + (self.timechunk_dt * 60)
        while start <= endtime:
            if end >= endtime:
                end = endtime
                times_list.append([start, end])
                break
            times_list.append([start, end])
            start = end - (self.waveform_overlap * 60)
            end = start + (self.waveform_overlap * 60) + (self.timechunk_dt * 60)

        self.times_list = times_list
    
    def _drain_worker_logs(self):
            while True:
                rec = self.log_queue.get()  # blocks until a record arrives
                if rec is None: break       # sentinel to stop thread
                try:
                    self.logger.handle(rec) # routes to file+console handlers
                except Exception:
                    # never crash on logging
                    self.logger.exception("Failed to handle worker log record")

    def configure_cpu(self): 
        # We need to configure the tf_environ for the CPU configuration that is being inputted
        self.logger.info(f"Running EQCCT over Requested MSeed Files using CPU(s)...")
        if self.best_usecase_config:
            # We use the best usecase configuration that was found using EvaluateSystem
            result = find_optimal_configuration_cpu(best_overall_usecase=True, eval_sys_results_dir=self.csv_dir)
            if result is None: 
                self.logger.info("")
                self.logger.info(f"Error: Could not retrieve an optimal CPU configuration. Please check that the CSV file exists and try again. Exiting...")
                exit()  # Exit gracefully
            cpus_to_use, num_concurrent_predictions, intra, inter, station_count = result
            self.logger.info("")
            self.logger.info(f"Using {cpus_to_use} CPUs, {num_concurrent_predictions} Conc. Predictions, {intra} Intra Threads, and {inter} Inter Threads...")
            tf_environ(gpu_id=-1, intra_threads=intra, inter_threads=inter, logger=self.logger)
        else:
            # We pass the requested parameters to the tf_environ 
            tf_environ(gpu_id=-1, intra_threads=self.intra_threads, inter_threads=self.inter_threads, logger=self.logger) 
            
    def configure_gpu(self):
        # We need to configure the tf_environ for the GPU configuration that is being inputted
        self.logger.info(f"Running EQCCT over Requested MSeed Files using GPU(s)...")
        if self.best_usecase_config: 
            result = find_optimal_configuration_gpu(True, self.csv_dir)
            if result is None:
                self.logger.info("")
                self.logger.error(f"Error: Could not retrieve an optimal GPU configuration. Please check that the CSV file exists and try again. Exiting...")
                exit()  # Exit gracefully

            self.logger.info("")
            cpus_to_use, num_concurrent_predictions, intra, inter, gpus, vram_mb, station_count = result # Unpack values only if result is valid
            self.logger.info(f"Using {cpus_to_use} CPUs, {num_concurrent_predictions} Conc. Predictions, {intra} Intra Threads, {inter} Inter Threads, {gpus} GPU IDs, and {vram_mb} MB VRAM per Task...")
            tf_environ(gpu_id=1, vram_limit_mb=vram_mb, gpus_to_use=gpus, intra_threads=intra, inter_threads=inter, logger=self.logger)
        
        else: 
            self.logger.info("")
            self.logger.info(f"User requested to use GPU(s): {self.selected_gpus} with {self.vram_mb} MB of VRAM per Raylet (intra-op threads = {self.intra_threads}, inter-op threads = {self.inter_threads})") # Use the selected GPUs 
            tf_environ(gpu_id=1, vram_limit_mb=self.vram_mb, gpus_to_use=self.selected_gpus, intra_threads=self.intra_threads, inter_threads=self.inter_threads, logger=self.logger)
    
    def eqcctpro_parallelization(self):

        if self.specific_stations is None: # We check if the station dirs are consistent, if not, exit
            statement, specific_stations_list, do_i_exit = check_station_dirs(input_dir=self.input_dir)
            self.logger.info(f"{statement}")
            if do_i_exit: exit()

        # We want to use a specified amount of stations
        else: specific_stations_list = [station.strip() for station in self.specific_stations.split(',')]
        statement = f"Using {len(specific_stations_list)} selected station(s)."
        self.logger.info(f"{statement}")
        self.logger.info("")           

        # Submit timechunk tasks to mseed_predictor
        tasks_queue = []
        log_queue = queue.Queue()  # Create a queue for log entries
        
        # Compute total analyis timeframe 
        total_analysis_time = datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
        
        max_pending_tasks = self.number_of_concurrent_timechunk_predictions 
        self.logger.info(f"------- Starting EQCCTPro... -------")
        self.logger.info(f"Detailed subprocess information can be found in the log file.")
        self.logger.info("")
        for i in range(len(self.tasks_picker)):
            mseed_timechunk_dir_name = self.tasks_picker[i][1]
            timechunk_dir_path = os.path.join(self.input_dir, mseed_timechunk_dir_name) 
        
            # Concurrent Timechunks 
            while True: 
                if len(tasks_queue) < max_pending_tasks: 
                    tasks_queue.append(mseed_predictor.options(num_gpus=0, num_cpus=1).remote(input_dir=timechunk_dir_path, output_dir=self.output_dir, log_queue=self.log_queue, 
                                        P_threshold=self.P_threshold, S_threshold=self.S_threshold, p_model=self.p_model_filepath, s_model=self.s_model_filepath, 
                                        number_of_concurrent_station_predictions=self.number_of_concurrent_station_predictions, ray_cpus=self.cpu_id_list, use_gpu=self.use_gpu, 
                                        gpu_id=self.selected_gpus, gpu_memory_limit_mb=self.vram_mb, specific_stations=specific_stations_list, 
                                        timechunk_id=mseed_timechunk_dir_name, waveform_overlap=self.waveform_overlap, total_timechunks=len(self.tasks_picker), 
                                        number_of_concurrent_timechunk_predictions=self.number_of_concurrent_timechunk_predictions, total_analysis_time=total_analysis_time,
                                        intra_threads=self.intra_threads, inter_threads=self.inter_threads))
                    break
                
                else: # If there are more tasks than maximum, just process them
                    tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
                    for finished_task in tasks_finished:
                        log_entry = ray.get(finished_task)
                        log_queue.put(log_entry)  # Add log entry to the queue

        # After adding all the tasks to queue, process what's left
        while tasks_queue:
            tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
            for finished_task in tasks_finished:
                log_entry = ray.get(finished_task)
                self.logger.info(log_entry)

        # stop log forwarder
        self.log_queue.put(None) # remember, log_queue is a Ray Queue actor, and will only exist while Ray is still active (cannot be after the .shutdown())
        self._log_thread.join(timeout=2)

        ray.shutdown()
        self.logger.info(f"Ray Successfully Shutdown.")
        self.logger.info("------- Successfully Picked All Waveform(s) from all Timechunk(s) -------")
        # self.logger.info("------- END OF FILE -------")
        
    def run_eqcctpro(self):
        # Set CPU affinity
        process = psutil.Process(os.getpid())
        process.cpu_affinity(self.cpu_id_list)  # Limit process to the given CPU IDs
        
        self.chunk_time() # Generates the UTC times for each of the timesets in the given time range 
        self.dt_task_generator() # Generates the task list so can know how many total tasks there are for our given time range 
        
        if self.use_gpu: # GPU
            self.configure_gpu()
            ray.init(ignore_reinit_error=True, num_gpus=len(self.selected_gpus), num_cpus=len(self.cpu_id_list), logging_level=logging.ERROR, log_to_driver=False) # Ray initalization using GPUs 
            self.log_queue = Queue() # Create a Ray-safe queue to recieve LogRecord objects from workers so we can write them to file 
            self._log_thread = threading.Thread(target=self._drain_worker_logs, daemon=True) # Creates background thread whose only job is to get() records from self.log_queue and hand them over to the actual logger
            self._log_thread.start() # Starts the thread
            # Log some import info to user 
            statement = f"Ray Successfully Initialized with {self.selected_gpus} GPU(s) and {len(self.cpu_id_list)} CPU(s)."
            self.logger.info(f"{statement}")
            self.logger.info(f"Analyzing {len(self.times_list)} time chunk(s) from {self.start_time} to {self.end_time} (dt={self.timechunk_dt}min, overlap={self.waveform_overlap}min).")
            
            # Running parllelization
            self.eqcctpro_parallelization()

        else: # CPU
            self.configure_cpu()
            ray.init(ignore_reinit_error=True, num_cpus=len(self.cpu_id_list), logging_level=logging.ERROR, log_to_driver=False) # Ray initalization using CPUs
            self.log_queue = Queue() # Create a Ray-safe queue to recieve LogRecord objects from workers so we can write them to file 
            self._log_thread = threading.Thread(target=self._drain_worker_logs, daemon=True) # Creates background thread whose only job is to get() records from self.log_queue and hand them over to the actual logger
            self._log_thread.start() # Starts the thread
            # Log some import info to user
            statement = f"Ray Successfully Initialized with {len(self.cpu_id_list)} CPU(s)."
            self.logger.info(f"{statement}")
            self.logger.info(f"Analyzing {len(self.times_list)} time chunk(s) from {self.start_time} to {self.end_time} (dt={self.timechunk_dt}min, overlap={self.waveform_overlap}min).")
            
            # Running parllelization
            self.eqcctpro_parallelization()

class EvaluateSystem(): 
    """Evaluate System class for running the evaluation system functions for multiple instances of the class"""
    def __init__(self,
                 eval_mode: str,
                 input_dir: str,
                 output_dir: str,
                 log_filepath: str,
                 csv_dir: str, 
                 p_model_filepath: str, 
                 s_model_filepath: str, 
                 P_threshold: float = 0.001, 
                 S_threshold: float = 0.02, 
                 intra_threads: int = 1,
                 inter_threads: int = 1,
                 stations2use:int = None, 
                 cpu_id_list:list = [1],
                 cpu_test_step_size:int = 1, 
                 starting_amount_of_stations: int = 1, 
                 station_list_step_size: int = 1, 
                 min_cpu_amount: int = 1,
                 min_conc_stations: int = 1, 
                 conc_station_tasks_step_size: int = 1,
                 vram_mb:float = None, 
                 selected_gpus:list = None,
                 start_time:str = None, 
                 end_time:str = None, 
                 conc_timechunk_tasks_step_size: int = 1, 
                 timechunk_dt:int = None,
                 waveform_overlap:int = None,
                 tmp_dir:str = None): 
        
        valid_modes = {"cpu", "gpu"}
        if eval_mode not in valid_modes: 
            raise ValueError(f"Invalid mode '{eval_mode}'. Choose either 'cpu' or 'gpu'.")
        
        self.eval_mode = eval_mode.lower()
        self.intra_threads = intra_threads
        self.inter_threads = inter_threads
        self.input_dir = input_dir  
        self.output_dir = output_dir
        self.log_filepath = log_filepath
        self.csv_dir = csv_dir
        self.P_threshold = P_threshold
        self.S_threshold = S_threshold
        self.p_model_filepath = p_model_filepath
        self.s_model_filepath = s_model_filepath
        self.stations2use = stations2use
        self.cpu_id_list = cpu_id_list
        self.vram_mb = vram_mb
        self.selected_gpus = selected_gpus
        self.cpu_count = len(cpu_id_list)
        self.cpu_test_step_size = cpu_test_step_size
        self.starting_amount_of_stations = starting_amount_of_stations
        self.station_list_step_size = station_list_step_size
        self.min_cpu_amount = min_cpu_amount
        self.min_conc_stations = min_conc_stations # default is = 1 
        self.conc_station_tasks_step_size = conc_station_tasks_step_size # default is = 1 
        self.stations2use_list = list(range(1, 11)) + list(range(15, 50, 5)) if stations2use is None else generate_station_list(self.starting_amount_of_stations, stations2use, self.station_list_step_size,)
        self.start_time = start_time
        self.end_time = end_time
        self.conc_timechunk_tasks_step_size = conc_timechunk_tasks_step_size
        self.timechunk_dt = timechunk_dt
        self.waveform_overlap = waveform_overlap
        self.home_tmp_dir = tmp_dir 
        
        # Ensures that the output_dir exists. If it doesn't, we create it 
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up main logger and logger queue to retrive queued logs from Raylets to be passed to the main logger
        self.logger = logging.getLogger("eqcctpro") # We named the logger eqcctpro (can be any name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False # if true, events logged to this logger will be passed to the handlers of higher level (ancestor) loggers, in addition to any handlers attached to this logger
        if not self.logger.handlers: # avoid duplicating inits 
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_h = logging.FileHandler(self.log_filepath) # Writes logs to file 
            stream_h = logging.StreamHandler() # Sends logs to console
            file_h.setFormatter(fmt)
            stream_h.setFormatter(fmt)
            self.logger.addHandler(file_h)
            self.logger.addHandler(stream_h)
        
        self.logger.info("")
        self.logger.info(f"------- Welcome to EQCCTPro's EvaluateSystem Functionality -------")
        self.logger.info("")
        # Set up temp dir 
        import tempfile
        tempfile.tempfile = self.home_tmp_dir

        os.environ['TMPDIR'] = self.home_tmp_dir
        os.environ['TEMP'] = self.home_tmp_dir
        os.environ['TMP'] = self.home_tmp_dir
        self.logger.info(f"Successfully set up temp files to be stored at {self.home_tmp_dir}")

        # We need to ensure that the vram specified does not exceed the capabilities of t  he system, if not, we need to exit safely before it happens
        self.chunk_time()
        intended_workers = int(len(self.stations2use_list)) * int(len(self.times_list) // 2)
        if self.eval_mode == 'gpu':
            if not self.selected_gpus:
                raise ValueError("selected_gpus must be set in GPU mode.")
            self.chunk_time()
            intended_workers = int(len(self.stations2use_list)) * int(len(self.times_list) // 2)

            per_gpu_free_mb = [get_gpu_vram(gpu_index=g)[1] * 1024.0 for g in self.selected_gpus]  # free_gb -> MB
            plan = evaluate_vram_capacity(
                intended_workers=intended_workers,
                vram_per_worker_mb=float(self.vram_mb),
                per_gpu_free_mb=per_gpu_free_mb,
                model_vram_mb=3000.0,
                safety_cap=0.90,
                eqcct_overhead_gb=1.1,
            )
            if not plan.ok_aggregate:
                unit = plan.per_worker_mb + plan.overhead_mb
                raise RuntimeError(
                    f"Insufficient aggregate VRAM. Cap={plan.aggregate_cap_mb:.0f} MB, "
                    f"Need={plan.aggregate_need_mb:.0f} MB (= {plan.model_vram_mb:.0f}×{len(self.selected_gpus)} + "
                    f"{plan.intended_workers}×{unit:.0f})."
                )
            self.logger.info(
                f"VRAM budget OK. Need {plan.aggregate_need_mb:.0f} MB ≤ Cap {plan.aggregate_cap_mb:.0f} MB "
                f"across {len(self.selected_gpus)} GPU(s)."
            )
        
    def _generate_stations_list(self):
        """Generates station list"""
        if self.station2use is None: 
            return list(range(1, 11)) + list(range(15, 50, 5))
        return generate_station_list(self.stations2use, self.starting_amount_of_stations, self.station_list_step_size)
    
    # def _prepare_environment(self):
    #     """Removed 'output_dir' so that there is no conflicts in the save for a clean output return"""
    #     remove_directory(self.output_dir)
        
    def chunk_time(self):
        starttime = UTCDateTime(self.start_time) - (self.waveform_overlap * 60)
        endtime = UTCDateTime(self.end_time)

        times_list = []
        start = starttime
        end = start + (self.waveform_overlap * 60) + (self.timechunk_dt * 60)
        while start <= endtime:
            if end >= endtime:
                end = endtime
                times_list.append([start, end])
                break
            times_list.append([start, end])
            start = end - (self.waveform_overlap * 60)
            end = start + (self.waveform_overlap * 60) + (self.timechunk_dt * 60)

        self.times_list = times_list

    def _drain_worker_logs(self):
            while True:
                rec = self.log_queue.get()  # blocks until a record arrives
                if rec is None: break       # sentinel to stop thread
                try:
                    self.logger.handle(rec) # routes to file+console handlers
                except Exception:
                    # never crash on logging
                    self.logger.exception("Failed to handle worker log record")
    
    def dt_task_generator(self): 
        tasks = [[f"({i+1}/{len(self.times_list)})", f"{self.times_list[i][0].strftime(format='%Y%m%dT%H%M%SZ')}_{self.times_list[i][1].strftime(format='%Y%m%dT%H%M%SZ')}"] for i in range((len(self.times_list)))]
        self.tasks_picker = tasks
        
    def evaluate_cpu(self): 
        """Evaluate system parallelization using CPUs"""
        statement = "Evaluating System Parallelization Capability using CPU"
        self.logger.info(f"{statement}")
        
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test results csv 
        csv_filepath = f"{self.csv_dir}/cpu_test_results.csv"
        prepare_csv(csv_file_path=csv_filepath, logger=self.logger)
        
        self.chunk_time()
        self.dt_task_generator()
        
        trial_num = 1
        log_queue = queue.Queue()  # Create a queue for log entries
        total_analysis_time = datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
        
        if self.eval_mode == 'gpu': 
            use_gpu = True 
        else: 
            use_gpu = False 

        if self.min_cpu_amount > len(self.cpu_id_list): 
            # Code won't execute because the minimum CPU amount of > the len(cpu id list)
            # In which the rest of the code is dependent on the len for generating cpu_count 
            print(f"CPU ID List provided has less CPUs than the minimum requested ({len(self.cpu_id_list)} vs. {self.min_cpu_amount}). Exiting...")
            quit()
        
        with open(self.log_filepath, mode="a+", buffering=1) as log: 
            for i in range(self.min_cpu_amount, self.cpu_count+1, self.cpu_test_step_size):
                # Set CPU affinity and initialize Ray
                cpus_to_use = self.cpu_id_list[:i]
                process = psutil.Process(os.getpid())
                process.cpu_affinity(cpus_to_use)  # Limit process to the given CPU IDs
                
                ray.init(ignore_reinit_error=True, num_cpus=len(cpus_to_use), logging_level=logging.FATAL, log_to_driver=False)
                self.log_queue = Queue() # Create a Ray-safe queue to recieve LogRecord objects from workers so we can write them to file 
                self._log_thread = threading.Thread(target=self._drain_worker_logs, daemon=True) # Creates background thread whose only job is to get() records from self.log_queue and hand them over to the actual logger
                self._log_thread.start() # Starts the thread
                self.logger.info(f"Ray Successfully Initialized with {len(cpus_to_use)} CPU(s).")
                
                timechunks_list = []
                timechunk = 1
                step = self.conc_timechunk_tasks_step_size # Use the class attribute
                while timechunk <= len(self.tasks_picker):
                    timechunks_list.append(timechunk)
                    if timechunk == 1:
                        timechunk += 1
                    else:
                        timechunk += step

                if len(self.tasks_picker) not in timechunks_list:
                    timechunks_list.append(len(self.tasks_picker))
                # sets are a set of multiple items stored in a single variable 
                # unchangable after being set, cannot have duplicates and is unordered
                timechunks_list = sorted(list(set(timechunks_list))) 
                for timechunks in timechunks_list:
                    tested_concurrency = set() # Rest for each cpu / timechunk
                    for num_stations in self.stations2use_list: 
                        concurrent_predictions_list = generate_station_list(self.min_conc_stations, num_stations, self.conc_station_tasks_step_size)
                        # We do this so that we don't repeat concurrent prediction tests 
                        # Because a number of concurrent predictions running can be equivilated to the number of total stations that need to be processed
                        # There is no need to duplicate more tests that will be doing the same amount of concurrent testing for a different number of total stations
                        new_concurrent_values = [x for x in concurrent_predictions_list if x not in tested_concurrency and x <= num_stations]
                        if not new_concurrent_values:
                            continue  # All concurrency values already tested
                        for num_concurrent_predictions in new_concurrent_values:           
                            mseed_timechunk_dir_name = self.tasks_picker[timechunks-1][1]
                            timechunk_dir_path = os.path.join(self.input_dir, mseed_timechunk_dir_name) 
                            max_pending_tasks = timechunks
                            
                            self.logger.info("")
                            self.logger.info(f"------- Trial Number: {trial_num} -------")
                            self.logger.info(f"CPU(s): {i}")
                            self.logger.info(f"Conc. Timechunks Being Analyzed: {timechunks} / Total Timechunks to be Analyzed: {len(self.tasks_picker)}")
                            self.logger.info(f"Total Amount of Stations to be Processed in Current Trial: {num_stations} / Number of Stations Being Processed Concurrently: {num_concurrent_predictions} / Total Overall Trial Station Count: {max(self.stations2use_list)}") 
                            
                            # Concurrent Timechunks
                            tasks_queue = []
                            log_queue = queue.Queue()  # Create a queue for log entries


                            # ===== RAM Baseline (before launching worker) =====
                            _rss = process.memory_info().rss
                            for _ch in process.children(recursive=True):
                                try:
                                    _rss += _ch.memory_info().rss
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                            mem_before_total_mb = _rss / 1e6

                            # peak before (platform-aware)
                            if resource is not None:  # Linux/macOS
                                _ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                if sys.platform.startswith("linux"):
                                    peak_before_mb = _ru / 1024.0            # ru_maxrss in KB on Linux
                                elif sys.platform == "darwin":
                                    peak_before_mb = _ru / (1024.0 * 1024.0) # ru_maxrss in bytes on macOS
                                else:
                                    peak_before_mb = mem_before_total_mb     # safe fallback
                            else:  # Windows: no 'resource'
                                try:
                                    peak_before_mb = process.memory_full_info().peak_wset / 1e6
                                except Exception:
                                    peak_before_mb = mem_before_total_mb

                            try: 
                                while True: 
                                    if len(tasks_queue) < max_pending_tasks: 
                                        tasks_queue.append(mseed_predictor.options(num_gpus=0, num_cpus=1).remote(input_dir=timechunk_dir_path, output_dir=self.output_dir, log_queue=self.log_queue, 
                                                            P_threshold=self.P_threshold, S_threshold=self.S_threshold, p_model=self.p_model_filepath, s_model=self.s_model_filepath, 
                                                            number_of_concurrent_station_predictions=num_concurrent_predictions, ray_cpus=cpus_to_use, use_gpu=use_gpu, 
                                                            gpu_id=self.selected_gpus, gpu_memory_limit_mb=self.vram_mb, stations2use=num_stations, 
                                                            timechunk_id=mseed_timechunk_dir_name, waveform_overlap=self.waveform_overlap, total_timechunks=len(self.tasks_picker), 
                                                            number_of_concurrent_timechunk_predictions=max_pending_tasks, total_analysis_time=total_analysis_time, testing_gpu=False, 
                                                            test_csv_filepath=csv_filepath, intra_threads=self.intra_threads, inter_threads=self.inter_threads, timechunk_dt=self.timechunk_dt))
                                    
                                        break
                                
                                    else: 
                                        tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
                                        for finished_task in tasks_finished:
                                            log_entry = ray.get(finished_task)
                                            log_queue.put(log_entry)  # Add log entry to the queue
                                
                                # After adding all the tasks to queue, process what's left
                                while tasks_queue:
                                    tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
                                    for finished_task in tasks_finished:
                                        log_entry = ray.get(finished_task)
                                        log_queue.put(log_entry)  # Add log entry to the queue

                                    update_csv(csv_filepath, success=1, error_message="")
                            except Exception as e:
                                # Failure occured, need to add to log 
                                error_msg = f"{type(e).__name__}: {str(e)}"
                                update_csv(csv_filepath, success=0, error_message=error_msg)
                                self.logger.error(f"Trial {trial_num} FAILED: {error_msg}")
                                
                            # Write log entries from the queue to the file
                            while not log_queue.empty():
                                log_entry = log_queue.get()
                                
                            remove_output_subdirs(self.output_dir, logger=self.logger)
                            trial_num += 1  
                            
                            # RAM cleanup
                            # ===== AFTER RUN (before cleanup) =====
                            _rss = process.memory_info().rss
                            for _ch in process.children(recursive=True):
                                try:
                                    _rss += _ch.memory_info().rss
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                            mem_after_run_total_mb = _rss / 1e6
                            delta_run_mb = mem_after_run_total_mb - mem_before_total_mb

                            # updated peak (platform-aware)
                            if resource is not None:
                                _ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                if sys.platform.startswith("linux"):
                                    peak_after_mb = _ru / 1024.0
                                elif sys.platform == "darwin":
                                    peak_after_mb = _ru / (1024.0 * 1024.0)
                                else:
                                    peak_after_mb = mem_after_run_total_mb
                            else:
                                try:
                                    peak_after_mb = process.memory_full_info().peak_wset / 1e6
                                except Exception:
                                    peak_after_mb = mem_after_run_total_mb

                            self.logger.info("")
                            self.logger.info(
                                f"[MEM] Baseline: {mem_before_total_mb:.2f} MB | After run: {mem_after_run_total_mb:.2f} MB "
                                f"| Δrun: {delta_run_mb:.2f} MB | Peak≈{max(peak_before_mb, peak_after_mb):.2f} MB"
                            )

                            # ===== CLEANUP =====
                            # drop strong refs so GC matters
                            try: del ref
                            except NameError: pass
                            try: del log_entry
                            except NameError: pass

                            _rss = process.memory_info().rss
                            for _ch in process.children(recursive=True):
                                try:
                                    _rss += _ch.memory_info().rss
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                            mem_before_clean_mb = _rss / 1e6

                            gc.collect()
                            time.sleep(0.1)

                            _rss = process.memory_info().rss
                            for _ch in process.children(recursive=True):
                                try:
                                    _rss += _ch.memory_info().rss
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                            mem_after_clean_mb = _rss / 1e6

                            freed_mb = mem_before_clean_mb - mem_after_clean_mb
                            self.logger.info(f"[MEM] Freed ~{max(freed_mb, 0):.2f} MB; Post-clean total: {mem_after_clean_mb:.2f} MB") # To-Do: Fix the Freed so its beeter (for cpu and gpu)
                            self.logger.info("")
                            
                        # tested_concurrency.update([x for x in concurrent_predictions_list if x <= num_stations])

                    # stop log forwarder
                    self.log_queue.put(None) # remember, log_queue is a Ray Queue actor, and will only exist while Ray is still active (cannot be after the .shutdown())
                    self._log_thread.join(timeout=2)

                    ray.shutdown() # Shutdown Ray after processing all timechunks for this CPU count 
                    self.logger.info(f"Ray Successfully Shutdown.")
                                
     
        self.logger.info(f"Testing complete.")
        self.logger.info(f"")
        self.logger.info(f"Finding Optimal Configurations...")
        # Compute optimal configurations (CPU)
        df = pd.read_csv(csv_filepath)
        optimal_configuration_df, best_overall_usecase_df = find_optimal_configurations_cpu(df)
        optimal_configuration_df.to_csv(f"{self.csv_dir}/optimal_configurations_cpu.csv", index=False)
        best_overall_usecase_df.to_csv(f"{self.csv_dir}/best_overall_usecase_cpu.csv", index=False)
        self.logger.info(f"Optimal Configurations Found. Findings saved to:") 
        self.logger.info(f" 1) Optimal CPU/Station/Concurrent Prediction Configurations: {self.csv_dir}/optimal_configurations_cpu.csv") 
        self.logger.info(f" 2) Best Overall Usecase Configuration: {self.csv_dir}/best_overall_usecase_cpu.csv")

    def evaluate_gpu(self): 
        """Evaluate system parallelization using GPUs"""
        statement = "Evaluating System Parallelization Capability using GPUs"
        self.logger.info(f"{statement}")
        
        # Set CPU affinity
        process = psutil.Process(os.getpid())
        process.cpu_affinity(self.cpu_id_list)  # Limit process to the given CPU IDs
        
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calculate these at the start
        self.chunk_time()
        self.dt_task_generator()
        total_analysis_time = datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
            
        # Create test results csv 
        csv_filepath = f"{self.csv_dir}/gpu_test_results.csv"
        prepare_csv(csv_file_path=csv_filepath, logger=self.logger)
        
        free_vram_mb = self.vram_mb if self.vram_mb else self.calculate_vram()
        self.selected_gpus = self.selected_gpus if self.selected_gpus else list_gpu_ids()
        self.logger.info(f"Using GPU(s): {self.selected_gpus}")
        
        trial_num = 1
        log_queue = queue.Queue()  # Create a queue for log entries
        
        # Initialize Ray with GPUs
        ray.init(ignore_reinit_error=True, num_gpus=len(self.selected_gpus), num_cpus=len(self.cpu_id_list), 
                logging_level=logging.FATAL, log_to_driver=False)
        self.log_queue = Queue() # Create a Ray-safe queue to recieve LogRecord objects from workers so we can write them to file 
        self._log_thread = threading.Thread(target=self._drain_worker_logs, daemon=True) # Creates background thread whose only job is to get() records from self.log_queue and hand them over to the actual logger
        self._log_thread.start() # Starts the thread
        self.logger.info(f"Ray Successfully Initialized with {len(self.selected_gpus)} GPU(s) and {len(self.cpu_id_list)} CPU(s).")
        
        for stations in self.stations2use_list:
            concurrent_predictions_list = generate_station_list(self.min_conc_stations, stations, self.conc_station_tasks_step_size)
            for predictions in concurrent_predictions_list:
                vram_per_task_mb = free_vram_mb / predictions
                step_size = vram_per_task_mb * 0.05
                vram_steps = np.arange(step_size, vram_per_task_mb + step_size, step_size)
                self.logger.info(f"Testing the following VRAM limitations (MB): {vram_steps}")
                
                for gpu_memory_limit_mb in vram_steps:
                    
                    self.logger.info("")
                    self.logger.info(f"------- Trial Number: {trial_num} -------")
                    self.logger.info(f"VRAM Limited to {gpu_memory_limit_mb:.2f} MB per Task")
                    
                    # Get the first timechunk for testing
                    mseed_timechunk_dir_name = self.tasks_picker[0][1]
                    timechunk_dir_path = os.path.join(self.input_dir, mseed_timechunk_dir_name)
                    
                    self.logger.info(f"Stations: {stations}")
                    self.logger.info(f"Concurrent Station Predictions: {predictions}")
                    self.logger.info(f"VRAM per Task: {gpu_memory_limit_mb:.2f} MB")
                    self.logger.info("")


                    # ===== Baseline RAM consumption (before launching worker) =====
                    _rss = process.memory_info().rss
                    for _ch in process.children(recursive=True):
                        try:
                            _rss += _ch.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    mem_before_total_mb = _rss / 1e6

                    # peak before (platform-aware)
                    if resource is not None:  # Linux/macOS
                        _ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        if sys.platform.startswith("linux"):
                            peak_before_mb = _ru / 1024.0            # ru_maxrss in KB on Linux
                        elif sys.platform == "darwin":
                            peak_before_mb = _ru / (1024.0 * 1024.0) # ru_maxrss in bytes on macOS
                        else:
                            peak_before_mb = mem_before_total_mb     # safe fallback
                    else:  # Windows: no 'resource'
                        try:
                            peak_before_mb = process.memory_full_info().peak_wset / 1e6
                        except Exception:
                            peak_before_mb = mem_before_total_mb
                    
                    try:
                        # Call mseed_predictor directly via Ray (just like evaluate_cpu does)
                        ref = mseed_predictor.options(num_gpus=0, num_cpus=1).remote(
                            input_dir=timechunk_dir_path, 
                            output_dir=self.output_dir, 
                            log_queue=self.log_queue, 
                            P_threshold=self.P_threshold, 
                            S_threshold=self.S_threshold, 
                            p_model=self.p_model_filepath, 
                            s_model=self.s_model_filepath, 
                            number_of_concurrent_station_predictions=predictions, 
                            ray_cpus=self.cpu_id_list, 
                            use_gpu=True, 
                            gpu_id=self.selected_gpus, 
                            gpu_memory_limit_mb=gpu_memory_limit_mb, 
                            stations2use=stations, 
                            timechunk_id=mseed_timechunk_dir_name, 
                            waveform_overlap=self.waveform_overlap, 
                            total_timechunks=len(self.tasks_picker), 
                            number_of_concurrent_timechunk_predictions=1,  # Testing one timechunk at a time
                            total_analysis_time=total_analysis_time, 
                            testing_gpu=True,  # Enable test mode
                            test_csv_filepath=csv_filepath, 
                            intra_threads=self.intra_threads, 
                            inter_threads=self.inter_threads, 
                            timechunk_dt=self.timechunk_dt
                        )
                        
                        # Wait for result
                        log_entry = ray.get(ref)
                        log_queue.put(log_entry)  # Add log entry to the queue
                        
                        # Success - update CSV
                        update_csv(csv_filepath, success=1, error_message="")
                        
                    except Exception as e:
                        # Failure occurred, need to add to log 
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        update_csv(csv_filepath, success=0, error_message=error_msg)
                        self.logger.info(f"Trial {trial_num} FAILED: {error_msg}")
                    
                    # Write log entries from the queue to the file
                    while not log_queue.empty():
                        log_entry = log_queue.get()
                        self.logger.info(f"{log_entry}") # FIX ME 
                    
                    remove_output_subdirs(self.output_dir, logger=self.logger) 
                    trial_num += 1
                    
                    # RAM cleanup
                    # ===== AFTER RUN (before cleanup) =====
                    _rss = process.memory_info().rss
                    for _ch in process.children(recursive=True):
                        try:
                            _rss += _ch.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    mem_after_run_total_mb = _rss / 1e6
                    delta_run_mb = mem_after_run_total_mb - mem_before_total_mb

                    # updated peak (platform-aware)
                    if resource is not None:
                        _ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        if sys.platform.startswith("linux"):
                            peak_after_mb = _ru / 1024.0
                        elif sys.platform == "darwin":
                            peak_after_mb = _ru / (1024.0 * 1024.0)
                        else:
                            peak_after_mb = mem_after_run_total_mb
                    else:
                        try:
                            peak_after_mb = process.memory_full_info().peak_wset / 1e6
                        except Exception:
                            peak_after_mb = mem_after_run_total_mb

                    self.logger.info(
                        f"[MEM] Baseline: {mem_before_total_mb:.2f} MB | After run: {mem_after_run_total_mb:.2f} MB "
                        f"| Δrun: {delta_run_mb:.2f} MB | Peak≈{max(peak_before_mb, peak_after_mb):.2f} MB"
                    )

                    # ===== CLEANUP =====
                    # drop strong refs so GC matters
                    try: del ref
                    except NameError: pass
                    try: del log_entry
                    except NameError: pass

                    _rss = process.memory_info().rss
                    for _ch in process.children(recursive=True):
                        try:
                            _rss += _ch.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    mem_before_clean_mb = _rss / 1e6

                    gc.collect()
                    time.sleep(0.1)

                    _rss = process.memory_info().rss
                    for _ch in process.children(recursive=True):
                        try:
                            _rss += _ch.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    mem_after_clean_mb = _rss / 1e6

                    freed_mb = mem_before_clean_mb - mem_after_clean_mb
                    self.logger.info(f"[MEM] Freed ~{max(freed_mb, 0):.2f} MB; Post-clean total: {mem_after_clean_mb:.2f} MB\n")
                    self.logger.info("")
        
        # stop log forwarder
        self.log_queue.put(None) # remember, log_queue is a Ray Queue actor, and will only exist while Ray is still active (cannot be after the .shutdown())
        self._log_thread.join(timeout=2)

        ray.shutdown()  # Shutdown Ray after all testing
        self.logger.info(f"Ray Successfully Shutdown.")

        self.logger.info(f"Testing complete.")
        self.logger.info(f"")
        self.logger.info(f"Finding Optimal Configurations...")
        # Compute optimal configurations (GPU)
        df = pd.read_csv(csv_filepath)
        optimal_configuration_df, best_overall_usecase_df = find_optimal_configurations_gpu(df)
        optimal_configuration_df.to_csv(f"{self.csv_dir}/optimal_configurations_gpu.csv", index=False)
        best_overall_usecase_df.to_csv(f"{self.csv_dir}/best_overall_usecase_gpu.csv", index=False)
        self.logger.info(f"Optimal Configurations Found. Findings saved to:") 
        self.logger.info(f" 1) Optimal GPU/Station/Concurrent Prediction Configurations: {self.csv_dir}/optimal_configurations_gpu.csv") 
        self.logger.info(f" 2) Best Overall Usecase Configuration: {self.csv_dir}/best_overall_usecase_gpu.csv")

    def evaluate(self):
        if self.eval_mode == "cpu":
            self.evaluate_cpu()
        elif self.eval_mode == "gpu":
            self.evaluate_gpu()
        else: 
            exit()
        
    def calculate_vram(self):
        """Calculate available VRAM for GPU testing."""
        self.logger.info(f"Utilizing available VRAM...")
        total_vram, available_vram = get_gpu_vram()
        self.logger.info(f"Total VRAM: {total_vram:.2f} GB.")
        self.logger.info(f"Available VRAM: {available_vram:.2f} GB.")

        free_vram = total_vram * 0.9485 if available_vram / total_vram >= 0.9486 else available_vram
        self.logger.info(f"Using up to {round(free_vram, 2)} GB of VRAM.")
        return free_vram * 1024  # Convert to MB

"""
Finds the optimal CPU configuration based on evaluation results
"""
class OptimalCPUConfigurationFinder: 
    def __init__(self, 
                 eval_sys_results_dir: str, 
                 log_file_path: str):
        
        self.eval_sys_results_dir = eval_sys_results_dir
        if not self.eval_sys_results_dir or not os.path.isdir(self.eval_sys_results_dir): 
            raise ValueError(f"Error: The provided directory path '{self.eval_sys_results_dir}' is invalid or does not exist.")
        self.log_file_path = log_file_path

        # Set up main logger and logger queue to retrive queued logs from Raylets to be passed to the main logger
        self.logger = logging.getLogger("eqcctpro") # We named the logger eqcctpro (can be any name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False # if true, events logged to this logger will be passed to the handlers of higher level (ancestor) loggers, in addition to any handlers attached to this logger
        if not self.logger.handlers: # avoid duplicating inits 
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            # ensure parent dir
            Path(self.log_file_path).parent.mkdir(parents=True, exist_ok=True)
            file_h = logging.FileHandler(self.log_file_path) # Writes logs to file 
            stream_h = logging.StreamHandler() # Sends logs to console
            file_h.setFormatter(fmt)
            stream_h.setFormatter(fmt)
            self.logger.addHandler(file_h)
            self.logger.addHandler(stream_h)


    def find_best_overall_usecase(self):
        """Finds the best overall CPU usecase configuation from eval results"""
        file_path = f"{self.eval_sys_results_dir}/best_overall_usecase_cpu.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{datetime.now()}] Error: The file '{file_path}' does not exist. Ensure it is in the correct directory.")

        df_best_overall = pd.read_csv(file_path)
        # best_config_dict = df_best_overall.set_index(df_best_overall.columns[0]).to_dict()[df_best_overall.columns[1]]
        best_config_dict = df_best_overall.to_dict(orient='records')[0]
        
        # Extract required values
        num_cpus = best_config_dict.get("Number of CPUs Allocated for Ray to Use")
        waveform_timespace = best_config_dict.get("Total Waveform Analysis Timespace (min)")
        total_num_timechunks = best_config_dict.get("Total Number of Timechunks")
        num_concurrent_timechunks = best_config_dict.get("Concurrent Timechunks Used")
        length_of_timechunks = best_config_dict.get("Length of Timechunk (min)")
        num_concurrent_stations = best_config_dict.get("Number of Concurrent Station Tasks per Timechunk")
        intra_threads = best_config_dict.get("Intra-parallelism Threads")
        inter_threads = best_config_dict.get("Inter-parallelism Threads")
        num_stations = best_config_dict.get("Number of Stations Used")
        total_runtime = best_config_dict.get("Total Run time for Picker (s)")
        
        self.logger.info("")
        self.logger.info(f"------- Finding the Best Overall CPU Usecase Configuration Based on Available Trial Data in {self.eval_sys_results_dir} -------")
        self.logger.info(f"CPU(s): {num_cpus}")
        self.logger.info(f"Intra-parallelism Threads: {intra_threads}")
        self.logger.info(f"Inter-parallelism Threads: {inter_threads}")
        self.logger.info(f"Waveform Timespace: {waveform_timespace}")
        self.logger.info(f"Total Number of Stations Used: {num_stations}")
        self.logger.info(f"Total Number of Timechunks: {total_num_timechunks}")
        self.logger.info(f"Length of Timechunks (min): {length_of_timechunks}")
        self.logger.info(f"Concurrent Timechunk Processes: {num_concurrent_timechunks}")
        self.logger.info(f"Concurrent Station Processes Per Timechunk: {num_concurrent_stations}")
        self.logger.info(f"Total Runtime (s): {total_runtime}")
        self.logger.info("")

        # return int(float(num_cpus)), int(float(intra_threads)), int(float(inter_threads)), int(float(num_concurrent_timechunks)), int(float(num_concurrent_stations)), int(float(num_stations))
    
    def find_optimal_for(self, cpu: int, station_count: int):
        """Finds the optimal configuration for a given number of CPUs and stations."""
        if cpu is None or station_count is None:
            raise ValueError("Error: CPU and station_count must have valid values.")

        file_path = f"{self.eval_sys_results_dir}/optimal_configurations_cpu.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{datetime.now()}] Error: The file '{file_path}' does not exist. Ensure it is in the correct directory.")

        df_optimal = pd.read_csv(file_path)

        # Convert relevant columns to numeric
        df_optimal["Number of Stations Used"] = pd.to_numeric(df_optimal["Number of Stations Used"], errors="coerce")
        df_optimal["Number of CPUs Allocated for Ray to Use"] = pd.to_numeric(df_optimal["Number of CPUs Allocated for Ray to Use"], errors="coerce")
        df_optimal["Number of Concurrent Station Tasks"] = pd.to_numeric(df_optimal["Number of Concurrent Station Tasks"], errors="coerce")
        df_optimal["Total Run time for Picker (s)"] = pd.to_numeric(df_optimal["Total Run time for Picker (s)"], errors="coerce")

        filtered_df = df_optimal[
            (df_optimal["Number of CPUs Allocated for Ray to Use"] == cpu) &
            (df_optimal["Number of Stations Used"] == station_count)]

        if filtered_df.empty:
            raise ValueError("No matching configuration found. Please enter a valid entry.")

        # Finds for the "Total Run time for Picker (s)" the row with the smallest value and the '1' is to say I only want 
        # only the single row where the smallest runtime is 
        # iloc gets the selection of data from a numerical index from the df and turns that access point into a Series
        best_config = filtered_df.nsmallest(1, "Total Run time for Picker (s)").iloc[0]

        self.logger.info(f"------- Best CPU-EQCCTPro Configuration for Requested Input Parameters Based on the available Trial Data in {self.eval_sys_results_dir} -------")
        self.logger.info(f"CPU(s): {cpu}")
        self.logger.info(f"Intra-parallelism Threads: {best_config['Intra-parallelism Threads']}")
        self.logger.info(f"Inter-parallelism Threads: {best_config['Inter-parallelism Threads']}")
        self.logger.info(f"Waveform Timespace: {best_config['Total Waveform Analysis Timespace (min)']}")
        self.logger.info(f"Total Number of Stations Used: {station_count}")
        self.logger.info(f"Total Number of Timechunks: {best_config['Total Number of Timechunks']}")
        self.logger.info(f"Length of Timechunks (min): {best_config['Length of Timechunk (min)']}")
        self.logger.info(f"Concurrent Timechunk Processes: {best_config['Concurrent Timechunks Used']}")
        self.logger.info(f"Concurrent Station Processes Per Timechunk: {best_config['Number of Concurrent Station Tasks']}")
        self.logger.info(f"Total Runtime (s): {best_config['Total Run time for Picker (s)']}")
        self.logger.info("")

        # return int(float(cpu)), int(float(best_config["Intra-parallelism Threads"])), int(float(best_config["Inter-parallelism Threads"])), int(float(best_config["Concurrent Timechunks Used"])), int(float(best_config["Number of Concurrent Station Tasks"])), int(float(station_count))


class OptimalGPUConfigurationFinder:
    """Finds the optimal GPU configuration based on evaluation system results."""

    def __init__(self, 
                 eval_sys_results_dir: str, 
                 log_file_path: str):
        
        self.eval_sys_results_dir = eval_sys_results_dir
        if not self.eval_sys_results_dir or not os.path.isdir(self.eval_sys_results_dir): 
            raise ValueError(f"Error: The provided directory path '{self.eval_sys_results_dir}' is invalid or does not exist.")
        self.log_file_path = log_file_path

        # Set up main logger and logger queue to retrive queued logs from Raylets to be passed to the main logger
        self.logger = logging.getLogger("eqcctpro") # We named the logger eqcctpro (can be any name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False # if true, events logged to this logger will be passed to the handlers of higher level (ancestor) loggers, in addition to any handlers attached to this logger
        if not self.logger.handlers: # avoid duplicating inits 
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            # ensure parent dir
            Path(self.log_file_path).parent.mkdir(parents=True, exist_ok=True)
            file_h = logging.FileHandler(self.log_file_path) # Writes logs to file 
            stream_h = logging.StreamHandler() # Sends logs to console
            file_h.setFormatter(fmt)
            stream_h.setFormatter(fmt)
            self.logger.addHandler(file_h)
            self.logger.addHandler(stream_h)

    def find_best_overall_usecase(self):
        """Finds the best overall GPU configuration from evaluation results."""
        file_path = f"{self.eval_sys_results_dir}/best_overall_usecase_gpu.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{datetime.now()}] Error: The file '{file_path}' does not exist. Ensure it is in the correct directory.")

        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"[{datetime.now()}] Error: '{file_path}' is empty.")

        row = df.iloc[0]  # the best row you wrote out

        # Some codepaths use two different column names for concurrency; support both
        conc_col = "Number of Concurrent Station Tasks per Timechunk" \
            if "Number of Concurrent Station Tasks per Timechunk" in df.columns \
            else "Number of Concurrent Station Tasks"

        # Robust GPU parse: accepts [0], (0,), "0", 0, "", None
        num_gpus_list = _parse_gpus_field(row.get("GPUs Used"))
        # Keep as tuple for display/consistency
        num_gpus = tuple(num_gpus_list)

        # Pull/normalize scalars
        num_cpus = row.get("Number of CPUs Allocated for Ray to Use")
        num_concurrent = row.get(conc_col)
        intra_threads = row.get("Intra-parallelism Threads")
        inter_threads = row.get("Inter-parallelism Threads")
        num_stations = row.get("Number of Stations Used")
        total_runtime = row.get("Total Run time for Picker (s)")
        vram_used = row.get("VRAM Used Per Task")

        self.logger.info("")
        self.logger.info(f"------- Finding the Best Overall GPU Usecase Configuration Based on Available Trial Data in {self.eval_sys_results_dir} -------")
        self.logger.info("")
        self.logger.info(f"CPU(s): {num_cpus}")
        self.logger.info(f"GPU ID(s): {num_gpus_list}")
        self.logger.info(f"Concurrent Predictions: {num_concurrent}")
        self.logger.info(f"Intra-parallelism Threads: {intra_threads}")
        self.logger.info(f"Inter-parallelism Threads: {inter_threads}")
        self.logger.info(f"Stations: {num_stations}")
        self.logger.info(f"VRAM Used per Task: {vram_used}")
        self.logger.info(f"Total Runtime (s): {total_runtime}")
        self.logger.info("")
        # return int(float(num_cpus)), int(float(num_concurrent_predictions)), int(float(intra_threads)), int(float(inter_threads)), num_gpus, int(float(vram_used)), int(float(num_stations))

    def find_optimal_for(self, num_cpus: int, gpu_list: list, station_count: int):
        """Finds the optimal configuration for a given number of CPUs, GPUs, and stations."""
        if num_cpus is None or station_count is None or gpu_list is None:
            raise ValueError("Error: num_cpus, station_count, and gpu_list must have valid values.")

        file_path = f"{self.eval_sys_results_dir}/optimal_configurations_gpu.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{datetime.now()}] Error: The file '{file_path}' does not exist. Ensure it is in the correct directory.")

        df_optimal = pd.read_csv(file_path)

        # Convert relevant columns to numeric, handling NaNs
        df_optimal["Number of Stations Used"] = pd.to_numeric(df_optimal["Number of Stations Used"], errors="coerce")
        df_optimal["Number of CPUs Allocated for Ray to Use"] = pd.to_numeric(df_optimal["Number of CPUs Allocated for Ray to Use"], errors="coerce")
        df_optimal["Number of Concurrent Station Tasks"] = pd.to_numeric(df_optimal["Number of Concurrent Station Tasks"], errors="coerce")
        df_optimal["Total Run time for Picker (s)"] = pd.to_numeric(df_optimal["Total Run time for Picker (s)"], errors="coerce")
        df_optimal["VRAM Used Per Task"] = pd.to_numeric(df_optimal["VRAM Used Per Task"], errors="coerce")

        # Convert "GPUs Used" from string representation to list
        df_optimal["GPUs Used"] = df_optimal["GPUs Used"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Convert GPU lists to tuples for comparison
        df_optimal["GPUs Used"] = df_optimal["GPUs Used"].apply(lambda x: tuple(x) if isinstance(x, list) else (x,))

        # Ensure gpu_list is in tuple format for comparison
        gpu_list_tuple = tuple(gpu_list) if isinstance(gpu_list, list) else (gpu_list,)

        filtered_df = df_optimal[
            (df_optimal["Number of CPUs Allocated for Ray to Use"] == num_cpus) &
            (df_optimal["GPUs Used"] == gpu_list_tuple) &
            (df_optimal["Number of Stations Used"] == station_count)
        ]

        if filtered_df.empty:
            raise ValueError("No matching configuration found. Please enter a valid entry.")

        best_config = filtered_df.nsmallest(1, "Total Run time for Picker (s)").iloc[0]

        self.logger.info(f"------- Best GPU-EQCCTPro Configuration for Requested Input Parameters Based on the Available Trial Data in {self.eval_sys_results_dir} -------")
        self.logger.info(f"CPU(s): {num_cpus}")
        self.logger.info(f"GPU(s): {gpu_list}")
        self.logger.info(f"Concurrent Predictions: {best_config['Number of Concurrent Station Tasks']}")
        self.logger.info(f"Intra-parallelism Threads: {best_config['Intra-parallelism Threads']}")
        self.logger.info(f"Inter-parallelism Threads: {best_config['Inter-parallelism Threads']}")
        self.logger.info(f"Stations: {station_count}")
        self.logger.info(f"VRAM Used per Task: {best_config['VRAM Used Per Task']}")
        self.logger.info(f"Total Runtime (s): {best_config['Total Run time for Picker (s)']}")

        # return int(float(best_config["Number of CPUs Allocated for Ray to Use"])), \
        #        int(float(best_config["Number of Concurrent Station Tasks"])), \
        #        int(float(best_config["Intra-parallelism Threads"])), \
        #        int(float(best_config["Inter-parallelism Threads"])), \
        #        gpu_list, \
        #        int(float(best_config["VRAM Used Per Task"])), \
        #        int(float(station_count))
