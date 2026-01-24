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
import numpy as np
import queue 
import psutil
import random
import numbers
import logging
import resource
import threading
from eqcctpro.tools import *
from pathlib import Path
from eqcctpro.parallelization import *
from obspy import UTCDateTime
from ray.util.queue import Queue
from datetime import datetime, timedelta
from eqcctpro.tools import _parse_gpus_field
from logging.handlers import QueueHandler, QueueListener


class RunEQCCTPro():  
    """RunEQCCTPro class for running the RunEQCCTPro functions for multiple instances of the class"""
    def __init__(self, # self is 'this instance' of the class 
                use_gpu: bool, 
                input_dir: str, 
                output_dir: str, 
                log_filepath: str, 
                p_model_filepath: str = None, 
                s_model_filepath: str = None, 
                number_of_concurrent_station_predictions: int = None,
                number_of_concurrent_timechunk_predictions: int = 1, 
                intra_threads: int = 1, 
                inter_threads: int = 1, 
                P_threshold: float = 0.001, 
                S_threshold: float = 0.02,
                specific_stations: str = None,
                csv_dir: str = None,
                best_usecase_config: bool = False,
                vram_mb: float = None,
                selected_gpus: list = None,
                gpu_vram_safety_cap: float = 0.95,
                cpu_id_list: list = [1],
                start_time:str = None, 
                end_time:str = None, 
                timechunk_dt:int = 1,
                waveform_overlap:int = 0,
                tmp_dir:str = None,
                # SeisBench model parameters
                model_type: str = 'eqcct',  # 'eqcct' or 'seisbench'
                seisbench_parent_model: str = None,  # e.g., 'PhaseNet', 'EQTransformer'
                seisbench_child_model: str = None,  # e.g., 'original', 'stead'
                Detection_threshold: float = 0.3,  # Detection threshold for SeisBench models
                # Ripper mode - uses old task-based approach instead of ModelActors
                ripper: bool = False):  # If True, use task-based parallel_predict instead of ModelActor pool
         
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
        self.gpu_vram_safety_cap = gpu_vram_safety_cap
        self.selected_gpus = selected_gpus if selected_gpus is not None else list_gpu_ids() # a list of the GPU IDs. If not provided, we use all available GPUs
        self.cpu_id_list = cpu_id_list 
        self.cpu_count = len(cpu_id_list)
        self.start_time = start_time
        self.end_time = end_time
        self.timechunk_dt = timechunk_dt
        self.waveform_overlap = waveform_overlap
        self.home_tmp_dir = tmp_dir  
        
        # SeisBench model parameters
        self.model_type = model_type.lower()
        self.seisbench_parent_model = seisbench_parent_model
        self.seisbench_child_model = seisbench_child_model
        self.Detection_threshold = Detection_threshold
        
        # Ripper mode - uses old task-based approach instead of ModelActors
        self.ripper = ripper

        # Validate model type and parameters
        if self.model_type not in ['eqcct', 'seisbench']:
            raise ValueError(f"model_type must be 'eqcct' or 'seisbench', got '{model_type}'")
        
        if self.model_type == 'eqcct':
            if p_model_filepath is None or s_model_filepath is None:
                raise ValueError("For EQCCT model_type, p_model_filepath and s_model_filepath are required")
            if number_of_concurrent_station_predictions is None:
                raise ValueError("number_of_concurrent_station_predictions is required for EQCCT")
        elif self.model_type == 'seisbench':
            if seisbench_parent_model is None or seisbench_child_model is None:
                raise ValueError("For SeisBench model_type, seisbench_parent_model and seisbench_child_model are required")
            if number_of_concurrent_station_predictions is None:
                raise ValueError("number_of_concurrent_station_predictions is required for SeisBench")

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

        # We need to ensure that the vram specified does not exceed the capabilities of the system, if not, we need to exit safely before it happens
        if self.use_gpu:
            if self.vram_mb is None:
                # Calculate automatic vram_mb per worker
                # We take the total safe budget across GPUs and divide by total potential concurrent tasks
                total_budget = self.calculate_vram() 
                self.vram_mb = total_budget / (self.number_of_concurrent_station_predictions * self.number_of_concurrent_timechunk_predictions)
                self.logger.info(f"vram_mb not provided. Automatically calculated per-worker budget: {self.vram_mb:.0f} MB")

            # If the user passed a GPU but no valid VRAM (and calculation failed somehow), need to exit 
            if not (isinstance(self.vram_mb, numbers.Real) and math.isfinite(self.vram_mb) and self.vram_mb > 0): 
                self.logger.error(f"No numerical VRAM passed or calculated. Please provide vram_mb (MB per Raylet per GPU) as a positive real number. Exiting...")
                sys.exit(1)

            # Determine model VRAM requirement based on model type
            if self.model_type == 'seisbench':
                from .parallelization import get_seisbench_model_vram_mb
                model_vram_mb = get_seisbench_model_vram_mb(
                    self.seisbench_parent_model, 
                    self.seisbench_child_model,
                    default_mb=2000.0,  # Default VRAM for SeisBench models
                    logger=self.logger
                )
                self.logger.info(f"Using VRAM requirement: {model_vram_mb:.0f} MB for SeisBench model {self.seisbench_parent_model}/{self.seisbench_child_model}")
            else:
                model_vram_mb = 1500.0  # Safety reserve for EQCCT
            
            check_vram_per_gpu_style(
                selected_gpus=self.selected_gpus,
                get_gpu_vram_fn=lambda gid: get_gpu_vram(gpu_index=gid),
                intended_workers=self.number_of_concurrent_station_predictions * self.number_of_concurrent_timechunk_predictions,
                vram_mb=self.vram_mb,
                model_vram_mb=model_vram_mb,
                safety_cap=self.gpu_vram_safety_cap,
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
        skip_tf = (self.model_type != 'eqcct')
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
            tf_environ(gpu_id=-1, intra_threads=intra, inter_threads=inter, logger=self.logger, skip_tf=skip_tf)
        else:
            # We pass the requested parameters to the tf_environ 
            tf_environ(gpu_id=-1, intra_threads=self.intra_threads, inter_threads=self.inter_threads, logger=self.logger, skip_tf=skip_tf) 
            
    def configure_gpu(self):
        # We need to configure the tf_environ for the GPU configuration that is being inputted
        self.logger.info(f"Running EQCCT over Requested MSeed Files using GPU(s)...")
        # In the main process (driver), we only set environment variables. 
        # We ALWAYS skip TensorFlow initialization here because the main process doesn't run models.
        # This avoids confusing "No GPUs visible" messages if the driver's environment differs from workers.
        skip_tf = True 
        if self.best_usecase_config: 
            result = find_optimal_configuration_gpu(True, self.csv_dir)
            if result is None:
                self.logger.info("")
                self.logger.error(f"Error: Could not retrieve an optimal GPU configuration. Please check that the CSV file exists and try again. Exiting...")
                exit()  # Exit gracefully

            self.logger.info("")
            cpus_to_use, num_concurrent_predictions, intra, inter, gpus, vram_mb, station_count = result # Unpack values only if result is valid
            self.logger.info(f"Using {cpus_to_use} CPUs, {num_concurrent_predictions} Conc. Predictions, {intra} Intra Threads, {inter} Inter Threads, {gpus} GPU IDs, and {vram_mb} MB VRAM per Task...")
            tf_environ(gpu_id=1, vram_limit_mb=vram_mb, gpus_to_use=gpus, intra_threads=intra, inter_threads=inter, logger=self.logger, skip_tf=skip_tf)
        
        else: 
            self.logger.info("")
            self.logger.info(f"User requested to use GPU(s): {self.selected_gpus} with {self.vram_mb} MB of VRAM per Raylet (intra-op threads = {self.intra_threads}, inter-op threads = {self.inter_threads})") # Use the selected GPUs 
            tf_environ(gpu_id=1, vram_limit_mb=self.vram_mb, gpus_to_use=self.selected_gpus, intra_threads=self.intra_threads, inter_threads=self.inter_threads, logger=self.logger, skip_tf=skip_tf)
    
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
                                        intra_threads=self.intra_threads, inter_threads=self.inter_threads,
                                        model_type=self.model_type, seisbench_parent_model=self.seisbench_parent_model, 
                                        seisbench_child_model=self.seisbench_child_model, Detection_threshold=self.Detection_threshold,
                                        ripper=self.ripper))
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
            ray.init(ignore_reinit_error=True, num_gpus=len(self.selected_gpus), num_cpus=len(self.cpu_id_list), logging_level=logging.ERROR, log_to_driver=False, _temp_dir=self.home_tmp_dir) # Ray initalization using GPUs 
            self.log_queue = Queue() # Create a Ray-safe queue to recieve LogRecord objects from workers so we can write them to file 
            self._log_thread = threading.Thread(target=self._drain_worker_logs, daemon=True) # Creates background thread whose only job is to get() records from self.log_queue and hand them over to the actual logger
            self._log_thread.start() # Starts the thread
            # Log some import info to user 
            statement = f"Ray Successfully Initialized with {self.selected_gpus} GPU(s) and {len(self.cpu_id_list)} CPU(s) ({list(self.cpu_id_list)} CPU Affinity Binding)."
            self.logger.info(f"{statement}")
            self.logger.info(f"Analyzing {len(self.times_list)} time chunk(s) from {self.start_time} to {self.end_time} (dt={self.timechunk_dt}min, overlap={self.waveform_overlap}min).")
            
            # Running parllelization
            self.eqcctpro_parallelization()

        else: # CPU
            self.configure_cpu()
            ray.init(ignore_reinit_error=True, num_cpus=len(self.cpu_id_list), logging_level=logging.ERROR, log_to_driver=False, _temp_dir=self.home_tmp_dir) # Ray initalization using CPUs
            self.log_queue = Queue() # Create a Ray-safe queue to recieve LogRecord objects from workers so we can write them to file 
            self._log_thread = threading.Thread(target=self._drain_worker_logs, daemon=True) # Creates background thread whose only job is to get() records from self.log_queue and hand them over to the actual logger
            self._log_thread.start() # Starts the thread
            # Log some import info to user
            statement = f"Ray Successfully Initialized with {len(self.cpu_id_list)} CPU(s) ({list(self.cpu_id_list)} CPU Affinity Binding)."
            self.logger.info(f"{statement}")
            self.logger.info(f"Analyzing {len(self.times_list)} time chunk(s) from {self.start_time} to {self.end_time} (dt={self.timechunk_dt}min, overlap={self.waveform_overlap}min).")
            
            # Running parllelization
            self.eqcctpro_parallelization()

    def calculate_vram(self):
        """
        Calculates the VRAM budget per GPU based on the safety cap and free VRAM.
        Used for automatic calculation of vram_mb if not provided.
        """
        cap = float(self.gpu_vram_safety_cap)
        if not (0.0 < cap <= 0.99):
            raise ValueError(f"gpu_vram_safety_cap must be in (0, 0.99], got {cap}.")

        gpus = self.selected_gpus if self.selected_gpus else list_gpu_ids()
        if not gpus:
            raise RuntimeError("No GPUs detected for VRAM calculation.")

        per_gpu_budget_mb = []
        for gid in gpus:
            total_gb, free_gb = get_gpu_vram(gpu_index=gid)
            total_mb = float(total_gb) * 1024.0
            free_mb  = float(free_gb)  * 1024.0

            # hard cap vs physical total, plus never exceed currently free memory
            budget_mb = min(total_mb * cap, free_mb)
            per_gpu_budget_mb.append(budget_mb)

            self.logger.info(
                f"GPU {gid}: total={total_gb:.2f} GB, free={free_gb:.2f} GB, "
                f"budget={budget_mb/1024.0:.2f} GB (cap={cap:.2f})"
            )

        budget_mb_min = float(min(per_gpu_budget_mb))
        # self.logger.info(f"Using per-GPU VRAM budget = {budget_mb_min:.0f} MB (min across selected GPUs).")
        return budget_mb_min

class EvaluateSystem(): 
    """Evaluate System class for running the evaluation system functions for multiple instances of the class"""
    def __init__(self,
                 eval_mode: str,
                 input_dir: str,
                 output_dir: str,
                 log_filepath: str,
                 csv_dir: str, 
                 p_model_filepath: str = None, 
                 s_model_filepath: str = None, 
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
                 max_vram_mb:float = None,
                 gpu_vram_safety_cap:float = 0.90, 
                 ram_safety_cap:float = 0.90,
                 selected_gpus:list = None,
                 start_time:str = None, 
                 end_time:str = None, 
                 conc_timechunk_tasks_step_size: int = 1, 
                 timechunk_dt:int = 1,
                 waveform_overlap:int = 0,
                 tmp_dir:str = None,
                 # SeisBench model parameters
                 model_type: str = 'eqcct',  # 'eqcct' or 'seisbench'
                 seisbench_parent_model: str = None,
                 seisbench_child_model: str = None,
                 Detection_threshold: float = 0.3,
                 # Ripper mode - uses old task-based approach instead of ModelActors
                 ripper: bool = False): 
        
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
        self.vram_mb = max_vram_mb
        self.gpu_vram_safety_cap = gpu_vram_safety_cap
        
        if ram_safety_cap > 0.98:
            raise ValueError(f"ram_safety_cap cannot exceed 0.98, got {ram_safety_cap}")
        self.ram_safety_cap = ram_safety_cap

        self.selected_gpus = selected_gpus
        self.use_gpu = True if self.eval_mode == 'gpu' else False
        self.cpu_count = len(cpu_id_list)
        self.cpu_test_step_size = cpu_test_step_size
        self.starting_amount_of_stations = starting_amount_of_stations
        self.station_list_step_size = station_list_step_size
        self.min_cpu_amount = min_cpu_amount
        self.min_conc_stations = min_conc_stations # default is = 1 
        self.conc_station_tasks_step_size = conc_station_tasks_step_size # default is = 1 
        self.stations2use_list = list(range(1, 11)) + list(range(15, 105, 5)) if stations2use is None else generate_station_list(self.starting_amount_of_stations, stations2use, self.station_list_step_size,)
        self.start_time = start_time
        self.end_time = end_time
        self.conc_timechunk_tasks_step_size = conc_timechunk_tasks_step_size
        self.timechunk_dt = timechunk_dt
        self.waveform_overlap = waveform_overlap
        self.home_tmp_dir = tmp_dir 

        # SeisBench model parameters
        self.model_type = model_type.lower()
        self.seisbench_parent_model = seisbench_parent_model
        self.seisbench_child_model = seisbench_child_model
        self.Detection_threshold = Detection_threshold
        
        # Ripper mode - uses old task-based approach instead of ModelActors
        self.ripper = ripper

        # Validate model type and parameters
        if self.model_type not in ['eqcct', 'seisbench']:
            raise ValueError(f"model_type must be 'eqcct' or 'seisbench', got '{model_type}'")
        
        if self.model_type == 'eqcct':
            if p_model_filepath is None or s_model_filepath is None:
                raise ValueError("For EQCCT model_type, p_model_filepath and s_model_filepath are required")
        elif self.model_type == 'seisbench':
            if seisbench_parent_model is None or seisbench_child_model is None:
                raise ValueError("For SeisBench model_type, seisbench_parent_model and seisbench_child_model are required")
        
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

        # Calculate these at the start to determine total potential concurrency
        self.chunk_time()
        max_stations = max(self.stations2use_list) if self.stations2use_list else 1
        max_timechunks = len(self.times_list) if self.times_list else 1
        intended_workers = max_stations * max_timechunks

        if self.eval_mode == 'gpu':
            if not self.selected_gpus:
                raise ValueError("selected_gpus must be set in GPU mode.")
            # Total potential workers across all GPUs
            intended_workers = max_stations * max_timechunks

            # Determine model VRAM requirement based on model type
            if self.model_type == 'seisbench':
                from eqcctpro.parallelization import get_seisbench_model_vram_mb
                model_vram_mb = get_seisbench_model_vram_mb(
                    self.seisbench_parent_model, 
                    self.seisbench_child_model,
                    default_mb=2000.0,
                    logger=self.logger
                )
            else:
                from eqcctpro.parallelization import get_eqcct_vram_mb
                model_vram_mb = get_eqcct_vram_mb()  # Use measured EQCCT VRAM requirement

            per_gpu_free_mb = [get_gpu_vram(gpu_index=g)[1] * 1024.0 for g in self.selected_gpus]  # free_gb -> MB
            
            # If vram_mb is None, we skip the initial capacity check here and let evaluate_gpu calculate it
            if self.vram_mb is not None:
                plan = evaluate_vram_capacity(
                    intended_workers=intended_workers,
                    vram_per_worker_mb=float(self.vram_mb),
                    per_gpu_free_mb=per_gpu_free_mb,
                    model_vram_mb=model_vram_mb,
                    safety_cap=self.gpu_vram_safety_cap,
                    eqcct_overhead_gb=1.0,
                )
                if not plan.ok_aggregate:
                    unit = plan.per_worker_mb + plan.overhead_mb
                    self.logger.warning(
                        f"Aggregate VRAM might be insufficient for the largest trials ({intended_workers} total tasks: {max_stations} stations * {max_timechunks} timechunks). "
                        f"Cap={plan.aggregate_cap_mb:.0f} MB, Need={plan.aggregate_need_mb:.0f} MB. "
                        f"The system will automatically lower concurrent predictions during evaluation to ensure testing reaches all station counts."
                    )
                else:
                    self.logger.info(
                        f"VRAM budget OK. Need {plan.aggregate_need_mb:.0f} MB ≤ Cap {plan.aggregate_cap_mb:.0f} MB "
                        f"across {len(self.selected_gpus)} GPU(s) for up to {intended_workers} concurrent tasks ({max_stations} stations * {max_timechunks} timechunks)."
                    )
            else:
                self.logger.info("vram_mb not provided. VRAM capacity will be calculated during evaluation based on safety cap.")
        
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

    def _trial_key(self, *, num_cpus: int, stations: int, predictions: int, gpu_memory_limit_mb, timechunks: int, model: str, gpus: list = None) -> str:
        # Use provided gpus parameter if available, otherwise fall back to self.selected_gpus
        if gpus is not None:
            gpus_to_use = gpus
        else:
            gpus_to_use = self.selected_gpus if self.selected_gpus is not None else []
        gpus_norm = tuple(sorted(int(x) for x in gpus_to_use))
        # Round VRAM to nearest 100MB to ensure stability across restarts despite small system fluctuations
        vram_norm = int(round(float(gpu_memory_limit_mb) / 100.0) * 100) if gpu_memory_limit_mb not in ("", None) else ""
        return f"cpus={int(num_cpus)}|gpus={gpus_norm}|stations={int(stations)}|pred={int(predictions)}|timechunks={int(timechunks)}|vram={vram_norm}|model={model}"
        
    def _load_existing_trial_keys(self, csv_path: str) -> set[str]:
        if not os.path.exists(csv_path):
            return set()
        try:
            df = pd.read_csv(csv_path, keep_default_na=False)
        except Exception:
            return set()

        keys = set()
        for _, row in df.iterrows():
            try:
                num_cpus = int(float(row.get("Number of CPUs Allocated for Ray to Use", 0) or 0))
                stations = int(float(row.get("Number of Stations Used", 0) or 0))
                predictions = int(float(row.get("Number of Concurrent Station Tasks", 0) or 0))
                timechunks = int(float(row.get("Concurrent Timechunks Used", 0) or 0))
                vram = row.get("Inference Actor Memory Limit (MB)", "")
                vram_mb = float(vram) if vram not in ("", None) else ""
                # Parse GPUs from this specific row (don't overwrite self.selected_gpus)
                row_gpus = _parse_gpus_field(row.get("GPUs Used")) or []
                # Extract model info
                model = row.get("Model Used", "eqcct") # Default to eqcct for legacy rows
                # Pass the row's info directly to _trial_key
                keys.add(self._trial_key(num_cpus=num_cpus, stations=stations, predictions=predictions, 
                                       gpu_memory_limit_mb=vram_mb, timechunks=timechunks, model=model, gpus=row_gpus))
            except Exception:
                continue
        return keys

    def evaluate_cpu(self): 
        """Evaluate system parallelization using CPUs"""
        statement = "Evaluating System Parallelization Capability using CPU"
        self.logger.info(f"{statement}")
        
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test results csv 
        csv_filepath = f"{self.csv_dir}/cpu_test_results.csv"
        prepare_csv(csv_file_path=csv_filepath, logger=self.logger)
        planned_keys = self._load_existing_trial_keys(csv_filepath)
        
        # Resume from existing trial count if CSV has trials
        existing_trial_count = 0
        if os.path.exists(csv_filepath):
            try:
                df_existing = pd.read_csv(csv_filepath, keep_default_na=False)
                existing_trial_count = len(df_existing)
                if existing_trial_count > 0:
                    self.logger.info(f"Resuming from trial {existing_trial_count + 1} (found {existing_trial_count} existing trial(s) in CSV).")
            except Exception:
                pass
        
        self.chunk_time()
        self.dt_task_generator()
        
        trial_num = existing_trial_count + 1  # Resume from where we left off
        log_queue = queue.Queue()  # Create a queue for log entries
        total_analysis_time = datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")

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
                
                # Ensure local workspace is in workers' sys.path to pick up local eqcctpro
                # and avoid conflicts with site-packages
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                runtime_env = {"env_vars": {"PYTHONPATH": f"{workspace_root}:{os.environ.get('PYTHONPATH', '')}"}}
                
                ray.init(ignore_reinit_error=True, num_cpus=len(cpus_to_use), 
                         logging_level=logging.FATAL, log_to_driver=False, 
                         _temp_dir=self.home_tmp_dir, runtime_env=runtime_env)
                self.log_queue = Queue() # Create a Ray-safe queue to recieve LogRecord objects from workers so we can write them to file 
                self._log_thread = threading.Thread(target=self._drain_worker_logs, daemon=True) # Creates background thread whose only job is to get() records from self.log_queue and hand them over to the actual logger
                self._log_thread.start() # Starts the thread
                self.logger.info(f"Ray Successfully Initialized with {len(cpus_to_use)} CPU(s) ({list(cpus_to_use)} CPU Affinity Binding).")
                
                # Track the highest number of actors spawned in this Ray session
                # This is used to detect when we need to restart Ray to avoid OOM
                actors_high_water_mark = 0
                
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
                # ===== MODEL-AWARE RESOURCE CALCULATION (CPU) =====
                # Determine model name and RAM requirement from empirical data
                if self.model_type == 'seisbench':
                    trial_model = f"{self.seisbench_parent_model}/{self.seisbench_child_model}"
                    from eqcctpro.parallelization import get_seisbench_model_ram_mb
                    model_ram_per_actor_mb = get_seisbench_model_ram_mb(
                        self.seisbench_parent_model, 
                        self.seisbench_child_model,
                        use_gpu=False,  # CPU mode
                        default_mb=600.0,
                        logger=self.logger
                    )
                else:
                    trial_model = "eqcct"
                    from eqcctpro.parallelization import get_eqcct_ram_mb
                    model_ram_per_actor_mb = get_eqcct_ram_mb(use_gpu=False)  # CPU mode
                
                # Import buffer constants for logging clarity
                from eqcctpro.parallelization import RAM_BUFFER_MB
                
                # Additional per-task overhead for waveform data buffers and intermediate results
                # (Note: The main safety buffer is already included in model_ram_per_actor_mb)
                overhead_ram_per_task_mb = 256.0  # Extra per-task RAM for waveform data
                
                # ===== RAM-BASED CAPACITY CHECK =====
                # Get total system RAM to determine absolute machine capacity
                system_mem = psutil.virtual_memory()
                system_ram_total_mb = system_mem.total / (1024 * 1024)
                system_ram_available_mb = system_mem.available / (1024 * 1024)
                
                # Usable RAM is a percentage of the TOTAL machine RAM
                # This ensures the check is based on what the computer can handle physically
                aggregate_ram_cap_mb = system_ram_total_mb * self.ram_safety_cap
                
                # ===== MEMORY-AWARE ACTOR CAPPING (CPU) =====
                # KEY INSIGHT: The constraint is MEMORY, not hardware count.
                # taskset already limits CPU visibility, so Ray can only see the CPUs we've bound to.
                # Actors are capped to MEMORY availability in parallelization.py.
                
                # Memory limit: how many actors fit in RAM (this is the primary constraint)
                ram_per_actor = model_ram_per_actor_mb + overhead_ram_per_task_mb
                max_actors_by_ram = int(aggregate_ram_cap_mb // ram_per_actor)
                
                # Hardware reference (for logging only - not the limiting factor)
                n_cpus = len(cpus_to_use)
                
                # The actual limit is RAM, not CPUs
                # parallelization.py will cap to min(requested, max_actors_by_ram)
                max_actors = max(1, max_actors_by_ram)
                
                # Check if at least one actor fits in memory
                one_actor_fits = ram_per_actor <= aggregate_ram_cap_mb
                
                self.logger.info(f"")
                self.logger.info(f"===== MODEL-AWARE RESOURCE ANALYSIS (CPU) =====")
                self.logger.info(f"Model Type: {trial_model}")
                self.logger.info(f"Model RAM per Actor: {model_ram_per_actor_mb:.0f} MB (includes {RAM_BUFFER_MB:.0f} MB safety buffer for Ray/framework overhead)")
                self.logger.info(f"Per-Task Data Overhead: RAM={overhead_ram_per_task_mb:.0f} MB (for waveform buffers)")
                self.logger.info(f"Total RAM per Actor: {ram_per_actor:.0f} MB")
                self.logger.info(f"")
                self.logger.info(f"RAM Capacity: {aggregate_ram_cap_mb:.0f} MB (Total: {system_ram_total_mb:.0f} MB × {self.ram_safety_cap:.0%} safety)")
                self.logger.info(f"Currently Available RAM: {system_ram_available_mb:.0f} MB")
                self.logger.info(f"")
                self.logger.info(f"===== MEMORY-AWARE ACTOR CAPPING (CPU) =====")
                self.logger.info(f"CPUs visible to Ray: {n_cpus}")
                self.logger.info(f"Max actors by RAM: {max_actors_by_ram} ({aggregate_ram_cap_mb:.0f} MB / {ram_per_actor:.0f} MB per actor)")
                self.logger.info(f"Effective max actors: {max_actors} (memory-constrained, not hardware-constrained)")
                self.logger.info(f"Ray will handle CPU scheduling (taskset already limits visibility)")
                self.logger.info(f"================================================")
                
                if not one_actor_fits:
                    self.logger.warning(f"Insufficient RAM to spawn even 1 actor ({ram_per_actor:.0f} MB > {aggregate_ram_cap_mb:.0f} MB). Skipping this CPU configuration.")
                    continue

                for timechunks in timechunks_list:
                    for num_stations in self.stations2use_list: 
                        tested_concurrency = set() # Reset for each configuration
                        # Use a 20% step size for concurrency testing as requested
                        # This tests 20%, 40%, 60%, 80%, and 100% of the current total stations
                        step = max(1, int(num_stations * 0.2))
                        concurrent_predictions_list = sorted(list(set(range(step, num_stations + 1, step))))
                        
                        # ===== FEASIBILITY CHECK (CPU) =====
                        # Since actors are CAPPED to MEMORY in parallelization.py, all concurrency levels
                        # are valid as long as the actors fit in memory (checked above).
                        # Tasks beyond actor count are simply queued and processed round-robin.
                        valid_predictions = concurrent_predictions_list
                        invalid_predictions = []  # No invalid predictions since actors are capped to memory
                        
                        # Only block if memory is truly insufficient (already checked above, so this is a safety fallback)
                        if not one_actor_fits:
                            invalid_predictions = concurrent_predictions_list
                            valid_predictions = []
                        
                        if invalid_predictions:
                            for inv_p in invalid_predictions:
                                # Efficiency check: don't record if already tested/skipped
                                key = self._trial_key(
                                    num_cpus=len(cpus_to_use),
                                    stations=num_stations,
                                    predictions=inv_p,
                                    gpu_memory_limit_mb="",
                                    timechunks=timechunks,
                                    model=trial_model
                                )
                                if key in planned_keys:
                                    continue
                                planned_keys.add(key)
                                
                                # Calculate actual memory that would be used (capped to memory availability)
                                actual_actors = min(inv_p, max_actors)
                                est_ram = actual_actors * ram_per_actor
                                
                                error_msg = (f"[OOM PREVENTION] Cannot fit {actual_actors} actor(s) in RAM. "
                                           f"Required: {est_ram:.0f}MB RAM. "
                                           f"Available: {aggregate_ram_cap_mb:.0f}MB RAM.")
                                
                                self.logger.warning(f"Recording OOM-risk trial skip for {num_stations} stations, {inv_p} tasks: {error_msg}")
                                
                                # Build minimal trial data for CSV recording
                                oom_comment = f"Requested {inv_p} actors, 0 created (RAM limited to {aggregate_ram_cap_mb:.0f} MB, {ram_per_actor:.0f} MB/actor) - Trial skipped"
                                trial_data = {
                                    "Trial Number": trial_num,
                                    "Number of Stations Used": num_stations,
                                    "Number of CPUs Allocated for Ray to Use": len(cpus_to_use),
                                    "GPUs Used": "[]",
                                    "N ModelActors": 0,  # No actors created - OOM prevention
                                    "Inference Actor Memory Limit (MB)": "",
                                    "Number of Concurrent Station Tasks": inv_p,
                                    # Include timechunk metadata for consistency
                                    "Concurrent Timechunks Used": getattr(self, 'number_of_concurrent_timechunk_predictions', 1),
                                    "Total Number of Timechunks": timechunks,
                                    "Length of Timechunk (min)": getattr(self, 'timechunk_dt', 1.0),
                                    "Total Waveform Analysis Timespace (min)": float(timechunks * getattr(self, 'timechunk_dt', 1.0)),
                                    "Model Used": trial_model,
                                    "Trial Success": 0,
                                    "Error Message": error_msg,
                                    "Comments": oom_comment,
                                }
                                from eqcctpro.parallelization import append_trial_row
                                append_trial_row(csv_filepath, trial_data)
                                
                                # Add memory metadata columns to the newly appended row
                                mem_data = build_memory_trial_data(
                                    mem_after=get_memory_snapshot(process),
                                    process=process,
                                    model_requested_ram_mb=est_ram,
                                    n_model_actors=0,  # No actors created - OOM prevention
                                    model_ram_per_actor_mb=model_ram_per_actor_mb,
                                    is_gpu_trial=False  # CPU trial: ModelActors use RAM
                                )
                                update_csv_with_memory(csv_filepath, mem_data)
                                trial_num += 1
                        
                        if not valid_predictions and num_stations > 0:
                            # If even the first step (20%) is too much, try just 1 prediction or the max possible
                            if max_actors >= 1:
                                valid_predictions = [min(step, max_actors)]
                                self.logger.info(f"Station count {num_stations}: 20% step ({step}) exceeds max actors ({max_actors}). Using max={valid_predictions[0]}")
                            else:
                                self.logger.warning(f"Station count {num_stations}: Cannot spawn any ModelActors. Skipping.")
                                continue
                        elif len(valid_predictions) < len(concurrent_predictions_list):
                            self.logger.info(f"Trimming concurrency for {num_stations} stations: {concurrent_predictions_list} → {valid_predictions} (max actors={max_actors})")
                        
                        concurrent_predictions_list = valid_predictions
                        
                        # Efficiency optimization: only test concurrency values we haven't seen yet 
                        # for this CPU/Timechunk combo to save compute time. 
                        new_concurrent_values = [x for x in concurrent_predictions_list if x not in tested_concurrency]
                        if not new_concurrent_values:
                            continue  # All concurrency values already tested
                        for num_concurrent_predictions in new_concurrent_values:
                            tested_concurrency.add(num_concurrent_predictions)
                            
                            # ===== RIPPER MODE: Fresh Ray instance per trial =====
                            # Ripper mode benefits from a clean RAM state at each trial start
                            # This ensures no memory fragmentation from previous trials affects the current one
                            if self.ripper:
                                self.logger.info(f"")
                                self.logger.info(f"===== RIPPER MODE: Restarting Ray for fresh RAM state =====")
                                
                                # Stop the log drain thread gracefully
                                self.log_queue.put(None)  # Sentinel to stop the drain thread
                                time.sleep(0.5)  # Give thread time to finish
                                
                                # Shutdown and reinitialize Ray
                                ray.shutdown()
                                time.sleep(1.0)  # Allow RAM to be fully released
                                
                                # Reinitialize Ray with the same configuration (CPU mode)
                                ray.init(ignore_reinit_error=True, num_cpus=len(cpus_to_use), 
                                        logging_level=logging.FATAL, log_to_driver=False, _temp_dir=self.home_tmp_dir)
                                self.log_queue = Queue()
                                self._log_thread = threading.Thread(target=self._drain_worker_logs, daemon=True)
                                self._log_thread.start()
                                
                                # Reset the high water mark since we cleared RAM
                                actors_high_water_mark = 0
                                self.logger.info(f"Ray restarted. RAM cleared for ripper trial.")
                                self.logger.info(f"=============================================")
                            
                            key = self._trial_key(
                                num_cpus=len(cpus_to_use),
                                stations=num_stations,
                                predictions=num_concurrent_predictions,
                                gpu_memory_limit_mb="",   # CPU eval has no per-task VRAM cap
                                timechunks=timechunks,
                                model=trial_model
                            )
                            if key in planned_keys:
                                self.logger.info(f"[SKIP] Already tested: {key}")
                                continue
                            planned_keys.add(key)
                                    
                            mseed_timechunk_dir_name = self.tasks_picker[timechunks-1][1]
                            timechunk_dir_path = os.path.join(self.input_dir, mseed_timechunk_dir_name) 
                            max_pending_tasks = timechunks
                            
                            # ===== CALCULATE REQUESTED MEMORY FOR THIS TRIAL =====
                            # N ModelActors = num_concurrent_predictions (each concurrent task gets its own ModelActor)
                            n_model_actors = num_concurrent_predictions
                            requested_ram_mb = n_model_actors * model_ram_per_actor_mb
                            task_overhead_ram_mb = n_model_actors * overhead_ram_per_task_mb
                            total_requested_ram_mb = requested_ram_mb + task_overhead_ram_mb
                            
                            # ===== OOM PREVENTION: Restart Ray if needed =====
                            # When increasing concurrency, actors from previous trials may still be in RAM.
                            # If the new actors + existing actors would exceed RAM capacity, restart Ray.
                            restart_note = ""
                            if num_concurrent_predictions > actors_high_water_mark and actors_high_water_mark > 0:
                                # Estimate total RAM if we add new actors while existing ones are still in memory
                                estimated_total_ram = (actors_high_water_mark + num_concurrent_predictions) * model_ram_per_actor_mb
                                if estimated_total_ram > aggregate_ram_cap_mb:
                                    self.logger.warning(f"")
                                    self.logger.warning(f"===== RAY RESTART: Clearing RAM =====")
                                    self.logger.warning(f"Increasing from {actors_high_water_mark} to {num_concurrent_predictions} actors")
                                    self.logger.warning(f"Estimated RAM if additive: {estimated_total_ram:.0f} MB > capacity {aggregate_ram_cap_mb:.0f} MB")
                                    self.logger.warning(f"Restarting Ray to prevent OOM...")
                                    
                                    # Stop the log drain thread gracefully
                                    self.log_queue.put(None)  # Sentinel to stop the drain thread
                                    time.sleep(0.5)  # Give thread time to finish
                                    
                                    # Shutdown and reinitialize Ray
                                    ray.shutdown()
                                    time.sleep(1.0)  # Allow RAM to be released
                                    
                                    # Reinitialize Ray with the same configuration
                                    ray.init(ignore_reinit_error=True, num_cpus=len(cpus_to_use), 
                                            logging_level=logging.FATAL, log_to_driver=False, _temp_dir=self.home_tmp_dir)
                                    self.log_queue = Queue()
                                    self._log_thread = threading.Thread(target=self._drain_worker_logs, daemon=True)
                                    self._log_thread.start()
                                    
                                    # Set the restart note BEFORE resetting the high water mark
                                    restart_note = f"[RAY RESTART] RAM cleared (Prev Peak: {actors_high_water_mark} actors). "
                                    # Reset the high water mark since we cleared RAM
                                    actors_high_water_mark = 0
                                    self.logger.info(f"Ray restarted successfully. Memory cleared.")
                                    self.logger.info(f"==========================================")
                                    self.logger.info(f"")
                            
                            self.logger.info("")
                            self.logger.info(f"------- Trial Number: {trial_num} -------")
                            self.logger.info(f"CPU(s): {len(cpus_to_use)}")
                            self.logger.info(f"Conc. Timechunks Being Analyzed: {timechunks} / Total Timechunks to be Analyzed: {len(self.tasks_picker)}")
                            self.logger.info(f"Stations: {num_stations}")
                            self.logger.info(f"Concurrent Predictions (N ModelActors): {num_concurrent_predictions}")
                            self.logger.info(f"Model RAM per Actor: {model_ram_per_actor_mb:.0f} MB")
                            self.logger.info(f"Requested Total RAM: {total_requested_ram_mb:.0f} MB ({n_model_actors} actors × {model_ram_per_actor_mb:.0f} MB + {task_overhead_ram_mb:.0f} MB overhead)")
                            self.logger.info("")
                            
                            # Concurrent Timechunks
                            tasks_queue = []
                            log_queue = queue.Queue()  # Create a queue for log entries

                            # ===== COMPREHENSIVE MEMORY TRACKING (BEFORE) =====
                            mem_before = get_memory_snapshot(process)

                            try: 
                                while True: 
                                    if len(tasks_queue) < max_pending_tasks: 
                                        tasks_queue.append(mseed_predictor.options(num_gpus=0, num_cpus=1).remote(input_dir=timechunk_dir_path, output_dir=self.output_dir, log_queue=self.log_queue, 
                                                            P_threshold=self.P_threshold, S_threshold=self.S_threshold, p_model=self.p_model_filepath, s_model=self.s_model_filepath, 
                                                            number_of_concurrent_station_predictions=num_concurrent_predictions, ray_cpus=cpus_to_use, use_gpu=self.use_gpu, 
                                                            gpu_id=self.selected_gpus, gpu_memory_limit_mb=self.vram_mb, stations2use=num_stations, 
                                                            timechunk_id=mseed_timechunk_dir_name, waveform_overlap=self.waveform_overlap, total_timechunks=len(self.tasks_picker), 
                                                            number_of_concurrent_timechunk_predictions=max_pending_tasks, total_analysis_time=total_analysis_time, testing_gpu=False, 
                                                            test_csv_filepath=csv_filepath, intra_threads=self.intra_threads, inter_threads=self.inter_threads, timechunk_dt=self.timechunk_dt,
                                                            model_type=self.model_type, seisbench_parent_model=self.seisbench_parent_model, 
                                                            seisbench_child_model=self.seisbench_child_model, Detection_threshold=self.Detection_threshold,
                                                            ripper=self.ripper))
                                    
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

                                # ===== COMPREHENSIVE MEMORY TRACKING (AFTER) =====
                                mem_after = get_memory_snapshot(process)
                                
                                # Build memory data for CSV with model-requested information
                                memory_trial_data = build_memory_trial_data(
                                    mem_after=mem_after,
                                    mem_before=mem_before,
                                    process=process,
                                    model_requested_ram_mb=total_requested_ram_mb,
                                    n_model_actors=n_model_actors,
                                    model_ram_per_actor_mb=model_ram_per_actor_mb,
                                    is_gpu_trial=False  # CPU trial: ModelActors use RAM
                                )
                                update_csv(csv_filepath, success=1, error_message=restart_note)
                                update_csv_with_memory(csv_filepath, memory_trial_data)
                                
                            except Exception as e:
                                # Failure occured, need to add to log 
                                error_msg = restart_note + f"{type(e).__name__}: {str(e)}"
                                # Still capture memory after failure
                                mem_after = get_memory_snapshot(process)
                                memory_trial_data = build_memory_trial_data(
                                    mem_after=mem_after,
                                    mem_before=mem_before,
                                    process=process,
                                    model_requested_ram_mb=total_requested_ram_mb,
                                    n_model_actors=n_model_actors,
                                    model_ram_per_actor_mb=model_ram_per_actor_mb,
                                    is_gpu_trial=False  # CPU trial: ModelActors use RAM
                                )
                                update_csv(csv_filepath, success=0, error_message=error_msg)
                                update_csv_with_memory(csv_filepath, memory_trial_data)
                                self.logger.error(f"Trial {trial_num} FAILED: {error_msg}")
                                
                            # Write log entries from the queue to the file
                            while not log_queue.empty():
                                log_entry = log_queue.get()
                                
                            remove_output_subdirs(self.output_dir, logger=self.logger)
                            trial_num += 1
                            
                            # Update the high water mark of actors spawned in this Ray session
                            # This is used to detect when we need to restart Ray to prevent OOM
                            actors_high_water_mark = max(actors_high_water_mark, num_concurrent_predictions)
                            
                            # ===== MEMORY LOGGING =====
                            delta = compute_memory_delta(mem_before, mem_after)
                            actual_ram_used = mem_after.combined_rss_mb - mem_before.combined_rss_mb

                            self.logger.info("")
                            self.logger.info(f"[MODEL REQUESTED] RAM: {total_requested_ram_mb:.0f} MB")
                            self.logger.info(f"[ACTUAL MEASURED] RAM Used: {max(0, actual_ram_used):.0f} MB")
                            self.logger.info(
                                f"[MEM] Baseline: {mem_before.process_rss_mb:.2f} MB | After run: {mem_after.process_rss_mb:.2f} MB "
                                f"| Δrun: {delta['process_ram_delta_mb']:.2f} MB | Peak≈{mem_after.process_peak_mb:.2f} MB"
                            )
                            self.logger.info(
                                f"[MEM] System Available: {mem_before.system_available_mb:.2f} MB → {mem_after.system_available_mb:.2f} MB "
                                f"| Raylets: {mem_after.num_raylet_processes} (Total: {mem_after.total_raylet_ram_mb:.2f} MB, Avg: {mem_after.avg_raylet_ram_mb:.2f} MB)"
                            )

                            # ===== CLEANUP =====
                            try: del ref
                            except NameError: pass
                            try: del log_entry
                            except NameError: pass

                            gc.collect()
                            time.sleep(0.1)

                            mem_after_clean = get_memory_snapshot(process)
                            freed_mb = mem_after.combined_rss_mb - mem_after_clean.combined_rss_mb
                            self.logger.info(f"[MEM] Freed ~{max(freed_mb, 0):.2f} MB; Post-clean total: {mem_after_clean.combined_rss_mb:.2f} MB")
                            self.logger.info("")
                            
                        # tested_concurrency.update([x for x in concurrent_predictions_list if x <= num_stations])
                        tested_concurrency.update(new_concurrent_values)

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
        
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create test results csv 
        csv_filepath = f"{self.csv_dir}/gpu_test_results.csv"
        prepare_csv(csv_file_path=csv_filepath, logger=self.logger)
        
        # Normalize existing CSV to ensure consistent "GPUs Used" formatting and quoting
        if os.path.exists(csv_filepath):
            from eqcctpro.tools import normalize_gpu_csv_quoting
            normalize_gpu_csv_quoting(csv_filepath)
            self.logger.info("Normalized existing CSV entries for consistent 'GPUs Used' formatting.")
        
        planned_keys = self._load_existing_trial_keys(csv_filepath)
        
        # Log summary of existing trials
        if planned_keys:
            self.logger.info(f"Loaded {len(planned_keys)} existing trial(s) from CSV. These will be skipped.")
            # Count trials by GPU configuration
            gpu_counts = {}
            for key in planned_keys:
                # Extract GPU info from key (format: cpus=X|gpus=(...)|...)
                if "gpus=" in key:
                    gpu_part = key.split("gpus=")[1].split("|")[0]
                    gpu_counts[gpu_part] = gpu_counts.get(gpu_part, 0) + 1
            if gpu_counts:
                self.logger.info("Existing trials by GPU configuration:")
                for gpu_config, count in sorted(gpu_counts.items()):
                    self.logger.info(f"  {gpu_config}: {count} trial(s)")
        else:
            self.logger.info("No existing trials found in CSV. Starting fresh evaluation.")

        # Resume from existing trial count if CSV has trials
        existing_trial_count = 0
        if os.path.exists(csv_filepath):
            try:
                df_existing = pd.read_csv(csv_filepath, keep_default_na=False)
                existing_trial_count = len(df_existing)
                if existing_trial_count > 0:
                    self.logger.info(f"Resuming from trial {existing_trial_count + 1} (found {existing_trial_count} existing trial(s) in CSV).")
            except Exception:
                pass

        # Calculate these at the start
        self.chunk_time()
        self.dt_task_generator()

        trial_num = existing_trial_count + 1  # Resume from where we left off
        log_queue = queue.Queue()  # Create a queue for log entries
        total_analysis_time = datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
        
        # Track statistics
        trials_skipped = 0
        trials_run = 0
        
        if self.min_cpu_amount > len(self.cpu_id_list): 
            # Code won't execute because the minimum CPU amount of > the len(cpu id list)
            # In which the rest of the code is dependent on the len for generating cpu_count 
            print(f"CPU ID List provided has less CPUs than the minimum requested ({len(self.cpu_id_list)} vs. {self.min_cpu_amount}). Exiting...")
            quit()

        for gpu in range(len(self.selected_gpus)):
            for cpu in range(self.min_cpu_amount, len(self.cpu_id_list)+1, self.cpu_test_step_size):
                # Set CPU affinity and initialize Ray
                cpus_to_use = self.cpu_id_list[:cpu] # 'cpu' is a count (e.g., 1, 2, 3). Slicing [:cpu] gets that many CPU IDs, we want :cpu bc we are using 0 index counting so 0-20 exclusive = 0-19 IDs = 20 CPUs to use explicitely. ([:n is exclusive])
                gpus_to_use = self.selected_gpus[:gpu+1] # 'gpu' is an index (0, 1, 2). We need +1 to include the current GPU.
                # Set CPU affinity
                process = psutil.Process(os.getpid())
                process.cpu_affinity(cpus_to_use)  # Limit process to the given CPU IDs
                
                # VRAM budget per GPU (MB). If vram_mb is provided, treat it as an explicit per-GPU budget override.
                free_vram_mb = float(self.vram_mb) if self.vram_mb else self.calculate_vram()
                self.logger.info("")
                self.logger.info("=" * 80)
                self.logger.info(f"Testing Using {len(gpus_to_use)} GPU(s) with IDs {gpus_to_use} and {len(cpus_to_use)} CPU(s)")
                self.logger.info("=" * 80)
                
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus_to_use))
                
                # Ensure local workspace is in workers' sys.path to pick up local eqcctpro
                # and avoid conflicts with site-packages
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                runtime_env = {"env_vars": {"PYTHONPATH": f"{workspace_root}:{os.environ.get('PYTHONPATH', '')}"}}
                
                # Initialize Ray with GPUs and runtime_env
                ray.init(ignore_reinit_error=True, num_gpus=len(gpus_to_use), num_cpus=len(cpus_to_use), 
                        logging_level=logging.FATAL, log_to_driver=False, 
                        _temp_dir=self.home_tmp_dir, runtime_env=runtime_env)
                self.log_queue = Queue() # Create a Ray-safe queue to recieve LogRecord objects from workers so we can write them to file 
                self._log_thread = threading.Thread(target=self._drain_worker_logs, daemon=True) # Creates background thread whose only job is to get() records from self.log_queue and hand them over to the actual logger
                self._log_thread.start() # Starts the thread
                self.logger.info(f"Ray Successfully Initialized with {len(gpus_to_use)} GPU(s) and {len(cpus_to_use)} CPU(s) ({list(cpus_to_use)} CPU Affinity Binding).")
                self.logger.info(f"Trials will evalute GPU(s) performance against Iterative Total Station Tasks ({self.stations2use_list}) with Varying Concurrent Predictions.")
                self.logger.info("")
                
                # Efficiency optimization: track tested configurations for this GPU/CPU combo
                tested_gpu_configs = set() 

                # Track the highest number of actors spawned in this Ray session
                # This is used to detect when we need to restart Ray to avoid OOM
                actors_high_water_mark = 0

                # ===== MODEL-AWARE RESOURCE CALCULATION =====
                # Determine model name, VRAM, and RAM requirements from empirical data
                if self.model_type == 'seisbench':
                    trial_model = f"{self.seisbench_parent_model}/{self.seisbench_child_model}"
                    from eqcctpro.parallelization import get_seisbench_model_vram_mb, get_seisbench_model_ram_mb
                    model_vram_per_actor_mb = get_seisbench_model_vram_mb(
                        self.seisbench_parent_model, 
                        self.seisbench_child_model,
                        default_mb=600.0,
                        logger=self.logger
                    )
                    model_ram_per_actor_mb = get_seisbench_model_ram_mb(
                        self.seisbench_parent_model, 
                        self.seisbench_child_model,
                        use_gpu=True,
                        default_mb=1000.0,
                        logger=self.logger
                    )
                else:
                    trial_model = "eqcct"
                    # Use the measured EQCCT requirements (includes CUDA context + TensorFlow + XLA)
                    from eqcctpro.parallelization import get_eqcct_vram_mb, get_eqcct_ram_mb
                    model_vram_per_actor_mb = get_eqcct_vram_mb()
                    model_ram_per_actor_mb = get_eqcct_ram_mb(use_gpu=True)

                # Import buffer constants for logging clarity
                from eqcctpro.parallelization import VRAM_BUFFER_MB, RAM_BUFFER_MB
                
                # Additional per-task overhead for waveform data buffers and intermediate results
                # (Note: The main safety buffer is already included in model_*_per_actor_mb)
                overhead_vram_per_task_mb = 128.0  # Extra per-task VRAM for waveform data
                overhead_ram_per_task_mb = 256.0   # Extra per-task RAM for waveform data
                
                # Use user-defined VRAM capacity if provided, otherwise calculate based on total hardware and safety cap.
                if self.vram_mb:
                    aggregate_vram_cap_mb = float(self.vram_mb)
                    self.logger.info(f"Using user-defined VRAM capacity: {aggregate_vram_cap_mb:.0f} MB")
                else:
                    per_gpu_total_mb = [get_gpu_vram(gpu_index=g)[0] * 1024.0 for g in gpus_to_use]
                    aggregate_vram_cap_mb = sum(per_gpu_total_mb) * self.gpu_vram_safety_cap
                    self.logger.info(f"Using calculated VRAM capacity: {aggregate_vram_cap_mb:.0f} MB ({sum(per_gpu_total_mb):.0f} MB total × {self.gpu_vram_safety_cap:.0%} safety)")
                
                # Get system RAM capacity (psutil already imported at top of file)
                system_ram_total_mb = psutil.virtual_memory().total / 1e6
                system_ram_available_mb = psutil.virtual_memory().available / 1e6
                # Apply user-defined RAM safety cap to TOTAL system RAM
                aggregate_ram_cap_mb = system_ram_total_mb * self.ram_safety_cap
                
                self.logger.info(f"")
                self.logger.info(f"===== MODEL-AWARE RESOURCE ANALYSIS (GPU) =====")
                self.logger.info(f"Model Type: {trial_model}")
                self.logger.info(f"Model VRAM per Actor: {model_vram_per_actor_mb:.0f} MB (includes {VRAM_BUFFER_MB:.0f} MB safety buffer for Ray/CUDA overhead)")
                self.logger.info(f"Model RAM per Actor: {model_ram_per_actor_mb:.0f} MB (includes {RAM_BUFFER_MB:.0f} MB safety buffer for Ray/framework overhead)")
                self.logger.info(f"")
                if self.vram_mb:
                    self.logger.info(f"VRAM Capacity: {aggregate_vram_cap_mb:.0f} MB (User-Defined)")
                else:
                    self.logger.info(f"VRAM Capacity: {aggregate_vram_cap_mb:.0f} MB (Total: {sum(per_gpu_total_mb):.0f} MB × {self.gpu_vram_safety_cap:.0%} safety)")
                self.logger.info(f"RAM Capacity: {aggregate_ram_cap_mb:.0f} MB (Total: {system_ram_total_mb:.0f} MB × {self.ram_safety_cap:.0%} safety)")
                self.logger.info(f"Currently Available RAM: {system_ram_available_mb:.0f} MB")
                
                # ===== MEMORY-AWARE ACTOR CAPPING (GPU) =====
                # KEY INSIGHT: With TF memory growth enabled, multiple actors CAN share GPU(s).
                # The constraint is MEMORY, not hardware count.
                # CUDA_VISIBLE_DEVICES already limits GPU visibility, so Ray can only see the GPUs we've set.
                # Actors are capped to MEMORY availability in parallelization.py.
                
                # Memory limits (primary constraint)
                max_actors_by_vram = int(aggregate_vram_cap_mb // model_vram_per_actor_mb)
                max_actors_by_ram = int(aggregate_ram_cap_mb // model_ram_per_actor_mb)
                
                # The effective limit is the minimum of VRAM and RAM constraints
                max_actors = max(1, min(max_actors_by_vram, max_actors_by_ram))
                
                # Hardware reference (for logging only - not the limiting factor)
                n_gpus = len(gpus_to_use)
                
                # Check if at least one actor fits in memory
                one_actor_fits_vram = model_vram_per_actor_mb <= aggregate_vram_cap_mb
                one_actor_fits_ram = model_ram_per_actor_mb <= aggregate_ram_cap_mb
                one_actor_fits = one_actor_fits_vram and one_actor_fits_ram
                
                self.logger.info(f"")
                self.logger.info(f"===== MEMORY-AWARE ACTOR CAPPING (GPU) =====")
                self.logger.info(f"GPUs visible to Ray: {n_gpus}")
                self.logger.info(f"Max actors by VRAM: {max_actors_by_vram} ({aggregate_vram_cap_mb:.0f} MB / {model_vram_per_actor_mb:.0f} MB per actor)")
                self.logger.info(f"Max actors by RAM: {max_actors_by_ram} ({aggregate_ram_cap_mb:.0f} MB / {model_ram_per_actor_mb:.0f} MB per actor)")
                self.logger.info(f"Effective max actors: {max_actors} (memory-constrained, not hardware-constrained)")
                self.logger.info(f"Ray will handle GPU scheduling (CUDA_VISIBLE_DEVICES already limits visibility)")
                self.logger.info(f"TF memory growth enabled: Multiple actors can share GPU memory dynamically")
                self.logger.info(f"==========================================")
                
                if not one_actor_fits:
                    mem_issue = "VRAM" if not one_actor_fits_vram else "RAM"
                    self.logger.warning(f"Insufficient {mem_issue} to spawn even 1 actor. Skipping this resource block.")
                    ray.shutdown()
                    continue

                for stations in self.stations2use_list:
                    # Use a 20% step size for concurrency testing as requested
                    step = max(1, int(stations * 0.2))
                    concurrent_predictions_list = sorted(list(set(range(step, stations + 1, step))))
                    
                    # ===== FEASIBILITY CHECK =====
                    # Since actors are CAPPED to MEMORY in parallelization.py, all concurrency levels
                    # are valid as long as the actors fit in memory (checked above).
                    # Tasks beyond actor count are simply queued and processed round-robin.
                    valid_predictions = concurrent_predictions_list
                    invalid_predictions = []  # No invalid predictions since actors are capped to memory
                    
                    # Only block if memory is truly insufficient (already checked above, so this is a safety fallback)
                    if not one_actor_fits:
                        invalid_predictions = concurrent_predictions_list
                        valid_predictions = []
                    
                    if invalid_predictions:
                        for inv_p in invalid_predictions:
                            # Efficiency check: don't record if already tested/skipped
                            key = self._trial_key(
                                num_cpus=len(cpus_to_use),
                                stations=stations,
                                predictions=inv_p,
                                gpu_memory_limit_mb=int(round(model_vram_per_actor_mb)),
                                timechunks=1,
                                model=trial_model,
                                gpus=gpus_to_use,
                            )
                            if key in planned_keys:
                                continue
                            planned_keys.add(key)
                            
                            # Calculate actual memory that would be used (capped to memory availability)
                            actual_actors = min(inv_p, max_actors)
                            est_vram = actual_actors * model_vram_per_actor_mb
                            est_ram = actual_actors * model_ram_per_actor_mb
                            
                            error_msg = (f"[OOM PREVENTION] Cannot fit {actual_actors} actor(s) in memory. "
                                       f"Required: {est_vram:.0f}MB VRAM / {est_ram:.0f}MB RAM. "
                                       f"Available: {aggregate_vram_cap_mb:.0f}MB VRAM / {aggregate_ram_cap_mb:.0f}MB RAM.")
                            
                            self.logger.warning(f"Recording OOM-risk trial skip for {stations} stations, {inv_p} tasks: {error_msg}")
                            
                            # Build minimal trial data for CSV recording
                            oom_comment = f"Requested {inv_p} actors, 0 created (VRAM limited to {aggregate_vram_cap_mb:.0f} MB, {model_vram_per_actor_mb:.0f} MB/actor) - Trial skipped"
                            trial_data = {
                                "Trial Number": trial_num,
                                "Number of Stations Used": stations,
                                "Number of CPUs Allocated for Ray to Use": len(cpus_to_use),
                                "GPUs Used": json.dumps(list(gpus_to_use)),
                                "N ModelActors": 0,  # No actors created - OOM prevention
                                "Inference Actor Memory Limit (MB)": int(round(model_vram_per_actor_mb)),
                                "Number of Concurrent Station Tasks": inv_p,
                                # Include timechunk metadata for consistency
                                "Concurrent Timechunks Used": getattr(self, 'number_of_concurrent_timechunk_predictions', 1),
                                "Total Number of Timechunks": 1, 
                                "Length of Timechunk (min)": getattr(self, 'timechunk_dt', 1.0),
                                "Total Waveform Analysis Timespace (min)": float(getattr(self, 'timechunk_dt', 1.0)),
                                "Model Used": trial_model,
                                "Trial Success": 0,
                                "Error Message": error_msg,
                                "Comments": oom_comment,
                            }
                            from eqcctpro.parallelization import append_trial_row
                            append_trial_row(csv_filepath, trial_data)
                            
                            # Add memory metadata columns to the newly appended row
                            mem_data = build_memory_trial_data(
                                mem_after=get_memory_snapshot(process),
                                process=process,
                                gpu_ids=list(gpus_to_use),
                                model_requested_vram_mb=est_vram,
                                model_requested_ram_mb=est_ram,
                                n_model_actors=0,  # No actors created - OOM prevention
                                model_vram_per_actor_mb=model_vram_per_actor_mb,
                                model_ram_per_actor_mb=model_ram_per_actor_mb,
                                is_gpu_trial=True  # GPU trial: ModelActors use VRAM
                            )
                            update_csv_with_memory(csv_filepath, mem_data)
                            trial_num += 1
                    
                    if not valid_predictions and stations > 0:
                        # If even the first step (20%) exceeds max actors (memory limit), try the max we can support
                        if max_actors >= 1:
                            valid_predictions = [min(step, max_actors)]
                            self.logger.info(f"Station count {stations}: 20% step ({step}) exceeds max actors ({max_actors}). Using max={valid_predictions[0]}")
                        else:
                            self.logger.warning(f"Station count {stations}: Cannot spawn any ModelActors. Skipping.")
                            continue
                    elif len(valid_predictions) < len(concurrent_predictions_list):
                        self.logger.info(f"Trimming concurrency for {stations} stations: {concurrent_predictions_list} → {valid_predictions} (max actors={max_actors})")

                    self.logger.info(f"")
                    self.logger.info(f"Evaluating GPU(s) against {stations} TOTAL STATION(s)")
                    self.logger.info(f"Testing N ModelActors (concurrent predictions): {valid_predictions}")
                    
                    for predictions in valid_predictions:
                        # ===== RIPPER MODE: Fresh Ray instance per trial =====
                        # Ripper mode benefits from a clean GPU state at each trial start
                        # This ensures no memory fragmentation from previous trials affects the current one
                        if self.ripper:
                            self.logger.info(f"")
                            self.logger.info(f"===== RIPPER MODE: Restarting Ray for fresh GPU state =====")
                            
                            # Stop the log drain thread gracefully
                            self.log_queue.put(None)  # Sentinel to stop the drain thread
                            time.sleep(0.5)  # Give thread time to finish
                            
                            # Shutdown and reinitialize Ray
                            ray.shutdown()
                            time.sleep(1.0)  # Allow GPU memory to be fully released
                            
                            # Reinitialize Ray with the same configuration
                            ray.init(ignore_reinit_error=True, num_gpus=len(gpus_to_use), num_cpus=len(cpus_to_use), 
                                    logging_level=logging.FATAL, log_to_driver=False, _temp_dir=self.home_tmp_dir)
                            self.log_queue = Queue()
                            self._log_thread = threading.Thread(target=self._drain_worker_logs, daemon=True)
                            self._log_thread.start()
                            
                            # Reset the high water mark since we cleared GPU memory
                            actors_high_water_mark = 0
                            self.logger.info(f"Ray restarted. GPU memory cleared for ripper trial.")
                            self.logger.info(f"=================================================")
                        
                        # ===== CALCULATE REQUESTED MEMORY FOR THIS TRIAL =====
                        # N ModelActors = predictions (each concurrent task gets its own ModelActor)
                        n_model_actors = predictions
                        requested_vram_mb = n_model_actors * model_vram_per_actor_mb
                        requested_ram_mb = n_model_actors * model_ram_per_actor_mb
                        task_overhead_vram_mb = n_model_actors * overhead_vram_per_task_mb
                        task_overhead_ram_mb = n_model_actors * overhead_ram_per_task_mb
                        total_requested_vram_mb = requested_vram_mb + task_overhead_vram_mb
                        total_requested_ram_mb = requested_ram_mb + task_overhead_ram_mb

                        # ===== OOM PREVENTION: Restart Ray if needed =====
                        # When increasing concurrency, actors from previous trials may still be in GPU memory.
                        # If the new actors + existing actors would exceed VRAM capacity, restart Ray.
                        restart_note = ""
                        if predictions > actors_high_water_mark and actors_high_water_mark > 0:
                            # Estimate total VRAM if we add new actors while existing ones are still in memory
                            estimated_total_vram = (actors_high_water_mark + predictions) * model_vram_per_actor_mb
                            if estimated_total_vram > aggregate_vram_cap_mb:
                                self.logger.warning(f"")
                                self.logger.warning(f"===== RAY RESTART: Clearing GPU Memory =====")
                                self.logger.warning(f"Increasing from {actors_high_water_mark} to {predictions} actors")
                                self.logger.warning(f"Estimated VRAM if additive: {estimated_total_vram:.0f} MB > capacity {aggregate_vram_cap_mb:.0f} MB")
                                self.logger.warning(f"Restarting Ray to prevent OOM...")
                                
                                # Stop the log drain thread gracefully
                                self.log_queue.put(None)  # Sentinel to stop the drain thread
                                time.sleep(0.5)  # Give thread time to finish
                                
                                # Shutdown and reinitialize Ray
                                ray.shutdown()
                                time.sleep(1.0)  # Allow GPU memory to be released
                                
                                # Reinitialize Ray with the same configuration
                                ray.init(ignore_reinit_error=True, num_gpus=len(gpus_to_use), num_cpus=len(cpus_to_use), 
                                        logging_level=logging.FATAL, log_to_driver=False, _temp_dir=self.home_tmp_dir)
                                self.log_queue = Queue()
                                self._log_thread = threading.Thread(target=self._drain_worker_logs, daemon=True)
                                self._log_thread.start()
                                
                                # Set the restart note BEFORE resetting the high water mark
                                restart_note = f"[RAY RESTART] GPU memory cleared (Prev Peak: {actors_high_water_mark} actors). "
                                # Reset the high water mark since we cleared GPU memory
                                actors_high_water_mark = 0
                                self.logger.info(f"Ray restarted successfully. GPU memory cleared.")
                                self.logger.info(f"==========================================")
                                self.logger.info(f"")

                        # Efficiency check: avoid redundant tests for same (stations, predictions) - no VRAM tiers needed
                        config_key = (stations, predictions)
                        if config_key in tested_gpu_configs:
                            continue
                        tested_gpu_configs.add(config_key)

                        # Use model's known VRAM requirement as the gpu_memory_limit_mb (no tiers)
                        gpu_memory_limit_mb = int(round(model_vram_per_actor_mb))

                        key = self._trial_key(
                            num_cpus=len(cpus_to_use),
                            stations=stations,
                            predictions=predictions,
                            gpu_memory_limit_mb=gpu_memory_limit_mb,
                            timechunks=1,  # your GPU eval is explicitly "one timechunk at a time"
                            model=trial_model,
                            gpus=gpus_to_use,  # Pass the actual GPUs being used in this iteration
                        )
                        if key in planned_keys:
                            self.logger.info(f"[SKIP] Already tested: {key}")
                            trials_skipped += 1
                            continue
                        planned_keys.add(key)
                        trials_run += 1
                        self.logger.info("")
                        self.logger.info(f"------- Trial Number: {trial_num} -------")
                        
                        # Get the first timechunk for testing
                        mseed_timechunk_dir_name = self.tasks_picker[0][1]
                        timechunk_dir_path = os.path.join(self.input_dir, mseed_timechunk_dir_name)
                        
                        self.logger.info(f"Stations: {stations}")
                        self.logger.info(f"Concurrent Predictions (N ModelActors): {predictions}")
                        self.logger.info(f"Model VRAM per Actor: {model_vram_per_actor_mb:.0f} MB")
                        self.logger.info(f"Model RAM per Actor: {model_ram_per_actor_mb:.0f} MB")
                        self.logger.info(f"Requested Total VRAM: {total_requested_vram_mb:.0f} MB ({n_model_actors} actors × {model_vram_per_actor_mb:.0f} MB + {task_overhead_vram_mb:.0f} MB overhead)")
                        self.logger.info(f"Requested Total RAM: {total_requested_ram_mb:.0f} MB ({n_model_actors} actors × {model_ram_per_actor_mb:.0f} MB + {task_overhead_ram_mb:.0f} MB overhead)")
                        self.logger.info("")

                        # ===== COMPREHENSIVE MEMORY TRACKING (BEFORE) =====
                        mem_before = get_memory_snapshot(process)
                        vram_before = get_gpu_vram_snapshot(gpus_to_use)
                            
                        try: # To Do: Add Concurrent Timechunks Testing for GPU/CPU too, reference eqcctpro_parallelization()
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
                                ray_cpus=cpus_to_use, 
                                use_gpu=self.use_gpu, 
                                gpu_id=gpus_to_use, 
                                gpu_memory_limit_mb=gpu_memory_limit_mb,  # Per-actor VRAM limit
                                total_vram_pool_mb=aggregate_vram_cap_mb,  # Total VRAM budget for all actors
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
                                timechunk_dt=self.timechunk_dt,
                                model_type=self.model_type, seisbench_parent_model=self.seisbench_parent_model, 
                                seisbench_child_model=self.seisbench_child_model, Detection_threshold=self.Detection_threshold,
                                ripper=self.ripper
                            )
                            
                            # Wait for result
                            log_entry = ray.get(ref)
                            log_queue.put(log_entry)  # Add log entry to the queue
                                
                            # ===== COMPREHENSIVE MEMORY TRACKING (AFTER) =====
                            mem_after = get_memory_snapshot(process)
                            vram_after = get_gpu_vram_snapshot(gpus_to_use)
                            
                            # Build memory data for CSV with model-requested information
                            # GPU trials also track RAM for Ray workers and data processing
                            memory_trial_data = build_memory_trial_data(
                                mem_after=mem_after,
                                mem_before=mem_before,
                                process=process,
                                gpu_ids=list(gpus_to_use),
                                model_requested_vram_mb=total_requested_vram_mb,
                                model_requested_ram_mb=total_requested_ram_mb,
                                n_model_actors=n_model_actors,
                                model_vram_per_actor_mb=model_vram_per_actor_mb,
                                model_ram_per_actor_mb=model_ram_per_actor_mb,
                                is_gpu_trial=True  # GPU trial: ModelActors use VRAM
                            )
                            update_csv(csv_filepath, success=1, error_message=restart_note)
                            update_csv_with_memory(csv_filepath, memory_trial_data)
                                
                        except Exception as e:
                            # Failure occurred, need to add to log 
                            error_msg = restart_note + f"{type(e).__name__}: {str(e)}"
                            # Still capture memory after failure
                            mem_after = get_memory_snapshot(process)
                            vram_after = get_gpu_vram_snapshot(gpus_to_use)
                            memory_trial_data = build_memory_trial_data(
                                mem_after=mem_after,
                                mem_before=mem_before,
                                process=process,
                                gpu_ids=list(gpus_to_use),
                                model_requested_vram_mb=total_requested_vram_mb,
                                model_requested_ram_mb=total_requested_ram_mb,
                                n_model_actors=n_model_actors,
                                model_vram_per_actor_mb=model_vram_per_actor_mb,
                                model_ram_per_actor_mb=model_ram_per_actor_mb,
                                is_gpu_trial=True  # GPU trial: ModelActors use VRAM
                            )
                            update_csv(csv_filepath, success=0, error_message=error_msg)
                            update_csv_with_memory(csv_filepath, memory_trial_data)
                            self.logger.info(f"Trial {trial_num} FAILED: {error_msg}")
                            
                        # Write log entries from the queue to the file
                        log_entry = None
                        while not log_queue.empty():
                            log_entry = log_queue.get()
                        if log_entry is not None:
                            self.logger.info(f"{log_entry}")
                            
                        remove_output_subdirs(self.output_dir, logger=self.logger) 
                        trial_num += 1
                            
                        # Update the high water mark of actors spawned in this Ray session
                        # This is used to detect when we need to restart Ray to prevent OOM
                        actors_high_water_mark = max(actors_high_water_mark, predictions)
                        
                        # ===== MEMORY LOGGING =====
                        delta = compute_memory_delta(mem_before, mem_after)
                        actual_vram_used = vram_before.get("gpu_free_vram_mb", 0) - vram_after.get("gpu_free_vram_mb", 0)
                        actual_ram_used = mem_after.combined_rss_mb - mem_before.combined_rss_mb
                        
                        self.logger.info(f"[MODEL REQUESTED] VRAM: {total_requested_vram_mb:.0f} MB | RAM: {total_requested_ram_mb:.0f} MB")
                        self.logger.info(f"[ACTUAL MEASURED] VRAM Used: {max(0, actual_vram_used):.0f} MB | RAM Used: {max(0, actual_ram_used):.0f} MB")
                        self.logger.info(
                            f"[MEM] Baseline: {mem_before.process_rss_mb:.2f} MB | After run: {mem_after.process_rss_mb:.2f} MB "
                            f"| Δrun: {delta['process_ram_delta_mb']:.2f} MB | Peak≈{mem_after.process_peak_mb:.2f} MB"
                        )
                        self.logger.info(
                            f"[MEM] System Available: {mem_before.system_available_mb:.2f} MB → {mem_after.system_available_mb:.2f} MB "
                            f"| Raylets: {mem_after.num_raylet_processes} (Total: {mem_after.total_raylet_ram_mb:.2f} MB)"
                        )
                        self.logger.info(
                            f"[VRAM] Total: {vram_before.get('gpu_total_vram_mb', 0):.2f} MB | Free Before: {vram_before.get('gpu_free_vram_mb', 0):.2f} MB "
                            f"| Free After: {vram_after.get('gpu_free_vram_mb', 0):.2f} MB"
                            )

                        # ===== CLEANUP =====
                        try: del ref
                        except NameError: pass
                        try: del log_entry
                        except NameError: pass

                        gc.collect()
                        time.sleep(0.1)

                        mem_after_clean = get_memory_snapshot(process)
                        freed_mb = mem_after.combined_rss_mb - mem_after_clean.combined_rss_mb
                        self.logger.info(f"[MEM] Freed ~{max(freed_mb, 0):.2f} MB; Post-clean total: {mem_after_clean.combined_rss_mb:.2f} MB")
                        self.logger.info("")
                
                # stop log forwarder
                self.log_queue.put(None) # remember, log_queue is a Ray Queue actor, and will only exist while Ray is still active (cannot be after the .shutdown())
                self._log_thread.join(timeout=2)

                ray.shutdown()  # Shutdown Ray after all testing
                self.logger.info(f"Ray Successfully Shutdown.")

        self.logger.info(f"Testing complete.")
        self.logger.info(f"")
        self.logger.info(f"Trial Summary: {trials_run} new trial(s) executed, {trials_skipped} trial(s) skipped (already in CSV)")
        self.logger.info(f"Finding Optimal Configurations...")
        self.logger.info(f"Recalculating optimal configurations from all trial data (including new trials)...")
        # Compute optimal configurations (GPU)
        df = pd.read_csv(csv_filepath)
        optimal_configuration_df, best_overall_usecase_df = find_optimal_configurations_gpu(df)
        # Overwrite existing files with recalculated optimal configurations
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
        cap = float(self.gpu_vram_safety_cap)
        if not (0.0 < cap <= 0.99):
            raise ValueError(f"gpu_vram_safety_cap must be in (0, 0.99], got {cap}.")

        gpus = self.selected_gpus if self.selected_gpus else list_gpu_ids()
        if not gpus:
            raise RuntimeError("No GPUs detected for VRAM calculation.")

        per_gpu_budget_mb = []
        for gid in gpus:
            total_gb, free_gb = get_gpu_vram(gpu_index=gid)
            total_mb = float(total_gb) * 1024.0
            free_mb  = float(free_gb)  * 1024.0

            # hard cap vs physical total, plus never exceed currently free memory
            budget_mb = min(total_mb * cap, free_mb)
            per_gpu_budget_mb.append(budget_mb)

            self.logger.info(
                f"GPU {gid}: total={total_gb:.2f} GB, free={free_gb:.2f} GB, "
                f"budget={budget_mb/1024.0:.2f} GB (cap={cap:.2f})"
            )

        budget_mb_min = float(min(per_gpu_budget_mb))
        self.logger.info(f"Using per-GPU VRAM budget = {budget_mb_min:.0f} MB (min across selected GPUs).")
        return budget_mb_min


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
        model_used = best_config_dict.get("Model Used")
        
        self.logger.info("")
        self.logger.info(f"------- Finding the Best Overall CPU Usecase Configuration Based on Available Trial Data in {self.eval_sys_results_dir} -------")
        self.logger.info(f"Model Used: {model_used}")
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
        self.logger.info(f"Model Used: {best_config.get('Model Used')}")
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
        vram_used = row.get("Inference Actor Memory Limit (MB)")
        model_used = row.get("Model Used")

        self.logger.info("")
        self.logger.info(f"------- Finding the Best Overall GPU Usecase Configuration Based on Available Trial Data in {self.eval_sys_results_dir} -------")
        self.logger.info("")
        self.logger.info(f"Model Used: {model_used}")
        self.logger.info(f"CPU(s): {num_cpus}")
        self.logger.info(f"GPU ID(s): {num_gpus_list}")
        self.logger.info(f"Concurrent Predictions: {num_concurrent}")
        self.logger.info(f"Intra-parallelism Threads: {intra_threads}")
        self.logger.info(f"Inter-parallelism Threads: {inter_threads}")
        self.logger.info(f"Stations: {num_stations}")
        self.logger.info(f"Inference Actor Memory Limit (MB): {vram_used}")
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
        df_optimal["Inference Actor Memory Limit (MB)"] = pd.to_numeric(df_optimal["Inference Actor Memory Limit (MB)"], errors="coerce")

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
        self.logger.info(f"Inference Actor Memory Limit (MB): {best_config['Inference Actor Memory Limit (MB)']}")
        self.logger.info(f"Total Runtime (s): {best_config['Total Run time for Picker (s)']}")
