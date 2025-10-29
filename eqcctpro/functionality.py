"""
functionality.py controls all the functionality of EQCCTPro, specifically how we access mseed_predictor() and parallel_predict. 
It is a level of abstraction so we can make the code more concise and cleaner
"""
import os 
import ray
import sys
import ast
import math
import queue 
import psutil
import random
import numbers
import logging
import threading
from tools import *
from parallelization import *
from obspy import UTCDateTime
from ray.util.queue import Queue
from datetime import datetime, timedelta
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

        self.logger.info(f"------- Welcome to EQCCTPro -------")
        self.logger.info("")

        # If the user passed a GPU but no valid VRAM, need to exit 
        if self.use_gpu and not (isinstance(self.vram_mb, numbers.Real) and math.isfinite(self.vram_mb) and self.vram_mb > 0): 
            self.logger.error(f"No numerical VRAM passed. Please provide vram_mb (MB per Raylet per GPU) as a positive real number. Exiting...")
            sys.exit(1)

        # We need to ensure that the vram specified does not exceed the capabilities of the system, if not, we need to exit safely before it happens
        if self.use_gpu: 
            check_vram(self, model_vram_mb=1500) # EQCCT takes up to 1.2 GB of RAM, round up to 1.5 GB for safety
    
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

    # Calculates the amount of available VRAM within an the first GPU
    # def calculate_vram(self):
    #     self.logger.info(f"Utilizing available VRAM within Ray Memory Usage Threshold Limit of 0.95...")
    #     total_vram, available_vram = get_gpu_vram()
    #     self.logger.info(f"Total VRAM: {total_vram:.2f} GB")
    #     self.logger.info(f"Available VRAM: {available_vram:.2f} GB")

    #     free_vram = total_vram * 0.95 if available_vram / total_vram >= 0.95 else available_vram
    #     self.logger.info(f"Using {round(free_vram, 2)} GB VRAM (within 95% VRAM threshold).")
    #     return free_vram * 1024  # Convert to MB

    def configure_cpu(self): 
        # We need to configure the tf_environ for the CPU configuration that is being inputted
        self.logger.info("")
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
        self.logger.info("------- END OF FILE -------")
        
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
