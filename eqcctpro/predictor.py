import re
import os
import gc 
import sys
import ast
import ray
import csv
import json 
import time
import glob
import queue
import obspy
import shutil
import psutil
import random
import pynvml 
import logging
import warnings
import platform
import importlib
import numpy as np
import pandas as pd
from os import listdir
from pathlib import Path
from datetime import datetime, timedelta

# GLOBAL VARIABLES 
tf = None 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'

CANONICAL_CSV_HEADER = [
    "Trial Number",
    "Stations Used",
    "Number of Stations Used",
    "Number of CPUs Allocated for Ray to Use",
    "Intra-parallelism Threads",
    "Inter-parallelism Threads",
    "GPUs Used",
    "VRAM Used Per Task",
    "Total Waveform Analysis Timespace (min)",
    "Total Number of Timechunks",
    "Concurrent Timechunks Used",
    "Length of Timechunk (min)",
    "Number of Concurrent Station Tasks",
    "Total Run time for Picker (s)",
    "Trial Success",
    "Error Message",
]

_TIMECHUNK_RE = re.compile(r"^\d{8}T\d{6}Z_\d{8}T\d{6}Z$")

def looks_like_timechunk_id(name: str) -> bool:
    return bool(_TIMECHUNK_RE.match(name or ""))

def build_station_list_from_dir(input_dir: str) -> list[str]:
    """
    Robustly discover stations under a timechunk directory.
    Accepts files like *.mseed/*.sac or one-dir-per-station structures.
    """
    stations = set()

    # 1) Files directly inside input_dir
    for p in glob.glob(os.path.join(input_dir, "*")):
        base = os.path.basename(p)
        if os.path.isfile(p):
            # file path — take stem without extension
            stations.add(os.path.splitext(base)[0])

    # 2) One subdir per station (e.g., input_dir/AT01/*.mseed)
    for p in glob.glob(os.path.join(input_dir, "*")):
        if os.path.isdir(p):
            stations.add(os.path.basename(p))

    # Filter out anything that looks like a timechunk id (safety)
    stations = [s for s in stations if not looks_like_timechunk_id(s)]

    return sorted(stations)

def tf_environ(gpu_id, gpu_memory_limit_mb=None, gpus_to_use=None, intra_threads=None, inter_threads=None, log_device=True,):
    """
    Configure TensorFlow to use fixed VRAM slices per visible GPU.
    Call this ONCE per Ray actor, BEFORE building/loading any TF model.
    """
    from datetime import datetime
    import os

    # 0) Visibility must be set BEFORE importing tensorflow
    if gpu_id == -1 or not gpus_to_use:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print(f"[{datetime.now()}] GPU disabled (CPU-only).")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus_to_use))
        print(f"[{datetime.now()}] GPU enabled. Visible GPU IDs: {gpus_to_use}")

    # 1) Now import TF (it will honor visibility)
    import tensorflow as tf

    if log_device:
        tf.debugging.set_log_device_placement(True)

    # 2) Threading (optional)
    if intra_threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra_threads))
        print(f"[{datetime.now()}] Intra-op threads = {intra_threads}")
    if inter_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(int(inter_threads))
        print(f"[{datetime.now()}] Inter-op threads = {inter_threads}")

    # 3) Configure fixed VRAM slices on all visible GPUs
    vis_gpus = tf.config.list_physical_devices("GPU")
    if not vis_gpus:
        print(f"[{datetime.now()}] No GPUs visible; proceeding on CPU.")
        return {"logical_gpus": [], "physical_gpus": []}

    if gpu_memory_limit_mb is None or gpu_memory_limit_mb <= 0:
        raise ValueError(
            "gpu_memory_limit_mb must be a positive integer when using fixed VRAM slicing."
        )

    try:
        for gpu in vis_gpus:
            # One logical device per physical GPU, each with a hard VRAM cap
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(gpu_memory_limit_mb))]
            )
        # Force logical devices to materialize
        logical = tf.config.list_logical_devices("GPU")
        print(
            f"[{datetime.now()}] Fixed VRAM slicing enabled: "
            f"{gpu_memory_limit_mb} MB per logical GPU "
            f"({len(logical)} logical over {len(vis_gpus)} physical)."
        )
    except RuntimeError as e:
        # Happens if any TF GPU context was already initialized
        raise RuntimeError(
            "Failed to set logical device configuration. "
            "Ensure tf_environ() is called before any TensorFlow GPU ops or model creation.\n"
            f"Original error: {e}"
        )

    # return {
    #     "logical_gpus": [d.name for d in tf.config.list_logical_devices("GPU")],
    #     "physical_gpus": [d.name for d in vis_gpus],
    #     "memory_limit_mb": int(gpu_memory_limit_mb),
    # }


def get_gpu_vram():
    """Retrieve total and free VRAM (in GB) for the current GPU."""
    pynvml.nvmlInit()  # Initialize NVML
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get first GPU
    total_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)  # Convert bytes to GB
    free_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024**3)  # Convert bytes to GB
    pynvml.nvmlShutdown()  # Shutdown NVML
    return total_vram, free_vram

def list_gpu_ids():
    """List all available GPU IDs on the system."""
    pynvml.nvmlInit()  # Initialize NVML
    gpu_count = pynvml.nvmlDeviceGetCount()  # Get number of GPUs
    gpu_ids = list(range(gpu_count))  # Create a list of GPU indices
    pynvml.nvmlShutdown()  # Shutdown NVML
    return gpu_ids          
            
def prepare_csv(csv_file_path, gpu:bool=False):
    """
    Loads or initializes the CSV file for storing test results.
    """
    if os.path.exists(csv_file_path):
        print(f"\n[{datetime.now()}] Loading existing CSV file from '{csv_file_path}'...")
        return pd.read_csv(csv_file_path)
    print(f"[{datetime.now()}] CSV file not found. Creating a new CSV file at '{csv_file_path}'...")
    
    columns = CANONICAL_CSV_HEADER
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_file_path, index=False)

def append_trial_row(csv_path: str, trial_data: dict):
    """
    Append a complete trial row to the CSV with all fields populated.
    """
    csvp = Path(csv_path) 
    
    # Ensure header exists with canonical order
    if not csvp.exists():
        pd.DataFrame(columns=CANONICAL_CSV_HEADER).to_csv(csvp, index=False)

    df_existing = pd.read_csv(csvp, keep_default_na=False)
    
    # Align row to the canonical header (use empty string for missing keys)
    row = {col: trial_data.get(col, "") for col in CANONICAL_CSV_HEADER}
    
    # Auto-number trials if not provided
    if pd.isna(row["Trial Number"]) or row["Trial Number"] == "" or row["Trial Number"] is None:
        row["Trial Number"] = len(df_existing) + 1

    df_new = pd.DataFrame([row], columns=CANONICAL_CSV_HEADER)
    df_out = pd.concat([df_existing, df_new], ignore_index=True)
    df_out.to_csv(csvp, index=False)
    
    print(f"[{datetime.now()}] Appended trial {row['Trial Number']} to {csv_path}")


def update_csv(csv_filepath, success, error_message):
    df = pd.read_csv(csv_filepath)
    last_idx = df.index[-1] # Get last row id number
    df.loc[last_idx, 'Trial Success'] = success # Access value at row last_idx, column 'Trial Success' 
    df.loc[last_idx, 'Error Message'] = error_message # Access value at row last_idx, column 'Error Message'

    df.to_csv(csv_filepath, index=False)

def generate_station_list(starting_amount_of_stations, total_num_stations_to_use, station_list_step_size):
    if total_num_stations_to_use == 1:
        return [1]
    elif total_num_stations_to_use <= 10:
        return list(range(1, total_num_stations_to_use + 1))
    elif starting_amount_of_stations == 1 and station_list_step_size == 1:
        # Numbers 1-10
        station_list = list(range(1, 11))
        
        # Multiples of 5 up to total_num_stations_to_use
        multiples_of_5 = list(range(15, total_num_stations_to_use + 1, 5))
        
        # Any additional numbers between 21 and total_num_stations_to_use
        additional_numbers = list(range(21, total_num_stations_to_use + 1))
        
        # Combine lists while ensuring uniqueness
        return sorted(set(station_list + multiples_of_5 + additional_numbers))
    else: 
        return list(range(starting_amount_of_stations, total_num_stations_to_use + 1, station_list_step_size))


def remove_directory(path):
    """
    Removes the specified directory if it exists.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[{datetime.now()}] Removed directory: {path}")
    else:
        print(f"[{datetime.now()}] Directory '{path}' does not exist anymore.")
        
def remove_output_subdirs(path): 
    """
    Removes all subdirectories within the specified directory, but not the directory itself.
    """
    if os.path.exists(path) and os.path.isdir(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                    print(f"[{datetime.now()}] Removed subdirectory: {item_path}")
                except Exception as e:
                    print(f"[{datetime.now()}] Failed to remove subdirectory: {item_path}. Error: {e}")
    elif not os.path.exists(path):
        print(f"[{datetime.now()}] Directory '{path}' does not exist.")
    elif not os.path.isdir(path):
        print(f"[{datetime.now()}] '{path}' is not a directory.")


def find_optimal_configurations_cpu(df):
    """
    Find:
    1. The best number of concurrent predictions for each (stations, CPUs) pair that results in the fastest runtime.
    2. The overall best configuration balancing stations, CPUs, and runtime.
    """

    # Convert relevant columns to numeric, handling NaNs gracefully
    df["Number of Stations Used"] = pd.to_numeric(df["Number of Stations Used"], errors="coerce")
    df["Number of CPUs Allocated for Ray to Use"] = pd.to_numeric(df["Number of CPUs Allocated for Ray to Use"], errors="coerce")
    df["Total Number of Timechunks"] = pd.to_numeric(df["Total Number of Timechunks"], errors="coerce")
    df["Concurrent Timechunks Used"] = pd.to_numeric(df["Concurrent Timechunks Used"], errors="coerce")
    df["Number of Concurrent Station Tasks"] = pd.to_numeric(df["Number of Concurrent Station Tasks"], errors="coerce")
    df["Total Run time for Picker (s)"] = pd.to_numeric(df["Total Run time for Picker (s)"], errors="coerce")
    

    # Drop rows with missing values in these essential columns
    df_cleaned = df.dropna(subset=["Number of Stations Used", "Number of CPUs Allocated for Ray to Use", 
                                "Concurrent Timechunks Used", "Number of Concurrent Station Tasks", "Total Run time for Picker (s)"])

    # Find the best concurrent prediction configuration for each combination of (Stations, Timechunks, CPUs)
    optimal_concurrent_preds = df_cleaned.loc[
        df_cleaned.groupby(["Number of Stations Used", "Concurrent Timechunks Used", "Number of CPUs Allocated for Ray to Use"])
        ["Total Run time for Picker (s)"].idxmin()
    ]

    # Define what "moderate" means in terms of CPU usage (e.g., middle 50% of available CPUs)
    cpu_min = df_cleaned["Number of CPUs Allocated for Ray to Use"].quantile(0.25)
    cpu_max = df_cleaned["Number of CPUs Allocated for Ray to Use"].quantile(0.75)

    # Filter for rows within the moderate CPU range
    df_moderate_cpus = df_cleaned[(df_cleaned["Number of CPUs Allocated for Ray to Use"] >= cpu_min) & 
                                (df_cleaned["Number of CPUs Allocated for Ray to Use"] <= cpu_max)]

    # Sort by the highest number of stations first, then by the fastest runtime
    best_overall_config = df_moderate_cpus.sort_values(
        by=["Number of Stations Used", "Total Run time for Picker (s)"], 
        ascending=[False, True]  # Maximize stations, minimize runtime
    ).iloc[0]
    
    # Format the output for human readability
    formatted_output = {
        "Trial Number": best_overall_config["Trial Number"],
        "Number of Stations Used": best_overall_config["Number of Stations Used"],
        "Total Number of Timechunks": best_overall_config["Total Number of Timechunks"],
        "Concurrent Timechunks Used": best_overall_config["Concurrent Timechunks Used"],
        "Length of Timechunk (min)": str(best_overall_config["Length of Timechunk (min)"]),
        "Total Waveform Analysis Timespace (min)": str(best_overall_config["Total Waveform Analysis Timespace (min)"]),
        "Number of Concurrent Station Tasks per Timechunk": best_overall_config["Number of Concurrent Station Tasks"],
        "Number of CPUs Allocated for Ray to Use": best_overall_config["Number of CPUs Allocated for Ray to Use"],
        "Intra-parallelism Threads": best_overall_config["Intra-parallelism Threads"],
        "Inter-parallelism Threads": best_overall_config["Inter-parallelism Threads"],
        "Total Run time for Picker (s)": best_overall_config["Total Run time for Picker (s)"],
        "Trial Success": best_overall_config["Trial Success"],
        "Error Message": best_overall_config["Error Message"],
    }
    
    best_overall_df = pd.DataFrame([formatted_output])


    return optimal_concurrent_preds, best_overall_df


def find_optimal_configuration_cpu(best_overall_usecase:bool, eval_sys_results_dir:str, cpu:int=None, station_count:int=None): 
    # Check if eval_sys_results_dir is valid
    if not eval_sys_results_dir or not os.path.isdir(eval_sys_results_dir):
        print(f"Error: The provided directory path '{eval_sys_results_dir}' is invalid or does not exist.")
        print("Please provide a valid directory path for the input parameter 'csv_dir'.")
        return exit()  # Exit early if the directory is invalid
    
    if best_overall_usecase is True: 
        file_path = f"{eval_sys_results_dir}/best_overall_usecase_cpu.csv"

        # Check if the CSV file exists before reading
        if not os.path.exists(file_path):
            print(f"[{datetime.now()}] Error: The file '{file_path}' does not exist. Ensure the file is in the correct directory.")
            return exit()

        # Load the CSV
        df_best_overall = pd.read_csv(file_path)
        # Convert into a dictionary for easy access
        best_config_dict = df_best_overall.set_index(df_best_overall.columns[0]).to_dict()[df_best_overall.columns[1]]

        # Extract required values
        num_cpus = best_config_dict.get("Number of CPUs Allocated for Ray to Use")
        waveform_timespace = best_config_dict.get("Total Waveform Analysis Timespace (min)")
        total_num_timechunks = best_config_dict.get("Total Number of Timechunks")
        num_concurrent_timechunks = best_config_dict.get("Concurrent Timechunks Used")
        length_of_timechunks = best_config_dict.get("Length of Timechunk (min)")
        num_concurrent_stations = best_config_dict.get("Number of Concurrent Station Tasks")
        intra_threads = best_config_dict.get("Intra-parallelism Threads")
        inter_threads = best_config_dict.get("Inter-parallelism Threads")
        num_stations = best_config_dict.get("Number of Stations Used")
        total_runtime = best_config_dict.get("Total Run time for Picker (s)")
        
        print("\nBest Overall Usecase Configuration Based on Trial Data:")
        print(f"CPU: {num_cpus}\n"
        f"Intra-parallelism Threads: {intra_threads}\n"
        f"Inter-parallelism Threads: {inter_threads}\n"
        f"Waveform Timespace: {waveform_timespace}"
        f"Total Number of Timechunks: {total_num_timechunks}"
        f"Length of Timechunks (min): {length_of_timechunks}"
        f"Concurrent Timechunks: {num_concurrent_stations}\n"
        f"Concurrent Stations: {num_concurrent_stations}\n"
        f"Stations: {num_stations}\n"
        f"Total Runtime (s): {total_runtime}")

        # Return the extracted values
        return int(float(num_cpus)), int(float(num_concurrent_stations)), int(float(intra_threads)), int(float(inter_threads)), int(float(num_stations))
    
    else: # Optimal Configuration for User-Specified CPUs and Number of Stations to use
        # Ensure valid CPU and station count values
        if cpu is None or station_count is None:
            print("Error: CPU and station_count must have valid values.")
            return exit()
        
        file_path = f"{eval_sys_results_dir}/optimal_configurations_cpu.csv"

        # Check if the CSV file exists before reading
        if not os.path.exists(file_path):
            print(f"[{datetime.now()}] Error: The file '{file_path}' does not exist. Ensure the file is in the correct directory.")
            return exit() 
        
        
        df_optimal = pd.read_csv(file_path)

        # Convert relevant columns to numeric, handling NaNs gracefully
        df_optimal["Number of Stations Used"] = pd.to_numeric(df_optimal["Number of Stations Used"], errors="coerce")
        df_optimal["Number of CPUs Allocated for Ray to Use"] = pd.to_numeric(df_optimal["Number of CPUs Allocated for Ray to Use"], errors="coerce")
        df_optimal["Number of Concurrent Station Tasks"] = pd.to_numeric(df_optimal["Number of Concurrent Station Tasks"], errors="coerce")
        df_optimal["Total Run time for Picker (s)"] = pd.to_numeric(df_optimal["Total Run time for Picker (s)"], errors="coerce")
        filtered_df = df_optimal[
        (df_optimal["Number of CPUs Allocated for Ray to Use"] == cpu) &
        (df_optimal["Number of Stations Used"] == station_count)]
        if filtered_df.empty:
            print("No matching configuration found. Please enter a valid entry.")
            exit() 

        # Find the best configuration (fastest runtime)
        best_config = filtered_df.nsmallest(1, "Total Run time for Picker (s)").iloc[0]
        
        print("\nBest Configuration for Requested Input Parameters Based on Trial Data:")
        print(f"CPU: {cpu}\nConcurrent Predictions: {best_config['Number of Concurrent Station Tasks']}\n"
            f"Intra-parallelism Threads: {best_config['Intra-parallelism Threads']}\n"
            f"Inter-parallelism Threads: {best_config['Inter-parallelism Threads']}\n"
            f"Stations: {station_count}\nTotal Runtime (s): {best_config['Total Run time for Picker (s)']}")

        return int(float(cpu)), int(float(best_config["Number of Concurrent Station Tasks"])), int(float(best_config["Intra-parallelism Threads"])), int(float(best_config["Inter-parallelism Threads"])), int(float(station_count))


def find_optimal_configurations_gpu(df):
    """
    Find:
    1. The best number of concurrent predictions for each (stations, GPUs, VRAM, CPUs) pair that results in the fastest runtime.
    2. The overall best configuration balancing stations, GPUs, CPUs, VRAM, and runtime.
    """
    # Convert relevant columns to numeric, handling NaNs gracefully
    numeric_cols = [
        "Number of Stations Used", "Number of CPUs Allocated for Ray to Use",
        "Number of Concurrent Station Tasks", "Total Run time for Picker (s)",
        "VRAM Used Per Task"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["GPUs Used"] = df["GPUs Used"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["GPUs Used"] = df["GPUs Used"].apply(lambda x: tuple(x) if isinstance(x, list) else (x,))

    # Drop rows where essential columns are missing
    df_cleaned = df.dropna(subset=numeric_cols + ["GPUs Used"])

    # Find the best number of concurrent predictions for each (Stations, CPUs, GPUs, VRAM) combination
    optimal_concurrent_preds = df_cleaned.loc[
        df_cleaned.groupby(["Number of Stations Used", "Number of CPUs Allocated for Ray to Use", 
                            "GPUs Used", "VRAM Used Per Task"])
        ["Total Run time for Picker (s)"].idxmin()
    ]

    optimal_concurrent_preds["GPUs Used"] = optimal_concurrent_preds["GPUs Used"].apply(lambda x: list(x) if isinstance(x, tuple) else x)

    # Define what "moderate" means in terms of VRAM usage (e.g., middle 50% of available VRAM)
    vram_min = df_cleaned["VRAM Used Per Task"].quantile(0.25)
    vram_max = df_cleaned["VRAM Used Per Task"].quantile(0.75)

    # Filter for rows within the moderate VRAM range
    df_moderate_vram = df_cleaned[
        (df_cleaned["VRAM Used Per Task"] >= vram_min) & 
        (df_cleaned["VRAM Used Per Task"] <= vram_max)
    ]

    # Sort by the highest number of stations first, then by the fastest runtime
    best_overall_config = df_moderate_vram.sort_values(
        by=["Number of Stations Used", "Total Run time for Picker (s)"], 
        ascending=[False, True]  # Maximize stations, minimize runtime
    ).iloc[0]

    formatted_output = {
        "Trial Number": best_overall_config["Trial Number"],
        "Number of Stations Used": best_overall_config["Number of Stations Used"],
        "Total Number of Timechunks": best_overall_config["Total Number of Timechunks"],
        "Concurrent Timechunks Used": best_overall_config["Concurrent Timechunks Used"],
        "Length of Timechunk (min)": str(best_overall_config["Length of Timechunk (min)"]),
        "Total Waveform Analysis Timespace (min)": str(best_overall_config["Total Waveform Analysis Timespace (min)"]),
        "Number of Concurrent Station Tasks per Timechunk": best_overall_config["Number of Concurrent Station Tasks"],
        "Number of CPUs Allocated for Ray to Use": best_overall_config["Number of CPUs Allocated for Ray to Use"],
        "GPUs Used": best_overall_config["GPUs Used"],
        "VRAM Used Per Task": best_overall_config["VRAM Used Per Task"],
        "Intra-parallelism Threads": best_overall_config["Intra-parallelism Threads"],
        "Inter-parallelism Threads": best_overall_config["Inter-parallelism Threads"],
        "Total Run time for Picker (s)": best_overall_config["Total Run time for Picker (s)"],
        "Trial Success": best_overall_config["Trial Success"],
        "Error Message": best_overall_config["Error Message"],
    }

    best_overall_df = pd.DataFrame([formatted_output])

    return optimal_concurrent_preds, best_overall_df


def find_optimal_configuration_gpu(best_overall_usecase: bool, eval_sys_results_dir: str, num_cpus: int = None, num_gpus: list = None, station_count: int = None):
    """
    Find the optimal GPU configuration for a given number of CPUs, GPUs, and stations.
    Returns the best configuration including CPUs, concurrent predictions, intra/inter parallelism threads,
    GPUs, VRAM, and stations.
    """

    # Check if eval_sys_results_dir is valid
    if not eval_sys_results_dir or not os.path.isdir(eval_sys_results_dir):
        print(f"Error: The provided directory path '{eval_sys_results_dir}' is invalid or does not exist.")
        print("Please provide a valid directory path for the input parameter 'csv_dir'.")
        return None  # Exit early if the directory is invalid

    if best_overall_usecase:
        file_path = f"{eval_sys_results_dir}/best_overall_usecase_gpu.csv"

        # Check if the CSV file exists before reading
        if not os.path.exists(file_path):
            print(f"[{datetime.now()}] Error: The file '{file_path}' does not exist. Ensure the file is in the correct directory.")
            return None

        # Load the CSV
        df_best_overall = pd.read_csv(file_path, header=None, index_col=0)

        # Convert into a dictionary for easy access
        best_config_dict = df_best_overall.to_dict()[1]  # Extract key-value pairs

        # Extract required values
        num_cpus = best_config_dict.get("Number of CPUs Allocated for Ray to Use")
        num_concurrent_stations = best_config_dict.get("Number of Concurrent Station Tasks")
        intra_threads = best_config_dict.get("Intra-parallelism Threads")
        inter_threads = best_config_dict.get("Inter-parallelism Threads")
        num_stations = best_config_dict.get("Number of Stations Used")
        total_runtime = best_config_dict.get("Total Run time for Picker (s)")
        vram_used = best_config_dict.get("VRAM Used Per Task")
        num_gpus_st = best_config_dict.get("GPUs Used")
        num_gpus = ast.literal_eval(num_gpus_st)
        
        print("\nBest Overall Usecase Configuration Based on Trial Data:")
        print(f"CPU: {num_cpus}\n"
              f"GPU ID(s): {num_gpus}\n"
              f"Concurrent Predictions: {num_concurrent_stations}\n"
              f"Intra-parallelism Threads: {intra_threads}\n"
              f"Inter-parallelism Threads: {inter_threads}\n"
              f"Stations: {num_stations}\n"
              f"VRAM Used per Task: {vram_used}\n"
              f"Total Runtime (s): {total_runtime}")

        return int(float(num_cpus)), int(float(num_concurrent_stations)), int(float(intra_threads)), int(float(inter_threads)), num_gpus, int(float(vram_used)), int(float(num_stations))

    else:  # Optimal Configuration for User-Specified CPUs, GPUs, and Number of Stations to use
        # Ensure valid CPU, GPU, and station count values
        if num_cpus is None or station_count is None or num_gpus is None:
            print("Error: num_cpus, station_count, and num_gpus must have valid values.")
            return None

        file_path = f"{eval_sys_results_dir}/optimal_configurations_gpu.csv"

        # Check if the CSV file exists before reading
        if not os.path.exists(file_path):
            print(f"[{datetime.now()}] Error: The file '{file_path}' does not exist. Ensure the file is in the correct directory.")
            return None

        df_optimal = pd.read_csv(file_path)

        # Convert relevant columns to numeric, handling NaNs gracefully
        df_optimal["Number of Stations Used"] = pd.to_numeric(df_optimal["Number of Stations Used"], errors="coerce")
        df_optimal["Number of CPUs Allocated for Ray to Use"] = pd.to_numeric(df_optimal["Number of CPUs Allocated for Ray to Use"], errors="coerce")
        df_optimal["Number of Concurrent Station Tasks"] = pd.to_numeric(df_optimal["Number of Concurrent Station Tasks"], errors="coerce")
        df_optimal["Total Run time for Picker (s)"] = pd.to_numeric(df_optimal["Total Run time for Picker (s)"], errors="coerce")
        df_optimal["VRAM Used Per Task"] = pd.to_numeric(df_optimal["VRAM Used Per Task"], errors="coerce")

        # Convert "GPUs Used" from string representation to list
        df_optimal["GPUs Used"] = df_optimal["GPUs Used"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Convert GPU lists to tuples for comparison
        df_optimal["GPUs Used"] = df_optimal["GPUs Used"].apply(lambda x: tuple(x) if isinstance(x, list) else (x,))

        # Ensure num_gpus is in tuple format for comparison
        num_gpus_tuple = tuple(num_gpus) if isinstance(num_gpus, list) else (num_gpus,)

        filtered_df = df_optimal[
            (df_optimal["Number of CPUs Allocated for Ray to Use"] == num_cpus) &
            (df_optimal["GPUs Used"] == num_gpus_tuple) &
            (df_optimal["Number of Stations Used"] == station_count)
        ]

        if filtered_df.empty:
            print("No matching configuration found. Please enter a valid entry.")
            exit()

        # Find the best configuration (fastest runtime)
        best_config = filtered_df.nsmallest(1, "Total Run time for Picker (s)").iloc[0]
        
        print("\nBest Configuration for Requested Application Usecase Based on Trial Data:")
        print(f"CPU: {num_cpus}\n"
              f"GPU: {num_gpus}\n"
              f"Concurrent Predictions: {best_config['Number of Concurrent Station Tasks']}\n"
              f"Intra-parallelism Threads: {best_config['Intra-parallelism Threads']}\n"
              f"Inter-parallelism Threads: {best_config['Inter-parallelism Threads']}\n"
              f"Stations: {station_count}\n"
              f"VRAM Used per Task: {best_config['VRAM Used Per Task']}\n"
              f"Total Runtime (s): {best_config['Total Run time for Picker (s)']}")

        return int(float(best_config["Number of CPUs Allocated for Ray to Use"])), \
               int(float(best_config["Number of Concurrent Station Tasks"])), \
               int(float(best_config["Intra-parallelism Threads"])), \
               int(float(best_config["Inter-parallelism Threads"])), \
               num_gpus, \
               int(float(best_config["VRAM Used Per Task"])), \
               int(float(station_count))

                    
class EQCCTMSeedRunner():  
    """run_EQCCT_Mseed class for running the run_EQCCT_Mseed functions for multiple instances of the class"""
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
                set_vram_mb: float = None,
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
        self.set_vram_mb = set_vram_mb
        self.selected_gpus = selected_gpus
        self.cpu_id_list = cpu_id_list 
        self.cpu_count = len(cpu_id_list)
        self.start_time = start_time
        self.end_time = end_time
        self.timechunk_dt = timechunk_dt
        self.waveform_overlap = waveform_overlap 

        # We need to ensure that the vram specified does not exceed the capabilities of the system, if not, we need to exit safely before it happens
        if set_vram_mb is not None: 
            _, available_vram = get_gpu_vram()
            available_vram_mb = available_vram * 1024 
            intended_workers = self.number_of_concurrent_station_predictions * self.number_of_concurrent_timechunk_predictions
            
            # UPDATED: Account for model actors
            model_vram_mb = 3000  # Reserve 3GB per GPU for model
            num_model_actors = len(self.selected_gpus)
            total_model_vram = model_vram_mb * num_model_actors
            
            eqcct_usage = 1.1 * 1024 * intended_workers
            requested_vram_mb = intended_workers * self.set_vram_mb
            total_vram_needed = total_model_vram + requested_vram_mb + eqcct_usage
            
            avail_vram_mb_90 = available_vram_mb * len(self.selected_gpus) * 0.90

            if total_vram_needed > avail_vram_mb_90:
                print(f"[{datetime.now()}] ERROR: Insufficient VRAM!")
                print(f"  Model actors need: {total_model_vram:.0f} MB")
                print(f"  Workers need: {requested_vram_mb:.0f} MB")
                print(f"  EQCCT overhead: {eqcct_usage:.0f} MB")
                print(f"  Total needed: {total_vram_needed:.0f} MB")
                print(f"  Available (90%): {avail_vram_mb_90:.0f} MB")
                print(f"  Reduce concurrent workers or vram_mb setting")
                exit()
         
    def configure_cpu(self): 
        print(f"\nRunning EQCCT over MSeed Files with CPUs...")
        if self.best_usecase_config:
            cpus_to_use, num_concurrent_predictions, intra, inter, station_count = (True, self.csv_dir)
            print(f"\n[{datetime.now()}] Using {cpus_to_use} CPUs, {num_concurrent_predictions} Conc. Predictions, {intra} Intra Threads, and {inter} Inter Threads...")
            tf_environ(gpu_id=-1, intra_threads=intra, inter_threads=inter)
        else:
            tf_environ(gpu_id=-1, intra_threads=self.intra_threads, inter_threads=self.inter_threads) 
            
    def configure_gpu(self):
        print(f"\nRunning EQCCT over MSeed Files with GPUs...")
        if self.best_usecase_config: 
            result = find_optimal_configuration_gpu(True, self.csv_dir)
            if result is None:
                print(f"\n[{datetime.now()}] Error: Could not retrieve an optimal GPU configuration. Please check the CSV file and try again.")
                exit()  # Exit gracefully
            # Unpack values only if result is valid
            cpus_to_use, num_concurrent_predictions, intra, inter, gpus, vram_mb, station_count = result
            print(f"\n[{datetime.now()}] Using {cpus_to_use} CPUs, {num_concurrent_predictions} Conc. Predictions, {intra} Intra Threads, {inter} Inter Threads, {gpus} GPU IDs, and {vram_mb} MB VRAM per Task...")
            tf_environ(gpu_id=1, gpu_memory_limit_mb=vram_mb, gpus_to_use=gpus, intra_threads=intra, inter_threads=inter)
        else: 
            free_vram_mb = self.set_vram_mb if self.set_vram_mb is not None else self.calculate_vram() 
            selected_gpus = self.selected_gpus if self.selected_gpus else list_gpu_ids() # will give a list back of all available GPUs and use them all
            print(f"[{datetime.now()}] Using GPU(s): {selected_gpus}")
            vram_per_task_mb = free_vram_mb / self.number_of_concurrent_station_predictions
            tf_environ(gpu_id=1, gpu_memory_limit_mb=vram_per_task_mb, gpus_to_use=selected_gpus, intra_threads=self.intra_threads, inter_threads=self.inter_threads)
            
    def calculate_vram(self):
        print(f"[{datetime.now()}] Utilizing available VRAM within Ray Memory Usage Threshold Limit of 0.95...")
        total_vram, available_vram = get_gpu_vram()
        print(f"[{datetime.now()}] Total VRAM: {total_vram:.2f} GB")
        print(f"[{datetime.now()}] Available VRAM: {available_vram:.2f} GB")

        free_vram = total_vram * 0.9485 if available_vram / total_vram >= 0.9486 else available_vram
        print(f"[{datetime.now()}] Using {round(free_vram, 2)} GB VRAM (within 94.85% VRAM threshold).")
        return free_vram * 1024  # Convert to MB
    
    def chunk_time(self):
        # Creates the timechunks, EI. from X specific time to Y specific time to generate the dt tasks (timechunk tasks that are run in parallel first at the top level)
        # EX. [[UTCDateTime(2024, 12, 15, 11, 58), UTCDateTime(2024, 12, 15, 13, 0)], [UTCDateTime(2024, 12, 15, 12, 58), UTCDateTime(2024, 12, 15, 14, 0)]]
        from obspy import UTCDateTime
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
        
    def dt_task_generator(self): 
        tasks = [[f"({i+1}/{len(self.times_list)})", f"{self.times_list[i][0].strftime(format='%Y%m%dT%H%M%SZ')}_{self.times_list[i][1].strftime(format='%Y%m%dT%H%M%SZ')}"] for i in range((len(self.times_list)))]
        self.tasks_picker = tasks
    
    
    def timechunk_parallelization(self, ray_statement):
        # Tell user how many stations they are using 
        if self.specific_stations is None: 
            first_timechunk = sorted(os.listdir(self.input_dir))[0]
            station_dir = os.path.join(self.input_dir, first_timechunk)
            specific_stations_list = [d for d in os.listdir(station_dir) if os.path.isdir(os.path.join(station_dir, d))]
        else: 
            specific_stations_list = [station.strip() for station in self.specific_stations.split(',')]
            if self.specific_stations is None: 
             first_timechunk = sorted(os.listdir(self.input_dir))[0]
             station_dir = os.path.join(self.input_dir, first_timechunk)
             specific_stations_list = [d for d in os.listdir(station_dir) if os.path.isdir(os.path.join(station_dir, d))]
            else: 
                specific_stations_list = [station.strip() for station in self.specific_stations.split(',')]
        print(f"[{datetime.now()}] Using {len(specific_stations_list)} selected station(s).\n")
    
        
        # Ensure Output Dir exists and Create if doesn't 
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create logfile
        if not os.path.exists(self.log_filepath): 
            print(f"[{datetime.now()}] Log file not found. Creating log file...")
            with open(self.log_filepath, "w") as f: 
                f.write("")
                print(f"[{datetime.now()}] Log file: {self.log_filepath} created.")
        else: 
            print(f"[{datetime.now()}] Log file '{self.log_filepath}' already exists.")
        
        # Calculate how much VRAM & GPU to use 
        free_vram_mb = self.set_vram_mb if self.set_vram_mb else self.calculate_vram()
        vram_per_task_mb = free_vram_mb / self.number_of_concurrent_station_predictions
        
        # Submit timechunk tasks to mseed_predictor
        tasks_queue = []
        log_queue = queue.Queue()  # Create a queue for log entries
        
        # Compute total analyis timeframe 
        total_analysis_time = datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
        
        max_pending_tasks = self.number_of_concurrent_timechunk_predictions
        with open(self.log_filepath, mode="a+", buffering=1) as log: 
            statement = f"[{datetime.now()}] Starting EQCCTPro..."
            print(f"{statement}")
            print(f"[{datetime.now()}] Detailed subprocess information can be found in the log file.")
            log.write(f"Starting EQCCTPro...\n-----------------------------\n")
            log.write(f"{ray_statement}")
            for i in range(len(self.tasks_picker)):
                mseed_timechunk_dir_name = self.tasks_picker[i][1]
                timechunk_dir_path = os.path.join(self.input_dir, mseed_timechunk_dir_name) 
            
                # Concurrent Timechunks 
                while True: 
                    if len(tasks_queue) < max_pending_tasks: 
                        tasks_queue.append(mseed_predictor.options(num_gpus=0, num_cpus=1).remote(input_dir=timechunk_dir_path, output_dir=self.output_dir, log_file=self.log_filepath, 
                                            P_threshold=self.P_threshold, S_threshold=self.S_threshold, p_model=self.p_model_filepath, s_model=self.s_model_filepath, 
                                            number_of_concurrent_station_predictions=self.number_of_concurrent_station_predictions, ray_cpus=self.cpu_id_list, use_gpu=self.use_gpu, 
                                            gpu_id=self.selected_gpus, gpu_memory_limit_mb=vram_per_task_mb, specific_stations=specific_stations_list, 
                                            timechunk_id=mseed_timechunk_dir_name, waveform_overlap=self.waveform_overlap, total_timechunks=len(self.tasks_picker), 
                                            number_of_concurrent_timechunk_predictions=self.number_of_concurrent_timechunk_predictions, total_analysis_time=total_analysis_time,
                                            intra_threads=self.intra_threads, inter_threads=self.inter_threads))
                        break
                    # If there are more tasks than maximum, just process them
                    else:
                        tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
                        for finished_task in tasks_finished:
                            log_entry = ray.get(finished_task)
                            log_queue.put(log_entry)  # Add log entry to the queue
                            # log.write(log_entry + "\n")
                            # log.flush() 
            # After adding all the tasks to queue, process what's left
            while tasks_queue:
                tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
                for finished_task in tasks_finished:
                    log_entry = ray.get(finished_task)
                    log_queue.put(log_entry)  # Add log entry to the queue
                    # log.write(log_entry + "\n")
                    # log.flush()
            
            ray.shutdown()
            print(f"[{datetime.now()}] Ray Successfully Shutdown.")
            # Write log entries from the queue to the file
            while not log_queue.empty():
                log_entry = log_queue.get()
                log.write(log_entry + "\n")
                log.flush()
            
            # log.write("\n------- EQCCTPro: Parallel Processing Complete -------\n")
            log.write("\n------- Successfully Picked All Waveform(s) from all Timechunk(s) -------\n")
            log.write("\n------- END OF FILE -------\n")
            log.flush()
        
            


    def run_eqcctpro(self):
        # Set CPU affinity
        process = psutil.Process(os.getpid())
        process.cpu_affinity(self.cpu_id_list)  # Limit process to the given CPU IDs
        
        self.chunk_time()
        self.dt_task_generator()
        
        # GPU
        if self.use_gpu: 
            self.configure_gpu()
            ray.init(ignore_reinit_error=True, num_gpus=len(self.selected_gpus), num_cpus=len(self.cpu_id_list), logging_level=logging.FATAL, log_to_driver=False) # Ray initalization using GPUs 
            statement = f"[{datetime.now()}] Ray Successfully Initialized with {self.selected_gpus} GPU(s) and {len(self.cpu_id_list)} CPU(s)."
            print(f"{statement}")
            self.timechunk_parallelization(statement)
        else: 
        # CPU
            self.configure_cpu()
            ray.init(ignore_reinit_error=True, num_cpus=len(self.cpu_id_list), logging_level=logging.FATAL, log_to_driver=False) # Ray initalization using CPUs
            statement = f"[{datetime.now()}] Ray Successfully Initialized with {len(self.cpu_id_list)} CPU(s)."
            print(f"{statement}")
            print(f"[{datetime.now()}] {len(self.times_list)} time chunk(s) from {self.start_time} to {self.end_time} (dt={self.timechunk_dt}min, overlap={self.waveform_overlap}min).")
            self.timechunk_parallelization(statement)
        print(f"[{datetime.now()}] Successfully picked all waveform(s) from all time chunk(s). Exiting...\n")
        


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
                 set_vram_mb:float = None, 
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
            exit()
        
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
        self.set_vram_mb = set_vram_mb
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
        
        # Set up temp dir 
        import tempfile
        tempfile.tempfile = self.home_tmp_dir

        os.environ['TMPDIR'] = self.home_tmp_dir
        os.environ['TEMP'] = self.home_tmp_dir
        os.environ['TMP'] = self.home_tmp_dir
        print(f"\n[{datetime.now()}] Successfully set up temp files to be stored at {self.home_tmp_dir}")
        
        # We need to ensure that the vram specified does not exceed the capabilities of t  he system, if not, we need to exit safely before it happens
        if self.set_vram_mb is not None:
            if self.eval_mode != "gpu":
                raise ValueError(
                    f"set_vram_mb is only meaningful in GPU mode; got eval_mode='{self.eval_mode}'."
                )
            if not self.selected_gpus or len(self.selected_gpus) == 0:
                raise ValueError(
                    "selected_gpus must be a non-empty list when using set_vram_mb."
                )
            if self.set_vram_mb <= 0:
                raise ValueError("set_vram_mb must be a positive number of MB.")

            # Ensure time chunks are computed so we know how many tasks we will run
            self.chunk_time()  # populates self.times_list

            # 1) Available VRAM per GPU (GB) -> MB, then multiply by number of selected GPUs
            _, available_vram_gb = get_gpu_vram()            # returns per-GPU free VRAM in GB
            available_vram_mb = float(available_vram_gb) * 1024.0
            total_available_mb = available_vram_mb * len(self.selected_gpus)
            avail_vram_mb_90 = 0.90 * total_available_mb     # 90% safety ceiling

            # 2) Concurrency / worker count
            # Use your current worst-case: stations * timechunks for this eval run
            intended_workers = int(self.stations2use) * int(len(self.times_list) // 2)

            # 3) Model actor reservation (per GPU), plus per-worker slice and EQCCT overhead
            model_vram_mb = 3000.0                                      # reserve 3 GB per GPU
            num_model_actors = len(self.selected_gpus)
            total_model_vram = model_vram_mb * num_model_actors

            requested_vram_mb = float(self.set_vram_mb) * float(intended_workers)
            eqcct_usage = 1.1 * 1024.0 * float(intended_workers)        # ~1.1 GB per worker overhead

            total_vram_needed = total_model_vram + requested_vram_mb + eqcct_usage

            if total_vram_needed > avail_vram_mb_90:
                # Fail fast with a precise diagnostic
                raise RuntimeError(
                    (
                        f"[{datetime.now()}] ERROR: Insufficient VRAM for requested configuration.\n"
                        f"  Selected GPUs: {self.selected_gpus}\n"
                        f"  Available (90% cap): {avail_vram_mb_90:.0f} MB "
                        f"(= 0.9 * {total_available_mb:.0f} MB across {num_model_actors} GPU(s))\n"
                        f"  Budget request breakdown:\n"
                        f"    • Model actors: {total_model_vram:.0f} MB "
                        f"({model_vram_mb:.0f} MB × {num_model_actors} GPU(s))\n"
                        f"    • Workers:      {requested_vram_mb:.0f} MB "
                        f"({self.set_vram_mb:.0f} MB × {intended_workers} workers)\n"
                        f"    • EQCCT overhead: {eqcct_usage:.0f} MB "
                        f"(~1.1 GB × {intended_workers} workers)\n"
                        f"  TOTAL requested: {total_vram_needed:.0f} MB\n\n"
                        f"Action: Reduce stations/timechunks concurrency or lower set_vram_mb."
                    )
                )
            else:
                print(
                    f"[{datetime.now()}] VRAM budget OK. "
                    f"Request {total_vram_needed:.0f} MB ≤ {avail_vram_mb_90:.0f} MB (90% cap) across {num_model_actors} GPU(s)."
                )
        
    def _generate_stations_list(self):
        """Generates station list"""
        if self.station2use is None: 
            return list(range(1, 11)) + list(range(15, 50, 5))
        return generate_station_list(self.stations2use, self.starting_amount_of_stations, self.station_list_step_size)
    
    def _prepare_environment(self):
        """Removed 'output_dir' so that there is no conflicts in the save for a clean output return"""
        remove_directory(self.output_dir)
        
    def chunk_time(self):
        from obspy import UTCDateTime
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
    
    def dt_task_generator(self): 
        tasks = [[f"({i+1}/{len(self.times_list)})", f"{self.times_list[i][0].strftime(format='%Y%m%dT%H%M%SZ')}_{self.times_list[i][1].strftime(format='%Y%m%dT%H%M%SZ')}"] for i in range((len(self.times_list)))]
        self.tasks_picker = tasks
        
    def evaluate_cpu(self): 
        """Evaluate system parallelization using CPUs"""
        statement = "Evaluating System Parallelization Capability using CPU"
        print(f"\n{statement}\n")
        
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create logfile 
        if not os.path.exists(self.log_filepath): 
            print(f"[{datetime.now()}] Log file not found. Creating log file...")
            with open(self.log_filepath, "w") as f: 
                f.write("")
                f.write(f"{statement}\n-----------------------------\n")
                print(f"[{datetime.now()}] Log file: {self.log_filepath} created.")
        else: 
            print(f"[{datetime.now()}] Log file '{self.log_filepath}' already exists.")
        
        # Create test results csv 
        csv_filepath = f"{self.csv_dir}/cpu_test_results.csv"
        prepare_csv(csv_filepath, False)
        
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
                print(f"[{datetime.now()}] Ray Successfully Initialized with {len(cpus_to_use)} CPU(s).")
                
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
                            
                            log.write(f"\nTrial Number: {trial_num}")
                            print(f"\n[{datetime.now()}] Trial Number: {trial_num}")
                            print(f"[{datetime.now()}] CPU(s): {i}")
                            print(f"[{datetime.now()}] Conc. Timechunks Being Analyzed: {timechunks} / Total Timechunks to be Analyzed: {len(self.tasks_picker)}")
                            print(f"[{datetime.now()}] Total Amount of Stations to be Processed in Current Trial: {num_stations} / Number of Stations Being Processed Concurrently: {num_concurrent_predictions} / Total Overall Trial Station Count: {max(self.stations2use_list)}") 
                            
                            # Concurrent Timechunks
                            tasks_queue = []
                            try: 
                                while True: 
                                    if len(tasks_queue) < max_pending_tasks: 
                                        tasks_queue.append(mseed_predictor.options(num_gpus=0, num_cpus=1).remote(input_dir=timechunk_dir_path, output_dir=self.output_dir, log_file=self.log_filepath, 
                                                            P_threshold=self.P_threshold, S_threshold=self.S_threshold, p_model=self.p_model_filepath, s_model=self.s_model_filepath, 
                                                            number_of_concurrent_station_predictions=num_concurrent_predictions, ray_cpus=cpus_to_use, use_gpu=use_gpu, 
                                                            gpu_id=self.selected_gpus, gpu_memory_limit_mb=self.set_vram_mb, stations2use=num_stations, 
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
                                log.write(f"\n[{datetime.now()}] Trial {trial_num} FAILED: {error_msg}\n")
                                print(f"[{datetime.now()}] Trial {trial_num} FAILED: {error_msg}")
                                
                            # Write log entries from the queue to the file
                            while not log_queue.empty():
                                log_entry = log_queue.get()
                                log.write(log_entry + "\n")
                                log.flush()
                                
                            remove_output_subdirs(self.output_dir)
                            trial_num += 1  
                            
                            # RAM cleanup
                            process = psutil.Process(os.getpid())
                            mem_before = process.memory_info().rss
                            gc.collect()
                            mem_after = process.memory_info().rss
                            mem_freed = mem_before - mem_after
                            print(f"[{datetime.now()}] Successfully cleaned up {mem_freed / 1e6:.2f} MB of RAM.")
                            
                        # tested_concurrency.update([x for x in concurrent_predictions_list if x <= num_stations])

                    ray.shutdown() # Shutdown Ray after processing all timechunks for this CPU count 
                    print(f"[{datetime.now()}] Ray Successfully Shutdown.")
                                
                         
                        
                    
        print(f"\n[{datetime.now()}] Testing complete.\n[{datetime.now()}] Finding Optimal Configurations...")
        # Compute optimal configurations (CPU)
        df = pd.read_csv(csv_filepath)
        optimal_configuration_df, best_overall_usecase_df = find_optimal_configurations_cpu(df)
        optimal_configuration_df.to_csv(f"{self.csv_dir}/optimal_configurations_cpu.csv", index=False)
        best_overall_usecase_df.to_csv(f"{self.csv_dir}/best_overall_usecase_cpu.csv", index=False)
        print(f"[{datetime.now()}] Optimal Configurations Found. Findings saved to:\n" 
                f" 1) Optimal CPU/Station/Concurrent Prediction Configurations: {self.csv_dir}/optimal_configurations_cpu.csv\n" 
                f" 2) Best Overall Usecase Configuration: {self.csv_dir}/best_overall_usecase_cpu.csv")

    def evaluate_gpu(self): 
        """Evaluate system parallelization using GPUs"""
        statement = "Evaluating System Parallelization Capability using GPUs"
        print(f"\n{statement}\n")
        
        # Set CPU affinity
        process = psutil.Process(os.getpid())
        process.cpu_affinity(self.cpu_id_list)  # Limit process to the given CPU IDs
        
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create logfile 
        if not os.path.exists(self.log_filepath): 
            print(f"[{datetime.now()}] Log file not found. Creating log file...")
            with open(self.log_filepath, "w") as f: 
                f.write("")
                f.write(f"{statement}\n-----------------------------\n")
                print(f"[{datetime.now()}] Log file: {self.log_filepath} created.")
        else: 
            print(f"[{datetime.now()}] Log file '{self.log_filepath}' already exists.")
        
        # Calculate these at the start
        self.chunk_time()
        self.dt_task_generator()
        total_analysis_time = datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
            
        # Create test results csv 
        csv_filepath = f"{self.csv_dir}/gpu_test_results.csv"
        prepare_csv(csv_filepath, True)
        
        free_vram_mb = self.set_vram_mb if self.set_vram_mb else self.calculate_vram()
        self.selected_gpus = self.selected_gpus if self.selected_gpus else list_gpu_ids()
        print(f"[{datetime.now()}] Using GPU(s): {self.selected_gpus}")
        
        trial_num = 1
        log_queue = queue.Queue()  # Create a queue for log entries
        
        with open(self.log_filepath, mode="a+", buffering=1) as log:
            # Initialize Ray with GPUs
            ray.init(ignore_reinit_error=True, num_gpus=len(self.selected_gpus), num_cpus=len(self.cpu_id_list), 
                    logging_level=logging.FATAL, log_to_driver=False)
            print(f"[{datetime.now()}] Ray Successfully Initialized with {len(self.selected_gpus)} GPU(s) and {len(self.cpu_id_list)} CPU(s).")
            
            for stations in self.stations2use_list:
                concurrent_predictions_list = generate_station_list(self.min_conc_stations, stations, self.conc_station_tasks_step_size)
                for predictions in concurrent_predictions_list:
                    vram_per_task_mb = free_vram_mb / predictions
                    step_size = vram_per_task_mb * 0.05
                    vram_steps = np.arange(step_size, vram_per_task_mb + step_size, step_size)
                    print(f"[{datetime.now()}] Testing the following VRAM limitations (MB): {vram_steps}")
                    
                    for gpu_memory_limit_mb in vram_steps:
                        print(f"\n[{datetime.now()}] VRAM Limited to {gpu_memory_limit_mb:.2f} MB per Task")
                        print(f"\nTrial Number: {trial_num}")
                        
                        # Get the first timechunk for testing
                        mseed_timechunk_dir_name = self.tasks_picker[0][1]
                        timechunk_dir_path = os.path.join(self.input_dir, mseed_timechunk_dir_name)
                        
                        log.write(f"\nTrial Number: {trial_num}\n")
                        print(f"[{datetime.now()}] Stations: {stations}")
                        print(f"[{datetime.now()}] Concurrent Station Predictions: {predictions}")
                        print(f"[{datetime.now()}] VRAM per Task: {gpu_memory_limit_mb:.2f} MB")
                        
                        try:
                            # Call mseed_predictor directly via Ray (just like evaluate_cpu does)
                            ref = mseed_predictor.options(num_gpus=0, num_cpus=1).remote(
                                input_dir=timechunk_dir_path, 
                                output_dir=self.output_dir, 
                                log_file=self.log_filepath, 
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
                            log.write(f"\n[{datetime.now()}] Trial {trial_num} FAILED: {error_msg}\n")
                            print(f"[{datetime.now()}] Trial {trial_num} FAILED: {error_msg}")
                        
                        # Write log entries from the queue to the file
                        while not log_queue.empty():
                            log_entry = log_queue.get()
                            log.write(log_entry + "\n")
                            log.flush()
                        
                        remove_output_subdirs(self.output_dir)
                        trial_num += 1
                        
                        # RAM cleanup
                        mem_before = process.memory_info().rss
                        gc.collect()
                        mem_after = process.memory_info().rss
                        mem_freed = mem_before - mem_after
                        print(f"[{datetime.now()}] Successfully cleaned up {mem_freed / 1e6:.2f} MB of RAM.")
            
            ray.shutdown()  # Shutdown Ray after all testing
            print(f"[{datetime.now()}] Ray Successfully Shutdown.")

        print(f"\n[{datetime.now()}] Testing complete.\n[{datetime.now()}] Finding Optimal Configurations...")
        # Compute optimal configurations (GPU)
        df = pd.read_csv(csv_filepath)
        optimal_configuration_df, best_overall_usecase_df = find_optimal_configurations_gpu(df)
        optimal_configuration_df.to_csv(f"{self.csv_dir}/optimal_configurations_gpu.csv", index=False)
        best_overall_usecase_df.to_csv(f"{self.csv_dir}/best_overall_usecase_gpu.csv", index=False)
        print(f"[{datetime.now()}] Optimal Configurations Found. Findings saved to:\n" 
                f" 1) Optimal GPU/Station/Concurrent Prediction Configurations: {self.csv_dir}/optimal_configurations_gpu.csv\n" 
                f" 2) Best Overall Usecase Configuration: {self.csv_dir}/best_overall_usecase_gpu.csv")

    def evaluate(self):
        if self.eval_mode == "cpu":
            self.evaluate_cpu()
        elif self.eval_mode == "gpu":
            self.evaluate_gpu()
        else: 
            exit()
        
    def calculate_vram(self):
        """Calculate available VRAM for GPU testing."""
        print(f"[{datetime.now()}] Utilizing available VRAM...")
        total_vram, available_vram = get_gpu_vram()
        print(f"[{datetime.now()}] Total VRAM: {total_vram:.2f} GB.")
        print(f"[{datetime.now()}] Available VRAM: {available_vram:.2f} GB.")

        free_vram = total_vram * 0.9485 if available_vram / total_vram >= 0.9486 else available_vram
        print(f"[{datetime.now()}] Using up to {round(free_vram, 2)} GB of VRAM.")
        return free_vram * 1024  # Convert to MB

class OptimalCPUConfigurationFinder: 

    """Finds the optimal CPU configuration based on evaluation results"""
    def __init__(self, eval_sys_results_dir: str):
        if not eval_sys_results_dir or not os.path.isdir(eval_sys_results_dir): 
            raise ValueError(f"Error: The provided directory path '{eval_sys_results_dir}' is invalid or does not exist.")
        self.eval_sys_results_dir = eval_sys_results_dir

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
        
        print("\nBest Overall Usecase Configuration Based on Trial Data:\n--------------------------\n")
        print(f"CPU(s): {num_cpus}\n"
        f"Intra-parallelism Threads: {intra_threads}\n"
        f"Inter-parallelism Threads: {inter_threads}\n"
        f"Waveform Timespace: {waveform_timespace}\n"
        f"Total Number of Stations Used: {num_stations}\n"
        f"Total Number of Timechunks: {total_num_timechunks}\n"
        f"Length of Timechunks (min): {length_of_timechunks}\n"
        f"Concurrent Timechunk Processes: {num_concurrent_timechunks}\n"
        f"Concurrent Station Processes Per Timechunk: {num_concurrent_stations}\n"
        f"Total Runtime (s): {total_runtime}")

        return int(float(num_cpus)), int(float(intra_threads)), int(float(inter_threads)), int(float(num_concurrent_timechunks)), int(float(num_concurrent_stations)), int(float(num_stations))
    
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

        print("\nBest Configuration for Requested Input Parameters Based on Trial Data:")
        print(f"CPU(s): {cpu}\n"
              f"Intra-parallelism Threads: {best_config['Intra-parallelism Threads']}\n"
              f"Inter-parallelism Threads: {best_config['Inter-parallelism Threads']}\n"
              f"Waveform Timespace: {best_config['Total Waveform Analysis Timespace (min)']}\n"
              f"Total Number of Stations Used: {station_count}\n"
              f"Total Number of Timechunks: {best_config['Total Number of Timechunks']}\n"
              f"Length of Timechunks (min): {best_config['Length of Timechunk (min)']}\n"
              f"Concurrent Timechunk Processes: {best_config['Concurrent Timechunks Used']}\n"
              f"Concurrent Station Processes Per Timechunk: {best_config['Number of Concurrent Station Tasks']}\n"
              f"Total Runtime (s): {best_config['Total Run time for Picker (s)']}")

        return int(float(cpu)), int(float(best_config["Intra-parallelism Threads"])), int(float(best_config["Inter-parallelism Threads"])), int(float(best_config["Concurrent Timechunks Used"])), int(float(best_config["Number of Concurrent Station Tasks"])), int(float(station_count))


class OptimalGPUConfigurationFinder:
    """Finds the optimal GPU configuration based on evaluation system results."""

    def __init__(self, eval_sys_results_dir: str):
        if not eval_sys_results_dir or not os.path.isdir(eval_sys_results_dir):
            raise ValueError(f"Error: The provided directory path '{eval_sys_results_dir}' is invalid or does not exist.")
        self.eval_sys_results_dir = eval_sys_results_dir

    def find_best_overall_usecase(self):
        """Finds the best overall GPU configuration from evaluation results."""
        file_path = f"{self.eval_sys_results_dir}/best_overall_usecase_gpu.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{datetime.now()}] Error: The file '{file_path}' does not exist. Ensure it is in the correct directory.")

        df_best_overall = pd.read_csv(file_path, header=None, index_col=0)
        best_config_dict = df_best_overall.to_dict()[1]

        num_cpus = best_config_dict.get("Number of CPUs Allocated for Ray to Use")
        num_concurrent_predictions = best_config_dict.get("Number of Concurrent Station Tasks")
        intra_threads = best_config_dict.get("Intra-parallelism Threads")
        inter_threads = best_config_dict.get("Inter-parallelism Threads")
        num_stations = best_config_dict.get("Number of Stations Used")
        total_runtime = best_config_dict.get("Total Run time for Picker (s)")
        vram_used = best_config_dict.get("VRAM Used Per Task")
        num_gpus = ast.literal_eval(best_config_dict.get("GPUs Used"))

        print("\nBest Overall Usecase Configuration Based on Trial Data:")
        print(f"CPU: {num_cpus}\n"
              f"GPU ID(s): {num_gpus}\n"
              f"Concurrent Predictions: {num_concurrent_predictions}\n"
              f"Intra-parallelism Threads: {intra_threads}\n"
              f"Inter-parallelism Threads: {inter_threads}\n"
              f"Stations: {num_stations}\n"
              f"VRAM Used per Task: {vram_used}\n"
              f"Total Runtime (s): {total_runtime}")

        return int(float(num_cpus)), int(float(num_concurrent_predictions)), int(float(intra_threads)), int(float(inter_threads)), num_gpus, int(float(vram_used)), int(float(num_stations))

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

        print("\nBest Configuration for Requested Application Usecase Based on Trial Data:")
        print(f"CPU: {num_cpus}\n"
              f"GPU: {gpu_list}\n"
              f"Concurrent Predictions: {best_config['Number of Concurrent Station Tasks']}\n"
              f"Intra-parallelism Threads: {best_config['Intra-parallelism Threads']}\n"
              f"Inter-parallelism Threads: {best_config['Inter-parallelism Threads']}\n"
              f"Stations: {station_count}\n"
              f"VRAM Used per Task: {best_config['VRAM Used Per Task']}\n"
              f"Total Runtime (s): {best_config['Total Run time for Picker (s)']}")

        return int(float(best_config["Number of CPUs Allocated for Ray to Use"])), \
               int(float(best_config["Number of Concurrent Station Tasks"])), \
               int(float(best_config["Intra-parallelism Threads"])), \
               int(float(best_config["Inter-parallelism Threads"])), \
               gpu_list, \
               int(float(best_config["VRAM Used Per Task"])), \
               int(float(station_count))
            
    

def parse_time_range(time_string):
    """
    Parses a time range string and returns start time, end time, and time delta.
    """
    try:
        start_str, end_str = time_string.split('_')
        start_time = datetime.strptime(start_str, "%Y%m%dT%H%M%SZ")
        end_time = datetime.strptime(end_str, "%Y%m%dT%H%M%SZ")
        time_delta = end_time - start_time

        return start_time, end_time, time_delta

    except ValueError as e:
        return None, None, None #Error handling.

@ray.remote
def mseed_predictor(input_dir='downloads_mseeds',
              output_dir="detections",
              P_threshold=0.1,
              S_threshold=0.1, 
              normalization_mode='std',
              dt=1,
              batch_size=500,              
              overlap=0.3,
              gpu_id=None,
              gpu_limit=None,
              overwrite=False,
              log_file="./results/logs/picker/eqcct.log",
              stations2use=None,
              stations_filters=None,
              p_model=None,
              s_model=None,
              number_of_concurrent_station_predictions=None,
              ray_cpus=None,
              use_gpu=False,
              gpu_memory_limit_mb=None,
              testing_gpu=None,
              test_csv_filepath=None,
              specific_stations=None,
              timechunk_id=None,
              waveform_overlap=None,
              total_timechunks=None,
              number_of_concurrent_timechunk_predictions=None,
              total_analysis_time=None,
              intra_threads=None,
              inter_threads=None, 
              timechunk_dt=None): 
    
    """ 
    
    To perform fast detection directly on mseed data.
    
    Parameters
    ----------
    input_dir: str
        Directory name containing hdf5 and csv files-preprocessed data.
            
    input_model: str
        Path to a trained model.
            
    stations_json: str
        Path to a JSON file containing station information. 
           
    output_dir: str
        Output directory that will be generated.
            
    P_threshold: float, default=0.1
        A value which the P probabilities above it will be considered as P arrival.                
            
    S_threshold: float, default=0.1
        A value which the S probabilities above it will be considered as S arrival.
            
    normalization_mode: str, default=std
        Mode of normalization for data preprocessing max maximum amplitude among three components std standard deviation.
             
    batch_size: int, default=500
        Batch size. This wont affect the speed much but can affect the performance. A value beteen 200 to 1000 is recommended.
             
    overlap: float, default=0.3
        If set the detection and picking are performed in overlapping windows.
             
    gpu_id: int
        Id of GPU used for the prediction. If using CPU set to None.        
             
    gpu_limit: float
       Set the maximum percentage of memory usage for the GPU. 

    overwrite: Bolean, default=False
        Overwrite your results automatically.
           
    Returns
    --------        
      
    """ 

    # We set up the tf_environ again for the Raylets, who adopt their own import state and TF runtime when created. 
    # We want to ensure that they are configured properly so that they won't die (bad)
    if not use_gpu: 
        tf_environ(gpu_id=-1, intra_threads=intra_threads, inter_threads=inter_threads)
        # tf_environ(gpu_id=1, gpu_memory_limit_mb=gpu_memory_limit_mb, gpus_to_use=gpu_id, intra_threads=intra_threads, inter_threads=inter_threads)


    args = {
    "input_dir": input_dir,
    "output_dir": output_dir,
    "P_threshold": P_threshold,
    "S_threshold": S_threshold,
    "normalization_mode": normalization_mode,
    "dt": dt,
    "overlap": overlap,
    "batch_size": batch_size,
    "overwrite": overwrite, 
    "gpu_id": gpu_id,
    "gpu_limit": gpu_limit,
    "p_model": p_model,
    "s_model": s_model,
    "stations_filters": stations_filters
    }

    log_messages = ""  # Accumulate log messages here
    log_messages += f"\n-----------------------------\nHardware Configuration...\n"
    try:
        process = psutil.Process(os.getpid())
        process.cpu_affinity(ray_cpus)  # ray_cpus should be a list of core IDs like [0, 1, 2]

        log_messages += f"\n[{datetime.now()}] CPU affinity set to cores: {ray_cpus}\n"
    except Exception as e:
        log_messages += f"[{datetime.now()}] Failed to set CPU affinity. Reason: {e}\n"

    
    # Ensure Output Dir exists 
    os.makedirs(output_dir, exist_ok=True)
    # Ensure logfile exists before continuing 
    if not os.path.exists(log_file):
        print(f"[{datetime.now()}] Log file not found. Creating log file...")
        with open(log_file, "a") as log:
            log.write(f"[{datetime.now()}] Created log file: {log_file}\n")
    else:
        with open(log_file, "a") as log:
            log.write(f"\n[{datetime.now()}] Log file already exists. Located at: '{log_file}'\n")
        
    with open(log_file, mode="a", buffering=1) as log:
        out_dir = os.path.join(os.getcwd(), str(args['output_dir']))    
        try:
            if platform.system() == 'Windows':
                station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("\\")[-1] != ".DS_Store"]
            else:     
                station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("/")[-1] != ".DS_Store"]

            station_list = sorted(set(station_list))
        except Exception as exp:
            log_messages += f"{exp}\n"
            return log_messages
        # log.write(f"[{datetime.now()}] GPU ID: {args['gpu_id']}; Batch size: {args['batch_size']}\n")
        log_messages += f"\n-----------------------------\nData Preprocessing for EQCCTPro...\n"
        log_messages += f"\n[{datetime.now()}] {len(station_list)} station(s) in {args['input_dir']}\n"
        
        
        if stations2use and stations2use <= len(station_list):  # For System Evaluation Execution
            station_list = random.sample(station_list, stations2use)  # Randomly choose stations from the sample size 
            # log.write(f"[{datetime.now()}] Using {len(station_list)} station(s) after selection.\n")

        if specific_stations is not None:  # For "One Use Run" Over a Given Set of Stations (Just Run EQCCTPro on specific_stations)
            station_list = [x for x in station_list if x in specific_stations]
        else:  
            station_list = station_list # someone put None thinking that they would be able to run the whole directory in one go

        log_messages += f"[{datetime.now()}] Using {len(station_list)} selected station(s).\n"
    
        if not station_list or any(looks_like_timechunk_id(x) for x in station_list):
            # Rebuild from the actual contents of the timechunk dir
            station_list = build_station_list_from_dir(args['input_dir'])
            log_messages += f"[{datetime.now()}] Station list rebuilt from directory because it contained a timechunk id or was empty.\n"

        tasks_predictor = [[f"({i+1}/{len(station_list)})", station_list[i], out_dir, args] for i in range(len(station_list))]
        
        if not tasks_predictor:
            return
        
        # CREATE MODEL ACTOR(S) - Add this before the task loop
        log_messages += f"[{datetime.now()}] Creating model actor(s)...\n"
        
        if use_gpu:
            # Allocate more VRAM to model actors (they need to hold the full model)
            # Reserve ~2-3GB per model actor, adjust based on your model size
            model_vram_mb = min(gpu_memory_limit_mb * 2, 3000)  # At least 2x task VRAM or 3GB
            
            # Create one model actor per GPU
            model_actors = []
            for gpu_idx in gpu_id:
                actor = ModelActor.options(num_gpus=1, num_cpus=0).remote(gpus_to_use=gpu_id, p_model_path=p_model, s_model_path=s_model, gpu_memory_limit_mb=model_vram_mb, use_gpu=True)
                model_actors.append(actor)
            
            log_messages += f"[{datetime.now()}] Created {len(model_actors)} GPU model actor(s) with {model_vram_mb/1024:.2f}GB VRAM each\n"
        else:
            # Create CPU model actor
            model_actors = [ModelActor.options(num_cpus=1).remote(p_model_path=p_model, s_model_path=s_model, gpu_memory_limit_mb=None, use_gpu=False)]
            log_messages += f"[{datetime.now()}] Created 1 CPU model actor\n"

        # Submit tasks to ray in a queue
        tasks_queue = []
        max_pending_tasks = number_of_concurrent_station_predictions
        log_messages += f"[{datetime.now()}] Starting EQCCTPro parallelized waveform processing...\n"
        start_time = time.time() 
        log_messages += f"\n-----------------------------\nAnalyzing Seismic Waveforms for P and S Picks via EQCCT...\n\n"

        if timechunk_id is None:
            # derive from the path if caller forgot to pass it
            cand = os.path.basename(input_dir)
            if "_" in cand and len(cand) >= 10:
                timechunk_id = cand
            else:
                raise ValueError("timechunk_id is None and could not be inferred from input_dir; "
                                "expected a dir named like YYYYMMDDThhmmssZ_YYYYMMDDThhmmssZ")
        starttime, endtime, time_delta = parse_time_range(timechunk_id)

        log_messages += f"Analyzing {time_delta} minute timechunk from {starttime} to {endtime} ({waveform_overlap} min overlap)\n"
        log_messages += f"\n[{datetime.now()}] Processing a total of {len(tasks_predictor)} stations, {max_pending_tasks} at a time.\n"
   

        # Concurrent Prediction(s) Parallel Processing
        try: 
            for i in range(len(tasks_predictor)):
                while True:
                    # Add new task to queue while max is not reached
                    if len(tasks_queue) < max_pending_tasks:
                        # SELECT WHICH MODEL ACTOR TO USE (round-robin across GPUs)
                        model_actor = model_actors[i % len(model_actors)]
                        
                        if use_gpu is False:
                            tasks_queue.append(parallel_predict.options(num_cpus=0).remote(tasks_predictor[i], model_actor, False, None))
                        elif use_gpu is True:
                            # Don't allocate GPUs to workers, only to model actors
                            tasks_queue.append(parallel_predict.options(num_cpus=0, num_gpus=0).remote(tasks_predictor[i], model_actor, True, gpu_memory_limit_mb))
                        break
                    # If there are more tasks than maximum, just process them
                    else:
                        tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
                        for finished_task in tasks_finished:
                            log_entry = ray.get(finished_task)
                            log_messages += f'{log_entry}\n'

            # After adding all the tasks to queue, process what's left
            while tasks_queue:
                tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
                for finished_task in tasks_finished:
                    log_entry = ray.get(finished_task)
                    log_messages += f'{log_entry}\n'

        except Exception as e:
            # Catch any error in the parallel processing
            with open(log_file, "w") as f:
                f.write(f"ERROR in parallel processing at {datetime.now()}\n")
                f.write(f"Error: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
            raise  # Re-raise to see the error

        log_messages += f"\n------- Parallel Station Waveform Processing Complete For {starttime} to {endtime} Timechunk-------\n"
        log.flush()
        end_time = time.time()
        log_messages += f"\n[{datetime.now()}] Picks saved at {output_dir}\n[{datetime.now()}] Process Runtime: {end_time - start_time:.2f} s"

        if testing_gpu is not None: 
            # Guard: make sure CPUs is an int, not a list
            num_ray_cpus = len(ray_cpus) if isinstance(ray_cpus, (list, tuple)) else int(len(list(ray_cpus)))

            # Parse the timechunk_id to get start/end times
            if timechunk_id:
                starttime, endtime, time_delta = parse_time_range(timechunk_id)
                timechunk_length_min = time_delta.total_seconds() / 60.0 if time_delta else None
            else:
                timechunk_length_min = None

            trial_data = {
                "Trial Number": None,  # Will be auto-filled by append_trial_row
                "Stations Used": str(station_list),
                "Number of Stations Used": len(station_list),
                "Number of CPUs Allocated for Ray to Use": num_ray_cpus,
                "Intra-parallelism Threads": intra_threads if intra_threads is not None else "",
                "Inter-parallelism Threads": inter_threads if inter_threads is not None else "",
                "GPUs Used": str(gpu_id) if use_gpu else "",
                "VRAM Used Per Task": float(gpu_memory_limit_mb) if (use_gpu and gpu_memory_limit_mb is not None) else "",
                "Total Waveform Analysis Timespace (min)": float(total_analysis_time.total_seconds() / 60.0) if hasattr(total_analysis_time, "total_seconds") else (float(total_analysis_time) if total_analysis_time else ""),
                "Total Number of Timechunks": int(total_timechunks) if total_timechunks is not None else "",
                "Concurrent Timechunks Used": int(number_of_concurrent_timechunk_predictions) if number_of_concurrent_timechunk_predictions is not None else "",
                "Length of Timechunk (min)": timechunk_length_min if timechunk_length_min is not None else "",
                "Number of Concurrent Station Tasks": int(number_of_concurrent_station_predictions) if number_of_concurrent_station_predictions is not None else "",
                "Total Run time for Picker (s)": round(end_time - start_time, 6),
                "Trial Success": "",
                "Error Message": "",
            }
                
            append_trial_row(csv_path=test_csv_filepath, trial_data=trial_data)
            log_messages += f"\n[{datetime.now()}] Successfully saved trial data to CSV at {test_csv_filepath}"
            
        log_messages += f"\n[{datetime.now()}] Successfully ran EQCCTPro, exiting..."
        return log_messages
    
@ray.remote
class ModelActor:
    def __init__(self,  p_model_path, s_model_path, gpus_to_use=False, intra_threads=1, inter_threads=1, gpu_memory_limit_mb=None, use_gpu=True):
        from eqcct_tf_models import load_eqcct_model
        
        if use_gpu and gpu_memory_limit_mb:
            # Configure GPU memory for this actor
            try: 
                tf_environ(
                    gpu_id=-1, 
                    gpus_to_use=gpus_to_use,
                    gpu_memory_limit_mb=gpu_memory_limit_mb,
                    intra_threads=intra_threads,
                    inter_threads=inter_threads,
                    log_device=False)
            except RuntimeError as e:
                print(f"[ModelActor] Error setting memory limit: {e}")
        
        # Load the model once
        self.model = load_eqcct_model(p_model_path, s_model_path)
        print(f"[ModelActor] Model loaded successfully")
    
    def predict(self, data_generator):
        """Perform prediction using the loaded model"""
        return self.model.predict(data_generator, verbose=0)


@ray.remote
def parallel_predict(predict_args, model_actor, gpu=False, gpu_memory_limit_mb=None):
    """
    Modified to use shared ModelActor instead of loading model per task
    """
    from eqcct_tf_models import Patches, PatchEncoder, StochasticDepth, PreLoadGeneratorTest, load_eqcct_model
    pos, station, out_dir, args = predict_args
    
    # NOTE: We removed the model loading code that was causing OOM errors
    # The model is now shared via the model_actor
    
    save_dir = os.path.join(out_dir, str(station)+'_outputs')
    csv_filename = os.path.join(save_dir,'X_prediction_results.csv')

    if os.path.isfile(csv_filename):
        if args['overwrite']:
            shutil.rmtree(save_dir)
        else:
            return f"[{datetime.now()}] {pos} {station}: Skipped (already exists - overwrite=False)."

    os.makedirs(save_dir)
    csvPr_gen = open(csv_filename, 'w')
    predict_writer = csv.writer(csvPr_gen, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    predict_writer.writerow(['file_name', 
                            'network',
                            'station',
                            'instrument_type',
                            'station_lat',
                            'station_lon',
                            'station_elv',
                            'p_arrival_time',
                            'p_probability',
                            's_arrival_time',
                            's_probability'])  
    csvPr_gen.flush()
    
    start_Predicting = time.time()
    files_list = glob.glob(f"{args['input_dir']}/{station}/*mseed")
    
    try:
        meta, data_set, hp, lp = _mseed2nparray(args, files_list, station)
    except Exception:
        return f"[{datetime.now()}] {pos} {station}: FAILED reading mSEED."

    try:
        params_pred = {'batch_size': args["batch_size"], 'norm_mode': args["normalization_mode"]}
        pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)
        
        # USE THE SHARED MODEL ACTOR INSTEAD OF LOADING MODEL
        predP, predS = ray.get(model_actor.predict.remote(pred_generator))
        
        detection_memory = []
        prob_memory = []
        for ix in range(len(predP)):
            Ppicks, Pprob = _picker(args, predP[ix,:, 0])   
            Spicks, Sprob = _picker(args, predS[ix,:, 0], 'S_threshold')

            detection_memory, prob_memory = _output_writter_prediction(
                meta, csvPr_gen, Ppicks, Pprob, Spicks, Sprob, 
                detection_memory, prob_memory, predict_writer, ix, len(predP), len(predS)
            )
                                        
        end_Predicting = time.time()
        delta = (end_Predicting - start_Predicting)
        return f"[{datetime.now()}] {pos} {station}: Finished the prediction in {round(delta,2)}s. (HP={hp}, LP={lp})"

    except Exception as exp:
        return f"[{datetime.now()}] {pos} {station}: FAILED the prediction. {exp}"


def _mseed2nparray(args, files_list, station):
    ' read miniseed files and from a list of string names and returns 3 dictionaries of numpy arrays, meta data, and time slice info'
          
    st = obspy.Stream()
    # Read and process files
    for file in files_list:
        temp_st = obspy.read(file)
        try:
            temp_st.merge(fill_value=0)
        except Exception:
            temp_st.merge(fill_value=0)
        temp_st.detrend('demean')
        if temp_st:
            st += temp_st
        else:
            return None  # No data to process, return early

    # Apply taper and bandpass filter
    max_percentage = 5 / (st[0].stats.delta * st[0].stats.npts) # 5s of data will be tapered
    st.taper(max_percentage=max_percentage, type='cosine')
    freqmin = 1.0
    freqmax = 45.0
    if args["stations_filters"] is not None:
        try:
            df_filters = args["stations_filters"]
            freqmin = df_filters[df_filters.sta == station].iloc[0]["hp"]
            freqmax = df_filters[df_filters.sta == station].iloc[0]["lp"]
        except:
            pass
    st.filter(type='bandpass', freqmin=freqmin, freqmax=freqmax, corners=2, zerophase=True)

    # Interpolate if necessary
    if any(tr.stats.sampling_rate != 100.0 for tr in st):
        try:
            st.interpolate(100, method="linear")
        except:
            st = _resampling(st)

    # Trim stream to the common start and end times
    st.trim(min(tr.stats.starttime for tr in st), max(tr.stats.endtime for tr in st), pad=True, fill_value=0)
    start_time = st[0].stats.starttime
    end_time = st[0].stats.endtime

    # Prepare metadata
    meta = {
        "start_time": start_time,
        "end_time": end_time,
        "trace_name": f"{files_list[0].split('/')[-2]}/{files_list[0].split('/')[-1]}"
    }
                
    # Prepare component mapping and types
    data_set = {}
    st_times = []
    components = {tr.stats.channel[-1]: tr for tr in st}
    time_shift = int(60 - (args['overlap'] * 60))

    # Define preferred components for each column
    components_list = [
        ['E', '1'],  # Column 0
        ['N', '2'],  # Column 1
        ['Z']        # Column 2
    ]

    current_time = start_time
    while current_time < end_time:
        window_end = current_time + 60
        st_times.append(str(current_time).replace('T', ' ').replace('Z', ''))
        npz_data = np.zeros((6000, 3))

        for col_idx, comp_options in enumerate(components_list):
            for comp in comp_options:
                if comp in components:
                    tr = components[comp].copy().slice(current_time, window_end)
                    data = tr.data[:6000]
                    # Pad with zeros if data is shorter than 6000 samples
                    if len(data) < 6000:
                        data = np.pad(data, (0, 6000 - len(data)), 'constant')
                    npz_data[:, col_idx] = data
                    break  # Stop after finding the first available component

        key = str(current_time).replace('T', ' ').replace('Z', '')
        data_set[key] = npz_data
        current_time += time_shift

    meta["trace_start_time"] = st_times

    # Metadata population with default placeholders for now
    try:
        meta.update({
            "receiver_code": st[0].stats.station,
            "instrument_type": 0,
            "network_code": 0,
            "receiver_latitude": 0,
            "receiver_longitude": 0,
            "receiver_elevation_m": 0
        })
    except Exception:
        meta.update({
            "receiver_code": station,
            "instrument_type": 0,
            "network_code": 0,
            "receiver_latitude": 0,
            "receiver_longitude": 0,
            "receiver_elevation_m": 0
        })
                    
    return meta, data_set, freqmin, freqmax


def _output_writter_prediction(meta, csvPr, Ppicks, Pprob, Spicks, Sprob, detection_memory,prob_memory,predict_writer, idx, cq, cqq):

    """ 
    
    Writes the detection & picking results into a CSV file.

    Parameters
    ----------
    dataset: hdf5 obj
        Dataset object of the trace.

    predict_writer: obj
        For writing out the detection/picking results in the CSV file. 
       
    csvPr: obj
        For writing out the detection/picking results in the CSV file.  

    matches: dic
        It contains the information for the detected and picked event.  
  
    snr: list of two floats
        Estimated signal to noise ratios for picked P and S phases.   
    
    detection_memory : list
        Keep the track of detected events.          
        
    Returns
    -------   
    detection_memory : list
        Keep the track of detected events.  
        
        
    """      

    station_name = meta["receiver_code"]
    station_lat = meta["receiver_latitude"]
    station_lon = meta["receiver_longitude"]
    station_elv = meta["receiver_elevation_m"]
    start_time = meta["trace_start_time"][idx]
    station_name = "{:<4}".format(station_name)
    network_name = meta["network_code"]
    network_name = "{:<2}".format(network_name)
    instrument_type = meta["instrument_type"]
    instrument_type = "{:<2}".format(instrument_type)  

    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        
    def _date_convertor(r):  
        if isinstance(r, str):
            mls = r.split('.')
            if len(mls) == 1:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
            else:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
        else:
            new_t = r
            
        return new_t
  
    
    p_time = []
    p_prob = []
    PdateTime = []
    if Ppicks[0]!=None: 
#for iP in range(len(Ppicks)):
#if Ppicks[iP]!=None: 
        p_time.append(start_time+timedelta(seconds= Ppicks[0]/100))
        p_prob.append(Pprob[0])
        PdateTime.append(_date_convertor(start_time+timedelta(seconds= Ppicks[0]/100)))
        detection_memory.append(p_time) 
        prob_memory.append(p_prob)  
    else:          
        p_time.append(None)
        p_prob.append(None)
        PdateTime.append(None)

    s_time = []
    s_prob = []    
    SdateTime=[]
    if Spicks[0]!=None: 
#for iS in range(len(Spicks)):
#if Spicks[iS]!=None: 
        s_time.append(start_time+timedelta(seconds= Spicks[0]/100))
        s_prob.append(Sprob[0])
        SdateTime.append(_date_convertor(start_time+timedelta(seconds= Spicks[0]/100)))
    else:
        s_time.append(None)
        s_prob.append(None)
        SdateTime.append(None)

    SdateTime = np.array(SdateTime)
    s_prob = np.array(s_prob)
    
    p_prob = np.array(p_prob)
    PdateTime = np.array(PdateTime)
        
    predict_writer.writerow([meta["trace_name"], 
                     network_name,
                     station_name, 
                     instrument_type,
                     station_lat, 
                     station_lon,
                     station_elv,
                     PdateTime[0], 
                     p_prob[0],
                     SdateTime[0], 
                     s_prob[0]
                     ]) 



    csvPr.flush()                


    return detection_memory,prob_memory  


def _get_snr(data, pat, window=200):
    
    """ 
    
    Estimates SNR.
    
    Parameters
    ----------
    data : numpy array
        3 component data.    
        
    pat: positive integer
        Sample point where a specific phase arrives. 
        
    window: positive integer, default=200
        The length of the window for calculating the SNR (in the sample).         
        
    Returns
   --------   
    snr : {float, None}
       Estimated SNR in db. 
       
        
    """      
    import math
    snr = None
    if pat:
        try:
            if int(pat) >= window and (int(pat)+window) < len(data):
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)           
            elif int(pat) < window and (int(pat)+window) < len(data):
                window = int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)
            elif (int(pat)+window) > len(data):
                window = len(data)-int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)    
        except Exception:
            pass
    return snr 


def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def _picker(args, yh3, thr_type='P_threshold'):
    """ 
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array
         probability. 

    Returns
    --------    
    Ppickall: Pick.
    Pproball: Pick Probability.                           
                
    """
    P_PICKall=[]
    Ppickall=[]
    Pproball = []
    perrorall=[]

    sP_arr = _detect_peaks(yh3, mph=args[thr_type], mpd=1)

    P_PICKS = []
    pick_errors = []
    if len(sP_arr) > 0:
        P_uncertainty = None  

        for pick in range(len(sP_arr)):        
            sauto = sP_arr[pick]


            if sauto: 
                P_prob = np.round(yh3[int(sauto)], 3) 
                P_PICKS.append([sauto,P_prob, P_uncertainty]) 

    so=[]
    si=[]
    P_PICKS = np.array(P_PICKS)
    P_PICKall.append(P_PICKS)
    for ij in P_PICKS:
        so.append(ij[1])
        si.append(ij[0])
    try:
        so = np.array(so)
        inds = np.argmax(so)
        swave = si[inds]
        Ppickall.append((swave))
        Pproball.append((np.max(so)))
    except:
        Ppickall.append(None)
        Pproball.append(None)

    #print(np.shape(Ppickall))
    #Ppickall = np.array(Ppickall)
    #Pproball = np.array(Pproball)
    
    return Ppickall, Pproball


def _resampling(st):
    'perform resampling on Obspy stream objects'
    
    need_resampling = [tr for tr in st if tr.stats.sampling_rate != 100.0]
    if len(need_resampling) > 0:
       # print('resampling ...', flush=True)    
        for indx, tr in enumerate(need_resampling):
            if tr.stats.delta < 0.01:
                tr.filter('lowpass',freq=45,zerophase=True)
            tr.resample(100)
            tr.stats.sampling_rate = 100
            tr.stats.delta = 0.01
            tr.data.dtype = 'int32'
            st.remove(tr)                    
            st.append(tr) 
    return st 


def _normalize(data, mode = 'max'):  
    """ 
    
    Normalize 3D arrays.
    
    Parameters
    ----------
    data : 3D numpy array
        3 component traces. 
        
    mode : str, default='std'
        Mode of normalization. 'max' or 'std'     
        
    Returns
    -------  
    data : 3D numpy array
        normalized data. 
            
    """  
       
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data              

    elif mode == 'std':               
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data