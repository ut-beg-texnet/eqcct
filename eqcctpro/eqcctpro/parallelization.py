"""
parallelization.py has access to all Ray functions: mseed_predictor(), parallel_predict(), ModelActor(), and their dependencies. 
It is a level of abstraction so we can make the code more concise and cleaner
"""
import os 
import ray
import csv
import sys
import ast
import math
import time
import json
import glob
import queue 
import obspy
import psutil
import random
import numbers
import logging
import platform
import traceback
import numpy as np
from eqcctpro.tools import *
from eqcctpro.timing_util import cuda_synchronize_best_effort, monotonic_s
from os import listdir
from obspy import UTCDateTime
from datetime import datetime, timedelta 
from logging.handlers import QueueHandler

# =============================================================================
# RESOURCE REQUIREMENTS (MB) - Based on isolated process testing
# =============================================================================
# These values represent the "first-load" memory footprint when each model
# is loaded into a fresh process (like a Ray ModelActor). This includes:
# - Library initialization (PyTorch/TensorFlow, cuDNN, CUDA context)
# - Model architecture definition
# - Model weights
# - Inference buffers and activations

# Safety buffers for unexpected allocations during inference (ModelActor mode)
# These account for Ray worker process overhead, framework (TF/PyTorch) initialization,
# and runtime memory spikes.
# 
# Calibrated from empirical GPU test data (2x 49GB GPUs, 93100 MB pool):
# - EQCCT: 26 actors succeeded with requested 2756 MB/actor (base 1732 + buffer 1024)
# - PhaseNet: 58 actors succeeded with requested 1524 MB/actor (base 500 + buffer 1024)
# - EQTransformer: 32 actors succeeded with requested 2064 MB/actor (base 528 + buffer 1536)
#
# Using 1024 MB as baseline, which proved stable for TensorFlow and smaller PyTorch models.
# The cudnn_headroom parameter provides additional safety margin.
VRAM_BUFFER_MB = 1024.0   # 1GB Extra VRAM headroom per model actor (empirically calibrated)
RAM_BUFFER_MB = 1536.0    # 1.5GB Extra RAM headroom per model actor (for data buffers/Ray overhead)

# GPU Mode - VRAM requirements (MB) for each model actor (first-load cost)
# These include the CUDA context overhead (~500MB) that each actor pays
# Measured via test_resource_usage.py using isolated process + NVML
SEISBENCH_MODEL_VRAM_MB = {
    ('PhaseNet', 'original'): 500.0,
    ('PhaseNet', 'stead'): 502.0,
    ('PhaseNet', 'ethz'): 502.0,
    ('PhaseNet', 'scedc'): 502.0,
    ('PhaseNet', 'pisdl'): 502.0,
    ('PhaseNet', 'instance'): 502.0,
    ('PhaseNetLight', 'stead'): 500.0,
    ('PhaseNetLight', 'ethz'): 500.0,
    ('PhaseNetLight', 'scedc'): 500.0,
    ('PhaseNetLight', 'instance'): 500.0,
    ('EQTransformer', 'original'): 528.0,
    ('EQTransformer', 'original_nonconservative'): 530.0,
    ('EQTransformer', 'stead'): 528.0,
    ('EQTransformer', 'ethz'): 528.0,
    ('EQTransformer', 'scedc'): 528.0,
    ('EQTransformer', 'instance'): 528.0,
    ('GPD', 'original'): 584.0
}

# GPU Mode - RAM requirements (MB) for each model actor (first-load cost)
# Includes PyTorch + ObsPy + CUDA runtime in system RAM
# Measured via test_resource_usage.py using isolated process + psutil RSS
SEISBENCH_MODEL_RAM_MB = {
    ('PhaseNet', 'original'): 870.0,
    ('PhaseNet', 'stead'): 889.0,
    ('PhaseNet', 'ethz'): 900.0,
    ('PhaseNet', 'scedc'): 889.0,
    ('PhaseNet', 'pisdl'): 887.0,
    ('PhaseNet', 'instance'): 897.0,
    ('PhaseNetLight', 'stead'): 861.0,
    ('PhaseNetLight', 'ethz'): 861.0,
    ('PhaseNetLight', 'scedc'): 873.0,
    ('PhaseNetLight', 'instance'): 861.0,
    ('EQTransformer', 'original'): 1001.0,
    ('EQTransformer', 'original_nonconservative'): 1017.0,
    ('EQTransformer', 'stead'): 1017.0,
    ('EQTransformer', 'ethz'): 1021.0,
    ('EQTransformer', 'scedc'): 1025.0,
    ('EQTransformer', 'instance'): 1019.0,
    ('GPD', 'original'): 876.0
    }

# CPU Mode - RAM requirements (MB) for each model (no CUDA overhead)
# Measured via test_resource_usage.py using isolated process + psutil RSS
SEISBENCH_MODEL_CPU_RAM_MB = {
    ('PhaseNet', 'original'): 502.0,
    ('PhaseNet', 'stead'): 511.0,
    ('PhaseNet', 'ethz'): 516.0,
    ('PhaseNet', 'scedc'): 514.0,
    ('PhaseNet', 'pisdl'): 501.0,
    ('PhaseNet', 'instance'): 501.0,
    ('PhaseNetLight', 'stead'): 502.0,
    ('PhaseNetLight', 'ethz'): 498.0,
    ('PhaseNetLight', 'scedc'): 512.0,
    ('PhaseNetLight', 'instance'): 512.0,
    ('EQTransformer', 'original'): 521.0,
    ('EQTransformer', 'original_nonconservative'): 524.0,
    ('EQTransformer', 'stead'): 509.0,
    ('EQTransformer', 'ethz'): 511.0,
    ('EQTransformer', 'scedc'): 509.0,
    ('EQTransformer', 'instance'): 522.0,
    ('GPD', 'original'): 576.0
}

# EQCCT Model Requirements
# Measured via test_resource_usage.py using isolated process
EQCCT_GPU_VRAM_MB = 1732.0   # TensorFlow + XLA compilation + inference buffers
EQCCT_GPU_RAM_MB = 2311.0    # Heavy due to XLA compiled graph stored in system RAM
EQCCT_CPU_RAM_MB = 728.0     # TensorFlow CPU-only runtime

def get_seisbench_model_vram_mb(parent_model_name, child_model_name, default_mb=500.0, logger=None):
    """
    Get VRAM requirement for a SeisBench model including buffer (ModelActor mode).
    
    Args:
        parent_model_name: SeisBench model class (e.g., 'PhaseNet')
        child_model_name: Pretrained version (e.g., 'original')
        default_mb: Default if model not found in lookup table
        logger: Optional logger to warn when using default values
        
    Returns:
        float: Total VRAM in MB needed for one model actor
    """
    key = (parent_model_name, child_model_name)
    if key not in SEISBENCH_MODEL_VRAM_MB:
        if logger:
            logger.warning(f"Unknown SeisBench model '{parent_model_name}/{child_model_name}' - using default VRAM estimate ({default_mb} MB). "
                          f"Consider running measure_model_memory_usage.py to get accurate values.")
        base_vram = default_mb
    else:
        base_vram = SEISBENCH_MODEL_VRAM_MB[key]
    return base_vram + VRAM_BUFFER_MB

def get_seisbench_model_vram_mb_ripper(parent_model_name, child_model_name, default_mb=500.0, logger=None):
    """
    Get VRAM requirement for a SeisBench model in Ripper mode.
    
    Uses empirically-calibrated initialization multipliers based on actual GPU test data.
    Smaller models have relatively higher overhead due to fixed CUDA context costs.
    
    Calibration data (2x 49GB GPUs, 93100 MB pool):
    - PhaseNet: 58 actors → ~1605 MB/actor → 1.65x base
    - EQTransformer: 32 actors → ~2909 MB/actor → 1.90x base (larger model)
    
    We add a small safety margin to empirical values for robustness.
    """
    # Model-specific multipliers calibrated from empirical GPU test results
    # Smaller models have higher relative overhead (CUDA context is fixed cost)
    RIPPER_MULTIPLIERS = {
        'PhaseNet': 1.7,       # Empirical 1.65x + margin
        'PhaseNetLight': 1.7,  # Similar architecture to PhaseNet
        'EQTransformer': 2.0,  # Empirical 1.90x + margin (larger model)
        'GPD': 1.8,            # Medium-sized model
    }
    
    multiplier = RIPPER_MULTIPLIERS.get(parent_model_name, 1.8)  # Default for unknown models
    
    key = (parent_model_name, child_model_name)
    if key not in SEISBENCH_MODEL_VRAM_MB:
        if logger:
            logger.warning(f"Unknown SeisBench model '{parent_model_name}/{child_model_name}' - using default VRAM estimate ({default_mb} MB). "
                          f"Consider running measure_model_memory_usage.py to get accurate values.")
        base_vram = default_mb
    else:
        base_vram = SEISBENCH_MODEL_VRAM_MB[key]
    return base_vram * multiplier

def get_seisbench_model_ram_mb(parent_model_name, child_model_name, use_gpu=True, default_mb=500.0, logger=None):
    """
    Get RAM requirement for a SeisBench model including buffer.
    
    Args:
        parent_model_name: SeisBench model class (e.g., 'PhaseNet')
        child_model_name: Pretrained version (e.g., 'original')
        use_gpu: Whether GPU mode is being used
        default_mb: Default if model not found in lookup table
        logger: Optional logger to warn when using default values
        
    Returns:
        float: Total RAM in MB needed for one model actor
    """
    key = (parent_model_name, child_model_name)
    lookup_table = SEISBENCH_MODEL_RAM_MB if use_gpu else SEISBENCH_MODEL_CPU_RAM_MB
    if key not in lookup_table:
        mode_str = "GPU" if use_gpu else "CPU"
        if logger:
            logger.warning(f"Unknown SeisBench model '{parent_model_name}/{child_model_name}' in {mode_str} mode - using default RAM estimate ({default_mb} MB). "
                          f"Consider running measure_model_memory_usage.py to get accurate values.")
        base_ram = default_mb
    else:
        base_ram = lookup_table[key]
    return base_ram + RAM_BUFFER_MB

def get_eqcct_vram_mb():
    """Get VRAM requirement for EQCCT model actor (ModelActor mode)."""
    return EQCCT_GPU_VRAM_MB + VRAM_BUFFER_MB

def get_eqcct_vram_mb_ripper():
    """
    Get VRAM requirement for EQCCT in Ripper mode.
    
    Ripper mode loads/unloads models per-task, which means:
    - Multiple tasks may initialize TensorFlow simultaneously
    - CUDA contexts are created/destroyed repeatedly
    - Memory fragmentation is higher
    
    Uses an empirically-calibrated multiplier based on actual GPU test data.
    
    Calibration data (2x 49GB GPUs, 93100 MB pool):
    - 28 concurrent EQCCT tasks succeeded
    - Process Tree VRAM: ~94,070 MB
    - Actual per task: 3,360 MB
    - Base VRAM: 1,732 MB
    - Empirical multiplier: 1.94x
    
    We use 2.0x as a small safety margin above the empirical value.
    """
    # Empirically calibrated multiplier (1.94x measured, 2.0x with margin)
    # Accounts for: TF graph build, XLA compilation, cuDNN workspace, fragmentation
    RIPPER_INIT_MULTIPLIER = 2.0
    return EQCCT_GPU_VRAM_MB * RIPPER_INIT_MULTIPLIER

def get_eqcct_ram_mb(use_gpu=True):
    """Get RAM requirement for EQCCT model actor."""
    if use_gpu:
        return EQCCT_GPU_RAM_MB + RAM_BUFFER_MB
    else:
        return EQCCT_CPU_RAM_MB + RAM_BUFFER_MB


# =============================================================================
# MEMORY AVAILABILITY FUNCTIONS
# =============================================================================
# These functions query available system memory to enable memory-aware actor creation.
# Used when creating ModelActors to ensure we don't exceed physical memory limits.

def get_available_vram_mb(gpu_ids=None, max_vram_mb=None, logger=None):
    """
    Get available VRAM in MB for the specified GPUs.
    
    If max_vram_mb is provided (user-defined cap), returns that value divided
    equally among the specified GPUs (per-GPU budget). Otherwise, queries
    the actual free VRAM from each GPU.
    
    Args:
        gpu_ids: List of GPU IDs to query. If None, queries all available GPUs.
        max_vram_mb: Optional user-defined total VRAM cap across all GPUs.
                     If provided, this is the max budget (already divided by n_gpus externally).
        logger: Optional logger for debug messages.
        
    Returns:
        float: Available VRAM in MB (total across specified GPUs).
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        
        if gpu_ids is None:
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpu_ids = list(range(gpu_count))
        
        # If user provided a max_vram_mb cap, use that
        if max_vram_mb is not None and max_vram_mb > 0:
            if logger:
                logger.info(f"Using user-defined VRAM cap: {max_vram_mb:.0f} MB total")
            pynvml.nvmlShutdown()
            return float(max_vram_mb)
        
        # Otherwise, query actual free VRAM from each GPU
        total_free_vram_mb = 0.0
        for gpu_id in gpu_ids:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_mb = mem_info.free / (1024 * 1024)  # Convert to MB
                total_free_vram_mb += free_mb
            except Exception as e:
                if logger:
                    logger.warning(f"Could not query VRAM for GPU {gpu_id}: {e}")
        
        pynvml.nvmlShutdown()
        
        if logger:
            logger.info(f"Total free VRAM across GPUs {gpu_ids}: {total_free_vram_mb:.0f} MB")
        
        return total_free_vram_mb
    
    except Exception as e:
        if logger:
            logger.error(f"Failed to query VRAM: {e}")
        return 0.0


def get_available_ram_mb(ram_safety_cap=0.90, logger=None):
    """
    Get available RAM in MB based on system memory and safety cap.
    
    Args:
        ram_safety_cap: Fraction of TOTAL system RAM that can be used (0.0-1.0).
                        This is applied to total RAM, not just available.
        logger: Optional logger for debug messages.
        
    Returns:
        float: Usable RAM in MB (total system RAM * safety_cap).
    """
    try:
        mem = psutil.virtual_memory()
        total_ram_mb = mem.total / (1024 * 1024)
        available_ram_mb = mem.available / (1024 * 1024)
        
        # Apply safety cap to TOTAL RAM (not just available)
        # This gives a consistent budget regardless of current system state
        usable_ram_mb = total_ram_mb * ram_safety_cap
        
        if logger:
            logger.info(f"System RAM: Total={total_ram_mb:.0f} MB, Available={available_ram_mb:.0f} MB")
            logger.info(f"Usable RAM budget: {usable_ram_mb:.0f} MB ({ram_safety_cap:.0%} of total)")
        
        return usable_ram_mb
    
    except Exception as e:
        if logger:
            logger.error(f"Failed to query RAM: {e}")
        return 0.0

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
    
def _eqcct_stream_to_nparray(args, st, station, files_list=None):
    """
    EQCCT preprocessing from an in-memory ObsPy Stream for a single station
    (taper, bandpass, resample, windowing). Used by disk path and RIPPER ray.get path.
    """
    if st is None or len(st) == 0:
        return None
    st = obspy.Stream(traces=[tr.copy() for tr in st])
    try:
        st.merge(fill_value=0)
    except Exception:
        try:
            st.merge(fill_value=0)
        except Exception:
            pass
    if len(st) == 0:
        return None

    max_percentage = 5 / (st[0].stats.delta * st[0].stats.npts)
    st.taper(max_percentage=max_percentage, type='cosine')
    freqmin = 1.0
    freqmax = 45.0
    if args["stations_filters"] is not None:
        try:
            df_filters = args["stations_filters"]
            freqmin = df_filters[df_filters.sta == station].iloc[0]["hp"]
            freqmax = df_filters[df_filters.sta == station].iloc[0]["lp"]
        except Exception:
            pass
    st.filter(type='bandpass', freqmin=freqmin, freqmax=freqmax, corners=2, zerophase=True)

    if any(tr.stats.sampling_rate != 100.0 for tr in st):
        try:
            st.interpolate(100, method="linear")
        except Exception:
            st = _resampling(st)

    st.trim(min(tr.stats.starttime for tr in st), max(tr.stats.endtime for tr in st), pad=True, fill_value=0)
    start_time = st[0].stats.starttime
    end_time = st[0].stats.endtime

    if files_list:
        trace_name = f"{files_list[0].split('/')[-2]}/{files_list[0].split('/')[-1]}"
    else:
        trace_name = f"{station}/ray_object_store"

    meta = {
        "start_time": start_time,
        "end_time": end_time,
        "trace_name": trace_name,
    }

    data_set = {}
    st_times = []
    components = {tr.stats.channel[-1]: tr for tr in st}
    time_shift = int(60 - (args['overlap'] * 60))

    components_list = [
        ['E', '1'],
        ['N', '2'],
        ['Z'],
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
                    if len(data) < 6000:
                        data = np.pad(data, (0, 6000 - len(data)), 'constant')
                    npz_data[:, col_idx] = data
                    break

        key = str(current_time).replace('T', ' ').replace('Z', '')
        data_set[key] = npz_data
        current_time += time_shift

    meta["trace_start_time"] = st_times

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


def _mseed2nparray(args, files_list, station):
    ' read miniseed files and from a list of string names and returns 3 dictionaries of numpy arrays, meta data, and time slice info'

    st = obspy.Stream()
    for file in files_list:
        temp_st = obspy.read(file)
        try:
            temp_st.merge(fill_value=0)
        except Exception:
            temp_st.merge(fill_value=0)
        temp_st.detrend('demean')
        if temp_st:
            st += temp_st

    if not st or len(st) == 0:
        return None
    return _eqcct_stream_to_nparray(args, st, station, files_list=files_list)


def _load_ripper_mseed_stream(input_dir: str, station_list: list) -> obspy.Stream:
    """
    Read all station miniSEED once on the driver (scmlpick-style) for ray.put.
    Traces are read, merged per file, demeaned — same as the start of per-task disk reads.
    """
    full = obspy.Stream()
    for station in station_list:
        files_list = sorted(glob.glob(os.path.join(input_dir, str(station), "*mseed")))
        for file in files_list:
            temp_st = obspy.read(file)
            try:
                temp_st.merge(fill_value=0)
            except Exception:
                temp_st.merge(fill_value=0)
            temp_st.detrend('demean')
            if temp_st:
                full += temp_st
    return full


def _trace_matches_station_task(tr, station_id: str) -> bool:
    """
    Map timechunk subdirectory names to ObsPy trace headers.

    ``build_station_list_from_dir`` uses directory basenames (e.g. ``TX_EF09``).
    MiniSEED stores network and station separately (``TX`` + ``EF09``), so comparing
    only ``tr.stats.station`` to ``TX_EF09`` never matches.
    """
    sid = str(station_id).strip()
    if not sid:
        return False
    net = str(tr.stats.network).strip()
    sta = str(tr.stats.station).strip()
    if sta == sid:
        return True
    if net and f"{net}_{sta}".upper() == sid.upper():
        return True
    if net and "_" in sid:
        prefix, rest = sid.split("_", 1)
        if prefix.upper() == net.upper() and rest.strip().upper() == sta.upper():
            return True
    dotted = sid.replace("_", ".", 1) if "_" in sid else sid
    if net and f"{net}.{sta}".upper() == dotted.upper():
        return True
    return False


def _stream_select_for_station_task(full_st: obspy.Stream, station: str) -> obspy.Stream:
    """Subset a merged Stream to traces for one task station id (subdir name)."""
    return obspy.Stream(traces=[tr.copy() for tr in full_st if _trace_matches_station_task(tr, station)])


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
              log_queue=None,
              stations2use=None,
              stations_filters=None,
              p_model=None,
              s_model=None,
              number_of_concurrent_station_predictions=None,
              ray_cpus=None,
              use_gpu=False,
              gpu_memory_limit_mb=None,
              total_vram_pool_mb=None,  # NEW: Total VRAM budget for all actors (aggregate cap)
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
              timechunk_dt=None,
              # SeisBench model parameters
              model_type='eqcct',
              seisbench_parent_model=None,
              seisbench_child_model=None,
              Detection_threshold=0.3,
              ram_safety_cap=None,
              cudnn_headroom=0.20,
              # Ripper mode - uses old task-based approach instead of ModelActors
              ripper=False,
              # If set, use this exact station order/count (deterministic benchmarks).
              # Skips random.sample(stations2use) and specific_stations filtering.
              fixed_station_list=None):
    
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

    cudnn_headroom: float, default=0.20
        Percentage of GPU VRAM to reserve for cuDNN workspace overhead (0.0 to 0.80).
        This prevents "DNN library is not found" errors during concurrent predictions.

    overwrite: Bolean, default=False
        Overwrite your results automatically.
           
    Returns
    --------        
      
    """ 

    # Set up logger that will write logs to this native process and add them to the log.queue to be added back to the main logger outside of this Raylet
    # worker logger ships records to driver
    logger = logging.getLogger("eqcctpro.worker")
    logger.setLevel(logging.INFO)
    logger.handlers[:] = []
    logger.propagate = False
    log_handler = QueueHandler(log_queue)
    if log_queue is not None:
        logger.addHandler(log_handler)  # Ray queue supports put()

    # ===== RAM SAFETY CAP VALIDATION =====
    if ram_safety_cap is not None:
        if ram_safety_cap > 0.97:
            logger.error(f"CRITICAL: ram_safety_cap ({ram_safety_cap:.2f}) exceeds the maximum allowed limit of 0.97. This is unsafe for system stability.")
            logger.error("Please reduce ram_safety_cap to 0.97 or lower and try again. Exiting...")
            sys.exit(1)
        logger.info(f"RAM safety cap validated: {ram_safety_cap:.1%}")

    # We set up the tf_environ again for the Raylets, who adopt their own import state and TF runtime when created. 
    # We want to ensure that they are configured properly so that they won't die (bad)
    skip_tf = (model_type.lower() != 'eqcct')
    if not use_gpu: 
        tf_environ(gpu_id=-1, intra_threads=intra_threads, inter_threads=inter_threads, logger=logger, skip_tf=skip_tf)
        # tf_environ(gpu_id=1, gpu_memory_limit_mb=gpu_memory_limit_mb, gpus_to_use=gpu_id, intra_threads=intra_threads, inter_threads=inter_threads)

    # ===== TIMING: Start tracking total trial time =====
    trial_start_time = monotonic_s()

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
    "stations_filters": stations_filters,
    "model_type": model_type,
    "seisbench_parent_model": seisbench_parent_model,
    "seisbench_child_model": seisbench_child_model,
    "Detection_threshold": Detection_threshold
    }

    logger.info(f"------- Hardware Configuration -------")
    try:
        process = psutil.Process(os.getpid())
        process.cpu_affinity(ray_cpus)  # ray_cpus should be a list of core IDs like [0, 1, 2]
        logger.info(f"CPU affinity set to cores: {list(ray_cpus)}")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to set CPU affinity. Reason: {e}")
        logger.error("")
        sys.exit(1)
    
    out_dir = os.path.join(os.getcwd(), str(args['output_dir']))    
    try:
        from eqcctpro.tools import build_station_list_from_dir
        station_list = build_station_list_from_dir(args['input_dir'])
    except Exception as e:
        logger.info(f"{e}") 
        return # To-Do: Fix so that it has a valid return? 
    # log.write(f"GPU ID: {args['gpu_id']}; Batch size: {args['batch_size']}")
    logger.info(f"------- Data Preprocessing for EQCCTPro -------")
    logger.info(f"{len(station_list)} station(s) in {args['input_dir']}")
    
    if fixed_station_list is not None:
        station_list = list(fixed_station_list)
    elif stations2use and stations2use <= len(station_list):  # For System Evaluation Execution
        station_list = random.sample(station_list, stations2use)  # Randomly choose stations from the sample size 
        # log.write(f"Using {len(station_list)} station(s) after selection.")

    if specific_stations is not None and fixed_station_list is None:
        station_list = [x for x in station_list if x in specific_stations] # For "One Use Run" Over a Given Set of Stations (Just Run EQCCTPro on specific_stations)
    else:
        station_list = station_list  # someone put None thinking that they would be able to run the whole directory in one go
    logger.info(f"Using {len(station_list)} selected station(s): {station_list}.") 

    if not station_list or (fixed_station_list is None and any(looks_like_timechunk_id(x) for x in station_list)):
        # Rebuild from the actual contents of the timechunk dir
        station_list = build_station_list_from_dir(args['input_dir'])
        logger.info(f"Station list rebuilt from directory because it contained a timechunk id or was empty.") 

    tasks_predictor = [[f"({i+1}/{len(station_list)})", station_list[i], out_dir, args] for i in range(len(station_list))]
    
    if not tasks_predictor: return
    
    # =====================================================================
    # RIPPER MODE: Use old task-based approach (model loaded per task)
    # This bypasses ModelActors and allows more flexible GPU sharing.
    #
    # Scheduling matches scmlpick ``run_picker`` (bounded queue + ray.wait drain +
    # backfill). Unlike ModelActor mode (1 actor per GPU with round-robin),
    # ripper mode launches concurrent tasks that each load their own model.
    # To prevent OOM, we limit in-flight tasks (max_pending_tasks) from VRAM/RAM.
    #
    # The same automatic Ray restart mechanism applies at EvaluateSystem level
    # for OOM prevention between trials.
    # =====================================================================
    if ripper:
        # ===== TIMING: Ripper mode has no actor creation, just setup time =====
        setup_start_time = monotonic_s()
        
        logger.info(f"===== RIPPER MODE ENABLED =====")
        logger.info(f"Using old task-based approach (model loaded per task)")
        logger.info(f"This allows more flexible GPU memory sharing but has model loading overhead.")
        
        model_type_lower = model_type.lower() if model_type else 'eqcct'
        
        # Calculate VRAM-aware concurrency limit for ripper mode
        # Unlike ModelActor mode, ripper tasks each load their own model
        # so we must limit concurrent tasks based on available VRAM
        #
        # IMPORTANT: We query ACTUAL free VRAM at runtime because:
        # 1. User's max_vram_mb might be for multiple GPUs, but we're only using gpu_id subset
        # 2. Other processes may be using GPU memory
        # 3. Previous trials may not have fully released VRAM
        if use_gpu and gpu_id:
            # Get VRAM requirement per task using RIPPER-SPECIFIC estimates
            # These use a single initialization multiplier (2.5×) instead of stacking buffers
            # This accounts for: TF graph build, XLA compilation, cuDNN workspace, fragmentation
            if model_type_lower == 'seisbench':
                vram_per_task_mb = get_seisbench_model_vram_mb_ripper(
                    seisbench_parent_model, seisbench_child_model, logger=logger)
            else:
                vram_per_task_mb = get_eqcct_vram_mb_ripper()
            
            # Calculate per-GPU user-defined VRAM budget
            # max_vram_mb (total_vram_pool_mb) is the total across ALL GPUs in selected_gpus
            # so we divide by the number of GPUs to get per-GPU budget
            num_gpus = len(gpu_id) if isinstance(gpu_id, (list, tuple)) else 1
            user_vram_per_gpu_mb = (total_vram_pool_mb / num_gpus) if total_vram_pool_mb else float('inf')
            
            # Query ACTUAL free VRAM from the GPU(s) at runtime using pynvml
            # This prevents OOM when:
            # - User specifies pool for multiple GPUs but only uses subset
            # - Other processes are using GPU memory
            # - Previous trials haven't fully released VRAM
            try:
                import pynvml
                pynvml.nvmlInit()
                actual_free_vram_mb = 0
                for gpu_idx in gpu_id:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    actual_free_vram_mb += mem_info.free / (1024 * 1024)  # Convert bytes to MB
                pynvml.nvmlShutdown()
                logger.info(f"RIPPER VRAM CHECK: Actual free VRAM across {num_gpus} GPU(s): {actual_free_vram_mb:.0f} MB")
            except Exception as e:
                logger.warning(f"Could not query GPU VRAM via pynvml: {e}. Using user-defined pool.")
                actual_free_vram_mb = user_vram_per_gpu_mb * num_gpus  # Fallback to user pool
            
            # Use the MINIMUM of: user-defined per-GPU budget OR actual free VRAM
            # This prevents OOM when user overestimates available VRAM
            effective_vram_pool = min(
                user_vram_per_gpu_mb * num_gpus,  # User's total budget for these GPUs
                actual_free_vram_mb               # Actual free VRAM on these GPUs
            )
            
            # =====================================================================
            # RIPPER-SPECIFIC CONCURRENCY HEADROOM
            # =====================================================================
            # The per-task VRAM multiplier (2.0× for EQCCT, 1.7× for PhaseNet) handles
            # SINGLE-TASK overhead: TF/PyTorch init, graph build, cuDNN workspace.
            #
            # However, when many tasks run CONCURRENTLY, there's additional overhead from:
            # - cuDNN workspace contention (multiple tasks allocating simultaneously)
            # - Memory fragmentation from parallel allocations
            # - CUDA context switching overhead
            #
            # Empirical testing on 2x 49GB GPUs (93100 MB pool) shows:
            # - 0% headroom (13 tasks/GPU = 26 total): FAILS with cuDNN errors
            # - 25% headroom (10 tasks/GPU = 20 total): WORKS but too conservative
            # - 10% headroom (12 tasks/GPU = 24 total): Target balance
            #
            # This is NOT double-counting with the multiplier because:
            # - Multiplier: Per-task initialization overhead
            # - Concurrency headroom: Multi-task interference overhead
            # =====================================================================
            RIPPER_CONCURRENCY_HEADROOM = 0.10  # 10% reserved for concurrent task interference
            usable_vram_mb = effective_vram_pool * (1.0 - RIPPER_CONCURRENCY_HEADROOM)
            usable_vram_per_gpu_mb = usable_vram_mb / num_gpus
            
            # =====================================================================
            # RIPPER CONCURRENCY CALCULATION
            # =====================================================================
            # vram_per_task_mb includes empirically-calibrated multiplier (2.0× for EQCCT)
            # Plus 10% concurrency headroom for multi-task interference
            # =====================================================================
            
            max_tasks_per_gpu = max(1, int(usable_vram_per_gpu_mb / vram_per_task_mb))
            max_safe_concurrent = max_tasks_per_gpu * num_gpus
            
            # Cap max_pending_tasks to the VRAM-safe limit
            requested_concurrency = number_of_concurrent_station_predictions
            if requested_concurrency > max_safe_concurrent:
                logger.warning(f"RIPPER VRAM LIMIT: Requested {requested_concurrency} concurrent tasks, "
                             f"but VRAM allows {max_safe_concurrent} "
                             f"({max_tasks_per_gpu} tasks/GPU × {num_gpus} GPUs)")
                max_pending_tasks = max_safe_concurrent
            else:
                max_pending_tasks = requested_concurrency
            
            # Calculate per-task VRAM limit for TensorFlow soft cap
            # Each task gets its fair share of the GPU's usable VRAM
            tasks_per_gpu = max(1, int((max_pending_tasks + num_gpus - 1) / num_gpus))  # Ceiling division
            ripper_vram_limit_per_task_mb = int(usable_vram_per_gpu_mb / tasks_per_gpu)
            
            # Calculate the effective multiplier for logging
            if model_type_lower == 'seisbench':
                base_vram = SEISBENCH_MODEL_VRAM_MB.get(
                    (seisbench_parent_model, seisbench_child_model), 500.0)
                effective_multiplier = vram_per_task_mb / base_vram if base_vram > 0 else 2.0
            else:
                effective_multiplier = vram_per_task_mb / EQCCT_GPU_VRAM_MB
            
            logger.info(f"VRAM-aware concurrency: {max_pending_tasks} concurrent tasks "
                       f"(User budget/GPU: {user_vram_per_gpu_mb:.0f} MB, Actual free: {actual_free_vram_mb:.0f} MB)")
            logger.info(f"RIPPER VRAM ({effective_multiplier:.1f}× task multiplier + {RIPPER_CONCURRENCY_HEADROOM*100:.0f}% concurrency headroom): "
                       f"{vram_per_task_mb:.0f} MB/task, Usable: {usable_vram_per_gpu_mb:.0f} MB/GPU → {max_tasks_per_gpu} tasks/GPU max")
            logger.info(f"RIPPER VRAM SLICING: {tasks_per_gpu} tasks/GPU × {ripper_vram_limit_per_task_mb} MB/task "
                       f"= {tasks_per_gpu * ripper_vram_limit_per_task_mb} MB/GPU (budget: {usable_vram_per_gpu_mb:.0f} MB/GPU)")
            
            # CRITICAL: Override gpu_memory_limit_mb with the computed per-task limit
            # This ensures each Ripper task sets the correct TensorFlow soft memory cap
            gpu_memory_limit_mb = ripper_vram_limit_per_task_mb
        else:
            # =====================================================================
            # CPU RIPPER MODE: RAM-Aware Concurrency Limiting
            # Similar to GPU mode, we query actual free RAM and cap concurrency
            # to prevent OOM when many tasks load models simultaneously.
            # =====================================================================
            
            # Get RAM requirement per task
            if model_type_lower == 'seisbench':
                ram_per_task_mb = get_seisbench_model_ram_mb(
                    seisbench_parent_model, seisbench_child_model, use_gpu=False, logger=logger)
            else:
                ram_per_task_mb = get_eqcct_ram_mb(use_gpu=False)
            
            # Query actual free RAM using psutil (already imported at module level)
            try:
                mem_info = psutil.virtual_memory()
                system_ram_total_mb = mem_info.total / (1024 * 1024)
                actual_free_ram_mb = mem_info.available / (1024 * 1024)
                logger.info(f"RIPPER RAM CHECK: Total RAM: {system_ram_total_mb:.0f} MB, Available: {actual_free_ram_mb:.0f} MB")
            except Exception as e:
                logger.warning(f"Could not query system RAM: {e}. Using unlimited concurrency.")
                max_pending_tasks = number_of_concurrent_station_predictions
                actual_free_ram_mb = None
            
            if actual_free_ram_mb is not None:
                # Budget = fraction of TOTAL installed RAM (same idea as get_available_ram_mb).
                # Using only psutil.available * cap wrongly caps Ripper when much RAM is
                # cached/freeable but not currently in the "available" counter — e.g. requesting
                # 150 tasks while "available" implies ~140 tasks worth of headroom.
                ripper_ram_cap = ram_safety_cap if ram_safety_cap is not None else 0.95
                usable_ram_mb = system_ram_total_mb * ripper_ram_cap
                max_safe_concurrent = max(1, int(usable_ram_mb / ram_per_task_mb))
                
                # Cap max_pending_tasks to the RAM-safe limit
                requested_concurrency = number_of_concurrent_station_predictions
                if requested_concurrency > max_safe_concurrent:
                    logger.warning(
                        f"RIPPER RAM LIMIT: Requested {requested_concurrency} concurrent tasks, "
                        f"but RAM budget only allows {max_safe_concurrent} "
                        f"({ripper_ram_cap:.0%} of total {system_ram_total_mb:.0f} MB / {ram_per_task_mb:.0f} MB per task)"
                    )
                    max_pending_tasks = max_safe_concurrent
                else:
                    max_pending_tasks = requested_concurrency
                
                logger.info(
                    f"RAM-aware concurrency: {max_pending_tasks} concurrent tasks "
                    f"(Total RAM: {system_ram_total_mb:.0f} MB, budget {ripper_ram_cap:.0%} → {usable_ram_mb:.0f} MB, "
                    f"currently available: {actual_free_ram_mb:.0f} MB, per-task estimate: {ram_per_task_mb:.0f} MB)"
                )
        
        # ===== TIMING: End of setup, start of processing =====
        setup_end_time = monotonic_s()
        setup_time_seconds = setup_end_time - setup_start_time
        logger.info(f"Ripper mode setup completed in {setup_time_seconds:.2f} seconds")
        
        logger.info(f"Starting EQCCTPro parallelized waveform processing (RIPPER MODE)...") 
        logger.info("")
        start_time = monotonic_s() 
        
        if model_type_lower == 'seisbench':
            logger.info(f"------- Analyzing Seismic Waveforms for P and S Picks via SeisBench ({seisbench_parent_model} - {seisbench_child_model}) [RIPPER] -------")
        else:
            logger.info(f"------- Analyzing Seismic Waveforms for P and S Picks via EQCCT [RIPPER] -------")

        if timechunk_id is None:
            cand = os.path.basename(input_dir)
            if "_" in cand and len(cand) >= 10:
                timechunk_id = cand
            else:
                raise ValueError("timechunk_id is None and could not be inferred from input_dir")
        starttime, endtime, time_delta = parse_time_range(timechunk_id)

        logger.info(f"Analyzing {time_delta} minute timechunk from {starttime} to {endtime} ({waveform_overlap} min overlap)")
        logger.info(f"Processing a total of {len(tasks_predictor)} stations, {max_pending_tasks} at a time.") 

        # scmlpick-style in-memory refs: one ray.put for the full Stream and one for args so each
        # .remote() only ships ObjectRefs, not N copies of waveforms / config.
        merged_stream = _load_ripper_mseed_stream(args["input_dir"], station_list)
        if len(merged_stream) == 0:
            logger.warning(
                "RIPPER: no traces preloaded from disk; tasks will read mSEED per station (no shared object store Stream)."
            )
            tasks_predictor_ripper = tasks_predictor
        else:
            logger.info(
                f"RIPPER: ray.put shared Stream ({len(merged_stream)} trace(s)) and args for {len(station_list)} station task(s)."
            )
            args_ref = ray.put(args)
            stream_ref = ray.put(merged_stream)
            tasks_predictor_ripper = [
                [f"({i+1}/{len(station_list)})", station_list[i], out_dir, args_ref, stream_ref]
                for i in range(len(station_list))
            ]

        # Concurrent Prediction(s) Parallel Processing - RIPPER MODE
        # Same scheduling pattern as scmlpick run_picker: seed queue, wait(1), backfill.
        try:
            def _ripper_submit_index(i: int):
                if model_type_lower == "seisbench":
                    if use_gpu is False:
                        return ripper_parallel_predict_seisbench.remote(
                            tasks_predictor_ripper[i],
                            False,
                            None,
                            parent_model_name=seisbench_parent_model,
                            child_model_name=seisbench_child_model,
                            Detection_threshold=Detection_threshold,
                        )
                    gpu_allocation_per_task = len(gpu_id) / max_pending_tasks
                    return ripper_parallel_predict_seisbench.options(
                        num_gpus=gpu_allocation_per_task, num_cpus=0
                    ).remote(
                        tasks_predictor_ripper[i],
                        True,
                        gpu_memory_limit_mb,
                        parent_model_name=seisbench_parent_model,
                        child_model_name=seisbench_child_model,
                        Detection_threshold=Detection_threshold,
                    )
                if use_gpu is False:
                    return ripper_parallel_predict_eqcct.remote(
                        tasks_predictor_ripper[i], False, None
                    )
                gpu_allocation_per_task = len(gpu_id) / max_pending_tasks
                return ripper_parallel_predict_eqcct.options(
                    num_gpus=gpu_allocation_per_task, num_cpus=0
                ).remote(tasks_predictor_ripper[i], True, gpu_memory_limit_mb)

            model_load_times, waveform_load_times = _run_ripper_parallel_queue_scmlpick(
                logger=logger,
                tasks_predictor=tasks_predictor_ripper,
                max_tasks_queue=max_pending_tasks,
                submit_index=_ripper_submit_index,
            )
            logger.info("")
            
            # Calculate average model load time
            if model_load_times:
                avg_model_load_time = sum(model_load_times) / len(model_load_times)
                logger.info(f"Average model load time (per task): {avg_model_load_time:.3f}s (across {len(model_load_times)} tasks)")
            else:
                avg_model_load_time = 0.0
            
            # Calculate average waveform load time
            if waveform_load_times:
                avg_waveform_load_time = sum(waveform_load_times) / len(waveform_load_times)
                logger.info(f"Average waveform load time (per task): {avg_waveform_load_time:.3f}s (across {len(waveform_load_times)} tasks)")
            else:
                avg_waveform_load_time = 0.0

        except Exception as e:
            avg_model_load_time = 0.0  # Default if error occurs before collecting any times
            avg_waveform_load_time = 0.0
            logger.error(f"ERROR in parallel processing (RIPPER MODE) at {datetime.now()}")
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        logger.info(f"------- Parallel Station Waveform Processing Complete [RIPPER MODE] -------")
        end_time = monotonic_s()
        logger.info(f"Picks saved at {output_dir}. Process Runtime: {end_time - start_time:.2f} s")

        if testing_gpu is not None: 
            num_ray_cpus = len(ray_cpus) if isinstance(ray_cpus, (list, tuple)) else int(len(list(ray_cpus)))
            if timechunk_id:
                starttime, endtime, time_delta = parse_time_range(timechunk_id)
                timechunk_length_min = time_delta.total_seconds() / 60.0 if time_delta else None
            else:
                timechunk_length_min = None
            if model_type_lower == 'seisbench':
                model_used = f"{seisbench_parent_model}/{seisbench_child_model}"
            else:
                model_used = "eqcct"
            
            # Calculate timing metrics for ripper mode
            total_trial_time_seconds = end_time - trial_start_time
            waveform_processing_time_seconds = end_time - start_time
            
            trial_data = {
                "Trial Number": None,
                "Stations Used": str(station_list),
                "Number of Stations Used": len(station_list),
                "Number of CPUs Allocated for Ray to Use": num_ray_cpus,
                "Intra-parallelism Threads": intra_threads if intra_threads is not None else "",
                "Inter-parallelism Threads": inter_threads if inter_threads is not None else "",
                "GPUs Used": json.dumps(list(gpu_id)) if (use_gpu and gpu_id is not None) else "[]",
                "Inference Actor Memory Limit (MB)": float(gpu_memory_limit_mb) if (use_gpu and gpu_memory_limit_mb is not None) else "",
                "Total Waveform Analysis Timespace (min)": float(total_analysis_time.total_seconds() / 60.0) if hasattr(total_analysis_time, "total_seconds") else (float(total_analysis_time) if total_analysis_time else ""),
                "Total Number of Timechunks": int(total_timechunks) if total_timechunks is not None else "",
                "Concurrent Timechunks Used": int(number_of_concurrent_timechunk_predictions) if number_of_concurrent_timechunk_predictions is not None else "",
                "Length of Timechunk (min)": timechunk_length_min if timechunk_length_min is not None else "",
                "N ModelActors": 0,  # RIPPER mode doesn't use ModelActors
                "Number of Concurrent Station Tasks": int(number_of_concurrent_station_predictions) if number_of_concurrent_station_predictions is not None else "",
                "Actual Ripper Concurrent Tasks": int(max_pending_tasks),  # Actual concurrent tasks after VRAM/RAM limiting
                # ===== TIMING METRICS =====
                "Total Trial Time (s)": round(total_trial_time_seconds, 6),  # Entire trial: setup + processing
                "Actor Creation Time (s)": "",  # N/A for RIPPER mode (no actors created)
                "Avg Model Load Time (s)": round(avg_model_load_time, 6),  # RIPPER: average time to load model per task
                "Waveform Processing Time (s)": round(avg_waveform_load_time, 6),  # Average time to load waveforms into memory per task
                "Total Run time for Picker (s)": round(waveform_processing_time_seconds, 6),  # Total time for all task processing
                "Model Used": model_used,
                "Trial Success": "",
                "Error Message": str(""),
                "Comments": "[RIPPER MODE] Task-based approach (no ModelActors); scmlpick-style task queue",
            }
            append_trial_row(csv_path=test_csv_filepath, trial_data=trial_data)
            logger.info(f"Successfully saved trial data to CSV at {test_csv_filepath}")
            
        return "Successfully ran EQCCTPro (RIPPER MODE), exiting..."
    
    # =====================================================================
    # STANDARD MODE: Use ModelActor pool (new methodology)
    # =====================================================================
    
    # ===== TIMING: Start tracking actor creation time =====
    actor_creation_start_time = monotonic_s()
    
    # CREATE MODEL ACTOR(S) - Add this before the task loop
    logger.info(f"Creating model actor(s)...") 
    
    model_type_lower = model_type.lower() if model_type else 'eqcct'
    model_vram_mb = None  # Defensive init; will be set in GPU branches, remains None for CPU
    
    # Track requested vs actual actors for CSV Comments column
    requested_concurrent_tasks = number_of_concurrent_station_predictions if number_of_concurrent_station_predictions else 1
    actor_cap_comment = ""  # Will be populated if actors are capped due to memory constraints
    
    # ===== cuDNN CONCURRENT PREDICTION LIMIT =====
    # This tracks the maximum safe number of concurrent predictions based on GPU constraints.
    # Key insight: cuDNN workspace memory is allocated dynamically during inference, and having
    # too many concurrent predictions causes resource contention regardless of total actors.
    # 
    # safe_concurrent_predictions is set to safe_max_per_gpu (from a single GPU's perspective)
    # in GPU branches, ensuring that concurrent predictions don't overwhelm cuDNN resources.
    # For CPU mode, this remains at the user-requested value.
    safe_concurrent_predictions = number_of_concurrent_station_predictions if number_of_concurrent_station_predictions else 1
    
    if model_type_lower == 'seisbench':
        # --- Validate model name once on the driver (lightweight, no model loading) ---
        # Actors will skip this network call (validate_pretrained=False) to avoid
        # a thundering herd of concurrent list_pretrained() requests causing HTTP 500s.
        logger.info(f"Validating SeisBench model name: {seisbench_parent_model}/{seisbench_child_model}...")
        try:
            from eqcctpro.seisbench_models import SeisBenchModels
            SeisBenchModels(seisbench_parent_model, seisbench_child_model, validate_pretrained=True)
            logger.info("SeisBench model name validated successfully.")
        except Exception as e:
            logger.warning(f"SeisBench model validation failed: {e}. Proceeding anyway, actors will attempt to load from cache.")

        # Create SeisBench model actors
        if use_gpu:
            # Get VRAM requirement for this SeisBench model
            model_vram_mb = get_seisbench_model_vram_mb(
                seisbench_parent_model, 
                seisbench_child_model,
                default_mb=2000.0
            )
            # Use max of requested VRAM or model requirement (similar to EQCCT logic)
            # gpu_memory_limit_mb is per-actor VRAM limit, model_vram_mb is the minimum requirement
            per_actor_vram_mb = max(gpu_memory_limit_mb, model_vram_mb) if gpu_memory_limit_mb else model_vram_mb
            
            # ===== MEMORY-AWARE GPU ACTOR CREATION =====
            # KEY INSIGHT: With TF memory growth enabled, multiple actors CAN share GPU(s).
            # The constraint is: total requested VRAM must not exceed available VRAM.
            # taskset/CUDA_VISIBLE_DEVICES already limits hardware visibility, so let Ray handle scheduling.
            n_gpus = len(gpu_id)
            requested_actors = number_of_concurrent_station_predictions if number_of_concurrent_station_predictions else 1
            
            # Get available VRAM (total pool)
            # Use total_vram_pool_mb if provided (aggregate VRAM cap across all actors)
            # Otherwise fall back to actual free VRAM query
            available_vram_mb = get_available_vram_mb(
                gpu_ids=gpu_id, 
                max_vram_mb=total_vram_pool_mb,  # Use total pool, not per-actor limit
                logger=logger
            )
            
            # Calculate max actors based on VRAM (memory constraint, not hardware count)
            max_actors_by_vram = int(available_vram_mb / per_actor_vram_mb) if per_actor_vram_mb > 0 else requested_actors
            
            # ===== cuDNN STABILITY HEADROOM (PER-GPU ENFORCEMENT) =====
            # IMPORTANT: Having too many concurrent TensorFlow/PyTorch processes on a single GPU 
            # causes cuDNN resource contention, resulting in:
            #   - "DNN library is not found"
            #   - "Attempting to perform BLAS operation using StreamExecutor without BLAS support"
            #   - "cudaSetDevice() on GPU:0 failed. Status: out of memory"
            # 
            # Solution: Enforce a PER-GPU maximum, not just a global total.
            # cuDNN requires workspace memory beyond just model weights, and concurrent
            # operations compete for these resources.
            #
            # The formula scales dynamically with any model's VRAM requirements:
            #   safe_max_per_gpu = floor(per_gpu_vram / per_actor_vram * CUDNN_SAFETY_FACTOR)
            #
            # CUDNN_SAFETY_FACTOR: Controls how much GPU VRAM to reserve for cuDNN workspace.
            # cuDNN requires significant workspace memory for concurrent convolution operations.
            # The headroom provides efficient concurrent inference while minimizing OOM risk.
            # The headroom accounts for:
            #   1. cuDNN workspace memory that scales with concurrent operations
            #   2. CUDA context overhead for multiple processes
            #   3. Memory fragmentation during concurrent allocation/deallocation
            CUDNN_SAFETY_FACTOR = 1.0 - cudnn_headroom  # Use user-defined headroom
            
            vram_per_single_gpu = available_vram_mb / n_gpus
            theoretical_max_per_gpu = vram_per_single_gpu / per_actor_vram_mb if per_actor_vram_mb > 0 else float('inf')
            safe_max_per_gpu = max(1, int(theoretical_max_per_gpu * CUDNN_SAFETY_FACTOR))
            
            # Total max actors is per-GPU limit × number of GPUs
            max_actors_with_headroom = safe_max_per_gpu * n_gpus
            n_actors = min(requested_actors, max_actors_with_headroom)
            
            # Cap concurrent predictions to the total safe actors across all GPUs.
            # This ensures we don't create "idle" actors that eat VRAM while others predict.
            safe_concurrent_predictions = max_actors_with_headroom
            
            logger.info(f"===== MEMORY-AWARE GPU ACTOR POOL =====")
            logger.info(f"Requested concurrent tasks: {requested_actors}")
            logger.info(f"Available GPUs: {n_gpus}")
            logger.info(f"Total VRAM Pool: {available_vram_mb:.0f} MB")
            logger.info(f"VRAM per model: {per_actor_vram_mb:.0f} MB")
            logger.info(f"Per-GPU VRAM: {vram_per_single_gpu:.0f} MB")
            logger.info(f"Theoretical max per GPU: {theoretical_max_per_gpu:.1f} actors")
            logger.info(f"Safe max per GPU (with {cudnn_headroom*100:.0f}% cuDNN headroom): {safe_max_per_gpu} actors")
            logger.info(f"Max actors total: {max_actors_with_headroom} ({safe_max_per_gpu} per GPU × {n_gpus} GPUs)")
            logger.info(f"Creating {n_actors} SeisBenchModelActor(s)")
            if requested_actors > n_actors:
                logger.info(f"NOTE: Tasks will be queued and round-robin distributed to the {n_actors} actor(s).")
                logger.info(f"      Concurrency limited by VRAM with cuDNN headroom ({max_actors_with_headroom} max).")
            
            # Calculate fractional GPU allocation so Ray knows these are GPU actors
            # 
            # CRITICAL: Ray places each actor on a SINGLE GPU, not spread across GPUs.
            # When n_actors > n_gpus, multiple actors must share one GPU.
            # 
            # Example of the BUG we're fixing:
            #   - 3 actors, 2 GPUs, old calculation: 0.95 * 2 / 3 = 0.63 per actor
            #   - Ray places: Actor1→GPU0 (0.63), Actor2→GPU1 (0.63)
            #   - Actor3 needs 0.63, but GPU0 has 0.37 left, GPU1 has 0.37 left → DEADLOCK!
            # 
            # CORRECT approach: Calculate based on actors_per_gpu (ceiling).
            # If 3 actors on 2 GPUs, worst case is 2 actors on 1 GPU, so:
            #   - actors_per_gpu = ceil(3/2) = 2
            #   - fractional_gpu = 0.95 / 2 = 0.475 per actor
            #   - Now 2 actors can fit on 1 GPU: 2 × 0.475 = 0.95 ≤ 1.0 ✓
            # 
            GPU_HEADROOM_FACTOR = 0.95  # Leave 5% headroom per Ray best practices
            
            # Calculate MIN_FRACTIONAL_GPU dynamically based on model's actual VRAM requirement
            # This allows smaller models (e.g., PhaseNet ~500MB) to have more actors per GPU
            # compared to larger models (e.g., EQCCT ~1700MB) on the same hardware
            vram_per_single_gpu = available_vram_mb / n_gpus
            # Calculate the fraction of GPU VRAM this model actually needs
            # Floor at 0.01 (1%) for Ray scheduling stability, cap at 0.50 to ensure at least 2 actors can fit
            MIN_FRACTIONAL_GPU = max(0.01, min(0.475, per_actor_vram_mb / vram_per_single_gpu))
            
            # Calculate how many actors might need to share a single GPU (worst case)
            actors_per_gpu = math.ceil(n_actors / n_gpus)
            
            # Calculate max actors per GPU based on model-specific fractional constraint
            max_actors_per_single_gpu = int(GPU_HEADROOM_FACTOR / MIN_FRACTIONAL_GPU)
            
            # If actors_per_gpu exceeds what can fit on one GPU, cap n_actors
            if actors_per_gpu > max_actors_per_single_gpu:
                # Reduce n_actors so that actors_per_gpu fits
                n_actors = max_actors_per_single_gpu * n_gpus
                actors_per_gpu = max_actors_per_single_gpu
                logger.warning(f"Capping actors to {n_actors} total ({actors_per_gpu} per GPU) "
                             f"(model needs {MIN_FRACTIONAL_GPU:.1%} GPU, max {max_actors_per_single_gpu} actors/GPU with {GPU_HEADROOM_FACTOR:.0%} headroom)")
            
            # Calculate fractional GPU based on worst-case actors per GPU
            # This ensures all actors can be placed even if unevenly distributed
            fractional_gpu = GPU_HEADROOM_FACTOR / actors_per_gpu if actors_per_gpu > 0 else 1.0
            fractional_gpu = min(fractional_gpu, 1.0)  # Cap at 1.0
            fractional_gpu = math.floor(fractional_gpu * 100) / 100  # Truncate to 2 decimal places
            
            logger.info(f"Using fractional GPU allocation: {fractional_gpu:.3f} GPU per actor")
            logger.info(f"  → Model VRAM fraction: {MIN_FRACTIONAL_GPU:.1%} of GPU ({per_actor_vram_mb:.0f} MB / {vram_per_single_gpu:.0f} MB)")
            logger.info(f"  → Actors per GPU (worst case): {actors_per_gpu}")
            logger.info(f"  → Max per-GPU usage: {actors_per_gpu} × {fractional_gpu:.3f} = {actors_per_gpu * fractional_gpu:.3f} / 1.0 GPU")
            logger.info(f"  → Headroom per GPU: {(1 - actors_per_gpu * fractional_gpu) * 100:.1f}% reserved for Ray/CUDA overhead")
            
            # Create all actors in parallel (non-blocking .remote() calls)
            logger.info(f"Creating {n_actors} SeisBenchModelActor(s) in parallel ({per_actor_vram_mb/1024:.2f}GB VRAM each)...")
            model_actors = [
                SeisBenchModelActor.options(num_gpus=fractional_gpu, num_cpus=0).remote(
                    parent_model_name=seisbench_parent_model,
                    child_model_name=seisbench_child_model,
                    gpus_to_use=gpu_id,  # Pass all available GPUs, actor will use what Ray assigns
                    use_gpu=True
                ) for _ in range(n_actors)
            ]

            # Wait for all actors to initialize in parallel
            logger.info(f"Waiting for {n_actors} actor(s) to initialize (loading models concurrently)...")
            try:
                ray.get([actor.ready.remote() for actor in model_actors])
            except Exception as e:
                logger.error(f"Failed to initialize SeisBenchModelActors: {e}")
                raise
            logger.info(f"All {n_actors} GPU actor(s) created successfully. Task queue will handle concurrency.")
            
            # Generate comment if actors were capped
            if len(model_actors) < requested_actors:
                actor_cap_comment = f"Requested {requested_actors} actors, created {len(model_actors)} (VRAM pool: {available_vram_mb:.0f} MB, {per_actor_vram_mb:.0f} MB/actor)"
        else:
            # ===== MEMORY-AWARE CPU ACTOR CREATION =====
            # KEY INSIGHT: The constraint is RAM, not CPU count.
            # taskset already limits CPU visibility, so let Ray handle scheduling.
            n_cpus = len(ray_cpus) if ray_cpus else 1
            requested_actors = number_of_concurrent_station_predictions if number_of_concurrent_station_predictions else 1
            
            # Get RAM requirement for this model
            model_ram_mb = get_seisbench_model_ram_mb(
                seisbench_parent_model,
                seisbench_child_model,
                use_gpu=False,
                default_mb=600.0
            )
            
            # Get available RAM based on system capacity
            # Note: ram_safety_cap would need to be passed here, using 0.90 as default
            available_ram_mb = get_available_ram_mb(ram_safety_cap=0.90, logger=logger)
            
            # Calculate max actors based on RAM (memory constraint)
            max_actors_by_ram = int(available_ram_mb / model_ram_mb) if model_ram_mb > 0 else requested_actors
            n_actors = min(requested_actors, max(1, max_actors_by_ram))
            
            logger.info(f"===== MEMORY-AWARE CPU ACTOR POOL =====")
            logger.info(f"Requested concurrent tasks: {requested_actors}")
            logger.info(f"Available CPUs: {n_cpus}")
            logger.info(f"Available RAM: {available_ram_mb:.0f} MB")
            logger.info(f"RAM per model: {model_ram_mb:.0f} MB")
            logger.info(f"Max actors by RAM: {max_actors_by_ram}")
            logger.info(f"Creating {n_actors} SeisBenchModelActor(s)")
            if requested_actors > n_actors:
                logger.info(f"NOTE: Tasks will be queued and round-robin distributed to the {n_actors} actor(s).")
                logger.info(f"      Concurrency limited by RAM, not CPU count.")
            logger.info(f"Let Ray handle scheduling (taskset already restricts CPU visibility)")
            
            # Create all actors in parallel (non-blocking .remote() calls)
            logger.info(f"Creating {n_actors} SeisBenchModelActor(s) in parallel ({model_ram_mb/1024:.2f}GB RAM each)...")
            model_actors = [
                SeisBenchModelActor.remote(
                    parent_model_name=seisbench_parent_model,
                    child_model_name=seisbench_child_model,
                    gpus_to_use=False,
                    use_gpu=False
                ) for _ in range(n_actors)
            ]

            # Wait for all actors to initialize in parallel
            logger.info(f"Waiting for {n_actors} actor(s) to initialize (loading models concurrently)...")
            try:
                ray.get([actor.ready.remote() for actor in model_actors])
            except Exception as e:
                logger.error(f"Failed to initialize SeisBenchModelActors: {e}")
                raise
            logger.info(f"All {n_actors} CPU actor(s) created successfully. Task queue will handle concurrency.")
            
            # Generate comment if actors were capped
            if len(model_actors) < requested_actors:
                actor_cap_comment = f"Requested {requested_actors} actors, created {len(model_actors)} (RAM limited to {available_ram_mb:.0f} MB, {model_ram_mb:.0f} MB/actor)"
    else:
        # Create EQCCT model actors
        if use_gpu:
            # ===== MEMORY-AWARE GPU ACTOR CREATION (EQCCT/TensorFlow) =====
            # KEY INSIGHT: With TF memory growth enabled, multiple actors CAN share GPU(s).
            # The constraint is: total requested VRAM must not exceed available VRAM.
            # taskset/CUDA_VISIBLE_DEVICES already limits hardware visibility, so let Ray handle scheduling.
            model_vram_mb = get_eqcct_vram_mb()  # Use measured EQCCT VRAM requirement
            # gpu_memory_limit_mb is per-actor VRAM limit for TF config, model_vram_mb is the minimum requirement
            per_actor_vram_mb = max(gpu_memory_limit_mb, model_vram_mb) if gpu_memory_limit_mb else model_vram_mb
            
            n_gpus = len(gpu_id)
            requested_actors = number_of_concurrent_station_predictions if number_of_concurrent_station_predictions else 1
            
            # Get available VRAM (total pool)
            # Use total_vram_pool_mb if provided (aggregate VRAM cap across all actors)
            # Otherwise fall back to actual free VRAM query
            available_vram_mb = get_available_vram_mb(
                gpu_ids=gpu_id, 
                max_vram_mb=total_vram_pool_mb,  # Use total pool, not per-actor limit
                logger=logger
            )
            
            # Calculate max actors based on VRAM (memory constraint, not hardware count)
            max_actors_by_vram = int(available_vram_mb / per_actor_vram_mb) if per_actor_vram_mb > 0 else requested_actors
            
            # ===== cuDNN STABILITY HEADROOM (PER-GPU ENFORCEMENT) =====
            # IMPORTANT: Having too many concurrent TensorFlow processes on a single GPU 
            # causes cuDNN resource contention, resulting in:
            #   - "DNN library is not found"
            #   - "Attempting to perform BLAS operation using StreamExecutor without BLAS support"
            #   - "cudaSetDevice() on GPU:0 failed. Status: out of memory"
            # 
            # Solution: Enforce a PER-GPU maximum, not just a global total.
            # cuDNN requires workspace memory beyond just model weights, and concurrent
            # operations compete for these resources.
            #
            # The formula scales dynamically with any model's VRAM requirements:
            #   safe_max_per_gpu = floor(per_gpu_vram / per_actor_vram * CUDNN_SAFETY_FACTOR)
            #
            # Example: EQCCT with 46550 MB/GPU, 2756 MB/actor:
            #   theoretical = 46550 / 2756 = 16.9 actors/GPU
            #   safe_max = floor(16.9 * 0.90) = 15 actors/GPU (with 10% headroom)
            #
            # CUDNN_SAFETY_FACTOR: Controls how much GPU VRAM to reserve for cuDNN workspace.
            # cuDNN requires significant workspace memory for concurrent convolution operations.
            # Empirically tested values:
            #   - 0.90 (10% headroom): Efficient concurrent inference
            #   - 0.88 (12% headroom): Previously tested for stability
            #   - 0.75 (25% headroom): Extremely conservative
            # The headroom accounts for:
            #   1. cuDNN workspace memory that scales with concurrent operations
            #   2. CUDA context overhead for multiple processes
            #   3. Memory fragmentation during concurrent allocation/deallocation
            CUDNN_SAFETY_FACTOR = 1.0 - cudnn_headroom  # Use user-defined headroom
            
            vram_per_single_gpu = available_vram_mb / n_gpus
            theoretical_max_per_gpu = vram_per_single_gpu / per_actor_vram_mb if per_actor_vram_mb > 0 else float('inf')
            safe_max_per_gpu = max(1, int(theoretical_max_per_gpu * CUDNN_SAFETY_FACTOR))
            
            # Total max actors is per-GPU limit × number of GPUs
            max_actors_with_headroom = safe_max_per_gpu * n_gpus
            n_actors = min(requested_actors, max_actors_with_headroom)
            
            # Cap concurrent predictions to the total safe actors across all GPUs.
            # This ensures we don't create "idle" actors that eat VRAM while others predict.
            safe_concurrent_predictions = max_actors_with_headroom
            
            logger.info(f"===== MEMORY-AWARE GPU ACTOR POOL (EQCCT) =====")
            logger.info(f"Requested concurrent tasks: {requested_actors}")
            logger.info(f"Available GPUs: {n_gpus}")
            logger.info(f"Total VRAM Pool: {available_vram_mb:.0f} MB")
            logger.info(f"VRAM per model: {per_actor_vram_mb:.0f} MB")
            logger.info(f"Per-GPU VRAM: {vram_per_single_gpu:.0f} MB")
            logger.info(f"Theoretical max per GPU: {theoretical_max_per_gpu:.1f} actors")
            logger.info(f"Safe max per GPU (with {cudnn_headroom*100:.0f}% cuDNN headroom): {safe_max_per_gpu} actors")
            logger.info(f"Max actors total: {max_actors_with_headroom} ({safe_max_per_gpu} per GPU × {n_gpus} GPUs)")
            logger.info(f"Creating {n_actors} ModelActor(s)")
            if requested_actors > n_actors:
                logger.info(f"NOTE: Tasks will be queued and round-robin distributed to the {n_actors} actor(s).")
                logger.info(f"      Concurrency limited by VRAM with cuDNN headroom ({max_actors_with_headroom} max).")
            
            # Calculate fractional GPU allocation so Ray knows these are GPU actors
            # 
            # CRITICAL: Ray places each actor on a SINGLE GPU, not spread across GPUs.
            # When n_actors > n_gpus, multiple actors must share one GPU.
            # 
            # Example of the BUG we're fixing:
            #   - 3 actors, 2 GPUs, old calculation: 0.95 * 2 / 3 = 0.63 per actor
            #   - Ray places: Actor1→GPU0 (0.63), Actor2→GPU1 (0.63)
            #   - Actor3 needs 0.63, but GPU0 has 0.37 left, GPU1 has 0.37 left → DEADLOCK!
            # 
            # CORRECT approach: Calculate based on actors_per_gpu (ceiling).
            # If 3 actors on 2 GPUs, worst case is 2 actors on 1 GPU, so:
            #   - actors_per_gpu = ceil(3/2) = 2
            #   - fractional_gpu = 0.95 / 2 = 0.475 per actor
            #   - Now 2 actors can fit on 1 GPU: 2 × 0.475 = 0.95 ≤ 1.0 ✓
            # 
            GPU_HEADROOM_FACTOR = 0.95  # Leave 5% headroom per Ray best practices
            
            # Calculate MIN_FRACTIONAL_GPU dynamically based on model's actual VRAM requirement
            # This allows smaller models (e.g., PhaseNet ~500MB) to have more actors per GPU
            # compared to larger models (e.g., EQCCT ~1700MB) on the same hardware
            vram_per_single_gpu = available_vram_mb / n_gpus
            # Calculate the fraction of GPU VRAM this model actually needs
            # Floor at 0.01 (1%) for Ray scheduling stability, cap at 0.50 to ensure at least 2 actors can fit
            MIN_FRACTIONAL_GPU = max(0.01, min(0.475, per_actor_vram_mb / vram_per_single_gpu))
            
            # Calculate how many actors might need to share a single GPU (worst case)
            actors_per_gpu = math.ceil(n_actors / n_gpus)
            
            # Calculate max actors per GPU based on model-specific fractional constraint
            max_actors_per_single_gpu = int(GPU_HEADROOM_FACTOR / MIN_FRACTIONAL_GPU)
            
            # If actors_per_gpu exceeds what can fit on one GPU, cap n_actors
            if actors_per_gpu > max_actors_per_single_gpu:
                # Reduce n_actors so that actors_per_gpu fits
                n_actors = max_actors_per_single_gpu * n_gpus
                actors_per_gpu = max_actors_per_single_gpu
                logger.warning(f"Capping actors to {n_actors} total ({actors_per_gpu} per GPU) "
                             f"(model needs {MIN_FRACTIONAL_GPU:.1%} GPU, max {max_actors_per_single_gpu} actors/GPU with {GPU_HEADROOM_FACTOR:.0%} headroom)")
            
            # Calculate fractional GPU based on worst-case actors per GPU
            # This ensures all actors can be placed even if unevenly distributed
            fractional_gpu = GPU_HEADROOM_FACTOR / actors_per_gpu if actors_per_gpu > 0 else 1.0
            fractional_gpu = min(fractional_gpu, 1.0)  # Cap at 1.0
            fractional_gpu = math.floor(fractional_gpu * 100) / 100  # Truncate to 2 decimal places
            
            logger.info(f"Using fractional GPU allocation: {fractional_gpu:.3f} GPU per actor")
            logger.info(f"  → Model VRAM fraction: {MIN_FRACTIONAL_GPU:.1%} of GPU ({per_actor_vram_mb:.0f} MB / {vram_per_single_gpu:.0f} MB)")
            logger.info(f"  → Actors per GPU (worst case): {actors_per_gpu}")
            logger.info(f"  → Max per-GPU usage: {actors_per_gpu} × {fractional_gpu:.3f} = {actors_per_gpu * fractional_gpu:.3f} / 1.0 GPU")
            logger.info(f"  → Headroom per GPU: {(1 - actors_per_gpu * fractional_gpu) * 100:.1f}% reserved for Ray/CUDA overhead")
            
            # Create all actors in parallel (non-blocking .remote() calls)
            logger.info(f"Creating {n_actors} ModelActor(s) in parallel ({per_actor_vram_mb/1024:.2f}GB VRAM each)...")
            model_actors = [
                ModelActor.options(num_gpus=fractional_gpu, num_cpus=0).remote(
                    gpus_to_use=gpu_id,  # Pass all available GPUs, actor will use what Ray assigns
                    p_model_path=p_model, 
                    s_model_path=s_model, 
                    gpu_memory_limit_mb=per_actor_vram_mb,  # Per-actor VRAM limit via TF config
                    use_gpu=True
                ) for _ in range(n_actors)
            ]

            # Wait for all actors to initialize in parallel
            logger.info(f"Waiting for {n_actors} actor(s) to initialize (loading models concurrently)...")
            try:
                ray.get([actor.ready.remote() for actor in model_actors])
            except Exception as e:
                logger.error(f"Failed to initialize ModelActors: {e}")
                raise
            logger.info(f"All {n_actors} GPU actor(s) created successfully. Task queue will handle concurrency.")
            logger.info(f"[ModelActor] Models successfully loaded onto GPU(s).")
            
            # Generate comment if actors were capped
            if len(model_actors) < requested_actors:
                actor_cap_comment = f"Requested {requested_actors} actors, created {len(model_actors)} (VRAM pool: {available_vram_mb:.0f} MB, {per_actor_vram_mb:.0f} MB/actor)"
        else:
            # ===== MEMORY-AWARE CPU ACTOR CREATION (EQCCT/TensorFlow) =====
            # KEY INSIGHT: The constraint is RAM, not CPU count.
            # taskset already limits CPU visibility, so let Ray handle scheduling.
            n_cpus = len(ray_cpus) if ray_cpus else 1
            requested_actors = number_of_concurrent_station_predictions if number_of_concurrent_station_predictions else 1
            
            # Get RAM requirement for EQCCT in CPU mode
            model_ram_mb = get_eqcct_ram_mb(use_gpu=False)
            
            # Get available RAM based on system capacity
            # Note: ram_safety_cap would need to be passed here, using 0.90 as default
            available_ram_mb = get_available_ram_mb(ram_safety_cap=0.90, logger=logger)
            
            # Calculate max actors based on RAM (memory constraint)
            max_actors_by_ram = int(available_ram_mb / model_ram_mb) if model_ram_mb > 0 else requested_actors
            n_actors = min(requested_actors, max(1, max_actors_by_ram))
            
            logger.info(f"===== MEMORY-AWARE CPU ACTOR POOL (EQCCT) =====")
            logger.info(f"Requested concurrent tasks: {requested_actors}")
            logger.info(f"Available CPUs: {n_cpus}")
            logger.info(f"Available RAM: {available_ram_mb:.0f} MB")
            logger.info(f"RAM per model: {model_ram_mb:.0f} MB")
            logger.info(f"Max actors by RAM: {max_actors_by_ram}")
            logger.info(f"Creating {n_actors} ModelActor(s)")
            if requested_actors > n_actors:
                logger.info(f"NOTE: Tasks will be queued and round-robin distributed to the {n_actors} actor(s).")
                logger.info(f"      Concurrency limited by RAM, not CPU count.")
            logger.info(f"Let Ray handle scheduling (taskset already restricts CPU visibility)")
            
            # Create all actors in parallel (non-blocking .remote() calls)
            logger.info(f"Creating {n_actors} ModelActor(s) in parallel ({model_ram_mb/1024:.2f}GB RAM each)...")
            model_actors = [
                ModelActor.remote(
                    p_model_path=p_model, 
                    s_model_path=s_model, 
                    gpu_memory_limit_mb=None, 
                    use_gpu=False
                ) for _ in range(n_actors)
            ]

            # Wait for all actors to initialize in parallel
            logger.info(f"Waiting for {n_actors} actor(s) to initialize (loading models concurrently)...")
            try:
                ray.get([actor.ready.remote() for actor in model_actors])
            except Exception as e:
                logger.error(f"Failed to initialize ModelActors: {e}")
                raise
            logger.info(f"All {n_actors} CPU actor(s) created successfully. Task queue will handle concurrency.")
            
            # Generate comment if actors were capped
            if len(model_actors) < requested_actors:
                actor_cap_comment = f"Requested {requested_actors} actors, created {len(model_actors)} (RAM limited to {available_ram_mb:.0f} MB, {model_ram_mb:.0f} MB/actor)"

    # ===== TIMING: End of actor creation =====
    actor_creation_end_time = monotonic_s()
    actor_creation_time_seconds = actor_creation_end_time - actor_creation_start_time
    logger.info(f"Actor creation completed in {actor_creation_time_seconds:.2f} seconds")

    # Submit tasks to ray in a queue
    tasks_queue = []
    
    # Cap max_pending_tasks to safe_concurrent_predictions to prevent cuDNN resource contention
    # This ensures that even with multiple actors/GPUs, we don't overwhelm cuDNN workspace allocation
    if number_of_concurrent_station_predictions > safe_concurrent_predictions:
        logger.info(f"cuDNN PREDICTION LIMIT: Capping concurrent predictions from {number_of_concurrent_station_predictions} to {safe_concurrent_predictions}")
        logger.info(f"  → Actors created: {len(model_actors)} across {len(gpu_id) if use_gpu else 0} GPU(s)")
        logger.info(f"  → Max concurrent predictions: {safe_concurrent_predictions} (total safe system limit)")
        logger.info(f"  → Tasks will queue and execute as actors become available")
        max_pending_tasks = safe_concurrent_predictions
    else:
        max_pending_tasks = number_of_concurrent_station_predictions
    
    logger.info(f"Starting EQCCTPro parallelized waveform processing...") 
    logger.info("")
    start_time = monotonic_s() 
    model_type_lower = model_type.lower() if model_type else 'eqcct'
    if model_type_lower == 'seisbench':
        logger.info(f"------- Analyzing Seismic Waveforms for P and S Picks via SeisBench ({seisbench_parent_model} - {seisbench_child_model}) -------")
    else:
        logger.info(f"------- Analyzing Seismic Waveforms for P and S Picks via EQCCT -------")

    if timechunk_id is None:
        # derive from the path if caller forgot to pass it
        cand = os.path.basename(input_dir)
        if "_" in cand and len(cand) >= 10:
            timechunk_id = cand
        else:
            raise ValueError("timechunk_id is None and could not be inferred from input_dir; "
                            "expected a dir named like YYYYMMDDThhmmssZ_YYYYMMDDThhmmssZ")
    starttime, endtime, time_delta = parse_time_range(timechunk_id)

    logger.info(f"Analyzing {time_delta} minute timechunk from {starttime} to {endtime} ({waveform_overlap} min overlap)")
    logger.info(f"Processing a total of {len(tasks_predictor)} stations, {max_pending_tasks} at a time.") 

    # scmlpick-style shared refs: one copy of waveforms + args in the object store (same as RIPPER).
    station_codes_for_stream = [tasks_predictor[i][1] for i in range(len(tasks_predictor))]
    merged_stream_ma = _load_ripper_mseed_stream(args["input_dir"], station_codes_for_stream)
    if len(merged_stream_ma) == 0:
        logger.warning(
            "ModelActor mode: no traces preloaded; tasks will read mSEED from disk per station."
        )
        tasks_predictor_ma = tasks_predictor
    else:
        logger.info(
            f"ModelActor mode: ray.put shared Stream ({len(merged_stream_ma)} trace(s)) and args for "
            f"{len(tasks_predictor)} station task(s)."
        )
        args_ref_ma = ray.put(args)
        stream_ref_ma = ray.put(merged_stream_ma)
        tasks_predictor_ma = [
            [tasks_predictor[i][0], tasks_predictor[i][1], tasks_predictor[i][2], args_ref_ma, stream_ref_ma]
            for i in range(len(tasks_predictor))
        ]

    # ===== TIMING: Collect waveform load times for averaging =====
    waveform_load_times = []

    # Concurrent Prediction(s) Parallel Processing
    try: 
        for i in range(len(tasks_predictor_ma)):
            while True:
                # Add new task to queue while max is not reached
                if len(tasks_queue) < max_pending_tasks:
                    # SELECT WHICH MODEL ACTOR TO USE (round-robin across GPUs)
                    model_actor = model_actors[i % len(model_actors)]

                    # Route to appropriate prediction function based on model type
                    if model_type_lower == 'seisbench':
                        # SeisBench models use parallel_predict_seisbench
                        if use_gpu is False:
                            tasks_queue.append(parallel_predict_seisbench.options(num_cpus=0).remote(tasks_predictor_ma[i], model_actor, False))
                        elif use_gpu is True:
                            # Don't allocate GPUs to workers, only to model actors
                            # Use num_cpus=0 to avoid deadlocks when Ray has limited CPUs
                            tasks_queue.append(parallel_predict_seisbench.options(num_cpus=0, num_gpus=0).remote(tasks_predictor_ma[i], model_actor, True))
                    else:
                        # EQCCT models use parallel_predict (original)
                        if use_gpu is False:
                            tasks_queue.append(parallel_predict.options(num_cpus=0).remote(tasks_predictor_ma[i], model_actor, False))
                        elif use_gpu is True:
                            # Don't allocate GPUs to workers, only to model actors
                            # Use num_cpus=0 to avoid deadlocks when Ray has limited CPUs
                            tasks_queue.append(parallel_predict.options(num_cpus=0, num_gpus=0).remote(tasks_predictor_ma[i], model_actor, True))
                    break
                # If there are more tasks than maximum, just process them
                else:
                    tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
                    for finished_task in tasks_finished:
                        result = ray.get(finished_task)
                        log_entry, load_time = result  # Unpack tuple: (log_message, waveform_load_time)
                        logger.info(f'{log_entry}')
                        if load_time is not None:
                            waveform_load_times.append(load_time)

        # After adding all the tasks to queue, process what's left
        while tasks_queue:
            tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
            for finished_task in tasks_finished:
                result = ray.get(finished_task)
                log_entry, load_time = result  # Unpack tuple: (log_message, waveform_load_time)
                logger.info(f'{log_entry}')
                if load_time is not None:
                    waveform_load_times.append(load_time)
        logger.info("")
        
        # Calculate average waveform load time
        if waveform_load_times:
            avg_waveform_load_time = sum(waveform_load_times) / len(waveform_load_times)
            logger.info(f"Average waveform load time (per task): {avg_waveform_load_time:.3f}s (across {len(waveform_load_times)} tasks)")
        else:
            avg_waveform_load_time = 0.0

    except Exception as e:
        avg_waveform_load_time = 0.0  # Default if error occurs before collecting any times
        # Catch any error in the parallel processing
        logger.error(f"ERROR in parallel processing at {datetime.now()}")
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise  # Re-raise to see the error

    logger.info(f"------- Parallel Station Waveform Processing Complete For {starttime} to {endtime} Timechunk-------")
    end_time = monotonic_s()
    logger.info(f"Picks saved at {output_dir}Process Runtime: {end_time - start_time:.2f} s")

    if testing_gpu is not None: 
        # Guard: make sure CPUs is an int, not a list
        num_ray_cpus = len(ray_cpus) if isinstance(ray_cpus, (list, tuple)) else int(len(list(ray_cpus)))

        # Parse the timechunk_id to get start/end times
        if timechunk_id:
            starttime, endtime, time_delta = parse_time_range(timechunk_id)
            timechunk_length_min = time_delta.total_seconds() / 60.0 if time_delta else None
        else:
            timechunk_length_min = None

        # Determine model name for logging
        if model_type_lower == 'seisbench':
            model_used = f"{seisbench_parent_model}/{seisbench_child_model}"
        else:
            model_used = "eqcct"

        # N ModelActors = actual actors created (capped to hardware limits)
        # This may be less than number_of_concurrent_station_predictions due to optimal capping
        actual_actors = len(model_actors) if model_actors else 1
        
        # Calculate timing metrics
        total_trial_time_seconds = end_time - trial_start_time
        waveform_processing_time_seconds = end_time - start_time
        
        trial_data = {
            "Trial Number": None,  # Will be auto-filled by append_trial_row
            "Stations Used": str(station_list),
            "Number of Stations Used": len(station_list),
            "Number of CPUs Allocated for Ray to Use": num_ray_cpus,
            "Intra-parallelism Threads": intra_threads if intra_threads is not None else "",
            "Inter-parallelism Threads": inter_threads if inter_threads is not None else "",
            "GPUs Used": json.dumps(list(gpu_id)) if (use_gpu and gpu_id is not None) else "[]",
            "Inference Actor Memory Limit (MB)": float(model_vram_mb) if (use_gpu and gpu_memory_limit_mb is not None) else "",
            "Total Waveform Analysis Timespace (min)": float(total_analysis_time.total_seconds() / 60.0) if hasattr(total_analysis_time, "total_seconds") else (float(total_analysis_time) if total_analysis_time else ""),
            "Total Number of Timechunks": int(total_timechunks) if total_timechunks is not None else "",
            "Concurrent Timechunks Used": int(number_of_concurrent_timechunk_predictions) if number_of_concurrent_timechunk_predictions is not None else "",
            "Length of Timechunk (min)": timechunk_length_min if timechunk_length_min is not None else "",
            "N ModelActors": actual_actors,  # Actual actors created (capped to hardware/memory)
            "Number of Concurrent Station Tasks": int(number_of_concurrent_station_predictions) if number_of_concurrent_station_predictions is not None else "",
            # ===== TIMING METRICS =====
            "Total Trial Time (s)": round(total_trial_time_seconds, 6),  # Entire trial: setup + actor creation + processing
            "Actor Creation Time (s)": round(actor_creation_time_seconds, 6),  # Time to spin up ModelActors
            "Avg Model Load Time (s)": "",  # N/A for ModelActor mode (models loaded once in actor, not per-task)
            "Waveform Processing Time (s)": round(avg_waveform_load_time, 6),  # Average time to load waveforms into memory per task
            "Total Run time for Picker (s)": round(waveform_processing_time_seconds, 6),  # Total time for all task processing
            "Model Used": model_used,
            "Trial Success": "",
            "Error Message": str(""),
            "Comments": actor_cap_comment,  # Note when actors were capped due to memory constraints
        }
            
        append_trial_row(csv_path=test_csv_filepath, trial_data=trial_data)
        logger.info(f"Successfully saved trial data to CSV at {test_csv_filepath}")
        
    return "Successfully ran EQCCTPro, exiting..."


@ray.remote
class ModelActor:
    def __init__(self,  p_model_path, s_model_path, gpus_to_use=False, intra_threads=1, inter_threads=1, gpu_memory_limit_mb=None, use_gpu=True):
        self.logger = logging.getLogger("eqcctpro.model_actor")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers[:] = []
        self.logger.propagate = False
        self.logger.addHandler(logging.StreamHandler())

        self.logger.info("=== ModelActor __init__ STARTED ===")
        self.logger.info(f"p_model_path = {p_model_path}")
        self.logger.info(f"s_model_path = {s_model_path}")
        self.logger.info(f"Exists? P: {os.path.exists(p_model_path)}, S: {os.path.exists(s_model_path)}")

        if use_gpu:
            # Configure GPU memory for this actor
            # We want one GPU per actor 
            try:
                self.logger.info("Calling tf_environ...")
                tf_environ(
                    gpu_id=gpus_to_use[0] if gpus_to_use else 0, 
                    gpus_to_use=None, # First visible GPU only
                    vram_limit_mb=gpu_memory_limit_mb,
                    intra_threads=intra_threads,
                    inter_threads=inter_threads,
                    log_device=True,
                    logger=self.logger)
                self.logger.info("tf_environ finished.")
            except (RuntimeError, ValueError) as e:
                self.logger.error(f"[ModelActor] Error setting memory limit: {e}")
        
        # Load the model once
        self.logger.info("Importing/load_eqcct_model...")
        from eqcctpro.eqcct_tf_models import load_eqcct_model
        self.model = load_eqcct_model(p_model_path, s_model_path)
        self.logger.info("Model loaded.")
    
    def ready(self):
        """Simple method to check if the actor is ready"""
        return True
    
    def predict(self, data_generator):
        """Perform prediction using the loaded model"""
        return self.model.predict(data_generator, verbose=0)
    
    def predict_from_arrays(self, trace_start_time, data_set, batch_size, norm_mode):
        from eqcctpro.eqcct_tf_models import PreLoadGeneratorTest
        pred_generator = PreLoadGeneratorTest(trace_start_time, data_set,
                                            batch_size=batch_size, norm_mode=norm_mode)
        return self.model.predict(pred_generator, verbose=0)


@ray.remote
class SeisBenchModelActor:
    """
    Ray actor for SeisBench models that loads the model once and shares it across predictions.
    Similar to ModelActor but for SeisBench models (PyTorch-based).
    """
    def __init__(self, parent_model_name, child_model_name, gpus_to_use=False, use_gpu=True):
        self.logger = logging.getLogger("eqcctpro.seisbench_model_actor")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers[:] = []
        self.logger.propagate = False
        self.logger.addHandler(logging.StreamHandler())

        self.logger.info("=== SeisBenchModelActor __init__ STARTED ===")
        self.logger.info(f"parent_model_name = {parent_model_name}")
        self.logger.info(f"child_model_name = {child_model_name}")
        self.use_gpu = use_gpu
        self.gpus_to_use = gpus_to_use

        # Set device for PyTorch (SeisBench uses PyTorch)
        try:
            import torch
        except ImportError:
            self.logger.error("PyTorch (torch) is not installed. SeisBench models require PyTorch.")
            raise ImportError("PyTorch (torch) is not installed. Please install it to use SeisBench models.")

        if use_gpu:
            # When using Ray with num_gpus=1, the assigned GPU is always visible as cuda:0
            # regardless of its physical ID (0, 1, etc.) because Ray sets CUDA_VISIBLE_DEVICES.
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Using device: {self.device} (mapped by Ray from physical {gpus_to_use})")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU device")

        # Load the SeisBench model (skip validation — driver already verified the model name)
        self.logger.info("Loading SeisBench model...")
        from eqcctpro.seisbench_models import SeisBenchModels
        self.model_wrapper = SeisBenchModels(parent_model_name, child_model_name, validate_pretrained=False)
        self.model_wrapper.load_model()
        
        # Move model to device if using GPU
        if use_gpu:
            try:
                if hasattr(self.model_wrapper.model, 'to'):
                    self.model_wrapper.model.to(self.device)
                cuda_synchronize_best_effort()
                self.logger.info(f"Model moved to {self.device}")
            except Exception as e:
                self.logger.warning(f"Could not move model to GPU: {e}")
        
        self.logger.info("SeisBench model loaded successfully.")
    
    def ready(self):
        """Simple method to check if the actor is ready"""
        return True
    
    def classify(self, stream, P_threshold=0.3, S_threshold=0.3, Detection_threshold=0.3, **kwargs):
        """
        Classify a stream and return picks.
        
        Parameters:
        -----------
        stream : obspy.Stream
            3-component ObsPy Stream
        P_threshold : float
            P phase detection threshold
        S_threshold : float
            S phase detection threshold
        Detection_threshold : float
            Detection threshold
        **kwargs : dict
            Additional arguments for model.classify()
        
        Returns:
        --------
        ClassifyOutput
            Object containing picks
        """
        out = self.model_wrapper.classify(
            stream,
            P_threshold=P_threshold,
            S_threshold=S_threshold,
            Detection_threshold=Detection_threshold,
            **kwargs,
        )
        if self.use_gpu:
            cuda_synchronize_best_effort()
        return out


@ray.remote
def parallel_predict_seisbench(predict_args, model_actor, gpu=False):
    """
    Prediction function for SeisBench models.
    Uses mseed2stream_3c for preprocessing and SeisBenchModelActor for predictions.
    ``predict_args`` may be ``(pos, station, out_dir, args)`` or 5-tuple with
    ``args_ref`` / ``stream_ref`` (Ray ObjectRefs, scmlpick-style).
    """
    import glob
    import shutil
    import csv
    import logging
    from logging.handlers import QueueHandler
    from eqcctpro.seisbench_models import mseed2stream_3c, process_raw_station_stream_3c

    if len(predict_args) == 5:
        pos, station, out_dir, args_ref, stream_ref = predict_args
        args = ray.get(args_ref) if isinstance(args_ref, ray.ObjectRef) else args_ref
        use_shared_stream = True
    else:
        pos, station, out_dir, args = predict_args
        use_shared_stream = False

    # Set up logger to forward to the main listener
    logger = logging.getLogger(f"eqcctpro.worker.{station}")
    logger.setLevel(logging.INFO)
    if args.get('log_queue') is not None:
        logger.addHandler(QueueHandler(args['log_queue']))
    
    save_dir = os.path.join(out_dir, str(station)+'_outputs')
    csv_filename = os.path.join(save_dir,'X_prediction_results.csv')

    if os.path.isfile(csv_filename):
        if args['overwrite']:
            shutil.rmtree(save_dir)
        else:
            return (f"{pos} {station}: Skipped (already exists - overwrite=False).", 0.0)

    os.makedirs(save_dir, exist_ok=True)
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
    
    start_Predicting = monotonic_s()

    # ===== TIMING: Track waveform loading time =====
    waveform_load_start = monotonic_s()
    try:
        if use_shared_stream:
            full_st = ray.get(stream_ref) if isinstance(stream_ref, ray.ObjectRef) else stream_ref
            st_sel = _stream_select_for_station_task(full_st, station)
            if len(st_sel) == 0:
                csvPr_gen.close()
                return (f"{pos} {station}: FAILED - No traces for station in shared Stream.", None)
            stream3c, freqmin, freqmax = process_raw_station_stream_3c(args, st_sel, station)
        else:
            files_list = glob.glob(f"{args['input_dir']}/{station}/*mseed")
            if not files_list:
                csvPr_gen.close()
                return (f"{pos} {station}: FAILED - No mSEED files found.", None)
            stream3c, freqmin, freqmax = mseed2stream_3c(args, files_list, station)
    except Exception as e:
        csvPr_gen.close()
        err_msg = f"FAILED reading mSEED: {str(e)}" if str(e) else "FAILED reading mSEED (unknown error)."
        return (f"{pos} {station}: {err_msg}", None)
    waveform_load_time = monotonic_s() - waveform_load_start

    try:
        # Get picks from SeisBench model
        # Use ray.get with a timeout or just normally if we fixed the CPU deadlock
        classify_output = ray.get(model_actor.classify.remote(
            stream3c,
            P_threshold=args.get('P_threshold', 0.3),
            S_threshold=args.get('S_threshold', 0.3),
            Detection_threshold=args.get('Detection_threshold', 0.3),
            strict=False,
            flexible_horizontal_components=True
        ))
        
        # Extract metadata from stream
        station_code = stream3c[0].stats.station if len(stream3c) > 0 else station
        network_code = stream3c[0].stats.network if len(stream3c) > 0 else ""
        # Try to get coordinates from stream metadata if available
        station_lat = getattr(stream3c[0].stats, 'coordinates', {}).get('latitude', 0.0) if len(stream3c) > 0 else 0.0
        station_lon = getattr(stream3c[0].stats, 'coordinates', {}).get('longitude', 0.0) if len(stream3c) > 0 else 0.0
        station_elv = getattr(stream3c[0].stats, 'coordinates', {}).get('elevation', 0.0) if len(stream3c) > 0 else 0.0
        
        # Extract picks from ClassifyOutput
        picks = classify_output.picks if hasattr(classify_output, 'picks') else []
        
        # Group picks by time to write to CSV
        # SeisBench picks are individual. We'll group them if they are very close or just write them.
        # To match EQCCT style, we'll try to find P and S pairs within a 10s window? 
        # Actually, let's just write them as they come for now, or use a simple grouping.
        
        p_picks = [p for p in picks if getattr(p, 'phase', 'P').upper() == 'P']
        s_picks = [p for p in picks if getattr(p, 'phase', 'P').upper() == 'S']
        
        # Simple pairing: for each P, find the first S that comes after it within 30s
        used_s = set()
        for p in p_picks:
            # Robust attribute extraction for SeisBench Pick objects
            p_time = getattr(p, 'peak_time', getattr(p, 'start_time', getattr(p, 'time', None)))
            p_prob = getattr(p, 'peak_value', getattr(p, 'score', getattr(p, 'value', 0.0)))
            
            if p_time is None:
                continue
            
            match_s = None
            for s in s_picks:
                s_time = getattr(s, 'peak_time', getattr(s, 'start_time', getattr(s, 'time', None)))
                if s not in used_s and s_time and 0 < (s_time - p_time) < 30:
                    match_s = s
                    used_s.add(s)
                    break
            
            if match_s:
                ms_time = getattr(match_s, 'peak_time', getattr(match_s, 'start_time', getattr(match_s, 'time', None)))
                ms_prob = getattr(match_s, 'peak_value', getattr(match_s, 'score', getattr(match_s, 'value', 0.0)))
                s_time_str = ms_time.strftime('%Y-%m-%d %H:%M:%S.%f') if ms_time else ''
                s_prob_str = f"{ms_prob:.6f}"
            else:
                s_time_str = ''
                s_prob_str = ''
            
            predict_writer.writerow([
                station_code,
                network_code,
                station_code,
                0,  # instrument_type
                station_lat,
                station_lon,
                station_elv,
                p_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                f"{p_prob:.6f}",
                s_time_str,
                s_prob_str
            ])
            
        # Write remaining S picks
        for s in s_picks:
            if s not in used_s:
                s_time = getattr(s, 'peak_time', getattr(s, 'start_time', getattr(s, 'time', None)))
                s_prob = getattr(s, 'peak_value', getattr(s, 'score', getattr(s, 'value', 0.0)))
                if s_time:
                    predict_writer.writerow([
                        station_code,
                        network_code,
                        station_code,
                        0,  # instrument_type
                        station_lat,
                        station_lon,
                        station_elv,
                        '',
                        '',
                        s_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        f"{s_prob:.6f}"
                    ])
        
        # If no picks found at all, write one row with station info
        if not picks:
            predict_writer.writerow([
                station_code,
                network_code,
                station_code,
                0,  # instrument_type
                station_lat,
                station_lon,
                station_elv,
                '', '', '', ''
            ])
            
        csvPr_gen.flush()
        csvPr_gen.close()
        
        end_Predicting = monotonic_s()
        delta = (end_Predicting - start_Predicting)
        # Return tuple: (log_message, waveform_load_time) for timing analysis
        return (f"{pos} {station}: Finished the prediction in {round(delta,2)}s. (HP={freqmin}, LP={freqmax}, picks={len(picks)})", waveform_load_time)

    except Exception as exp:
        if 'csvPr_gen' in locals():
            csvPr_gen.close()
        # Return tuple with waveform_load_time if available
        load_time = waveform_load_time if 'waveform_load_time' in locals() else None
        return (f"{pos} {station}: FAILED the prediction. {exp}", load_time)


@ray.remote
def parallel_predict(predict_args, model_actor, gpu=False):
    """
    Uses shared ModelActor for EQCCT inference.
    ``predict_args`` may be ``(pos, station, out_dir, args)`` or 5-tuple with
    Ray ObjectRefs for args and merged mSEED Stream (scmlpick-style).
    """
    import glob
    import shutil
    import csv
    import logging
    from logging.handlers import QueueHandler
    # --- QUIET TF C++/Python LOGS BEFORE ANY TF IMPORT --- 
    # We were getting info messages from TF because we were importing it natively from eqcct_tf_models
    # We need to supress TF first before we import it fully
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 3=ERROR
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # hide oneDNN banner
    if not gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # don't probe CUDA on CPU tasks

    # Python-side TF/absl logging
    try:
        import tensorflow as tf
        try:
            tf.get_logger().setLevel(logging.ERROR)
        except AttributeError:
            # tf.get_logger() not available in some TF configurations
            pass
        try:
            from absl import logging as absl_logging
            absl_logging.set_verbosity(absl_logging.ERROR)
        except Exception:
            pass
    except Exception:
        # If eqcct_tf_models imports TF later, env vars above will still suppress C++ logs.
        pass

    from eqcctpro.eqcct_tf_models import Patches, PatchEncoder, StochasticDepth, PreLoadGeneratorTest, load_eqcct_model

    if len(predict_args) == 5:
        pos, station, out_dir, args_ref, stream_ref = predict_args
        args = ray.get(args_ref) if isinstance(args_ref, ray.ObjectRef) else args_ref
        use_shared_stream = True
    else:
        pos, station, out_dir, args = predict_args
        use_shared_stream = False

    logger = logging.getLogger(f"eqcctpro.worker.{station}")

    # NOTE: Model is shared via model_actor when model_actor is not None

    save_dir = os.path.join(out_dir, str(station)+'_outputs')
    csv_filename = os.path.join(save_dir,'X_prediction_results.csv')

    if os.path.isfile(csv_filename):
        if args['overwrite']:
            shutil.rmtree(save_dir)
        else:
            return (f"{pos} {station}: Skipped (already exists - overwrite=False).", 0.0)

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
    
    start_Predicting = monotonic_s()

    # ===== TIMING: Track waveform loading time =====
    waveform_load_start = monotonic_s()
    try:
        if use_shared_stream:
            full_st = ray.get(stream_ref) if isinstance(stream_ref, ray.ObjectRef) else stream_ref
            st_sel = _stream_select_for_station_task(full_st, station)
            if len(st_sel) == 0:
                return (f"{pos} {station}: FAILED no traces for station in shared Stream.", None)
            packed = _eqcct_stream_to_nparray(args, st_sel, station, files_list=None)
            if packed is None:
                return (f"{pos} {station}: FAILED reading mSEED (corrupted or empty files).", None)
            meta, data_set, hp, lp = packed
        else:
            files_list = glob.glob(f"{args['input_dir']}/{station}/*mseed")
            meta, data_set, hp, lp = _mseed2nparray(args, files_list, station)
            if meta is None:
                return (f"{pos} {station}: FAILED reading mSEED (corrupted or empty files).", None)
    except Exception as e:
        err_msg = f"FAILED reading mSEED: {str(e)}" if str(e) else "FAILED reading mSEED (corrupted or empty files)."
        return (f"{pos} {station}: {err_msg}", None)
    waveform_load_time = monotonic_s() - waveform_load_start

    try:
        # Load model ONLY if we don't have a shared model_actor (RIPPER mode)
        if model_actor is None:
            logger.info("RIPPER MODE: Loading EQCCT model inside task...")
            # Configure GPU for this specific task process
            if gpu:
                from eqcctpro.tools import tf_environ
                # Set a per-task VRAM limit if provided in args
                vram_limit = args.get('gpu_memory_limit_mb')
                tf_environ(gpu_id=0, vram_limit_mb=vram_limit, use_gpu=True, logger=logger)
            
            from eqcctpro.eqcct_tf_models import load_eqcct_model
            model = load_eqcct_model(args['p_model'], args['s_model'])
            logger.info("Model loaded inside task.")
            
            params_pred = {'batch_size': args["batch_size"], 'norm_mode': args["normalization_mode"]}
            pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)
            predP, predS = model.predict(pred_generator, verbose=0)
        else:
            # Standard mode: Use the shared ModelActor
            params_pred = {'batch_size': args["batch_size"], 'norm_mode': args["normalization_mode"]}
            pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)
            
            # USE THE SHARED MODEL ACTOR INSTEAD OF LOADING MODEL
            # predP, predS = ray.get(model_actor.predict.remote(pred_generator))\
            predP, predS = ray.get(model_actor.predict_from_arrays.remote(
                                meta["trace_start_time"], data_set, args["batch_size"], args["normalization_mode"]))
        
        detection_memory = []
        prob_memory = []
        for ix in range(len(predP)):
            Ppicks, Pprob = _picker(args, predP[ix,:, 0])   
            Spicks, Sprob = _picker(args, predS[ix,:, 0], 'S_threshold')

            detection_memory, prob_memory = _output_writter_prediction(
                meta, csvPr_gen, Ppicks, Pprob, Spicks, Sprob, 
                detection_memory, prob_memory, predict_writer, ix, len(predP), len(predS)
            )
                                        
        end_Predicting = monotonic_s()
        delta = (end_Predicting - start_Predicting)
        # Return tuple: (log_message, waveform_load_time) for timing analysis
        return (f"{pos} {station}: Finished the prediction in {round(delta,2)}s. (HP={hp}, LP={lp})", waveform_load_time)

    except Exception as exp:
        # Return tuple with waveform_load_time if available
        load_time = waveform_load_time if 'waveform_load_time' in locals() else None
        return (f"{pos} {station}: FAILED the prediction. {exp}", load_time)


# =====================================================================
# RIPPER MODE FUNCTIONS - Task-based approach (old methodology)
# These functions load the model inside each task, then release it.
# This allows dynamic GPU memory sharing but has model loading overhead.
#
# Task scheduling in mseed_predictor (ripper=True) follows the same pattern as
# scmlpick ``run_picker`` (scmlpick/seiscomp/bin/scmlpick): seed up to
# ``max_tasks_queue`` in-flight ``ray.remote`` calls, then ``ray.wait`` with
# ``num_returns=1`` and backfill the queue while more jobs remain.
# =====================================================================


def _run_ripper_parallel_queue_scmlpick(
    *,
    logger,
    tasks_predictor: list,
    max_tasks_queue: int,
    submit_index,
):
    """
    Bounded Ripper task pool matching scmlpick's run_picker loop:
    prefill min(max_tasks_queue, total), drain with ray.wait(..., 1), backfill.
    ``submit_index(i)`` must return an ObjectRef for tasks_predictor[i].
    """
    total_tasks = len(tasks_predictor)
    model_load_times: list = []
    waveform_load_times: list = []

    if total_tasks == 0:
        return model_load_times, waveform_load_times

    tasks_queue: list = []
    idx_iter = iter(range(total_tasks))

    for _ in range(min(max_tasks_queue, total_tasks)):
        try:
            i = next(idx_iter)
            tasks_queue.append(submit_index(i))
        except StopIteration:
            break

    while tasks_queue:
        tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
        for finished_task in tasks_finished:
            try:
                result = ray.get(finished_task)
            except ray.exceptions.RayTaskError as e:
                logger.warning("RIPPER task failed: %s", e.as_instanceof_cause())
                continue
            except ray.exceptions.RayError as e:
                logger.warning("RIPPER Ray error: %s", e)
                continue
            log_entry, ml_time, wf_time = result
            logger.info("%s", log_entry)
            if ml_time is not None:
                model_load_times.append(ml_time)
            if wf_time is not None:
                waveform_load_times.append(wf_time)
        try:
            while len(tasks_queue) < max_tasks_queue:
                i = next(idx_iter)
                tasks_queue.append(submit_index(i))
        except StopIteration:
            pass

    return model_load_times, waveform_load_times


@ray.remote(max_calls=1, max_retries=1)
def ripper_parallel_predict_eqcct(predict_args, gpu=False, gpu_memory_limit_mb=None):
    """
    RIPPER MODE: Old task-based parallel_predict for EQCCT models.
    Each task loads the model, runs prediction, and releases it.
    This allows more flexible GPU memory sharing than the ModelActor approach.
    
    Args:
        predict_args: ``(pos, station, out_dir, args)`` or
            ``(pos, station, out_dir, args_ref, stream_ref)`` (Ray ObjectRefs, scmlpick-style).
        gpu: Whether to use GPU
        gpu_memory_limit_mb: VRAM limit per task in MB
    """
    import glob
    import shutil
    import csv
    import logging
    import sys
    
    # --- QUIET TF C++/Python LOGS BEFORE ANY TF IMPORT --- 
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 3=ERROR
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # hide oneDNN banner
    
    if gpu is True: 
        # Use unified tf_environ for stable GPU memory management
        from eqcctpro.tools import tf_environ
        try:
            # Ripper mode tasks see 1 fractional GPU each (via Ray scheduling)
            # so we just initialize that one GPU
            tf_environ(
                gpu_id=0, 
                vram_limit_mb=gpu_memory_limit_mb,
                logger=logging.getLogger("eqcctpro.ripper")
            )
        except (RuntimeError, ValueError):
            pass  # Already initialized or logical devices configured
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # don't probe CUDA on CPU tasks

    # Python-side TF/absl logging
    try:
        import tensorflow as tf
        try:
            tf.get_logger().setLevel(logging.ERROR)
        except AttributeError:
            pass
        try:
            from absl import logging as absl_logging
            absl_logging.set_verbosity(absl_logging.ERROR)
        except Exception:
            pass
    except Exception:
        pass

    from eqcctpro.eqcct_tf_models import Patches, PatchEncoder, StochasticDepth, PreLoadGeneratorTest, load_eqcct_model

    if len(predict_args) == 5:
        pos, station, out_dir, args_ref, stream_ref = predict_args
        args = ray.get(args_ref) if isinstance(args_ref, ray.ObjectRef) else args_ref
        use_shared_stream = True
    else:
        pos, station, out_dir, args = predict_args
        use_shared_stream = False

    # RIPPER MODE: Load the model inside this task (old approach)
    # ===== TIMING: Track model load time for ripper mode analysis =====
    model_load_start = monotonic_s()
    model = load_eqcct_model(args["p_model"], args["s_model"])
    model_load_time = monotonic_s() - model_load_start
    
    save_dir = os.path.join(out_dir, str(station)+'_outputs')
    csv_filename = os.path.join(save_dir,'X_prediction_results.csv')

    if os.path.isfile(csv_filename):
        if args['overwrite']:
            shutil.rmtree(save_dir)
        else:
            # Return 3-tuple for consistency with caller unpacking logic
            return (f"{pos} {station}: Skipped (already exists - overwrite=False).", model_load_time, 0.0)

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
    
    start_Predicting = monotonic_s()

    # ===== TIMING: Track waveform loading time =====
    waveform_load_start = monotonic_s()
    try:
        if use_shared_stream:
            full_st = ray.get(stream_ref) if isinstance(stream_ref, ray.ObjectRef) else stream_ref
            st_sel = _stream_select_for_station_task(full_st, station)
            if len(st_sel) == 0:
                return (
                    f"{pos} {station}: FAILED no traces for station in shared Stream.",
                    model_load_time,
                    None,
                )
            packed = _eqcct_stream_to_nparray(args, st_sel, station, files_list=None)
        else:
            files_list = glob.glob(f"{args['input_dir']}/{station}/*mseed")
            packed = _mseed2nparray(args, files_list, station)
        if packed is None:
            return (
                f"{pos} {station}: FAILED reading mSEED (corrupted or empty files).",
                model_load_time,
                None,
            )
        meta, data_set, hp, lp = packed
    except Exception as e:
        err_msg = f"FAILED reading mSEED: {str(e)}" if str(e) else "FAILED reading mSEED (corrupted or empty files)."
        return (f"{pos} {station}: {err_msg}", model_load_time, None)
    waveform_load_time = monotonic_s() - waveform_load_start

    try:
        params_pred = {'batch_size': args["batch_size"], 'norm_mode': args["normalization_mode"]}
        pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)
        
        # RIPPER MODE: Use the locally loaded model directly
        predP, predS = model.predict(pred_generator, verbose=0)
        
        detection_memory = []
        prob_memory = []
        for ix in range(len(predP)):
            Ppicks, Pprob = _picker(args, predP[ix,:, 0])   
            Spicks, Sprob = _picker(args, predS[ix,:, 0], 'S_threshold')

            detection_memory, prob_memory = _output_writter_prediction(
                meta, csvPr_gen, Ppicks, Pprob, Spicks, Sprob, 
                detection_memory, prob_memory, predict_writer, ix, len(predP), len(predS)
            )
                                        
        end_Predicting = monotonic_s()
        delta = (end_Predicting - start_Predicting)
        
        # Clean up model to free GPU memory
        del model
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass
        
        # Return tuple: (log_message, model_load_time, waveform_load_time) for timing analysis
        return (f"{pos} {station}: Finished the prediction in {round(delta,2)}s. (HP={hp}, LP={lp})", model_load_time, waveform_load_time)

    except Exception as exp:
        # Return tuple with available times
        wf_time = waveform_load_time if 'waveform_load_time' in locals() else None
        return (f"{pos} {station}: FAILED the prediction. {exp}", model_load_time, wf_time)


@ray.remote(max_calls=1, max_retries=1)
def ripper_parallel_predict_seisbench(predict_args, gpu=False, gpu_memory_limit_mb=None, 
                                       parent_model_name=None, child_model_name=None,
                                       Detection_threshold=0.3):
    """
    RIPPER MODE: Old task-based parallel_predict for SeisBench models.
    Each task loads the model, runs prediction, and releases it.
    This allows more flexible GPU memory sharing than the ModelActor approach.
    
    Args:
        predict_args: ``(pos, station, out_dir, args)`` or
            ``(pos, station, out_dir, args_ref, stream_ref)`` (Ray ObjectRefs).
        gpu: Whether to use GPU
        gpu_memory_limit_mb: VRAM limit per task in MB (not used for PyTorch, but kept for API compatibility)
        parent_model_name: SeisBench parent model name (e.g., 'PhaseNet')
        child_model_name: SeisBench child model name (e.g., 'original')
        Detection_threshold: Detection threshold for picks
    """
    import glob
    import shutil
    import csv
    import logging
    import sys

    if len(predict_args) == 5:
        pos, station, out_dir, args_ref, stream_ref = predict_args
        args = ray.get(args_ref) if isinstance(args_ref, ray.ObjectRef) else args_ref
        use_shared_stream = True
    else:
        pos, station, out_dir, args = predict_args
        use_shared_stream = False

    # RIPPER MODE: Load the SeisBench model inside this task using SeisBenchModels class
    from eqcctpro.seisbench_models import SeisBenchModels, mseed2stream_3c, process_raw_station_stream_3c
    import torch
    
    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
    
    # ===== TIMING: Track model load time for ripper mode analysis =====
    model_load_start = monotonic_s()
    
    # Create and load the model (skip validation — driver already verified the model name)
    model_wrapper = SeisBenchModels(parent_model_name, child_model_name, validate_pretrained=False)
    model_wrapper.load_model()
    
    # Move model to device if using GPU
    if gpu and torch.cuda.is_available():
        try:
            if hasattr(model_wrapper.model, 'to'):
                model_wrapper.model.to(device)
        except Exception:
            pass
    if gpu:
        cuda_synchronize_best_effort()
    model_load_time = monotonic_s() - model_load_start
    
    save_dir = os.path.join(out_dir, str(station)+'_outputs')
    csv_filename = os.path.join(save_dir,'X_prediction_results.csv')

    if os.path.isfile(csv_filename):
        if args['overwrite']:
            shutil.rmtree(save_dir)
        else:
            # Return 3-tuple for consistency with caller unpacking logic
            return (f"{pos} {station}: Skipped (already exists - overwrite=False).", model_load_time, 0.0)

    os.makedirs(save_dir, exist_ok=True)
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
    
    start_Predicting = monotonic_s()

    # ===== TIMING: Track waveform loading time =====
    waveform_load_start = monotonic_s()
    try:
        if use_shared_stream:
            full_st = ray.get(stream_ref) if isinstance(stream_ref, ray.ObjectRef) else stream_ref
            st_sel = _stream_select_for_station_task(full_st, station)
            if len(st_sel) == 0:
                csvPr_gen.close()
                return (
                    f"{pos} {station}: FAILED no traces for station in shared Stream.",
                    model_load_time,
                    None,
                )
            stream, freqmin, freqmax = process_raw_station_stream_3c(args, st_sel, station)
        else:
            files_list = glob.glob(f"{args['input_dir']}/{station}/*mseed")
            result = mseed2stream_3c(args, files_list, station)
            if result is None:
                csvPr_gen.close()
                return (f"{pos} {station}: FAILED reading mSEED (no valid 3C stream).", model_load_time, None)
            stream, freqmin, freqmax = result
    except Exception as e:
        csvPr_gen.close()
        err_msg = f"FAILED reading mSEED: {str(e)}" if str(e) else "FAILED reading mSEED (corrupted or empty files)."
        return (f"{pos} {station}: {err_msg}", model_load_time, None)
    waveform_load_time = monotonic_s() - waveform_load_start

    try:
        # Run SeisBench model prediction using the model wrapper's classify method
        # IMPORTANT: strict=False and flexible_horizontal_components=True are needed
        # to handle streams that don't perfectly match expected channel names
        classify_output = model_wrapper.classify(
            stream,
            P_threshold=args.get('P_threshold', 0.3),
            S_threshold=args.get('S_threshold', 0.3),
            Detection_threshold=Detection_threshold,
            strict=False,
            flexible_horizontal_components=True,
        )
        if gpu:
            cuda_synchronize_best_effort()
        
        # Extract picks from ClassifyOutput
        picks = classify_output.picks if hasattr(classify_output, 'picks') else []
        
        # Process picks and write to CSV
        for pick in picks:
            pick_time = getattr(pick, 'peak_time', getattr(pick, 'start_time', getattr(pick, 'time', None)))
            pick_prob = getattr(pick, 'peak_value', getattr(pick, 'score', getattr(pick, 'value', 0.0)))
            pick_phase = getattr(pick, 'phase', 'P').upper()
            
            if pick_time is not None:
                predict_writer.writerow([
                    args['input_dir'].split('/')[-1],  # file_name
                    '',  # network
                    station,  # station
                    '',  # instrument_type
                    '',  # station_lat
                    '',  # station_lon
                    '',  # station_elv
                    str(pick_time) if pick_phase == 'P' else '',  # p_arrival_time
                    f"{pick_prob:.6f}" if pick_phase == 'P' else '',  # p_probability
                    str(pick_time) if pick_phase == 'S' else '',  # s_arrival_time
                    f"{pick_prob:.6f}" if pick_phase == 'S' else ''  # s_probability
                ])
        csvPr_gen.flush()
        csvPr_gen.close()
                                        
        end_Predicting = monotonic_s()
        delta = (end_Predicting - start_Predicting)
        
        # Clean up model to free GPU memory
        del model_wrapper
        if gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return tuple: (log_message, model_load_time, waveform_load_time) for timing analysis
        return (f"{pos} {station}: Finished the prediction in {round(delta,2)}s. (HP={freqmin}, LP={freqmax}, picks={len(picks)})", model_load_time, waveform_load_time)

    except Exception as exp:
        if 'csvPr_gen' in locals():
            csvPr_gen.close()
        # Return tuple with available times
        ml_time = model_load_time if 'model_load_time' in locals() else None
        wf_time = waveform_load_time if 'waveform_load_time' in locals() else None
        return (f"{pos} {station}: FAILED the prediction. {exp}", ml_time, wf_time)