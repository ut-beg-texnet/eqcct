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
import queue 
import obspy
import psutil
import random
import numbers
import logging
import platform
import traceback
import numpy as np
from .tools import *
from os import listdir
from obspy import UTCDateTime
from datetime import datetime, timedelta 
from logging.handlers import QueueHandler

# Dictionary of VRAM requirements (MB) for SeisBench models
# Format: (parent_model_name, child_model_name): vram_mb
# This will be populated with values provided by the user later
SEISBENCH_MODEL_VRAM_MB = {
    # Example entries:
    ('PhaseNet', 'original'): 2000.0,
    ('EQTransformer', 'stead'): 2500.0,
}

def get_seisbench_model_vram_mb(parent_model_name, child_model_name, default_mb=2000.0):
    """
    Get VRAM requirement for a SeisBench model.
    """
    key = (parent_model_name, child_model_name)
    return SEISBENCH_MODEL_VRAM_MB.get(key, default_mb)

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
              Detection_threshold=0.3): 
    
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

    # Set up logger that will write logs to this native process and add them to the log.queue to be added back to the main logger outside of this Raylet
    # worker logger ships records to driver
    logger = logging.getLogger("eqcctpro.worker")
    logger.setLevel(logging.INFO)
    logger.handlers[:] = []
    logger.propagate = False
    log_handler = QueueHandler(log_queue)
    if log_queue is not None:
        logger.addHandler(log_handler)  # Ray queue supports put()

    # We set up the tf_environ again for the Raylets, who adopt their own import state and TF runtime when created. 
    # We want to ensure that they are configured properly so that they won't die (bad)
    skip_tf = (model_type.lower() != 'eqcct')
    if not use_gpu: 
        tf_environ(gpu_id=-1, intra_threads=intra_threads, inter_threads=inter_threads, logger=logger, skip_tf=skip_tf)
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
        if platform.system() == 'Windows': station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("\\")[-1] != ".DS_Store"]
        else: station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("/")[-1] != ".DS_Store"]
        station_list = sorted(set(station_list))
    except Exception as e:
        logger.info(f"{e}") 
        return # To-Do: Fix so that it has a valid return? 
    # log.write(f"GPU ID: {args['gpu_id']}; Batch size: {args['batch_size']}")
    logger.info(f"------- Data Preprocessing for EQCCTPro -------")
    logger.info(f"{len(station_list)} station(s) in {args['input_dir']}")
    
    if stations2use and stations2use <= len(station_list):  # For System Evaluation Execution
        station_list = random.sample(station_list, stations2use)  # Randomly choose stations from the sample size 
        # log.write(f"Using {len(station_list)} station(s) after selection.")

    if specific_stations is not None: station_list = [x for x in station_list if x in specific_stations] # For "One Use Run" Over a Given Set of Stations (Just Run EQCCTPro on specific_stations)
    else: station_list = station_list # someone put None thinking that they would be able to run the whole directory in one go
    logger.info(f"Using {len(station_list)} selected station(s): {station_list}.") 

    if not station_list or any(looks_like_timechunk_id(x) for x in station_list):
        # Rebuild from the actual contents of the timechunk dir
        station_list = build_station_list_from_dir(args['input_dir'])
        logger.info(f"Station list rebuilt from directory because it contained a timechunk id or was empty.") 

    tasks_predictor = [[f"({i+1}/{len(station_list)})", station_list[i], out_dir, args] for i in range(len(station_list))]
    
    if not tasks_predictor: return
    
    # CREATE MODEL ACTOR(S) - Add this before the task loop
    logger.info(f"Creating model actor(s)...") 
    
    model_type_lower = model_type.lower() if model_type else 'eqcct'
    
    if model_type_lower == 'seisbench':
        # Create SeisBench model actors
        if use_gpu:
            # Get VRAM requirement for this SeisBench model
            model_vram_mb = get_seisbench_model_vram_mb(
                seisbench_parent_model, 
                seisbench_child_model,
                default_mb=2000.0
            )
            # Use max of requested VRAM or model requirement (similar to EQCCT logic)
            model_vram_mb = max(gpu_memory_limit_mb, model_vram_mb) if gpu_memory_limit_mb else model_vram_mb
            
            model_actors = []
            logger.info(f"Using GPUs: {gpu_id}")
            for gpu_idx in gpu_id:
                logger.info(f"Creating SeisBenchModelActor on GPU {gpu_idx} with {model_vram_mb/1024:.2f}GB VRAM requirement...")
                actor = SeisBenchModelActor.options(num_gpus=1, num_cpus=0).remote(
                    parent_model_name=seisbench_parent_model,
                    child_model_name=seisbench_child_model,
                    gpus_to_use=[gpu_idx],
                    use_gpu=True
                )
                try:
                    ray.get(actor.ready.remote())
                except Exception as e:
                    logger.error(f"Failed to create SeisBenchModelActor on GPU {gpu_idx}: {e}")
                    raise
                logger.info(f"SeisBenchModelActor created on GPU {gpu_idx}.")
                model_actors.append(actor)
            logger.info(f"Created {len(model_actors)} GPU-sized SeisBenchModelActor(s).")
        else:
            model_actors = [SeisBenchModelActor.options(num_cpus=1).remote(
                parent_model_name=seisbench_parent_model,
                child_model_name=seisbench_child_model,
                gpus_to_use=False,
                use_gpu=False
            )]
            ray.get(model_actors[0].ready.remote())
            logger.info(f"Created a 1 CPU-sized SeisBenchModelActor")
    else:
        # Create EQCCT model actors (original logic)
    if use_gpu:
        # Allocate more VRAM to model actors (they need to hold the full model)
        # Reserve ~2-3GB per model actor, adjust based on your model size
        model_vram_mb = max(gpu_memory_limit_mb, 3000)  # At least VRAM or 3GB for EQCCT (subject to change) 
        
        # Create one model actor per GPU
        model_actors = []
        logger.info(f"Using GPUs: {gpu_id}")
        for gpu_idx in gpu_id: # gpu_id is a list of GPU IDs and gpu_idx is the current GPU ID in the loop 
            logger.info(f"Creating ModelActor on GPU {gpu_idx} with {model_vram_mb/1024:.2f}GB VRAM limit...")
                actor = ModelActor.options(num_gpus=1, num_cpus=0).remote(gpus_to_use=[gpu_idx], p_model_path=p_model, s_model_path=s_model, gpu_memory_limit_mb=model_vram_mb, use_gpu=True)
            # Wait for __init__ to complete and raise if error
            try:
                ray.get(actor.ready.remote())
            except Exception as e:
                logger.error(f"Failed to create ModelActor on GPU {gpu_idx}: {e}")
                raise
            logger.info(f"ModelActor created on GPU {gpu_idx}.")
            model_actors.append(actor)
            
        logger.info(f"Created {len(model_actors)} GPU-sized ModelActor(s).") 
        # Using CUDA_VISIBLE_DEVICES is not a reliable way to report which physical GPU is being used bc Ray can overwrite, clear, or remap the assigned GPU so that each worker sees them as local indices (often starting from 0)
        logger.info(f"[ModelActor] Model successfully loaded onto {'GPU' if use_gpu else 'CPU'}.") # Better way to log is to use ray.get_gpu_ids()
    else:
        # Create CPU model actor
        model_actors = [ModelActor.options(num_cpus=1).remote(p_model_path=p_model, s_model_path=s_model, gpu_memory_limit_mb=None, use_gpu=False)]
        logger.info(f"Created a 1 CPU-sized ModelActor") 

    # Submit tasks to ray in a queue
    tasks_queue = []
    max_pending_tasks = number_of_concurrent_station_predictions
    logger.info(f"Starting EQCCTPro parallelized waveform processing...") 
    logger.info("")
    start_time = time.time() 
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


    # Concurrent Prediction(s) Parallel Processing
    try: 
        for i in range(len(tasks_predictor)):
            while True:
                # Add new task to queue while max is not reached
                if len(tasks_queue) < max_pending_tasks:
                    # SELECT WHICH MODEL ACTOR TO USE (round-robin across GPUs)
                    model_actor = model_actors[i % len(model_actors)]

                    # Route to appropriate prediction function based on model type
                    if model_type_lower == 'seisbench':
                        # SeisBench models use parallel_predict_seisbench
                        if use_gpu is False:
                            tasks_queue.append(parallel_predict_seisbench.options(num_cpus=0).remote(tasks_predictor[i], model_actor, False))
                        elif use_gpu is True:
                            # Don't allocate GPUs to workers, only to model actors
                            # Use num_cpus=0 to avoid deadlocks when Ray has limited CPUs
                            tasks_queue.append(parallel_predict_seisbench.options(num_cpus=0, num_gpus=0).remote(tasks_predictor[i], model_actor, True))
                    else:
                        # EQCCT models use parallel_predict (original)
                    if use_gpu is False:
                        tasks_queue.append(parallel_predict.options(num_cpus=0).remote(tasks_predictor[i], model_actor, False))
                    elif use_gpu is True:
                        # Don't allocate GPUs to workers, only to model actors
                            # Use num_cpus=0 to avoid deadlocks when Ray has limited CPUs
                            tasks_queue.append(parallel_predict.options(num_cpus=0, num_gpus=0).remote(tasks_predictor[i], model_actor, True))
                    break
                # If there are more tasks than maximum, just process them
                else:
                    tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
                    for finished_task in tasks_finished:
                        log_entry = ray.get(finished_task)
                        logger.info(f'{log_entry}')

        # After adding all the tasks to queue, process what's left
        while tasks_queue:
            tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
            for finished_task in tasks_finished:
                log_entry = ray.get(finished_task)
                logger.info(f'{log_entry}')
        logger.info("")

    except Exception as e:
        # Catch any error in the parallel processing
        logger.error(f"ERROR in parallel processing at {datetime.now()}")
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise  # Re-raise to see the error

    logger.info(f"------- Parallel Station Waveform Processing Complete For {starttime} to {endtime} Timechunk-------")
    end_time = time.time()
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

        # To-Do: Add column for CPU IDs 
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
            "Number of Concurrent Station Tasks": int(number_of_concurrent_station_predictions) if number_of_concurrent_station_predictions is not None else "",
            "Total Run time for Picker (s)": round(end_time - start_time, 6),
            "Model Used": model_used,
            "Trial Success": "",
            "Error Message": str(""),
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
            except RuntimeError as e:
                self.logger.error(f"[ModelActor] Error setting memory limit: {e}")
        
        # Load the model once
        self.logger.info("Importing/load_eqcct_model...")
        from .eqcct_tf_models import load_eqcct_model
        self.model = load_eqcct_model(p_model_path, s_model_path)
        self.logger.info("Model loaded.")
    
    def ready(self):
        """Simple method to check if the actor is ready"""
        return True
    
    def predict(self, data_generator):
        """Perform prediction using the loaded model"""
        return self.model.predict(data_generator, verbose=0)
    
    def predict_from_arrays(self, trace_start_time, data_set, batch_size, norm_mode):
        from .eqcct_tf_models import PreLoadGeneratorTest
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

        # Load the SeisBench model
        self.logger.info("Loading SeisBench model...")
        from .seisbench_models import SeisBenchModels
        self.model_wrapper = SeisBenchModels(parent_model_name, child_model_name)
        self.model_wrapper.load_model()
        
        # Move model to device if using GPU
        if use_gpu:
            try:
                if hasattr(self.model_wrapper.model, 'to'):
                    self.model_wrapper.model.to(self.device)
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
        return self.model_wrapper.classify(
            stream, 
            P_threshold=P_threshold,
            S_threshold=S_threshold,
            Detection_threshold=Detection_threshold,
            **kwargs
        )


@ray.remote
def parallel_predict_seisbench(predict_args, model_actor, gpu=False):
    """
    Prediction function for SeisBench models.
    Uses mseed2stream_3c for preprocessing and SeisBenchModelActor for predictions.
    """
    import glob
    import shutil
    import csv
    import logging
    from logging.handlers import QueueHandler
    from pathlib import Path
    from .seisbench_models import mseed2stream_3c
    
    pos, station, out_dir, args = predict_args
    
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
            return f"{pos} {station}: Skipped (already exists - overwrite=False)."

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
    
    start_Predicting = time.time()
    files_list = glob.glob(f"{args['input_dir']}/{station}/*mseed")
    
    if not files_list:
        csvPr_gen.close()
        return f"{pos} {station}: FAILED - No mSEED files found."
    
    try:
        # Use SeisBench preprocessing
        stream3c, freqmin, freqmax = mseed2stream_3c(args, files_list, station)
    except Exception as e:
        csvPr_gen.close()
        return f"{pos} {station}: FAILED reading mSEED: {str(e)}"

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
        
        end_Predicting = time.time()
        delta = (end_Predicting - start_Predicting)
        return f"{pos} {station}: Finished the prediction in {round(delta,2)}s. (HP={freqmin}, LP={freqmax}, picks={len(picks)})"

    except Exception as exp:
        if 'csvPr_gen' in locals():
            csvPr_gen.close()
        return f"{pos} {station}: FAILED the prediction. {exp}"


@ray.remote
def parallel_predict(predict_args, model_actor, gpu=False):
    """
    Modified to use shared ModelActor instead of loading model per task
    """
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
        tf.get_logger().setLevel(logging.ERROR)
        try:
            from absl import logging as absl_logging
            absl_logging.set_verbosity(absl_logging.ERROR)
        except Exception:
            pass
    except Exception:
        # If eqcct_tf_models imports TF later, env vars above will still suppress C++ logs.
        pass

    from .eqcct_tf_models import Patches, PatchEncoder, StochasticDepth, PreLoadGeneratorTest, load_eqcct_model
    pos, station, out_dir, args = predict_args
    
    # NOTE: We removed the model loading code that was causing OOM errors
    # The model is now shared via the model_actor
    
    save_dir = os.path.join(out_dir, str(station)+'_outputs')
    csv_filename = os.path.join(save_dir,'X_prediction_results.csv')

    if os.path.isfile(csv_filename):
        if args['overwrite']:
            shutil.rmtree(save_dir)
        else:
            return f"{pos} {station}: Skipped (already exists - overwrite=False)."

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
        return f"{pos} {station}: FAILED reading mSEED."

    try:
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
                                        
        end_Predicting = time.time()
        delta = (end_Predicting - start_Predicting)
        return f"{pos} {station}: Finished the prediction in {round(delta,2)}s. (HP={hp}, LP={lp})"

    except Exception as exp:
        return f"{pos} {station}: FAILED the prediction. {exp}"